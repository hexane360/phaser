"""
Gradient-descent solvers

Much of this is adapted from [Optax](https://github.com/google-deepmind/optax),
but modified to use our generic array and pytree utilities.

Optax is released under the Apache license:

> Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
> 
> Licensed under the Apache License, Version 2.0 (the "License");
> you may not use this file except in compliance with the License.
> You may obtain a copy of the License at
> 
>     http://www.apache.org/licenses/LICENSE-2.0
> 
> Unless required by applicable law or agreed to in writing, software
> distributed under the License is distributed on an "AS IS" BASIS,
> WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
> See the License for the specific language governing permissions and
> limitations under the License.
"""

import typing as t

import numpy
from numpy.typing import NDArray

from phaser.hooks.schedule import Schedule, ScheduleLike
from phaser.hooks.solver import GradientSolver, GradientSolverArgs
from phaser.plan import AdamSolverPlan, PolyakSGDSolverPlan, SGDSolverPlan
from phaser.state import ReconsState
from phaser.types import ReconsVar, process_schedule
from phaser.utils import tree
from phaser.utils.num import get_array_module, to_real_dtype, xp_is_torch

from .run import extract_params

OptState: t.TypeAlias = tree.Tree
Params: t.TypeAlias = tree.Tree
Updates: t.TypeAlias = Params


class TransformInitFn(t.Protocol):
    def __call__(self, params: Params) -> OptState:
        ...


class TransformUpdateFn(t.Protocol):
    def __call__(
      self, updates: Updates, state: OptState, params: t.Optional[Params] = None,
      **extra_args: t.Any,
  ) -> t.Tuple[Updates, OptState]:
        ...


class GradientTransformation(t.NamedTuple):
    init: TransformInitFn
    update: TransformUpdateFn


ScheduledSolverState: t.TypeAlias = t.Tuple[t.Any, t.Dict[str, t.Optional[float]]]


class ScheduledSolver(GradientSolver[ScheduledSolverState]):
    def __init__(self, name: str, factory: t.Callable[..., GradientTransformation], hyperparams: t.Mapping[str, ScheduleLike],
                 params: t.Iterable[ReconsVar]):
        self.factory: t.Callable[..., GradientTransformation] = factory
        #self.inner: GradientTransformationExtraArgs = optax.with_extra_args_support(solver)

        self.hyperparams: t.Dict[str, Schedule] = {k: process_schedule(v) for (k, v) in hyperparams.items()}
        self.params: t.FrozenSet[ReconsVar] = frozenset(params)

        self.name: str = name # or self.inner.__class__.__name__

    def init_state(self, sim: ReconsState) -> ScheduledSolverState:
        return (
            None,
            {k: None for (k, v) in self.hyperparams.items()},
        )

    def _resolve(self, hparams: t.Mapping[str, t.Optional[float]]) -> GradientTransformation:
        return self.factory(**{k: hparams[k] for k in self.hyperparams})

    def update_for_iter(self, sim: ReconsState, state: ScheduledSolverState, niter: int) -> ScheduledSolverState:
        hparams_state: t.Dict[str, t.Optional[float]] = {k: v({'state': sim, 'niter': niter}) for (k, v) in self.hyperparams.items()}
        return (
            self._resolve(hparams_state).init(params=extract_params(sim, self.params)) if state[0] is None else state[0],
            hparams_state
        )

    def update(
        self, sim: 'ReconsState', state: ScheduledSolverState, grad: t.Dict[ReconsVar, numpy.ndarray], loss: float,
    ) -> t.Tuple[t.Dict[ReconsVar, numpy.ndarray], ScheduledSolverState]:
        (inner_state, hparams_state) = state
        (updates, inner_state) = self._resolve(hparams_state).update(
            grad, inner_state, params=extract_params(sim, self.params), value=loss, loss=loss
        )
        return (t.cast(t.Dict[ReconsVar, t.Any], updates), (inner_state, hparams_state))


class SGDSolver(ScheduledSolver):
    def __init__(self, args: GradientSolverArgs, props: SGDSolverPlan):
        hparams = {
            'learning_rate': props.learning_rate
        }

        if props.momentum is not None:
            hparams['momentum'] = props.momentum
            def factory(**kwargs: t.Any) -> GradientTransformation:
                return chain(
                    trace(kwargs['momentum'], props.nesterov),
                    scale_by_learning_rate(kwargs['learning_rate']),
                )
        else:
            def factory(**kwargs: t.Any) -> GradientTransformation:
                return scale_by_learning_rate(kwargs['learning_rate'])

        super().__init__('sgd', factory, hparams, args['params'])


class AdamSolver(ScheduledSolver):
    def __init__(self, args: GradientSolverArgs, props: AdamSolverPlan):
        hparams = {
            'learning_rate': props.learning_rate
        }

        def factory(**kwargs) -> GradientTransformation:
            return chain(
                scale_by_adam(props.b1, props.b2, props.eps, props.eps_root, nesterov=props.nesterov),
                scale_by_learning_rate(learning_rate=kwargs['learning_rate']),
            )

        super().__init__('adam', factory, hparams, args['params'])


class PolyakSGDSolver(ScheduledSolver):
    def __init__(self, args: GradientSolverArgs, props: PolyakSGDSolverPlan):
        hparams = {
            'max_learning_rate': props.max_learning_rate,
            'scaling': props.scaling,
        }

        def factory(**kwargs) -> GradientTransformation:
            return chain(
                scale_by_learning_rate(kwargs['scaling']),
                scale_by_polyak(
                    max_learning_rate=kwargs['max_learning_rate'], f_min=props.f_min,
                    eps=props.eps, #variant='sps',
                )
            )

        super().__init__('polyak_sgd', factory, hparams, args['params'])


def chain(
    *args: GradientTransformation
) -> GradientTransformation:
    init_fns = tuple(arg.init for arg in args)
    update_fns = tuple(arg.update for arg in args)

    def init_fn(params: Params):
        return tuple(fn(params) for fn in init_fns)

    def update_fn(updates, state, params=None, **extra_args):
        new_state = []
        for s, fn in zip(state, update_fns):
            updates, new_s = fn(updates, s, params, **extra_args)
            new_state.append(new_s)
        return updates, tuple(new_state)

    return GradientTransformation(init_fn, update_fn)


def trace(
    decay: float,
    nesterov: bool = False,
    accumulator_dtype: t.Optional[t.Any] = None,
) -> GradientTransformation:

    def init_fn(params):
        return tree.zeros_like(params, dtype=accumulator_dtype)

    def update_fn(updates: Updates, state: Updates, params=None, **extra_args: t.Any):
        del params
        f = lambda g, t: g + decay * t  # noqa: E731
        new_trace = tree.map(
            lambda g, t: None if g is None else f(g, t),
            updates,
            state,
            is_leaf=lambda g: g is None,
        )
        updates = tree.map(f, updates, new_trace) if nesterov else new_trace
        new_trace = tree.cast(new_trace, accumulator_dtype)
        return updates, new_trace

    return GradientTransformation(init_fn, update_fn)


def scale_by_learning_rate(
    learning_rate: float, *,
    flip_sign: bool = False,
) -> GradientTransformation:
    if flip_sign:
        learning_rate *= -1

    def update_fn(updates: Updates, state: None, params=None, **extra_args: t.Any):
        del params
        updates = tree.map(lambda g: learning_rate * g, updates)
        return updates, state

    return GradientTransformation(lambda params: None, update_fn)


class ScaleByAdamState(t.NamedTuple):
    n: NDArray[numpy.int32]  # shape ()
    mu: Updates
    nu: Updates


def scale_by_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: t.Optional[t.Any] = None,
    *,
    nesterov: bool = False,
) -> GradientTransformation:
    def init_fn(params: Params) -> ScaleByAdamState:
        xp = get_array_module(params)
        mu = tree.zeros_like(params, dtype=mu_dtype)  # First moment
        nu = tree.map(lambda x: xp.zeros_like(x, dtype=to_real_dtype(x.dtype)), params)  # Second moment
        return ScaleByAdamState(n=xp.zeros((), dtype=xp.int32), mu=mu, nu=nu)

    def update_fn(
        updates: Updates, state: ScaleByAdamState, params: t.Any = None, **kwargs: t.Any
    ) -> t.Tuple[Updates, ScaleByAdamState]:
        xp = get_array_module(updates)
        del params
        mu = tree.update_moment(updates, state.mu, b1, 1)
        nu = tree.update_moment_per_elem_norm(updates, state.nu, b2, 2)
        n_inc = safe_increment(state.n)

        # HACK: on mps we need to prevent small mu values from returning nan
        if xp_is_torch(xp) and any(
            leaf.device.type == 'mps' for leaf in tree.leaves(updates)
        ):
            mu = tree.map(
                lambda arr: xp.nan_to_num(arr, nan=0.),
                mu, is_leaf=lambda x: x is None
            )

        if nesterov:
            mu_hat = tree.map(
                lambda m, g: b1 * m + (1 - b1) * g,
                tree.bias_correction(mu, b1, safe_increment(n_inc)),
                tree.bias_correction(updates, b1, n_inc),
            )
        else:
            mu_hat = tree.bias_correction(mu, b1, n_inc)

        nu_hat = tree.bias_correction(nu, b2, n_inc)
        updates = tree.map(
            lambda m, v: None if m is None else m / (xp.sqrt(v + eps_root) + eps),
            mu_hat,
            nu_hat,
            is_leaf=lambda x: x is None,
        )
        mu = tree.cast(mu, mu_dtype)
        return updates, ScaleByAdamState(n=n_inc, mu=mu, nu=nu)

    return GradientTransformation(init_fn, update_fn)


def scale_by_polyak(
    f_min: float = 0.0,
    max_learning_rate: float = 1.0,
    eps: float = 0.0
) -> GradientTransformation:
    def update_fn(
        updates: Updates, state: None, params: t.Any = None, *, value: float, **kwargs: t.Any
    ):
        del params
        del kwargs
        xp = get_array_module(updates)
        grad_sq_norm = tree.squared_norm(updates)
        gap = xp.array(value - f_min).astype(grad_sq_norm.dtype)
        step = xp.where(
            grad_sq_norm + eps <= xp.finfo(float).eps,
            xp.array(0.0),
            xp.minimum(gap / (grad_sq_norm + eps), max_learning_rate),
        )
        updates = tree.scale(step, updates)
        return updates, state

    return GradientTransformation(lambda params: None, update_fn)


def safe_increment(n: NDArray[numpy.int32]) -> NDArray[numpy.int32]:
    xp = get_array_module(n)

    max_value = xp.iinfo(n.dtype).max
    max_value = xp.array(max_value, dtype=n.dtype)
    return xp.where(
        n < max_value, n + xp.ones_like(n), max_value
    )