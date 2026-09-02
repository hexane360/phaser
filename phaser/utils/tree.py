import dataclasses
import functools
import typing as t

import numpy
from numpy.typing import ArrayLike, DTypeLike, NDArray
from typing_extensions import Self, dataclass_transform

T = t.TypeVar('T')
Leaf: t.TypeAlias = t.Any
Tree: t.TypeAlias = t.Any
field = dataclasses.field


class TreeSpec(t.Protocol):
    @property
    def num_leaves(self) -> int:
        ...

    @property
    def num_nodes(self) -> int:
        ...

    def unflatten(self, leaves: t.Iterable[Leaf], /) -> Tree:
        ...

    def flatten_up_to(self, xs: Tree, /) -> t.List[Tree]:
        ...

    def __eq__(self, other: Self, /) -> bool: # pyright: ignore[reportIncompatibleMethodOverride]
        ...

    def __ne__(self, other: Self, /) -> bool: # pyright: ignore[reportIncompatibleMethodOverride]
        ...

class Key(t.Protocol):
    def __hash__(self) -> int:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __str__(self) -> str:
        ...

class GetAttrKey(Key, t.Protocol):
    @property
    def name(self) -> str:
        ...


KeyPath: t.TypeAlias = t.Tuple[Key, ...]


def flatten(
    tree: Tree,
    is_leaf: t.Optional[t.Callable[..., t.Any]] = None,
) -> t.Tuple[t.List[Leaf], TreeSpec]:
    from phaser.utils.num import is_torch

    if is_torch(tree):
        from torch.utils._pytree import tree_flatten  # type: ignore
        return tree_flatten(tree, is_leaf)

    import jax.tree  # type: ignore
    return jax.tree.flatten(tree, is_leaf)


def flatten_with_path(
    tree: Tree,
    is_leaf: t.Optional[t.Callable[..., t.Any]] = None,
) -> t.Tuple[t.List[t.Tuple[KeyPath, Leaf]], TreeSpec]:
    from phaser.utils.num import is_torch

    if is_torch(tree):
        from torch.utils._pytree import tree_flatten_with_path  # type: ignore
        return tree_flatten_with_path(tree, is_leaf)  # type: ignore

    from jax.tree_util import tree_flatten_with_path
    return tree_flatten_with_path(tree, is_leaf)


def unflatten(
    leaves: t.Iterable[t.Any],
    treespec: TreeSpec
) -> Tree:
    try:
        from torch.utils._pytree import TreeSpec
        if isinstance(treespec, TreeSpec):
            return treespec.unflatten(leaves)
    except ImportError:
        pass
    try:
        from jax.tree_util import PyTreeDef
        if isinstance(treespec, PyTreeDef):
            return treespec.unflatten(leaves)
    except ImportError:
        pass

    raise TypeError(
        f"tree_unflatten expected `treespec` to be a TreeSpec, "
        f"got item of type {type(treespec)} instead."
    )


def map(
    f: t.Callable[..., t.Any],
    tree: Tree,
    *rest: Tree,
    is_leaf: t.Optional[t.Callable[..., t.Any]] = None,
) -> t.Any:
    from phaser.utils.num import is_torch

    if is_torch(tree):
        from torch.utils._pytree import tree_map  # type: ignore
        return tree_map(f, tree, *rest, is_leaf=is_leaf)

    import jax.tree  # type: ignore
    return jax.tree.map(f, tree, *rest, is_leaf=is_leaf)


def reduce(
    f: t.Callable[[T, t.Any], T], tree: Tree, initializer: T, *,
    is_leaf: t.Optional[t.Callable[..., t.Any]] = None,
) -> T:
    return functools.reduce(f, leaves(tree, is_leaf=is_leaf), initializer)


def sum(tree: Tree) -> numpy.ndarray:
    from phaser.utils.num import get_array_module

    xp = get_array_module(tree)
    sums = map(xp.sum, tree)
    return reduce(lambda lhs, rhs: lhs + rhs, sums, initializer=0)


def map_with_path(
    f: t.Callable[..., t.Any],
    tree: Tree,
    *rest: Tree,
    is_leaf: t.Optional[t.Callable[..., t.Any]] = None,
) -> t.Any:
    from phaser.utils.num import is_torch

    if is_torch(tree):
        from torch.utils._pytree import tree_map_with_path  # type: ignore

        def wrapper(path: KeyPath, *leaves: t.Any):
            return f(tuple(path), *leaves) 

        return tree_map_with_path(wrapper, tree, *rest, is_leaf=is_leaf)

    from jax.tree_util import tree_map_with_path  # type: ignore
    return tree_map_with_path(f, tree, *rest, is_leaf=is_leaf)


def grad(
    f: t.Callable,
    argnums: t.Union[int, t.Tuple[int, ...]] = 0,
    has_aux: bool = False, *, xp: t.Optional[t.Any] = None,
    sign: float = 1.0,
) -> t.Callable[..., Tree]:
    from phaser.utils.num import xp_is_jax, xp_is_torch

    if xp is None or xp_is_jax(xp):
        import jax  # type: ignore
        jax_f = jax.grad(f, argnums, has_aux=has_aux)

        @functools.wraps(jax_f)
        def jax_wrapper(*args: t.Any, **kwargs: t.Any) -> Tree:
            (grad, aux) = jax_f(*args, **kwargs) if has_aux else (jax_f(*args, **kwargs), None)
            # conjugate to get Wirtinger derivative (not required on torch)
            grad = map(lambda arr: arr.conj() * sign, grad, is_leaf=lambda x: x is None)
            return (grad, aux) if has_aux else grad

        return jax_wrapper

    if not xp_is_torch(xp):
        raise ValueError("`grad` is only supported for backends 'jax' and 'torch'")

    import torch.func  # type: ignore
    from torch.utils._pytree import tree_flatten, tree_unflatten  # type: ignore

    # hack to hide non-tensor aux types from torch.func
    box: t.List[t.Tuple[t.Any, t.List[bool], t.List[t.Any]]] = []

    @functools.wraps(f)
    def tensor_only_aux(*args: t.Any, **kwargs: t.Any) -> t.Tuple[t.Any, t.List[t.Any]]:
        (out, aux) = f(*args, **kwargs)
        (leaves, spec) = tree_flatten(aux)
        mask = [isinstance(leaf, torch.Tensor) for leaf in leaves]
        box.append((spec, mask, [leaf for (leaf, m) in zip(leaves, mask) if not m]))
        return (out, [leaf for (leaf, m) in zip(leaves, mask) if m])

    torch_f = torch.func.grad(tensor_only_aux if has_aux else f, argnums, has_aux=has_aux)

    @functools.wraps(torch_f)
    def torch_wrapper(*args: t.Any, **kwargs: t.Any) -> Tree:
        # torch conjugates automatically
        if not has_aux:
            return map(lambda arr: arr * sign, torch_f(*args, **kwargs), is_leaf=lambda x: x is None)

        (grad, aux_leaves) = torch_f(*args, **kwargs)
        (spec, mask, others) = box.pop()
        (tensors, rest) = (iter(aux_leaves), iter(others))
        aux = tree_unflatten([next(tensors) if m else next(rest) for m in mask], spec)
        return (map(lambda arr: arr * sign, grad, is_leaf=lambda x: x is None), aux)

    return torch_wrapper


def value_and_grad(
    f: t.Callable,
    argnums: t.Union[int, t.Tuple[int, ...]] = 0,
    has_aux: bool = False, *, xp: t.Optional[t.Any] = None,
    sign: float = 1.0,
) -> t.Callable[..., t.Tuple[Tree, Tree]]:
    from phaser.utils.num import xp_is_jax, xp_is_torch

    if xp is None or xp_is_jax(xp):
        import jax  # type: ignore
        f = jax.value_and_grad(f, argnums, has_aux=has_aux)

        @functools.wraps(f)
        def jax_wrapper(*args: t.Any, **kwargs: t.Any) -> t.Tuple[Tree, Tree]:
            (value, grad) = f(*args, **kwargs)
            # conjugate to get Wirtinger derivative, multiply by sign
            grad = map(lambda arr: arr.conj() * sign, grad, is_leaf=lambda x: x is None)
            return (value, grad)

        return jax_wrapper

    if not xp_is_torch(xp):
        raise ValueError("`value_and_grad` is only supported for backends 'jax' and 'torch'")

    import torch.func  # type: ignore
    f = torch.func.grad_and_value(f, argnums, has_aux=has_aux)

    @functools.wraps(f)
    def torch_wrapper(*args: t.Any, **kwargs: t.Any) -> t.Tuple[Tree, Tree]:
        # flip order of return values
        (grad, value) = f(*args, **kwargs)
        # multiply by sign
        grad = map(lambda arr: arr * sign, grad, is_leaf=lambda x: x is None)
        return (value, grad)

    return torch_wrapper


def leaves(
    tree: Tree,
    is_leaf: t.Optional[t.Callable[..., t.Any]] = None,
) -> t.List[Leaf]:
    return flatten(tree, is_leaf)[0]


def structure(
    tree: Tree,
    is_leaf: t.Optional[t.Callable[..., t.Any]] = None,
) -> TreeSpec:
    return flatten(tree, is_leaf)[1]


def leaves_with_path(
    tree: Tree,
    is_leaf: t.Optional[t.Callable[..., t.Any]] = None,
) -> t.List[t.Tuple[KeyPath, Leaf]]:
    return flatten_with_path(tree, is_leaf)[0]


def zeros_like(
    tree: Tree, dtype: t.Optional[DTypeLike] = None,
) -> Tree:
    from phaser.utils.num import get_array_module
    xp = get_array_module(tree)
    kwargs: t.Dict[str, t.Any] = {'dtype': dtype} if dtype is not None else {}
    return map(lambda x: xp.zeros_like(x, **kwargs), tree)


def ones_like(
    tree: Tree, dtype: t.Optional[DTypeLike] = None,
) -> Tree:
    from phaser.utils.num import get_array_module
    xp = get_array_module(tree)
    kwargs: t.Dict[str, t.Any] = {'dtype': dtype} if dtype is not None else {}
    return map(lambda x: xp.ones_like(x, **kwargs), tree)


def full_like(
    tree: Tree, fill_value: ArrayLike,
    dtype: t.Optional[DTypeLike] = None,
) -> Tree:
    from phaser.utils.num import get_array_module
    xp = get_array_module(tree)
    kwargs: t.Dict[str, t.Any] = {'dtype': dtype} if dtype is not None else {}
    return map(lambda x: xp.full_like(x, fill_value, **kwargs), tree)


def cast(
    tree: Tree, dtype: t.Optional[DTypeLike],
) -> Tree:
    if dtype is None:
        return tree
    return map(lambda x: x.astype(dtype), tree)


def clip(
    tree: Tree,
    min_value: t.Optional[ArrayLike] = None,
    max_value: t.Optional[ArrayLike] = None,
) -> Tree:
    from phaser.utils.num import get_array_module
    xp = get_array_module(tree)
    return map(lambda x: xp.clip(x, min_value, max_value), tree)


def conj(
    tree: Tree
) -> Tree:
    from phaser.utils.num import get_array_module
    xp = get_array_module(tree)
    return map(xp.conj, tree)


def update_moment(updates: Tree, moments: Tree, decay: float, order: int) -> Tree:
    return map(
        lambda g, t: (
            (1 - decay) * (g**order) + decay * t
            if g is not None else None
        ),
        updates,
        moments,
        is_leaf=lambda x: x is None,
    )


def update_moment_per_elem_norm(updates: Tree, moments: Tree, decay: float, order: int) -> Tree:
    from phaser.utils.num import abs2, get_array_module
    xp = get_array_module(updates, moments)

    def orderth_norm(g):
        if xp.isrealobj(g):
            return g ** order

        half_order = order / 2
        # JAX generates different HLO for int and float `order`
        if half_order.is_integer():
            half_order = int(half_order)
        return abs2(g) ** half_order

    return map(
        lambda g, t: (
            (1 - decay) * orderth_norm(g) + decay * t if g is not None else None
        ),
        updates,
        moments,
        is_leaf=lambda x: x is None,
    )


def bias_correction(moment: Tree, decay: float, count: t.Union[int, NDArray[numpy.integer]]) -> Tree:
    bias_correction = t.cast(NDArray[numpy.floating], 1 - decay**count)
    return map(lambda t: t / bias_correction.astype(t.dtype), moment)


def scale(
    scalar: t.Union[float, numpy.floating, NDArray[numpy.floating]],
    tree: Tree
) -> Tree:
    return map(lambda x: scalar * x, tree)


def squared_norm(
    tree: Tree
) -> NDArray[numpy.floating]:
    return sum(map(lambda x: x**2, tree))


@t.overload
@dataclass_transform(kw_only_default=False, frozen_default=False)
def tree_dataclass(cls: t.Type[T], /, *,
    init: bool = True, kw_only: bool = False, frozen: bool = False,
    static_fields: t.Sequence[str] = (), drop_fields: t.Sequence[str] = (),
) -> t.Type[T]:
    ...

@t.overload
@dataclass_transform(kw_only_default=False, frozen_default=False)
def tree_dataclass(*,
    init: bool = True, kw_only: bool = False, frozen: bool = False,
    static_fields: t.Sequence[str] = (), drop_fields: t.Sequence[str] = (),
) -> t.Callable[[t.Type[T]], t.Type[T]]:
    ...

def tree_dataclass(cls: t.Optional[t.Type[T]] = None, /, *,
    init: bool = True, kw_only: bool = False, frozen: bool = False,
    static_fields: t.Sequence[str] = (), drop_fields: t.Sequence[str] = (),
) -> t.Union[t.Type[T], t.Callable[[t.Type[T]], t.Type[T]]]:
    if cls is None:
        return lambda cls: tree_dataclass(cls, init=init, kw_only=kw_only, frozen=frozen,
                                         static_fields=static_fields, drop_fields=drop_fields)

    cls = dataclasses.dataclass(init=init, kw_only=kw_only, frozen=frozen)(cls)
    _register_dataclass(cls, static_fields=static_fields, drop_fields=drop_fields)
    return cls


def _register_dataclass(cls: type, static_fields: t.Sequence[str], drop_fields: t.Sequence[str]):
    fields = dataclasses.fields(cls)
    field_names = {field.name for field in fields}

    if (extra := set(static_fields).difference(field_names)):
        raise ValueError(f"Unknown field(s) passed to 'static_fields': {', '.join(map(repr, extra))}")
    if (extra := set(drop_fields).difference(field_names)):
        raise ValueError(f"Unknown field(s) passed to 'drop_fields': {', '.join(map(repr, extra))}")

    data_fields = tuple(field_names.difference(static_fields).difference(drop_fields))

    def make_flatten_with_keys(
        key_type: t.Callable[[str], Key]
    ) -> t.Callable[[t.Any], t.Tuple[t.List[t.Tuple[Key, t.Any]], t.Hashable]]:
        def flatten_with_keys(x: t.Any, /) -> tuple[list[tuple[Key, t.Any]], t.Hashable]:
            meta = tuple(getattr(x, name) for name in static_fields)
            trees = list((key_type(name), getattr(x, name)) for name in data_fields)
            return trees, meta

        return flatten_with_keys

    def unflatten(meta: t.Hashable, trees: t.Iterable[t.Any], /) -> t.Any:
        if not isinstance(meta, tuple):
            raise TypeError
        static_args = dict(zip(static_fields, meta, strict=True))
        data_args = dict(zip(data_fields, trees, strict=True))
        return cls(**static_args, **data_args)

    def flatten(x: t.Any, /) -> tuple[list[t.Any], t.Hashable]:
        hashed = tuple(getattr(x, name) for name in static_fields)
        trees = list(getattr(x, name) for name in data_fields)
        return trees, hashed

    def _register_jax():
        from jax.tree_util import GetAttrKey, register_pytree_with_keys

        flatten_with_keys = make_flatten_with_keys(GetAttrKey)
        register_pytree_with_keys(cls, flatten_with_keys, unflatten, flatten)

    def _register_torch():
        from torch.utils._pytree import GetAttrKey, register_pytree_node

        flatten_with_keys = make_flatten_with_keys(GetAttrKey)
        register_pytree_node(
            cls, flatten, lambda trees, meta: unflatten(meta, trees),
            flatten_with_keys_fn=flatten_with_keys,  # type: ignore
        )

    from phaser.utils.num import _BACKEND_LOADER
    _BACKEND_LOADER._schedule_on_load('jax', _register_jax)
    _BACKEND_LOADER._schedule_on_load('torch', _register_torch)


__all__ = [
    'flatten', 'flatten_with_path', 'unflatten', 'map', 'reduce', 'sum',
    'map_with_path', 'grad', 'value_and_grad', 'leaves', 'structure', 'leaves_with_path',
    'zeros_like', 'ones_like', 'full_like', 'cast', 'clip', 'conj', 'update_moment',
    'update_moment_per_elem_norm', 'bias_correction', 'scale', 'tree_dataclass', 'field',
]