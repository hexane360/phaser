"""Flag and schedule hooks."""

import logging
import typing as t

import numpy

from ..types import Dataclass, SimpleFlag, process_schedule
from .hook import Hook

if t.TYPE_CHECKING:
    from ..state import ReconsState


class FlagArgs(t.TypedDict):
    """Arguments passed to flag and schedule hooks."""

    state: 'ReconsState'
    """Current reconstruction state."""
    niter: int
    """Total number of iterations of the engine."""


class FlagHook(Hook[FlagArgs, bool]):
    """Hook returning a boolean each iteration."""

    known = {}


class ScheduleHook(Hook[FlagArgs, float]):
    """Hook returning a value each iteration."""

    known = {}


Flag: t.TypeAlias = t.Callable[['FlagArgs'], bool]
"""Callable returning a boolean each iteration."""
Schedule: t.TypeAlias = t.Callable[['FlagArgs'], float]
"""Callable returning a value each iteration."""
FlagLike: t.TypeAlias = t.Union[bool, SimpleFlag, FlagHook]
"""Value convertible to a `Flag`."""
ScheduleLike: t.TypeAlias = t.Union[float, ScheduleHook]
"""Value convertible to a `Schedule`."""


# TODO: make these hooks two-step


class ConstantScheduleProps(Dataclass):
    """Properties for a constant schedule."""

    value: float
    """Value returned every iteration."""


def constant_schedule(args: FlagArgs, props: ConstantScheduleProps) -> float:
    """Return `props.value`."""
    return props.value


class PiecewiseScheduleProps(Dataclass):
    """Properties for a piecewise-constant schedule."""

    init: ScheduleLike
    """Value before the first threshold in `steps`."""
    steps: t.Dict[int, ScheduleLike]
    """Map from engine iteration to the value used from that iteration on."""


def piecewise_schedule(args: FlagArgs, props: PiecewiseScheduleProps) -> float:
    """Return the value for the largest threshold in `steps` that is at most the current engine iteration, or `init`."""
    i = args['state'].iter.engine_iter

    for thresh, val in sorted(props.steps.items(), key=lambda t: t[0], reverse=True):
        if i >= thresh:
            return process_schedule(val)(args)

    return process_schedule(props.init)(args)


class ExprScheduleProps(Dataclass):
    """Properties for an expression schedule."""

    expr: str
    """Python expression evaluated each iteration. Variables: `i` (engine iteration), `iter`, `state`, `niter`, `np`."""


def expr_schedule(args: FlagArgs, props: ExprScheduleProps) -> float:
    """Return the value of `props.expr` as a float."""
    val = float(eval(props.expr, {
        'i': args['state'].iter.engine_iter,
        'iter': args['state'].iter,
        'state': args['state'],
        'niter': args['niter'],
        'np': numpy,
    }))
    logging.getLogger(__name__).debug(f"expr_schedule expr: {props.expr} value: {val}")

    return val


ScheduleHook.known['constant'] = ('phaser.hooks.schedule:constant_schedule', ConstantScheduleProps)
ScheduleHook.known['piecewise'] = ('phaser.hooks.schedule:piecewise_schedule', PiecewiseScheduleProps)
ScheduleHook.known['expr'] = ('phaser.hooks.schedule:expr_schedule', ExprScheduleProps)