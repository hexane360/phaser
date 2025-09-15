import logging
import typing as t

import numpy

from ..types import Dataclass, SimpleFlag, process_schedule
from .hook import Hook

if t.TYPE_CHECKING:
    from ..state import ReconsState


class FlagArgs(t.TypedDict):
    state: 'ReconsState'
    niter: int


class FlagHook(Hook[FlagArgs, bool]):
    known = {}


class ScheduleHook(Hook[FlagArgs, float]):
    known = {}


Flag: t.TypeAlias = t.Callable[['FlagArgs'], bool]
Schedule: t.TypeAlias = t.Callable[['FlagArgs'], float]
FlagLike: t.TypeAlias = t.Union[bool, SimpleFlag, FlagHook]
ScheduleLike: t.TypeAlias = t.Union[float, ScheduleHook]


# TODO: make these hooks two-step


class ConstantScheduleProps(Dataclass):
    value: float


def constant_schedule(args: FlagArgs, props: ConstantScheduleProps) -> float:
    return props.value


class PiecewiseScheduleProps(Dataclass):
    init: ScheduleLike
    steps: t.Dict[int, ScheduleLike]


def piecewise_schedule(args: FlagArgs, props: PiecewiseScheduleProps) -> float:
    i = args['state'].iter.engine_iter

    for thresh, val in sorted(props.steps.items(), key=lambda t: t[0], reverse=True):
        if i >= thresh:
            return process_schedule(val)(args)

    return process_schedule(props.init)(args)


class ExprScheduleProps(Dataclass):
    expr: str


def expr_schedule(args: FlagArgs, props: ExprScheduleProps) -> float:
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