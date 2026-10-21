"""Observers that receive reconstruction progress events."""

import contextlib
import logging
import time
import typing as t
from functools import wraps
from pathlib import Path

from phaser.plan import EnginePlan, ReconsPlan, SaveOptions
from phaser.state import PartialReconsState, ProgressState, ReconsState
from phaser.types import EarlyTermination, flag_any_true, process_flag

if t.TYPE_CHECKING:
    from typing_extensions import Self

    from phaser.hooks.schedule import FlagArgs


P = t.ParamSpec('P')

class Observer(contextlib.AbstractContextManager):
    """Base class for reconstruction observers. Methods are called at the named events and do nothing by default.

    Usable as a context manager, which calls `close` on exit.
    """

    def init_recons(self, plan: ReconsPlan, *, name: str):
        """Called when initialization of a reconstruction plan begins."""

    def start_recons(self, init_state: ReconsState):
        """Called when a reconstruction plan is initialized."""

    def init_engine(
        self, init_state: ReconsState, *, recons_name: str,
        plan: EnginePlan, **kwargs: t.Any
    ):
        """Called when initialization of an engine begins."""

    def start_engine(self, init_state: ReconsState):
        """Called after an engine is initialized, before it starts."""

    def heartbeat(self):
        """Called periodically by the engine."""

    def update_group(self, state: t.Union[ReconsState, PartialReconsState], force: bool = False):
        """Called when a group finishes, with the updated state. `force` is deprecated and ignored."""

    def update_iteration(self, state: ReconsState, i: int, n: int, errors: t.Dict[str, float]):
        """Called when iteration `i` of `n` finishes, with the updated state and errors by name (`'total_loss'` is the total)."""

    def finish_engine(self, state: ReconsState):
        """Called when an engine finishes."""

    def finish_recons(self, state: ReconsState):
        """Called when a reconstruction finishes."""

    def close(self, exc: t.Optional[BaseException] = None):
        """Called to clean up, whether the reconstruction succeeded or failed."""

    @t.final
    def __enter__(self) -> 'Self':
        return self

    @t.final
    def __exit__(self, type: t.Optional[t.Type[BaseException]],
                 value: t.Optional[BaseException], traceback: object) -> None:
        self.close(value)


class LoggingObserver(Observer):
    """Observer that logs progress and timing, and records `'utc'` and `'time'` progress in the state."""

    def __init__(self):
        self.logger = logging

        self.init_start_time: t.Optional[float] = None
        self.recons_start_time: t.Optional[float] = None
        self.init_start_utc: t.Optional[float] = None
        self.recons_start_utc: t.Optional[float] = None
        self.engine_start_time: t.Optional[float] = None
        self.iter_start_time: t.Optional[float] = None

    def _format_hhmmss(self, seconds: float) -> str:
        hh, ss = divmod(seconds, (60 * 60))
        mm, ss = divmod(ss, 60)
        return f"{int(hh):02d}:{int(mm):02d}:{ss:06.3f}"

    def _format_mmss(self, seconds: float) -> str:
        mm, ss = divmod(seconds, 60)
        return f"{int(mm):02d}:{ss:06.3f}"

    def get_utc(self) -> float:
        """Return the current time [s since the Unix epoch]."""
        return time.time_ns() * 1e-9

    def init_recons(self, plan: ReconsPlan, *, name: str):
        self.logger.info("Initializing reconstruction...")
        self.init_start_time = time.monotonic()
        self.init_start_utc = self.get_utc()

    def start_recons(self, init_state: ReconsState):
        self.recons_start_time = time.monotonic()
        self.recons_start_utc = self.get_utc()

        if self.init_start_time is not None:
            delta = self.recons_start_time - self.init_start_time
            self.logger.info(f"Initialized reconstruction in {self._format_mmss(delta)}")
        else:
            self.logger.info("Initialized reconstruction")

        if init_state.iter.total_iter == 0:
            utc_prog = ProgressState()

            if self.init_start_utc is not None:
                utc_prog.iters.append(-1)
                utc_prog.values.append(self.init_start_utc)
            utc_prog.iters.append(0)
            utc_prog.values.append(self.recons_start_utc)
            init_state.progress['utc'] = utc_prog

            if self.init_start_time is not None:
                init_state.progress['time'] = ProgressState([0], [self.recons_start_time - self.init_start_time])

    def init_engine(
        self, init_state: ReconsState, *, recons_name: str,
        plan: EnginePlan, **kwargs: t.Any
    ):
        self.logger.info("Initializing engine...")
        self.engine_start_time = time.monotonic()

    def start_engine(self, init_state: ReconsState):
        self.logger.info("Engine initialized")
        self.iter_start_time = time.monotonic()

    def update_iteration(self, state: ReconsState, i: int, n: int, errors: t.Dict[str, float]):
        finish_time = time.monotonic()

        if self.iter_start_time is not None:
            delta = finish_time - self.iter_start_time
            time_s = f" [{self._format_mmss(delta)}]"
        else:
            time_s = ""

        w = len(str(n))

        error_s = f" Error: {error:.3e}" if (error := errors.get('total_loss')) is not None else ""
        other_errors = ", ".join(f"{k}: {v:.3e}" for (k, v) in errors.items() if k != 'total_loss')
        other_errors = f"\n    Error breakdown: {other_errors}" if other_errors else ""
        self.logger.info(f"Finished iter {i:{w}}/{n}{time_s}{error_s}{other_errors}")
        self.iter_start_time = finish_time

        if 'utc' in state.progress:
            state.progress['utc'].iters.append(int(state.iter.total_iter))
            state.progress['utc'].values.append(self.get_utc())
        if 'time' in state.progress and self.init_start_time is not None:
            state.progress['time'].iters.append(int(state.iter.total_iter))
            state.progress['time'].values.append(finish_time - self.init_start_time)

    def finish_engine(self, state: ReconsState):
        self.logger.info("Engine finished!")
        if self.engine_start_time is not None:
            delta = time.monotonic() - self.engine_start_time
            self.logger.info(f"Total engine time: {self._format_hhmmss(delta)}")

    def finish_recons(self, state: ReconsState):
        self.logger.info("Finished reconstruction!")
        if self.recons_start_time is not None:
            delta = time.monotonic() - self.recons_start_time
            self.logger.info(f"Total reconstruction time: {self._format_hhmmss(delta)}")


class PatienceObserver(Observer):
    """Observer that raises `EarlyTermination` after `patience` iterations without improvement of the smoothed `'total_loss'`.

    Args:
        patience: Number of iterations without improvement.
        smoothing: Smoothing factor of the error moving average, in `(0, 1]`. Lower values smooth more.
        continue_next_engine: Whether the reconstruction continues with the next engine after termination.
    """

    def __init__(self, patience: int, smoothing: float = 0.9, continue_next_engine: bool = True):
        self.patience: int = patience
        self.no_improvement_iter: int = 0
        self.best_error: t.Optional[float] = None
        self.smoothed_error: t.Optional[float] = None
        self.smoothing: float = smoothing
        self.continue_next_engine: bool = continue_next_engine

    def init_engine(
        self, init_state: ReconsState, *, recons_name: str,
        plan: EnginePlan, **kwargs: t.Any
    ):
        self.no_improvement_iter = 0
        self.best_error = None
        self.smoothed_error = None

    def update_iteration(self, state: ReconsState, i: int, n: int, errors: t.Dict[str, float]):
        if (error := errors.get('total_loss')) is None:
            return

        # Exponential moving average
        self.smoothed_error = error if self.smoothed_error is None else \
            (1. - self.smoothing) * self.smoothed_error + self.smoothing * error

        if self.best_error is None or self.smoothed_error < self.best_error:
            self.best_error = self.smoothed_error
            self.no_improvement_iter = 0
        else:
            self.no_improvement_iter += 1

        if self.no_improvement_iter >= self.patience:
            logging.info(f"Early termination: no improvement for {self.patience} iterations")
            raise EarlyTermination(state, self.continue_next_engine)


class SaveObserver(Observer):
    """Observer that writes states and images to the output directory, per the plan's save options."""

    def __init__(self):
        self.out_dir: Path = Path.cwd()
        self.engine_out_dir: t.Optional[Path] = None
        self.save_options: t.Optional[SaveOptions] = None

        self.save_flag: t.Optional[t.Callable[[FlagArgs], bool]] = None
        self.save_images_flag: t.Optional[t.Callable[[FlagArgs], bool]] = None
        self.save_initial: bool = False
        self.save_initial_images: bool = False
        self.any_state_output: bool = False
        self.any_image_output: bool = False
        # whether any engine has finished
        self.any_engine_output: bool = False

    def init_recons(self, plan: ReconsPlan, *, name: str):
        if plan.out_dir is not None:
            fmt_str = plan.out_dir
            try:
                out_dir = fmt_str.format(
                    name=name,
                )
                self.out_dir = Path(out_dir).expanduser().absolute()
            except KeyError as e:
                raise ValueError(f"Invalid format string in 'out_dir' (unknown key {e})") from None
            except Exception as e:
                raise ValueError("Invalid format string in 'out_dir'") from e

    def init_engine(
        self, init_state: ReconsState, *, recons_name: str,
        plan: EnginePlan, **kwargs: t.Any
    ):
        self.save_initial = plan.save_initial
        self.save_initial_images = plan.save_initial_images
        self.save_flag = process_flag(plan.save)
        self.save_images_flag = process_flag(plan.save_images)
        self.save_options = plan.save_options
        self.any_state_output = self.save_initial or flag_any_true(self.save_flag, plan.niter)
        self.any_image_output = self.save_initial_images or flag_any_true(self.save_images_flag, plan.niter)
        engine_num = init_state.iter.engine_num

        try:
            fmt_str = plan.save_options.out_dir
            engine_out_dir = fmt_str.format(
                engine_num=engine_num, name=recons_name,
                grouping=plan.grouping, niter=plan.niter,
                **kwargs
            )
            # if engine_out_dir is absolute, it overrides self.out_dir
            # if neither are absolute .absolute() takes over
            engine_out_dir = (self.out_dir / Path(engine_out_dir).expanduser()).absolute()
        except KeyError as e:
            raise ValueError(f"Invalid format string in 'out_dir' (unknown key {e})") from None
        except Exception as e:
            raise ValueError("Invalid format string in 'out_dir'") from e

        self.engine_out_dir = engine_out_dir

        if self.any_state_output or self.any_image_output:
            # TODO: add option to clear out_dir

            # first we try to make out_dir
            if self.out_dir is not None:
                try:
                    self.out_dir.mkdir(exist_ok=True)
                except Exception as e:
                    e.add_note(f"Unable to create output dir '{self.out_dir}'")
                    raise

                (self.out_dir / 'finished').unlink(missing_ok=True)

            # then engine_out_dir
            try:
                self.engine_out_dir.mkdir(exist_ok=True)
            except Exception as e:
                e.add_note(f"Unable to create output dir '{self.engine_out_dir}'")
                raise

            (self.engine_out_dir / 'finished').unlink(missing_ok=True)

    def start_engine(self, init_state: ReconsState):
        assert self.engine_out_dir is not None
        assert self.save_options is not None

        if self.save_initial:
            from phaser.engines.common.output import output_state
            output_state(init_state, self.engine_out_dir, self.save_options, initial=True)

        if self.save_initial_images:
            from phaser.engines.common.output import output_images
            output_images(init_state, self.engine_out_dir, self.save_options, initial=True)

    def update_iteration(self, state: ReconsState, i: int, n: int, errors: t.Dict[str, float]):
        assert self.engine_out_dir is not None
        assert self.save_options is not None

        if self.save_flag and self.save_flag({'state': state, 'niter': n}):
            from phaser.engines.common.output import output_state
            output_state(state, self.engine_out_dir, self.save_options)

        if self.save_images_flag and self.save_images_flag({'state': state, 'niter': n}):
            from phaser.engines.common.output import output_images
            output_images(state, self.engine_out_dir, self.save_options)

    def finish_engine(self, state: ReconsState):
        assert self.engine_out_dir is not None
        assert self.save_options is not None

        any_output = self.any_image_output or self.any_state_output
        if not any_output:
            return

        self.any_engine_output = True

        if self.any_state_output:
            from phaser.engines.common.output import output_state
            output_state(state, self.engine_out_dir, self.save_options)
        if self.any_image_output:
            from phaser.engines.common.output import output_images
            output_images(state, self.engine_out_dir, self.save_options)
        if self.engine_out_dir != self.out_dir:
            (self.engine_out_dir / 'finished').touch(mode=0o664)

    def close(self, exc: t.Optional[BaseException] = None):
        # here rather than finish_recons for more flexibility in calling
        if exc is None and self.any_engine_output:
            (self.out_dir / 'finished').touch(mode=0o664)


def _fwd_to_children(f: t.Callable[t.Concatenate['ObserverSet', P], None]) -> t.Callable[t.Concatenate['ObserverSet', P], None]:
    @wraps(f)
    def wrapper(self: 'ObserverSet', *args: P.args, **kwargs: P.kwargs):
        for observer in self.inner:
            getattr(observer, f.__name__)(*args, **kwargs)

    return wrapper


class ObserverSet(Observer):
    """Observer that forwards all events to a collection of observers."""

    def __init__(self, observers: t.Iterable[Observer]):
        self.inner: t.Tuple[Observer, ...] = tuple(observers)

    @_fwd_to_children
    def init_recons(self, plan: ReconsPlan, *, name: str):
        """Called when initialization of a reconstruction plan begins."""

    @_fwd_to_children
    def start_recons(self, init_state: ReconsState):
        """Called when a reconstruction plan is initialized."""

    @_fwd_to_children
    def init_engine(
        self, init_state: ReconsState, *, recons_name: str,
        plan: EnginePlan, **kwargs: t.Any
    ):
        """Called when initialization of an engine begins."""

    @_fwd_to_children
    def start_engine(self, init_state: ReconsState):
        """Called after an engine is initialized, before it starts."""

    @_fwd_to_children
    def heartbeat(self):
        """Called periodically by the engine."""

    @_fwd_to_children
    def update_group(self, state: t.Union[ReconsState, PartialReconsState], force: bool = False):
        """Called when a group finishes, with the updated state. `force` is deprecated and ignored."""

    @_fwd_to_children
    def update_iteration(self, state: ReconsState, i: int, n: int, errors: t.Dict[str, float]):
        """Called when iteration `i` of `n` finishes, with the updated state and errors by name (`'total_loss'` is the total)."""

    @_fwd_to_children
    def finish_engine(self, state: ReconsState):
        """Called when an engine finishes."""

    @_fwd_to_children
    def finish_recons(self, state: ReconsState):
        """Called when a reconstruction finishes."""

    @_fwd_to_children
    def close(self, exc: t.Optional[BaseException] = None):
        """Called to clean up, whether the reconstruction succeeded or failed."""

    def __enter__(self) -> 'Self':  # type: ignore
        for observer in self.inner:
            observer.__enter__()
        return self

    def __exit__(self, type: t.Optional[t.Type[BaseException]],  # type: ignore
                 value: t.Optional[BaseException], traceback: object) -> None:
        self.close(value)
