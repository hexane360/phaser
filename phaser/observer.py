import contextlib
from functools import wraps
import logging
from pathlib import Path
import time
import typing as t

from phaser.plan import ReconsPlan, EnginePlan, SaveOptions
from phaser.state import ReconsState, PartialReconsState, ProgressState
from phaser.types import EarlyTermination, flag_any_true, process_flag

if t.TYPE_CHECKING:
    from phaser.hooks.schedule import FlagArgs
    from typing_extensions import Self

P = t.ParamSpec('P')

class Observer(contextlib.AbstractContextManager):
    def init_recons(self, plan: ReconsPlan):
        """Called when a reconstruction plan beings initialization."""
        pass

    def start_recons(self, init_state: ReconsState):
        """Called when a reconstruction plan is initialized."""
        pass

    def init_engine(
        self, init_state: ReconsState, *, recons_name: str,
        plan: EnginePlan, **kwargs: t.Any
    ):
        """Called when an engine begins initialization"""
        pass

    def start_engine(self, init_state: ReconsState):
        """Called after engine initialization, before it starts"""
        pass

    def heartbeat(self):
        """Called reasonably often by the engine, to e.g. periodically send data"""
        pass

    def update_group(self, state: t.Union[ReconsState, PartialReconsState], force: bool = False):
        """Called when a group is finished, with updated reconstruction state."""
        pass

    def update_iteration(self, state: ReconsState, i: int, n: int, errors: t.Dict[str, float]):
        """Called when an iteration is finished, with updated reconstruction state."""
        pass

    def finish_engine(self, state: ReconsState):
        """Called when an engine is finished"""
        pass

    def finish_recons(self, state: ReconsState):
        """Called when the reconstruction is finished"""
        pass

    def close(self, exc: t.Optional[BaseException] = None):
        """Called to clean up, whether the reconstruction succeeded or failed."""
        pass

    @t.final
    def __enter__(self) -> 'Self':
        return self

    @t.final
    def __exit__(self, type: t.Optional[t.Type[BaseException]],
                 value: t.Optional[BaseException], traceback: t.Any) -> None:
        self.close(value)


class LoggingObserver(Observer):
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
        return time.time_ns() * 1e-9

    def init_recons(self, plan: ReconsPlan):
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

        error_s = f" Error: {error:.3e}" if (error := errors.get('total_loss')) else ""
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
    def __init__(self, patience: int, smoothing: float = 0.1, continue_next_engine: bool = True):
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

    def _error_from_state(self, state: t.Union[ReconsState, PartialReconsState]) -> t.Optional[float]:
        if state.progress is None or (progress := state.progress['total_loss']) is None or not len(progress.values):
            return None
        return progress.values[-1]

    def update_iteration(self, state: ReconsState, i: int, n: int, errors: t.Dict[str, float]):
        if (error := errors.get('total_loss')) is None:
            return

        if self.best_error is None or error < self.best_error:
            self.best_error = error
            self.no_improvement_iter = 0
        else:
            self.no_improvement_iter += 1

        # Exponential moving average
        if self.smoothed_error is None:
            self.smoothed_error = error
        else:
            self.smoothed_error = (1 - self.smoothing) * self.smoothed_error + self.smoothing * error

        if self.no_improvement_iter >= self.patience:
            logging.info(f"Early termination: no improvement for {self.patience} iterations")
            raise EarlyTermination(state, self.continue_next_engine)


class SaveObserver(Observer):
    def __init__(self):
        self.out_dir: t.Optional[Path] = None
        self.save_options: t.Optional[SaveOptions] = None

        self.save_flag: t.Optional[t.Callable[['FlagArgs'], bool]] = None
        self.save_images_flag: t.Optional[t.Callable[['FlagArgs'], bool]] = None
        self.any_state_output: bool = False
        self.any_image_output: bool = False

    def init_engine(
        self, init_state: ReconsState, *, recons_name: str,
        plan: EnginePlan, **kwargs: t.Any
    ):
        self.save_flag = process_flag(plan.save)
        self.save_images_flag = process_flag(plan.save_images)
        self.save_options = plan.save_options
        self.any_state_output = flag_any_true(self.save_flag, plan.niter)
        self.any_image_output = flag_any_true(self.save_images_flag, plan.niter)
        engine_num = init_state.iter.engine_num

        try:
            fmt_str = plan.save_options.out_dir
            out_dir = fmt_str.format(
                engine_num=engine_num, name=recons_name,
                group=plan.grouping, niter=plan.niter,
                **kwargs
            )
            out_dir = Path(out_dir).expanduser().absolute()
        except KeyError as e:
            raise ValueError(f"Invalid format string in 'out_dir' (unknown key {e})") from None
        except Exception as e:
            raise ValueError("Invalid format string in 'out_dir'") from e

        if self.out_dir is not None and self.out_dir != out_dir:
            self.close()  # close out_dir from previous engine
        self.out_dir = out_dir

        if self.any_state_output or self.any_image_output:
            # TODO: add option to clear out_dir
            try:
                self.out_dir.mkdir(exist_ok=True)
            except Exception as e:
                e.add_note(f"Unable to create output dir '{self.out_dir}'")
                raise

            (self.out_dir / 'finished').unlink(missing_ok=True)

    def update_iteration(self, state: ReconsState, i: int, n: int, errors: t.Dict[str, float]):
        from phaser.engines.common.output import output_images, output_state

        assert self.out_dir is not None
        assert self.save_options is not None

        if self.save_flag and self.save_flag({'state': state, 'niter': n}):
            output_state(state, self.out_dir, self.save_options)

        if self.save_images_flag and self.save_images_flag({'state': state, 'niter': n}):
            output_images(state, self.out_dir, self.save_options)

    def finish_engine(self, state: ReconsState):
        from phaser.engines.common.output import output_images, output_state
        assert self.out_dir is not None
        assert self.save_options is not None

        if self.any_state_output:
            output_state(state, self.out_dir, self.save_options)

        if self.any_image_output:
            output_images(state, self.out_dir, self.save_options)

    def close(self, exc: t.Optional[BaseException] = None):
        if exc is None and self.out_dir is not None:
            if self.any_state_output or self.any_image_output:
                (self.out_dir / 'finished').touch(mode=0o664)


def _fwd_to_children(f: t.Callable[t.Concatenate['ObserverSet', P], None]) -> t.Callable[t.Concatenate['ObserverSet', P], None]:
    @wraps(f)
    def wrapper(self: 'ObserverSet', *args: P.args, **kwargs: P.kwargs):
        for observer in self.inner:
            getattr(observer, f.__name__)(*args, **kwargs)

    return wrapper


class ObserverSet(Observer):
    def __init__(self, observers: t.Iterable[Observer]):
        self.inner: t.Tuple[Observer, ...] = tuple(observers)

    @_fwd_to_children
    def init_recons(self, plan: ReconsPlan):
        """Called when a reconstruction plan beings initialization."""
        ...

    @_fwd_to_children
    def start_recons(self, init_state: ReconsState):
        """Called when a reconstruction plan is initialized."""
        ...

    @_fwd_to_children
    def init_engine(
        self, init_state: ReconsState, *, recons_name: str,
        plan: EnginePlan, **kwargs: t.Any
    ):
        ...

    @_fwd_to_children
    def start_engine(self, init_state: ReconsState):
        """Called after engine initialization, before it starts"""
        ...

    @_fwd_to_children
    def heartbeat(self):
        """Called reasonably often by the engine, to e.g. periodically send data"""
        ...

    @_fwd_to_children
    def update_group(self, state: t.Union[ReconsState, PartialReconsState], force: bool = False):
        """Called when a group is finished, with updated reconstruction state."""
        ...

    @_fwd_to_children
    def update_iteration(self, state: ReconsState, i: int, n: int, errors: t.Dict[str, float]):
        """Called when an iteration is finished, with updated reconstruction state."""
        ...

    @_fwd_to_children
    def finish_engine(self, state: ReconsState):
        """Called when an engine is finished"""
        ...

    @_fwd_to_children
    def finish_recons(self, state: ReconsState):
        """Called when the reconstruction is finished"""
        ...

    @_fwd_to_children
    def close(self, exc: t.Optional[BaseException] = None):
        """Called to clean up, whether the reconstruction succeeded or failed."""
        ...

    def __enter__(self) -> 'Self':  # type: ignore
        for observer in self.inner:
            observer.__enter__()
        return self

    def __exit__(self, type: t.Optional[t.Type[BaseException]],  # type: ignore
                 value: t.Optional[BaseException], traceback: t.Any) -> None:
        self.close(value)
