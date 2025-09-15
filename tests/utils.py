import builtins
from contextlib import contextmanager
from itertools import chain
import inspect
from pathlib import Path
import sys
import typing as t

import numpy
import pytest

CallableT = t.TypeVar('CallableT', bound=t.Callable)
P = t.ParamSpec('P')
T = t.TypeVar('T')

EXPECTED_PATH = Path(__file__).parent / 'expected'
ACTUAL_PATH = Path(__file__).parent / 'actual'
INPUT_FILES_PATH = Path(__file__).parent / 'input_files'
OVERWRITE_EXPECTED = object()


def _wrap_pytest(wrapper: CallableT, wrapped: t.Callable,
                 mod_params: t.Optional[t.Callable[[t.Sequence[inspect.Parameter]], t.Sequence[inspect.Parameter]]] = None
) -> CallableT:
    # hacks to allow pytest to find fixtures in wrapped functions
    old_sig = inspect.signature(wrapped)
    params = tuple(old_sig.parameters.values())
    if mod_params is not None:
        params = mod_params(params)
    new_sig = old_sig.replace(parameters=params)

    testmark = getattr(wrapped, "pytestmark", []) + getattr(wrapper, "pytestmark", [])
    if len(testmark) > 0:
        wrapper.pytestmark = testmark  # type: ignore
    wrapper.__name__ = wrapped.__name__
    wrapper.__doc__ = wrapped.__doc__
    wrapper.__signature__ = new_sig  # type: ignore
    return wrapper


def with_backends(
    *backends: t.Union[str, t.Iterable[str]]
) -> t.Callable[[t.Callable[P, T]], t.Callable[P, T]]:
    """Run a test on the specified compute backends"""
    backends = t.cast(t.Tuple[str, ...],
        tuple(chain.from_iterable((b,) if isinstance(b, str) else b for b in backends))
    )

    def decorator(f: t.Callable[P, T]) -> t.Callable[P, T]:
        return pytest.mark.parametrize('backend', [
            pytest.param(backend, marks=getattr(pytest.mark, backend))
            for backend in backends
        ])(f)

    return decorator


def read_array(path: Path) -> numpy.ndarray:
    ext = path.suffix.lower()

    try:
        # load with tifffile
        if ext in ('.tif', '.tiff'):
            import tifffile

            mag_path = path.with_stem(path.stem + '_mag')
            phase_path = path.with_stem(path.stem + '_phase')

            if mag_path.exists() or phase_path.exists():
                mag = numpy.asarray(tifffile.imread(mag_path))
                phase = numpy.asarray(tifffile.imread(phase_path))
                return mag * numpy.exp(1.j * phase)

            return numpy.asarray(tifffile.imread(path))
        # load with numpy
        elif ext in ('.npy',):
            return numpy.load(path, allow_pickle=False)
        raise ValueError(f"Don't know how to load file of type '{path.suffix}'")
    except Exception as e:
        raise RuntimeError(f"Unable to load file '{path.name}'") from e


def write_array(path: Path, arr: numpy.ndarray):
    ext = path.suffix.lower()

    try:
        if ext in ('.tif', '.tiff'):
            import tifffile

            if numpy.iscomplexobj(arr):
                mag_path = path.with_stem(path.stem + '_mag')
                phase_path = path.with_stem(path.stem + '_phase')

                tifffile.imwrite(mag_path, numpy.abs(arr))
                tifffile.imwrite(phase_path, numpy.angle(arr))
            else:
                tifffile.imwrite(path, arr)
        elif ext in ('.npy',):
            numpy.save(path, arr, allow_pickle=False)
        else:
            raise ValueError(f"Don't know how to save file of type '{path.suffix}'")
    except Exception as e:
        raise RuntimeError(f"Unable to save file '{path.name}'") from e


def check_array_equals_file(name: str, *, out_name: t.Optional[str] = None, decimal: int = 6) -> t.Callable[[t.Callable[..., numpy.ndarray]], t.Callable[..., None]]:
    def decorator(f: t.Callable[..., numpy.ndarray]):
        @pytest.mark.expected_filename(name)
        def wrapper(*args, file_contents_array: numpy.ndarray, **kwargs):
            from numpy.testing import assert_array_almost_equal
            actual = f(*args, **kwargs)

            # instead of comparing, overwrite expected path with the output
            if file_contents_array is OVERWRITE_EXPECTED:
                try:
                    out_path = EXPECTED_PATH / name.format(*args, **kwargs)
                    print(f"Overwriting expected result in '{out_path}'...")
                    write_array(out_path, actual)
                except Exception as e:
                    raise RuntimeError("Failed to overwrite expected result") from e
                return

            out_path = ACTUAL_PATH / (out_name if out_name is not None else name).format(*args, **kwargs)

            try:
                assert_array_almost_equal(actual, file_contents_array, decimal=decimal)
            except AssertionError:
                try:
                    print(f"Saving actual result to '{out_path}'")
                    write_array(out_path, actual)
                except Exception:
                    print("Failed to save result.")
                raise

        return _wrap_pytest(wrapper, f, lambda params: [*params, inspect.Parameter('file_contents_array', inspect.Parameter.KEYWORD_ONLY)])

    return decorator


_import = builtins.__import__


class Importer:
    def __init__(self, prevented: t.Iterable[str] = ()):
        self.prevented: set[str] = set(prevented)

    def __call__(self, name, globals, locals, fromlist, level):
        if name in self.prevented:
            raise ImportError(f"Mocked ImportError for '{name}'")
        return _import(name, globals, locals, fromlist, level)


@contextmanager
def mock_importerror(*modulenames: t.Iterable[str]):
    modules = set(chain.from_iterable((n,) if isinstance(n, str) else n for n in modulenames))

    if not len(modules):
        # just run the body and return
        yield
        return

    # remove modules if they're already imported
    for module in modules:
        sys.modules.pop(module, None)

    # replace __import__ with our copy
    if not isinstance(builtins.__import__, Importer):
        builtins.__import__ = Importer()
    # and specify which modules to prevent
    builtins.__import__.prevented.update(modules)

    # run the context manager's body
    yield

    # un-prevent modules
    builtins.__import__.prevented.difference_update(modules)

    # if we're the last ones, change __import__ back
    if not len(builtins.__import__.prevented):
        builtins.__import__ = _import