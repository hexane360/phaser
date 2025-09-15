
import pytest
import numpy

BACKENDS: set[str] = set(('cpu', 'jax', 'cupy', 'torch'))
AVAILABLE_BACKENDS: set[str] = set(('cpu',))

try:
    import cupy  # pyright: ignore[reportMissingImports]
    if cupy.cuda.runtime.getDeviceCount() > 0:
        AVAILABLE_BACKENDS.add('cupy')
except ImportError:
    pass

try:
    import jax  # pyright: ignore[reportMissingImports]  # noqa: F401
    AVAILABLE_BACKENDS.add('jax')
except ImportError:
    pass


try:
    import torch  # pyright: ignore[reportMissingImports]  # noqa: F401
    torch.asarray([1, 2, 3, 4]).numpy(force=True)  # ensures torch is loaded, fixes a strange error with pytest
    AVAILABLE_BACKENDS.add('torch')
except ImportError:
    pass


def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--save-expected", action="store_true", dest='save-expected', default=False,
                     help="Overwrite expected files with the results of tests.")


def pytest_runtest_setup(item):
    required_backends = BACKENDS.intersection(mark.name for mark in item.iter_markers())
    missing = required_backends - AVAILABLE_BACKENDS
    if len(missing):
        missing = ', '.join(missing)
        pytest.skip(f"Backend {missing} is not available")


@pytest.fixture(scope='function')
def file_contents_array(request: pytest.FixtureRequest, pytestconfig: pytest.Config) -> numpy.ndarray:
    from tests.utils import read_array, EXPECTED_PATH, OVERWRITE_EXPECTED

    marker = request.node.get_closest_marker('expected_filename')
    assert marker is not None
    name = str(marker.args[0])

    if pytestconfig.getoption("save-expected"):
        return OVERWRITE_EXPECTED  # type: ignore

    try:
        params = request._pyfuncitem.callspec.params
        name = name.format(**params)
    except AttributeError:
        pass

    try:
        return read_array(EXPECTED_PATH / name)
    except Exception as e:
        raise RuntimeError("Failed to load expected result. Use --save-expected to overwrite expected result.") from e