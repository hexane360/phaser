import typing as t
from pathlib import Path

import pytest

from phaser.execute import execute_plan
from phaser.hooks import EngineArgs, EngineHook
from phaser.plan import EnginePlan, ReconsPlan
from phaser.state import ReconsState


class FakeEnginePlan(EnginePlan, kw_only=True):
    fail_at: t.Optional[int] = None
    """Iteration to raise an exception at"""
    plateau_at: t.Optional[int] = None
    """Iteration after which the loss stops improving"""


def run_fake_engine(args: EngineArgs, props: FakeEnginePlan) -> ReconsState:
    """Engine which follows the observer protocol of the real engines, without computing anything."""
    state, observer = args['state'], args['observer']

    observer.init_engine(state, recons_name=args['recons_name'], plan=props)
    observer.start_engine(state)

    start_i = int(state.iter.total_iter)
    for i in range(1, props.niter + 1):
        if props.fail_at == i:
            raise RuntimeError(f"fake engine failed at iter {i}")

        state.iter.engine_iter = i
        state.iter.total_iter = start_i + i
        loss = 1. / (i if props.plateau_at is None else min(i, props.plateau_at))
        observer.update_iteration(state, i, props.niter, {'total_loss': loss})

    observer.finish_engine(state)
    return state


@pytest.fixture(autouse=True)
def fake_engine(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(EngineHook.known, 'fake', ('tests.test_execute:run_fake_engine', FakeEnginePlan))


@pytest.fixture
def cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def make_plan(engines: t.Sequence[dict[str, t.Any]], **kwargs: t.Any) -> ReconsPlan:
    return ReconsPlan.from_data({
        'name': 'test',
        'backend': 'numpy',
        'raw_data': 'tests.test_engines:load_random',
        'engines': [{'type': 'fake', 'niter': 2, **engine} for engine in engines],
        **kwargs,
    })


def layout(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}


ENGINE_DIRS = {'out_dir': 'engine{engine_num}'}
PROBE_ONLY = {'images': ['probe']}


@pytest.mark.parametrize(('engines', 'plan_kwargs', 'expected'), [
    pytest.param([{}], {}, set(), id='no_output'),
    pytest.param([{'save': True}], {}, {
        'test/iter001.h5', 'test/iter002.h5', 'test/finished',
    }, id='save_true'),
    pytest.param([{'niter': 5, 'save': {'every': 2}}], {}, {
        'test/iter002.h5', 'test/iter004.h5', 'test/iter005.h5', 'test/finished',
    }, id='save_every'),
    pytest.param([{'niter': 5, 'save': {'every': 2, 'first': True}}], {}, {
        'test/iter001.h5', 'test/iter002.h5', 'test/iter004.h5', 'test/iter005.h5', 'test/finished',
    }, id='save_every_first'),
    pytest.param([{'niter': 3, 'save': {'after': 10, 'first': True}}], {}, {
        'test/iter001.h5', 'test/iter003.h5', 'test/finished',
    }, id='save_first_only'),
    pytest.param([{'niter': 5, 'save': {'before': 3}}], {}, {
        'test/iter001.h5', 'test/iter002.h5', 'test/iter005.h5', 'test/finished',
    }, id='save_before'),
    pytest.param([{'niter': 3, 'save_images': {'every': 2}, 'save_options': PROBE_ONLY}], {}, {
        'test/probe_iter002.tiff', 'test/probe_iter003.tiff', 'test/finished',
    }, id='save_images'),
    pytest.param([{'save_initial': True}], {}, {
        'test/initial.h5', 'test/iter002.h5', 'test/finished',
    }, id='save_initial'),
    pytest.param([{'save_initial_images': True, 'save_options': PROBE_ONLY}], {}, {
        'test/probe_initial.tiff', 'test/probe_iter002.tiff', 'test/finished',
    }, id='save_initial_images'),
    pytest.param([{'save': True, 'save_options': {'hdf5_fmt': 'e{iter.engine_num}_i{iter.engine_iter}.{ext}'}}], {}, {
        'test/e1_i1.h5', 'test/e1_i2.h5', 'test/finished',
    }, id='hdf5_fmt'),
    pytest.param([{'save': True}, {'save': True}], {}, {
        'test/iter001.h5', 'test/iter002.h5', 'test/iter003.h5', 'test/iter004.h5', 'test/finished',
    }, id='shared_dir'),
    pytest.param([{'save': True}, {}], {}, {
        'test/iter001.h5', 'test/iter002.h5', 'test/finished',
    }, id='shared_dir_on_off'),
    pytest.param([{'save': True, 'save_options': ENGINE_DIRS}, {'save': True, 'save_options': ENGINE_DIRS}], {}, {
        'test/engine1/iter001.h5', 'test/engine1/iter002.h5', 'test/engine1/finished',
        'test/engine2/iter003.h5', 'test/engine2/iter004.h5', 'test/engine2/finished',
        'test/finished',
    }, id='engine_dirs'),
    pytest.param([{'save': True, 'save_options': ENGINE_DIRS}, {'save_options': ENGINE_DIRS}], {}, {
        'test/engine1/iter001.h5', 'test/engine1/iter002.h5', 'test/engine1/finished',
        'test/finished',
    }, id='engine_dirs_on_off'),
    pytest.param([{'save_options': ENGINE_DIRS}, {'save': True, 'save_options': ENGINE_DIRS}], {}, {
        'test/engine2/iter003.h5', 'test/engine2/iter004.h5', 'test/engine2/finished',
        'test/finished',
    }, id='engine_dirs_off_on'),
    pytest.param([{'save': True, 'grouping': 16, 'save_options': {'out_dir': '{name}_g{grouping}_n{niter}'}}], {}, {
        'test/test_g16_n2/iter001.h5', 'test/test_g16_n2/iter002.h5', 'test/test_g16_n2/finished',
        'test/finished',
    }, id='engine_dir_keys'),
    pytest.param([{'save': True}], {'out_dir': 'runs_{name}'}, {
        'runs_test/iter001.h5', 'runs_test/iter002.h5', 'runs_test/finished',
    }, id='out_dir'),
    pytest.param([{'save': True}], {'out_dir': None}, {
        'iter001.h5', 'iter002.h5', 'finished',
    }, id='out_dir_null'),
])
def test_output_layout(
    cwd: Path, engines: list[dict[str, t.Any]], plan_kwargs: dict[str, t.Any], expected: set[str]
):
    execute_plan(make_plan(engines, **plan_kwargs))
    assert layout(cwd) == expected


def test_output_layout_name_override(cwd: Path):
    execute_plan(make_plan([{'save': True, 'save_options': {'out_dir': '{name}_e{engine_num}'}}]), name='other')
    assert layout(cwd) == {
        'other/other_e1/iter001.h5', 'other/other_e1/iter002.h5', 'other/other_e1/finished',
        'other/finished',
    }


def test_output_layout_absolute_engine_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / 'cwd').mkdir()
    monkeypatch.chdir(tmp_path / 'cwd')

    execute_plan(make_plan([{'save': True, 'save_options': {'out_dir': str(tmp_path / 'abs')}}]))
    assert layout(tmp_path) == {
        'abs/iter001.h5', 'abs/iter002.h5', 'abs/finished',
        'cwd/test/finished',
    }


@pytest.mark.parametrize(('engines', 'expected'), [
    pytest.param([{'save': True, 'fail_at': 2}], {
        'test/iter001.h5',
    }, id='single_engine'),
    pytest.param([{'save': True}, {'save': True, 'fail_at': 2}], {
        'test/iter001.h5', 'test/iter002.h5', 'test/iter003.h5',
    }, id='shared_dir'),
    pytest.param([{'save': True, 'save_options': ENGINE_DIRS}, {'save': True, 'fail_at': 2, 'save_options': ENGINE_DIRS}], {
        'test/engine1/iter001.h5', 'test/engine1/iter002.h5', 'test/engine1/finished',
        'test/engine2/iter003.h5',
    }, id='engine_dirs'),
    pytest.param([{'save': True, 'save_options': ENGINE_DIRS}, {'fail_at': 2, 'save_options': ENGINE_DIRS}], {
        'test/engine1/iter001.h5', 'test/engine1/iter002.h5', 'test/engine1/finished',
    }, id='engine_dirs_on_off'),
])
def test_failure_layout(cwd: Path, engines: list[dict[str, t.Any]], expected: set[str]):
    with pytest.raises(RuntimeError, match="fake engine failed"):
        execute_plan(make_plan(engines))
    assert layout(cwd) == expected


@pytest.mark.parametrize(('engine', 'plan_kwargs', 'match'), [
    pytest.param({'save': True}, {'out_dir': '{bogus}'}, r"'out_dir' \(unknown key 'bogus'\)", id='out_dir'),
    pytest.param({'save': True, 'save_options': {'out_dir': '{bogus}'}}, {}, r"'out_dir' \(unknown key 'bogus'\)", id='engine_out_dir'),
    pytest.param({'save': True, 'save_options': {'hdf5_fmt': '{bogus}.h5'}}, {}, r"'hdf5_fmt' \(unknown key 'bogus'\)", id='hdf5_fmt'),
    pytest.param({'save_initial': True, 'save_options': {'hdf5_fmt_init': '{bogus}.h5'}}, {}, r"'hdf5_fmt_init' \(unknown key 'bogus'\)", id='hdf5_fmt_init'),
    pytest.param({'save_images': True, 'save_options': {**PROBE_ONLY, 'img_fmt': '{bogus}.tiff'}}, {}, r"'img_fmt' \(unknown key 'bogus'\)", id='img_fmt'),
])
def test_invalid_format(cwd: Path, engine: dict[str, t.Any], plan_kwargs: dict[str, t.Any], match: str):
    with pytest.raises(ValueError, match=match):
        execute_plan(make_plan([engine], **plan_kwargs))
    assert layout(cwd) == set()


def test_rerun_clears_finished(cwd: Path):
    engines = [{'save': True, 'save_options': ENGINE_DIRS}, {'save': True, 'save_options': ENGINE_DIRS}]
    execute_plan(make_plan(engines))
    assert {p for p in layout(cwd) if p.endswith('finished')} == {
        'test/finished', 'test/engine1/finished', 'test/engine2/finished',
    }

    engines[1] = {**engines[1], 'fail_at': 2}
    with pytest.raises(RuntimeError, match="fake engine failed"):
        execute_plan(make_plan(engines))
    assert {p for p in layout(cwd) if p.endswith('finished')} == {'test/engine1/finished'}


def test_early_termination(cwd: Path):
    execute_plan(make_plan([
        {'niter': 5, 'save': True, 'early_termination': 1, 'plateau_at': 2, 'save_options': ENGINE_DIRS},
        {'save': True, 'save_options': ENGINE_DIRS},
    ]))
    assert layout(cwd) == {
        'test/engine1/iter001.h5', 'test/engine1/iter002.h5', 'test/engine1/iter003.h5', 'test/engine1/finished',
        'test/engine2/iter004.h5', 'test/engine2/iter005.h5', 'test/engine2/finished',
        'test/finished',
    }
