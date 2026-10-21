import json

import pytest
from pane.errors import ConvertError

from phaser.io.empad import EmpadMetadata, UnscannedDataset

from .utils import INPUT_FILES_PATH


def test_read_empad_metadata():
    metadata = EmpadMetadata.from_json(INPUT_FILES_PATH / 'metadata_v2.json')

    print(repr(metadata))
    assert metadata.file_type == 'empad_metadata'
    assert metadata.name == 'acquisition_5_7.2Mx_17.9mrad_460mmcl_-20nmdefocus'
    assert metadata.raw_filename == 'scan_x128_y128.raw'
    assert metadata.author == "Abinash"
    assert metadata.path == INPUT_FILES_PATH
    assert metadata.time == '2021-05-02T06:58:26.775565'
    assert metadata.voltage == 200000.0
    assert metadata.conv_angle == 17.9
    assert metadata.camera_length == 0.4575
    assert metadata.scan_fov == (1.2435077703182981e-08,) * 2
    assert metadata.scan_step == (0.9714904455611704e-10,) * 2
    assert metadata.beam_current == 30.0e-12
    assert metadata.notes is None
    assert metadata.adu == 375.0
    assert metadata.defocus is None


def test_read_empad2_toml_metadata(tmp_path):
    import shutil

    toml_path = tmp_path / 'metadata.toml'
    shutil.copy(INPUT_FILES_PATH / 'empad2_metadata.toml', toml_path)
    metadata = EmpadMetadata.from_toml(toml_path)

    print(repr(metadata))
    assert metadata.file_type == 'empad_metadata'
    assert metadata.name == 'acquisition_53'
    assert metadata.empad_version == 2
    # raw file not present, conventional name is used
    assert metadata.raw_filename == 'scan_x200_y200.raw'
    assert metadata.orig_path == tmp_path
    assert metadata.author is None
    assert metadata.time == '2026-10-10T19:45:30.408447+00:00'
    assert metadata.time_unix == pytest.approx(1791661530.408447)
    assert metadata.bg_unix == pytest.approx(1791661179.214721)
    assert metadata.has_bg is True
    assert metadata.voltage == 300000.0
    assert metadata.conv_angle == pytest.approx(25.011686787433686)
    assert metadata.camera_length == 0.5766
    assert metadata.scan_rotation == 0.0
    assert metadata.scan_shape == (200, 200)
    assert metadata.scan_fov == (9.585889570552145e-09,) * 2
    assert metadata.scan_step == (4.769099351542572e-11,) * 2
    assert metadata.exposure_time == 0.0005
    assert metadata.notes == "15nm"
    assert metadata.defocus is None

    # single raw file on disk is used
    (tmp_path / 'other.raw').touch()
    metadata = EmpadMetadata.from_toml(toml_path)
    assert metadata.raw_filename == 'other.raw'

    # scan_rotation is stored in radians
    toml_path.write_text(toml_path.read_text().replace('scan_rotation = 0.0', 'scan_rotation = 1.5707963267948966'))
    metadata = EmpadMetadata.from_toml(toml_path)
    assert metadata.scan_rotation == pytest.approx(90.0)

    # only raster scans are 4D STEM datasets
    toml_path.write_text(toml_path.read_text().replace('"stem_raster"', '"other"'))
    with pytest.raises(UnscannedDataset):
        EmpadMetadata.from_toml(toml_path)


def test_read_empad_xml_metadata(tmp_path):
    metadata = EmpadMetadata.from_xml(INPUT_FILES_PATH / 'empad1_metadata.xml')

    print(repr(metadata))
    assert metadata.file_type == 'empad_metadata'
    assert metadata.name == 'acquisition_1'
    assert metadata.empad_version == 1
    assert metadata.raw_filename == 'scan_x128_y128.raw'
    assert metadata.orig_path == INPUT_FILES_PATH
    assert metadata.author is None
    assert metadata.time == '2026-07-20T18:56:59.701766'
    assert metadata.time_unix == pytest.approx(1784588219.701766)
    assert metadata.bg_unix == pytest.approx(1784588075.038519)
    assert metadata.has_bg is True
    assert metadata.voltage == 300000.0
    # not recorded in EMPAD1 metadata
    assert metadata.conv_angle is None
    assert metadata.camera_length == 0.4575
    assert metadata.scan_rotation == 0.0
    # 'acquire' scan parameters are used, not 'search'
    assert metadata.scan_shape == (128, 128)
    assert metadata.scan_fov == pytest.approx((4.880338213710828e-09 / 0.72,) * 2)
    assert metadata.scan_step == pytest.approx((4.880338213710828e-09 / 0.72 / 128,) * 2)
    assert metadata.exposure_time == 0.001
    assert metadata.notes is None

    # series acquisitions are not 4D STEM datasets
    xml_path = tmp_path / 'acquisition_1.xml'
    xml_path.write_text((INPUT_FILES_PATH / 'empad1_metadata.xml').read_text().replace(
        '<scan_parameters mode="search">', '<scan_parameters mode="search"><series_count>10</series_count>'
    ))
    with pytest.raises(UnscannedDataset):
        EmpadMetadata.from_xml(xml_path)


def empad1_xml():
    return ('acquisition.xml', (INPUT_FILES_PATH / 'empad1_metadata.xml').read_text())


def empad2_toml(conv_angle=25.011686787433686, author=''):
    """EMPAD2 dataset, with `conv_angle` in mrad (None to leave it out)"""
    toml = (INPUT_FILES_PATH / 'empad2_metadata.toml').read_text()
    (author_line, conv_line) = ('operator_name = ""\n', 'convergence_angle = 0.025011686787433686\n')
    assert author_line in toml and conv_line in toml
    toml = toml.replace(author_line, f'operator_name = "{author}"\n')
    toml = toml.replace(conv_line, '' if conv_angle is None else f'convergence_angle = {conv_angle * 1e-3}\n')
    return ('metadata.toml', toml)


def process_datasets(path, monkeypatch, datasets, answers, config=None):
    """
    Run `process_dir` on `datasets` (list of (filename, contents)), one per acquisition folder.
    `answers` maps prompt names to the answers to give ('' accepts the default). Unlisted prompts
    accept their default, or are answered with '1' if there is none (a number prompt would otherwise
    re-prompt forever). Returns the defaults offered for each prompt, and the written metadata.
    """
    import io
    from rich.console import Console
    from rich.prompt import Prompt
    from phaser.cli.process_empad import ProcessEmpadConfig, make_params, process_dir

    for (i, (filename, contents)) in enumerate(datasets, 1):
        acq_dir = path / f'acquisition_{i}'
        acq_dir.mkdir()
        (acq_dir / filename).write_text(contents)

    answers = {name: list(vals) for (name, vals) in answers.items()}
    offered = {}

    def ask(prompt, default=None, **kwargs):
        # emulates rich, where '' accepts `default`
        name = prompt.split(' \\[')[0]
        offered.setdefault(name, []).append(default)
        answer = answers[name].pop(0) if name in answers else ('' if default else '1')
        return answer or default

    monkeypatch.setattr(Prompt, 'ask', ask)

    config = config or ProcessEmpadConfig()
    process_dir(path, config, make_params(config), Console(file=io.StringIO()))

    written = [
        json.loads((path / f'acquisition_{i}' / filename).with_suffix('.json').read_text())
        for (i, (filename, _)) in enumerate(datasets, 1)
    ]
    return (offered, written)


@pytest.mark.parametrize(('dataset', 'empad_version', 'det_flips'), [
    (empad1_xml(), 1, [True, False, False]),
    (empad2_toml(), 2, [False, False, False]),
])
def test_process_empad_version(tmp_path, monkeypatch, dataset, empad_version, det_flips):
    (_, [written]) = process_datasets(tmp_path, monkeypatch, [dataset], {})
    assert written['empad_version'] == empad_version
    assert written['det_flips'] == det_flips


@pytest.mark.parametrize(('datasets', 'answers', 'offered', 'written', 'config_conv_angle'), [
    # readouts are followed
    ([empad2_toml(25.0)] * 3, ['', '', ''], ['25.0', '25.0', '25.0'], [25.0, 25.0, 25.0], None),
    # overrides stick to their readout only
    ([empad2_toml(25.0), empad2_toml(19.0), empad2_toml(25.0)], ['24', '', ''], ['25.0', '19.0', '24'], [24.0, 19.0, 24.0], None),
    ([empad2_toml(25.0), empad2_toml(19.0), empad2_toml(19.0)], ['', '18', ''], ['25.0', '19.0', '18'], [25.0, 18.0, 18.0], None),
    # overrides can be changed back
    ([empad2_toml(25.0)] * 3, ['24', '25', ''], ['25.0', '24', '25'], [24.0, 25.0, 25.0], None),
    # accepting (or typing the shown default) writes the exact readout, other values are written as typed
    ([empad2_toml(25.0117)], [''], ['25.0'], [25.0117], None),
    ([empad2_toml(25.0117)], ['25.0'], ['25.0'], [25.0117], None),
    ([empad2_toml(25.0117)], ['25'], ['25.0'], [25.0], None),
    # readouts are keyed to 0.1 mrad
    ([empad2_toml(25.04), empad2_toml(25.06)], ['24', ''], ['25.0', '25.1'], [24.0, 25.06], None),
    # missing readout is its own key, starting from the config default
    ([empad2_toml(25.0), empad2_toml(None), empad2_toml(25.0)], ['24', '', ''], ['25.0', '25.0', '24'], [24.0, 25.0, 24.0], None),
    # zero is a real readout
    ([empad2_toml(0.0)] * 2, ['', ''], ['0.0', '0.0'], [0.0, 0.0], None),
    # EMPAD1 has no readout
    ([empad1_xml()] * 3, ['', '', ''], ['25.0', '25.0', '25.0'], [25.0, 25.0, 25.0], None),
    ([empad1_xml()] * 3, ['19', '', ''], ['25.0', '19', '19'], [19.0, 19.0, 19.0], None),
    ([empad1_xml()] * 2, ['19', ''], ['21.4', '19'], [19.0, 19.0], 21.4),
])
def test_process_empad_conv_angle(tmp_path, monkeypatch, datasets, answers, offered, written, config_conv_angle):
    from phaser.cli.process_empad import ProcessEmpadConfig

    config = ProcessEmpadConfig(conv_angle=config_conv_angle) if config_conv_angle is not None else None
    (actual_offered, actual_written) = process_datasets(
        tmp_path, monkeypatch, datasets, {'Convergence angle': answers}, config
    )
    assert actual_offered['Convergence angle'] == offered
    assert [m['conv_angle'] for m in actual_written] == pytest.approx(written)


@pytest.mark.parametrize(('datasets', 'answers', 'offered', 'written'), [
    # metadata is followed
    ([empad2_toml(author='Alice')] * 3, ['', '', ''], ['Alice', 'Alice', 'Alice'], ['Alice', 'Alice', 'Alice']),
    ([empad2_toml(author='Alice'), empad2_toml(author='Bob'), empad2_toml(author='Alice')], ['', '', ''],
     ['Alice', 'Bob', 'Alice'], ['Alice', 'Bob', 'Alice']),
    ([empad2_toml(author=''), empad2_toml(author='Alice')], ['', ''], [None, 'Alice'], [None, 'Alice']),
    ([empad2_toml(author='Alice'), empad2_toml(author='')], ['', ''], ['Alice', None], ['Alice', None]),
    # overrides stick to their metadata author only
    ([empad2_toml(author='Alice')] * 3, ['Bob', '', ''], ['Alice', 'Bob', 'Bob'], ['Bob', 'Bob', 'Bob']),
    ([empad2_toml(author='Alice'), empad2_toml(author='Bob')], ['Carol', ''], ['Alice', 'Bob'], ['Carol', 'Bob']),
    ([empad2_toml(author=''), empad2_toml(author='Bob'), empad2_toml(author=''), empad2_toml(author='')],
     ['Alice', 'Carol', '', ''], [None, 'Bob', 'Alice', 'Alice'], ['Alice', 'Carol', 'Alice', 'Alice']),
    # EMPAD1 has no author
    ([empad1_xml()], [''], [None], [None]),
    ([empad1_xml()] * 3, ['Bob', '', ''], [None, 'Bob', 'Bob'], ['Bob', 'Bob', 'Bob']),
])
def test_process_empad_author(tmp_path, monkeypatch, datasets, answers, offered, written):
    (actual_offered, actual_written) = process_datasets(tmp_path, monkeypatch, datasets, {'Author': answers})
    assert actual_offered['Author'] == offered
    assert [m.get('author') for m in actual_written] == written


@pytest.mark.parametrize(('prompt', 'key', 'answers', 'offered', 'written'), [
    # scaled defaults are formatted, accepting them writes the exact value
    ('Diffraction pixel spacing', 'diff_step', ['', ''], ['0.529', '0.529'], [0.52882, 0.52882]),
    ('Single-electron intensity', 'adu', ['', ''], ['191', '191'], [190.95, 190.95]),
    # entered values are offered as typed
    ('Diffraction pixel spacing', 'diff_step', ['0.52882', ''], ['0.529', '0.52882'], [0.52882, 0.52882]),
    ('Single-electron intensity', 'adu', ['190.5', ''], ['191', '190.5'], [190.5, 190.5]),
])
def test_process_empad_scaled_defaults(tmp_path, monkeypatch, prompt, key, answers, offered, written):
    from phaser.cli.process_empad import ProcessEmpadConfig

    # 300 kV, 0.5766 m camera length datasets, with defaults at 200 kV and twice the camera length
    config = ProcessEmpadConfig(diff_step={300: (1.1532, 0.26441)}, adu=(200, 127.3))
    (actual_offered, actual_written) = process_datasets(
        tmp_path, monkeypatch, [empad2_toml()] * 2, {prompt: answers}, config
    )
    assert actual_offered[prompt] == offered
    assert [m[key] for m in actual_written] == pytest.approx(written)


def test_read_pymultislicer_metadata():
    metadata = EmpadMetadata.from_json(INPUT_FILES_PATH / 'sim_metadata.json')

    print(repr(metadata))
    assert metadata.file_type == 'pyMultislicer_metadata'
    assert metadata.name == 'SiC-ErV_kh_-20focus_320mm'
    assert metadata.voltage == 200000.0
    assert metadata.conv_angle == 18.9
    assert metadata.defocus == 20.0e-9
    assert metadata.diff_step == 1.1083125
    assert metadata.scan_rotation == 0.0
    assert metadata.scan_shape == (44, 46)
    assert metadata.scan_fov == pytest.approx((1.4607789440000002e-09, 1.5272094792000002e-09))
    assert metadata.scan_step == pytest.approx((3.32e-11,) * 2)


def test_metadata_bad_version():
    with open(INPUT_FILES_PATH / 'metadata_v2.json') as f:
        obj = json.load(f)

    obj['version'] = '2.5'
    with pytest.raises(ConvertError, match="Version 2.5 is not supported, version 2.0 required"):
        EmpadMetadata.from_data(obj)

    obj['version'] = '1_5'
    with pytest.raises(ConvertError, match="Invalid version string"):
        EmpadMetadata.from_data(obj)