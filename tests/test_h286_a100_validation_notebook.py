"""Offline run-all packaging checks; no GPU, package install, or model runtime."""
# ruff: noqa: S102 -- Execute authored notebook cells with explicitly stubbed runtimes.
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('validation_builder', ROOT / 'notebooks/build_h286_a100_validation.py')
assert SPEC is not None and SPEC.loader is not None
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


def fixture_document(private=True):
    references = {name: json.dumps({'synthetic': name}).encode() for name in B.REFERENCE_SHA256}
    anchors = {name: hashlib.sha256(raw).hexdigest() for name, raw in references.items()}
    modules = {name: '"""Synthetic source; never executed in this packaging test."""\n'
               for name in ('attention_probe/__init__.py', 'attention_probe/reference.py',
                            'attention_probe/fixed_refw.py', 'attention_probe/interventions.py',
                            'qwen3_h286_runtime.py', 'qwen3_h286_validation_runtime.py',
                            'attention_probe/answer_validation.py')}
    modules['attention_probe/pilot_items.json'] = '[{"fixture":true}]\n'
    return B.notebook_document(modules, references if private else {}, expected_references=anchors), references


def execute(tmp_path, monkeypatch, *, private=True, cuda=True, free_gib=20, helper_code=0,
            helper_state='complete', loaded=True, metrics_mode='fresh', pip_code=0,
            status_loaded=False, flood=False, default_directory=False):
    document, references = fixture_document(private)
    namespace = {'__name__': 'offline_validation_test'}
    codes = [''.join(cell['source']) for cell in document['cells'] if cell['cell_type'] == 'code']
    exec(compile(codes[0], 'config', 'exec'), namespace)
    exec(compile(codes[1], 'payload', 'exec'), namespace)
    namespace['RUN_DIRECTORY'] = '' if default_directory else str(tmp_path / 'work')
    versions = {'torch': '2.14.1+cu130', 'transformers': 'old', 'accelerate': 'old'}
    commands, downloads = [], []
    fake_torch = types.ModuleType('torch')
    fake_torch.version = types.SimpleNamespace(cuda='13.0')
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda, get_device_capability=lambda _index: (8, 0),
        mem_get_info=lambda _index: (int(free_gib * 1024**3), 40 * 1024**3),
        get_device_name=lambda _index: 'synthetic A100',
    )

    class Child:
        def __init__(self, code):
            self.code = code
            self.stdout = io.StringIO(('library noise\n' * 5000 if flood else '')
                                      + 'Status: validation\nRequest 1: fixture\nGespeichert fixture\n')

        def wait(self, timeout=None):
            return self.code

        def poll(self):
            return self.code

    def popen(command, **kwargs):
        commands.append(command)
        assert kwargs['env']['HF_HUB_DISABLE_PROGRESS_BARS'] == '1'
        assert kwargs['env']['TQDM_DISABLE'] == '1'
        if command[1:4] == ['-m', 'pip', 'install']:
            assert '--constraint' in command and 'torch' not in command
            constraint = Path(command[command.index('--constraint') + 1])
            assert constraint.read_text() == 'torch==2.14.1+cu130\n'
            assert 'transformers==5.22.2' in command and 'accelerate==1.16.0' in command
            versions.update(transformers=B.TRANSFORMERS_VERSION, accelerate=B.ACCELERATE_VERSION)
            return Child(pip_code)
        assert command[1].endswith('/qwen3_h286_validation_runtime.py')
        assert Path(command[command.index('--cache-dir') + 1]) == tmp_path / 'h286_model_cache'
        assert kwargs['env']['PYTHONPATH'] == str(namespace['code'])
        for flag, name in (('--hn-set', 'hn_set.json'), ('--policy', 'htq_policy.json'), ('--refw', 'refw_h286.json')):
            assert Path(command[command.index(flag) + 1]).read_bytes() == references[name]
        output = Path(command[command.index('--out-dir') + 1])
        (output / 'status.json').write_text(json.dumps({'state': helper_state, 'pretrained_weights_loaded': status_loaded}))
        if metrics_mode != 'keep':
            metrics = {'weights_loaded': loaded,
                       'wrapper_attempt_id': kwargs['env']['H286_WRAPPER_ATTEMPT'] if metrics_mode == 'fresh' else 'old-attempt'}
            (output / 'metrics.json').write_text(json.dumps(metrics))
        (output / 'validation.json').write_text('{"synthetic":true}')
        return Child(helper_code)

    google, colab = types.ModuleType('google'), types.ModuleType('google.colab')
    colab.files = types.SimpleNamespace(download=lambda path: downloads.append(Path(path)))
    google.colab = colab
    original_is_dir = Path.is_dir
    error, console = None, io.StringIO()
    with monkeypatch.context() as patch:
        patch.chdir(tmp_path)
        patch.setattr(Path, 'is_dir', lambda path: False if str(path) == '/content' else original_is_dir(path))
        patch.setitem(sys.modules, 'torch', fake_torch)
        patch.setitem(sys.modules, 'google', google)
        patch.setitem(sys.modules, 'google.colab', colab)
        patch.setattr(importlib.metadata, 'version', lambda name: versions[name])
        patch.setattr(subprocess, 'Popen', popen)
        try:
            with contextlib.redirect_stdout(console):
                exec(compile(codes[2], 'run', 'exec'), namespace)
        except RuntimeError as exc:
            error = exc
    target = tmp_path / ('h286_a100_validation_diagnostic.zip' if error else 'h286_a100_validation_results.zip')
    assert downloads == [target]
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries['WRAPPER_MANIFEST.json'])
    assert set(manifest) == set(entries) - {'WRAPPER_MANIFEST.json'}
    assert all(hashlib.sha256(entries[name]).hexdigest() == value for name, value in manifest.items())
    return error, json.loads(entries['status.json']), entries, commands, console.getvalue()


@pytest.mark.parametrize('loaded', [True, False])
def test_success_uses_fresh_child_metrics_and_new_runtime(tmp_path, monkeypatch, loaded):
    error, status, entries, commands, _ = execute(tmp_path, monkeypatch, loaded=loaded, status_loaded=not loaded)
    assert error is None and status['state'] == 'complete'
    assert status['pretrained_weights_loaded'] is loaded
    assert status['weights_loaded_evidence'] == 'fresh_child_metrics'
    assert status['purpose'] == 'original_qwen3_h286_answer_validation'
    assert len(commands) == 2 and 'validation.json' in entries
    assert json.loads(entries['environment_installed.json'])['torch'] == '2.14.1+cu130'


@pytest.mark.parametrize('loaded', [True, False])
def test_child_failure_keeps_fresh_load_evidence_even_if_child_claims_complete(tmp_path, monkeypatch, loaded):
    error, status, entries, _, _ = execute(tmp_path, monkeypatch, helper_code=1, loaded=loaded,
                                         status_loaded=not loaded)
    assert error and status['state'] == 'failed' and status['wrapper_verified'] is False
    assert status['pretrained_weights_loaded'] is loaded
    assert json.loads(entries['helper_status.json'])['state'] == 'complete'


def test_stale_metrics_from_previous_attempt_cannot_claim_loaded_weights(tmp_path, monkeypatch):
    execute(tmp_path, monkeypatch, loaded=True)
    error, status, _, _, _ = execute(tmp_path, monkeypatch, helper_code=1, metrics_mode='keep')
    assert error and status['pretrained_weights_loaded'] is None
    assert status['weights_loaded_evidence'] == 'missing_unreadable_or_stale_child_metrics'


@pytest.mark.parametrize('kwargs', [{'cuda': False}, {'free_gib': 11}, {'private': False}, {'pip_code': 1}])
def test_early_failure_never_starts_model_runtime(tmp_path, monkeypatch, kwargs):
    error, status, _, commands, _ = execute(tmp_path, monkeypatch, **kwargs)
    assert error and status['state'] == 'failed' and status['pretrained_weights_loaded'] is False
    assert len(commands) == (1 if kwargs.get('pip_code') else 0)


def test_paused_is_explicit_and_console_stays_small(tmp_path, monkeypatch):
    error, status, entries, _, console = execute(tmp_path, monkeypatch, helper_state='paused', flood=True)
    assert error is None and status['state'] == 'paused' and 'kein PASS' in console
    assert len(console) < 15000 and 'library noise' not in console
    assert entries['runtime.log'].count(b'library noise') == 5000


def test_new_default_folder_is_stable_and_isolates_old_pilot(tmp_path, monkeypatch):
    old = tmp_path / 'h286_attention_pilot_previous'
    old.mkdir()
    (old / 'original').write_text('preserve')
    execute(tmp_path, monkeypatch, default_directory=True)
    folders = list(tmp_path.glob('h286_a100_validation_*'))
    folders = [path for path in folders if path.is_dir()]
    assert len(folders) == 1
    execute(tmp_path, monkeypatch, default_directory=True)
    assert (old / 'original').read_text() == 'preserve'
    assert [path for path in tmp_path.glob('h286_a100_validation_*') if path.is_dir()] == folders


def test_renderer_has_three_code_cells_and_does_not_mutate_base():
    before_run, before_intro = B.BASE.RUN, B.BASE.INTRO
    old_sources = B.BASE.source_bundle()
    document, _ = fixture_document(private=False)
    assert B.BASE.RUN == before_run and B.BASE.INTRO == before_intro and B.BASE.source_bundle() == old_sources
    assert len([cell for cell in document['cells'] if cell['cell_type'] == 'code']) == 3
    assert len({cell['id'] for cell in document['cells']}) == len(document['cells'])
    namespace = {}
    exec(''.join(document['cells'][2]['source']), namespace)
    assert namespace['REFERENCE_FILES_B64'] == {}
    assert document['metadata']['colab']['name'] == 'h286_a100_validation.ipynb'


def test_all_source_bindings_and_public_notebook_reproduce(tmp_path):
    sources = B.source_bundle()
    old = B.BASE.source_bundle()
    assert len(sources) == 8 and all(sources[name] == value for name, value in old.items())
    assert sources['qwen3_h286_runtime.py'] == (ROOT / 'notebooks/qwen3_h286_runtime.py').read_text()
    assert sources['qwen3_h286_validation_runtime.py'] == (ROOT / 'notebooks/qwen3_h286_validation_runtime.py').read_text()
    assert sources['attention_probe/answer_validation.py'] == (ROOT / 'attention_probe/answer_validation.py').read_text()
    path = B.build(tmp_path / 'public.ipynb')
    assert json.loads(path.read_text()) == json.loads((ROOT / 'notebooks/h286_a100_validation.ipynb').read_text())
    namespace = {}
    exec(''.join(json.loads(path.read_text())['cells'][2]['source']), namespace)
    assert namespace['REFERENCE_FILES_B64'] == {} and namespace['MODULE_SOURCES'] == sources


def test_private_build_cannot_write_inside_repo(tmp_path):
    refs = {}
    for name in B.REFERENCE_SHA256:
        refs[name] = tmp_path / name
        refs[name].write_bytes(b'synthetic')
    with pytest.raises(ValueError, match='outside'):
        B.build(ROOT / 'notebooks/forbidden_private.ipynb', reference_paths=refs)
    assert not (ROOT / 'notebooks/forbidden_private.ipynb').exists()


def test_ambiguous_base_wrapper_anchor_fails_closed():
    with pytest.raises(ValueError, match='ambiguous'):
        B._replace_once('anchor twice anchor', 'anchor', 'new')
