"""Actual authored notebook wrapper + real compact exporter, no model/network.

Only pip/helper subprocesses and Colab download are stubbed. The compact exporter
runs from the notebook's hash-checked extracted source, on real synthetic NPZs.
"""
# ruff: noqa: S102 -- Execute only the repository-authored notebook cells under explicit stubs.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _builder():
    spec = importlib.util.spec_from_file_location('b14_b18_compact_wrapper_builder', ROOT/'notebooks/build_b14_b18.py')
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _zip_entries(path):
    with zipfile.ZipFile(path) as packed:
        assert packed.testzip() is None
        assert len(packed.namelist()) == len(set(packed.namelist()))
        return {name: packed.read(name) for name in packed.namelist()}


def test_actual_wrapper_downloads_lossless_compact_and_original_raw_archives(tmp_path, monkeypatch):
    builder = _builder()
    refs = {name: json.dumps({'synthetic':name}).encode() for name in builder.REFERENCE_SHA256}
    anchors = {name:_sha(raw) for name,raw in refs.items()}
    # Only the real exporter is imported by the wrapper. Scientific source files
    # are inert placeholders because the model subprocess is deliberately fake.
    sources = {'attention_probe/__init__.py':'',
               'attention_probe/result_bundle.py':(ROOT/'attention_probe/result_bundle.py').read_text(),
               'qwen3_b14_b18_runtime.py':'"""Synthetic helper; never executed."""\n'}
    document = builder.notebook_document(sources,refs,expected_references=anchors)
    codes = [''.join(cell['source']) for cell in document['cells'] if cell['cell_type']=='code']
    namespace = {'__name__':'b14_b18_offline_compact_wrapper_test'}
    exec(compile(codes[0],'config','exec'),namespace)
    exec(compile(codes[1],'payload','exec'),namespace)
    namespace['RUN_DIRECTORY'] = str(tmp_path/'work')
    versions = {'torch':'2.14.1+cu130','transformers':'old','accelerate':'old'}
    calls,downloads = [],[]
    original_npz,original_record,original_plan = None,None,None
    states = {
        'o0_clean_raw_z':np.arange(286,dtype=np.float32).reshape(1,1,286)/32,
        'o0_clean_residual_14':np.array([0.,-0.,np.finfo(np.float32).tiny,-np.finfo(np.float32).tiny,
                                       .25,-.5,1.,-2.],np.float32).reshape(1,1,8),
        'jac_source_delta':np.linspace(-.1,.1,8,dtype=np.float32),
        'jac_target_actual':np.linspace(.3,-.1,8,dtype=np.float32),
        'jac_target_predicted':np.linspace(.2,-.15,8,dtype=np.float32),
        'jac_fd_0_25':np.linspace(.19,-.16,8,dtype=np.float32),
        'transport_delta_ridge_transport':np.linspace(.01,.08,8,dtype=np.float32),
        'generation_clean_raw_z':np.arange(4*286,dtype=np.float32).reshape(1,4,286)/16,
        'generation_clean_residual_layer_27':np.arange(32,dtype=np.float32).reshape(1,4,8)/8,
    }
    rng = np.random.default_rng(20261118)
    logits = {name:rng.standard_normal(65536).astype(np.float32) for name in
              ('o0_clean_logits','math_clean_logits','math_pulse_logits','patch18_projection_replay_logits')}
    all_arrays = dict(states,**logits)
    fake_torch = types.ModuleType('torch')
    fake_torch.version = types.SimpleNamespace(cuda='13.0')
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda:True,get_device_capability=lambda _index:(8,0),
        mem_get_info=lambda _index:(79*1024**3,80*1024**3),
        get_device_name=lambda _index:'synthetic A100; no actual GPU accessed')

    class Child:
        def __init__(self):
            self.stdout = io.StringIO('Status: synthetic wrapper/export integration\nGespeichert fixture\n')
        def wait(self,timeout=None):
            return 0
        def poll(self):
            return 0

    def popen(command,**kwargs):
        nonlocal original_npz,original_record,original_plan
        calls.append(command)
        if command[1:4] == ['-m','pip','install']:
            assert 'torch' not in command
            versions.update(transformers=builder.TRANSFORMERS_VERSION,accelerate=builder.ACCELERATE_VERSION)
            return Child()
        assert command[1].endswith('/qwen3_b14_b18_runtime.py')
        assert kwargs['env']['PYTHONPATH'] == str(namespace['code'])
        for flag,name in (('--hn-set','hn_set.json'),('--policy','htq_policy.json'),('--refw','refw_h286.json'),
                          ('--items','b14_b18_items.json'),('--plan','b14_b18_plan.json')):
            assert Path(command[command.index(flag)+1]).read_bytes() == refs[name]
        output = Path(command[command.index('--out-dir')+1])
        folder = output/'requests/b14b18_000'
        folder.mkdir(parents=True)
        np.savez_compressed(folder/'arrays.npz',**all_arrays)
        original_npz = (folder/'arrays.npz').read_bytes()
        original_record = json.dumps({'schema':'synthetic_export_fixture','state':'complete',
            'request_id':'b14b18_000','npz_sha256':_sha(original_npz),
            'measurements':[{'metric_dtype':'float64','clean_to_arm_KL':.125}],
            'note':'States are synthetic; no model inference was performed.'},sort_keys=True).encode()
        (folder/'result.json').write_bytes(original_record)
        original_plan = json.dumps({'schema':'synthetic_export_fixture','resume_binding':'original unchanged plan'},sort_keys=True).encode()
        (output/'run_plan.json').write_bytes(original_plan)
        (output/'status.json').write_text(json.dumps({'state':'complete','stage':'complete','completed_inputs':1}))
        (output/'metrics.json').write_text(json.dumps({'weights_loaded':False,
            'wrapper_attempt_id':kwargs['env']['H286_WRAPPER_ATTEMPT'],'no_model_executed':True}))
        return Child()

    google,colab = types.ModuleType('google'),types.ModuleType('google.colab')
    colab.files = types.SimpleNamespace(download=lambda path:downloads.append(Path(path)))
    google.colab = colab
    original_is_dir = Path.is_dir
    console = io.StringIO()
    with monkeypatch.context() as patch:
        patch.chdir(tmp_path)
        patch.setattr(Path,'is_dir',lambda path:False if str(path)=='/content' else original_is_dir(path))
        patch.setitem(sys.modules,'torch',fake_torch)
        patch.setitem(sys.modules,'google',google)
        patch.setitem(sys.modules,'google.colab',colab)
        patch.setattr(importlib.metadata,'version',lambda name:versions[name])
        patch.setattr(subprocess,'Popen',popen)
        with contextlib.redirect_stdout(console):
            exec(compile(codes[2],'run','exec'),namespace)
    compact,raw = tmp_path/'b14_b18_results.zip',tmp_path/'b14_b18_results_RAW.zip'
    assert downloads == [compact,raw],console.getvalue()
    assert len(calls) == 2  # Only the fake pip and fake model helper, no other process.
    raw_entries,compact_entries = _zip_entries(raw),_zip_entries(compact)
    assert compact.stat().st_size < raw.stat().st_size/4
    request_npz = 'requests/b14b18_000/arrays.npz'
    compact_npz = 'requests/b14b18_000/state_arrays.npz'
    assert raw_entries[request_npz] == original_npz
    assert raw_entries['requests/b14b18_000/result.json'] == original_record
    assert raw_entries['run_plan.json'] == original_plan
    # The exact original resumable checkpoint is kept only in RAW, never silently
    # replaced by the reduced state archive or a changed plan/result checksum.
    assert request_npz not in compact_entries and compact_npz in compact_entries
    with np.load(io.BytesIO(raw_entries[request_npz]),allow_pickle=False) as loaded:
        assert set(loaded.files) == set(all_arrays)
        assert all(loaded[key].tobytes(order='C') == value.tobytes(order='C') for key,value in all_arrays.items())
    with np.load(io.BytesIO(compact_entries[compact_npz]),allow_pickle=False) as loaded:
        assert set(loaded.files) == set(states)
        for key,value in states.items():
            assert loaded[key].dtype == value.dtype and loaded[key].shape == value.shape
            assert loaded[key].tobytes(order='C') == value.tobytes(order='C')  # Includes signed zero/subnormals.
    full_manifest = json.loads(raw_entries['WRAPPER_MANIFEST.json'])
    assert set(full_manifest) == set(raw_entries)-{'WRAPPER_MANIFEST.json'}
    assert all(_sha(raw_entries[name]) == sha for name,sha in full_manifest.items())
    assert compact_entries['full_archive_manifest.json'] == raw_entries['WRAPPER_MANIFEST.json']
    for name,value in raw_entries.items():
        if name.endswith('.npz') or name == 'WRAPPER_MANIFEST.json':
            continue
        assert compact_entries[name] == value
    package = json.loads(compact_entries['COMPACT_PACKAGE.json'])
    assert package['archive_kind'] == 'analysis_summary_not_resume' and package['resume_eligible'] is False
    assert package['full_archive'] == {'filename':raw.name,'sha256':_sha(raw.read_bytes()),'bytes':raw.stat().st_size}
    assert package['metrics_source'] == 'runtime_recorded_fp64'
    assert package['independent_recompute_available']['state_geometry'] is True
    assert package['independent_recompute_available']['jvp_error'] is True
    assert package['independent_recompute_available']['full_vocab_KL'] is False
    transforms = package['npz_transforms']
    assert len(transforms) == 1
    transform = transforms[0]
    assert transform['original_path'] == request_npz and transform['original_sha256'] == _sha(original_npz)
    assert transform['transformed_path'] == compact_npz
    assert transform['transformed_sha256'] == _sha(compact_entries[compact_npz])
    assert set(transform['included_keys']) == set(states) and set(transform['excluded_keys']) == set(logits)
    for name,value in states.items():
        metadata = transform['included_arrays'][name]
        assert np.dtype(metadata['dtype']) == value.dtype and metadata['shape'] == list(value.shape)
        assert metadata['nbytes'] == value.nbytes and metadata['sha256'] == _sha(value.tobytes(order='C'))
    for name,value in logits.items():
        assert transform['excluded_arrays'][name]['sha256'] == _sha(value.tobytes(order='C'))
    manifest = json.loads(compact_entries['COMPACT_MANIFEST.json'])
    assert set(manifest) == set(compact_entries)-{'COMPACT_MANIFEST.json'}
    for name,metadata in manifest.items():
        assert metadata == {'sha256':_sha(compact_entries[name]),'bytes':len(compact_entries[name])}
