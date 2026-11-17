"""Original-model runtime tests: no pretrained downloads or GPU execution."""
from __future__ import annotations

import importlib.util
import json
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('h286_runtime', ROOT / 'notebooks/qwen3_h286_runtime.py')
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def test_loader_kwargs_keep_original_numerics_and_pins():
    fake = types.SimpleNamespace(float32='FP32')
    assert R.model_load_kwargs(fake) == {'revision': R.REVISION, 'dtype': 'FP32', 'device_map': {'': 0},
                                        'attn_implementation': 'sdpa', 'trust_remote_code': False,
                                        'use_safetensors': True, 'local_files_only': True}
    assert len(R.ARMS) == 5
    assert R.ARMS[3]['beta'] == R.ARMS[4]['beta'] == 1 / 1.08
    assert R.ARMS[3]['mode'] == 'random'


def test_missing_reference_fails_with_diagnostic_zip_before_runtime(tmp_path):
    result = R.run(tmp_path / 'missing_hn.json', tmp_path / 'p.json', tmp_path / 'ref.json',
                   tmp_path / 'items.json', tmp_path / 'out')
    assert result['state'] == 'failed' and result['stage'] == 'reference'
    with zipfile.ZipFile(tmp_path / 'out/qwen3_h286_results.zip') as bundle:
        assert {'status.json', 'metrics.json', 'traceback.txt'} <= set(bundle.namelist())
        assert json.loads(bundle.read('metrics.json'))['pretrained_model_run'] is False


def test_model_plan_binds_module_sources(monkeypatch):
    from attention_probe.reference import ModelPins, Reference
    ref = Reference(tuple((l, j) for l in range(28) for j in range(16))[:286], '1' * 64, '2' * 64,
                    ModelPins(R.MODEL, R.REVISION, (('test', '3' * 64),)))
    plan = R.plan_document(ref, {'source_hn_meta_sha256': '4' * 64}, [])
    assert set(plan['module_sha256']) == {'reference.py', 'interventions.py', 'fixed_refw.py'}
    original = R.sha_file
    monkeypatch.setattr(R, 'sha_file', lambda p: '5' * 64 if str(p).endswith('interventions.py') else original(p))
    assert R.plan_document(ref, {'source_hn_meta_sha256': '4' * 64}, []) != plan


def _request():
    return {'request_id': 'r000', 'input_ids': [1, 2, 3], 'capture_positions': [0, 1, 2],
            'pulse_positions': [1], 'answer_ids': [3], 'answer_logit_positions': [1]}


def test_paired_atomic_checkpoint_resume_and_tamper_detection(tmp_path):
    output = tmp_path / 'first'
    output.mkdir()
    plan = {'test': 'fixed'}
    plan_hash = R.hashlib.sha256(R.canonical(plan)).hexdigest()
    R.atomic_json(output / 'run_plan.json', plan)
    results = [{'arm': arm} for arm in R.ARMS]
    arrays = [{'raw_z': np.zeros((1, 4, 286), dtype=np.float32)} for _ in R.ARMS]
    record = R.write_request(output, _request(), results, arrays, plan_hash, {'gpu': 'synthetic'}, {'state': 'passed', 'plan_sha256': plan_hash})
    assert R.verified_request(output, 'r000', plan_hash) == record
    with np.load(output / 'requests/r000/arrays.npz', allow_pickle=False) as data:
        assert data['raw_z'].shape == (5, 1, 4, 286)
    with pytest.raises(ValueError, match='overwrite'):
        R.write_request(output, _request(), results, arrays, plan_hash, {})
    archive = R.archive(output)
    resumed = tmp_path / 'resumed'
    resumed.mkdir()
    R.restore_resume(archive, resumed, plan_hash)
    assert R.verified_request(resumed, 'r000', plan_hash) == record
    (resumed / 'requests/r000/arrays.npz').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='SHA mismatch'):
        R.verified_request(resumed, 'r000', plan_hash)
    with pytest.raises(ValueError, match='different'):
        R.restore_resume(archive, tmp_path / 'other', 'wrong')


def test_interrupted_unpublished_chunk_is_not_a_checkpoint(tmp_path, monkeypatch):
    output = tmp_path / 'out'
    output.mkdir()
    results = [{'arm': arm} for arm in R.ARMS]
    arrays = [{'raw_z': np.zeros((1, 4, 286), dtype=np.float32)} for _ in R.ARMS]
    monkeypatch.setattr(R.os, 'rename', lambda *_: (_ for _ in ()).throw(OSError('simulated interrupted publish')))
    with pytest.raises(OSError, match='interrupted'):
        R.write_request(output, _request(), results, arrays, 'plan', {})
    assert R.verified_request(output, 'r000', 'plan') is None
    assert not list((output / 'requests').iterdir())


def test_prespecified_items_are_fixed_paired_and_short():
    raw = (ROOT / 'attention_probe/pilot_items.json').read_bytes()
    assert R.hashlib.sha256(raw).hexdigest() == R.ITEMS_SHA256
    items = json.loads(raw)
    assert len(items) == 16
    for pair in {item['pair_id'] for item in items}:
        values = [item for item in items if item['pair_id'] == pair]
        assert len(values) == 2 and values[0]['target'] == values[1]['target']


@pytest.fixture(scope='module')
def tiny():
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    if transformers.__version__ != R.TRANSFORMERS:
        pytest.skip('Pinned optional runtime is transformers==5.22.2')
    from transformers import Qwen3Config, Qwen3ForCausalLM
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(72)
        config = Qwen3Config(vocab_size=48, hidden_size=16, intermediate_size=16, num_hidden_layers=28,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=8)
        config._attn_implementation = 'sdpa'
        model = Qwen3ForCausalLM(config).float().eval().requires_grad_(False)
    yield torch, model
    torch.set_num_threads(old)


def test_native_5222_five_arms_raw_cett_teacher_forcing_and_cleanup(tiny):
    torch, model = tiny
    coordinates = tuple((layer, channel) for layer in range(28) for channel in range(16))[:286]
    request = {'request_id': 'r000', 'ids': torch.tensor([[1, 2, 3, 4, 5, 6]]),
               'answer_ids': [5, 6], 'answer_logit_positions': [3, 4],
               'capture_positions': [2, 3, 4, 5], 'pulse_positions': [3]}
    stats = {'refw': {'c_mu': [0.] * 286, 'c_sd': [1.] * 286}}
    initial = R.parameter_evidence(model)
    rng = torch.random.get_rng_state().clone()
    with torch.inference_mode():
        base = model(input_ids=request['ids'], attention_mask=torch.ones_like(request['ids']), use_cache=False).logits
    outputs = [R.run_arm(model, request, coordinates, stats, arm, efficient_gpu=False, include_logits=True)
               for arm in R.ARMS]
    sham, sham_arrays = outputs[0]
    assert np.array_equal(sham_arrays['technical_logits'], base[:, [3, 4]].numpy())
    manual = base[:, [3, 4]].log_softmax(-1).gather(-1, torch.tensor([[[5], [6]]])).flatten()
    np.testing.assert_allclose(sham['answer_token_logprobs'], manual.numpy())
    assert sham_arrays['raw_z'].shape == (1, 4, 286)
    assert all(f'residual_layer_{layer}' in sham_arrays for layer in (3, 7, 14, 27))
    assert np.array_equal(sham_arrays['cett'], sham_arrays['zcett'])
    mask = [i for i, (layer, _) in enumerate(coordinates) if layer < 14]
    np.testing.assert_array_equal(sham_arrays['raw_z'][..., mask], outputs[1][1]['raw_z'][..., mask])
    np.testing.assert_array_equal(sham_arrays['raw_z'][:, :1], outputs[1][1]['raw_z'][:, :1])
    assert all(result['pulse_fired'] == result['capture_fired'] == 1 for result, _ in outputs)
    assert all(np.isfinite(arr['zcett']).all() for _, arr in outputs)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert R.parameter_evidence(model) == initial
    assert sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules()) == 0


def test_cett_native_denominator_is_raw_and_formula_matches(tiny):
    torch, model = tiny
    coord = [(0, 1), (0, 3)]
    with torch.inference_mode(), R.cett_tap(model, coord, [2]) as observed:
        hook_values = []
        h = model.model.layers[0].mlp.down_proj.register_forward_hook(
            lambda _m, args, out: hook_values.append((args[0].clone(), out.clone())))
        try:
            model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
        finally:
            h.remove()
    z, native = hook_values[0]
    den = native.norm(dim=-1)[:, [2]]
    expected = z[:, [2]][:, :, [1, 3]].abs() * model.model.layers[0].mlp.down_proj.weight[:, [1, 3]].norm(dim=0) / den[..., None].clamp_min(1e-9)
    torch.testing.assert_close(observed['denominator'], den[..., None].expand_as(expected))
    torch.testing.assert_close(observed['cett'], expected)


def test_complete_resume_keeps_bound_technical_evidence_without_gpu(tmp_path, monkeypatch):
    output = tmp_path / 'complete'
    output.mkdir()
    plan = {'test': 'fixed_complete_resume'}
    plan_hash = R.hashlib.sha256(R.canonical(plan)).hexdigest()
    R.atomic_json(output / 'run_plan.json', plan)
    proof = {'state': 'passed', 'plan_sha256': plan_hash, 'checks': {'sham': {'bitwise_equal': True}}}
    results = [{'arm': arm} for arm in R.ARMS]
    arrays = [{'raw_z': np.zeros((1, 1, 1), dtype=np.float32)} for _ in R.ARMS]
    for index in range(16):
        request = {**_request(), 'request_id': f'r{index:03d}'}
        R.write_request(output, request, results, arrays, plan_hash, {'gpu': 'previous'}, proof)
    monkeypatch.setattr(R, 'load_inputs', lambda *_: (object(), {}, [None] * 16))
    monkeypatch.setattr(R, 'plan_document', lambda *_: plan)
    monkeypatch.setattr(R, 'hardware_preflight', lambda *_: (_ for _ in ()).throw(AssertionError('must not acquire GPU')))
    status = R.run('hn', 'policy', 'refw', 'items', output)
    assert status['state'] == 'complete' and status['resumed_without_gpu'] is True
    metrics = json.loads((output / 'metrics.json').read_text())
    assert metrics['checks_executed_this_attempt'] is False
    assert metrics['technical_checks_from_completed_requests'] == [proof] * 16
