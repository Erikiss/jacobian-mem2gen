"""Synthetic temporal protocol tests; no private data or pretrained downloads."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))


def runtime():
    return importlib.import_module("qwen3_temporal_functional_runtime")


@pytest.fixture(scope="module")
def native_engine():
    """Small native random model, with EOS forced to bound generation tests."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from attention_probe.reference import ModelPins, Reference

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(4031)
        config = Qwen3Config(
            vocab_size=41, hidden_size=16, intermediate_size=64,
            num_hidden_layers=28, num_attention_heads=2,
            num_key_value_heads=1, head_dim=8, eos_token_id=0,
            bos_token_id=1, pad_token_id=0,
        )
        config._attn_implementation = "sdpa"
        model = Qwen3ForCausalLM(config).float().eval().requires_grad_(False)
        # Equal logits force the deterministic argmax/EOS id zero immediately,
        # without changing the nonzero random backbone or cache dynamics.
        with torch.no_grad():
            model.lm_head.weight.zero_()

    class Tokenizer:
        eos_token_id = 0
        pad_token_id = 0

        def __call__(self, _text, **_kwargs):
            return {"input_ids": [11 + i % 20 for i in range(32)]}

        def decode(self, values, **_kwargs):
            return " ".join(str(int(value)) for value in values)

    coordinates = tuple((layer, channel) for layer in range(28) for channel in range(10))
    coordinates += tuple((layer, 10) for layer in range(6))
    reference = Reference(coordinates, "1" * 64, "2" * 64,
                          ModelPins("Qwen/Qwen3-1.7B",
                                    "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                                    (("synthetic.safetensors", "3" * 64),)))
    stats = {"refw": {"c_mu": [0.0] * 286, "c_sd": [1.0] * 286},
             "source_hn_meta_sha256": "4" * 64}
    items = [
        {"id": f"synthetic-{index}",
         "request_id": f"{'s3' if index % 8 < 4 else 'adet'}:synthetic-{index}",
         "family": "s3" if index % 8 < 4 else "adet",
         "split": "calibration" if index < 8 else "test",
         "prompt_ids": [1, 5, 7, 9, 2, 3, 4, 8],
         "prompt_ids_sha256": hashlib.sha256(b'[1,5,7,9,2,3,4,8]').hexdigest(),
         "fixed_continuation_text": f"synthetic gold {index}",
         "gold_token_ids": [3, 4], "foil_token_ids": [5, 6, 7],
         "gold_text": "3 4", "foil_text": "5 6 7",
         "prompt": f"synthetic prompt {index}"}
        for index in range(16)
    ]
    engine = {"model": model, "tokenizer": Tokenizer(),
              "hardware": {"device": "cpu", "scope": "synthetic native test"}}
    yield torch, engine, reference, stats, items
    torch.set_num_threads(old_threads)


def fitted_candidate(width=16, *, invalid=False, early_value=1e-3, late_value=-2e-4,
                     cache_error8=0.0, cache_error32=0.0):
    from attention_probe.temporal_candidate import fit_candidate

    ids = [f"synthetic-{index}" for index in range(8)]
    early = np.zeros(width, dtype=np.float32)
    late = np.zeros(width, dtype=np.float32)
    if not invalid:
        early[0], late[0] = early_value, late_value
    rows = [
        {"id": name, "split": "calibration", "delta8": early.tolist(),
         "delta32": late.tolist(), "cache_error8": cache_error8, "cache_error32": cache_error32,
         "checkpoint_sha256": f"{index:064x}"}
        for index, name in enumerate(ids)
    ]
    return fit_candidate(rows, calibration_ids=ids,
                         source_binding={"synthetic_plan_sha256": "5" * 64})


def bound_record(record, plan_hash='a'*64):
    R = runtime()
    value = copy.deepcopy(record)
    value.update(schema=R.SCHEMA, state='complete', plan_sha256=plan_hash)
    value['technical_checks'] = {'state':'passed', 'schema':R.SCHEMA,
                                 'plan_sha256':plan_hash, 'checks':value['checks']}
    return value


def document_and_split(items):
    return ({'items':items, 'neutral_token_ids':[11+i % 20 for i in range(32)]},
            {'calibration_ids':[item['id'] for item in items[:8]],
             'test_ids':[item['id'] for item in items[8:]]})


@pytest.fixture(scope="module")
def baselines(native_engine):
    R = runtime()
    torch, engine, reference, stats, items = native_engine
    before = tuple((name, p.data_ptr(), p._version) for name, p in engine['model'].named_parameters())
    rng = torch.random.get_rng_state().clone()
    neutral = [11 + i % 20 for i in range(32)]
    record, arrays, histories = R.run_baselines(engine, reference, stats, items[8], neutral)
    assert before == tuple((name, p.data_ptr(), p._version) for name, p in engine['model'].named_parameters())
    assert torch.equal(rng, torch.random.get_rng_state())
    assert R.hook_count(engine['model']) == 0
    return record, arrays, histories, neutral


@pytest.fixture(scope='module')
def native_cases(native_engine):
    R = runtime()
    _, engine, reference, stats, items = native_engine
    neutral = [11+i % 20 for i in range(32)]
    calibration = R.compute_calibration(engine, reference, stats, items[0], neutral)
    valid, invalid = fitted_candidate(), fitted_candidate(invalid=True)
    test = R.compute_test(engine, reference, stats, items[8], neutral, valid)
    skipped = R.compute_test(engine, reference, stats, items[8], neutral, invalid)
    return calibration, (valid, *test), (invalid, *skipped)


@pytest.fixture(scope='module')
def frozen_store(tmp_path_factory, native_engine, native_cases):
    """Eight real synthetic-calibration checkpoints plus a matched test case."""
    R = runtime()
    _, engine, reference, stats, items = native_engine
    output = tmp_path_factory.mktemp('temporal-frozen')
    document, split = document_and_split(items)
    plan = {'schema':R.SCHEMA, 'synthetic':'source-bound calibration and test'}
    plan_hash = hashlib.sha256(R.base.canonical(plan)).hexdigest()
    R.base.atomic_json(output/'run_plan.json', plan)
    original_record, arrays = native_cases[0]
    for item in items[:8]:
        record = copy.deepcopy(original_record)
        record['item'] = item
        R.write_checkpoint(output, record, arrays, plan_hash, {'device':'cpu/synthetic'})
    candidate = R.freeze_candidate(output, items[:8], plan_hash)
    test, test_arrays = R.compute_test(engine, reference, stats, items[8], document['neutral_token_ids'], candidate)
    saved = R.write_checkpoint(output, test, test_arrays, plan_hash, {'device':'cpu/synthetic'}, candidate)
    return output, document, split, plan, plan_hash, candidate, saved


def test_fixed_temporal_offsets_endpoints_and_generation_scope():
    R = runtime()
    assert R.OFFSETS == (0, 1, 8, 32)
    assert R.MAIN_ARMS == ('clean', 'pulse', 'clean_add_shared', 'pulse_remove_shared')
    assert len(R.ENDPOINT_ARMS) == len(set(R.ENDPOINT_ARMS)) == 23
    assert len(R.CACHE_ARMS) == 3
    assert sum('noise_' in name for name in R.ENDPOINT_ARMS) == 16
    assert R.GREEDY_CAP == 64


def test_native_cached_trajectories_use_fixed_chunks_and_original_pulse_position(native_engine, baselines):
    R = runtime()
    torch, _, _, _, items = native_engine
    record, arrays, histories, neutral = baselines
    length = len(items[8]['prompt_ids'])
    assert record['native_capture_exact'] is True and record['beta_one_sham_exact'] is True
    assert len(record['snapshots']) == 8
    assert [(row['arm'], row['offset']) for row in record['cached_full_parity']] == [
        ('clean', 8), ('clean', 32), ('pulse', 8), ('pulse', 32)]
    for arm, history in histories.items():
        assert list(history) == [0, 1, 8, 31, 32]
        assert [history[k].evidence['consumed_input_tokens'] for k in (1, 8, 31, 32)] == [1, 7, 23, 1]
        assert history[0].evidence['pulse_absolute_positions'] == ([length-1] if arm == 'pulse' else [])
        for offset in R.OFFSETS:
            step = history[offset]
            assert step.evidence['absolute_capture_positions'] == [length+offset-1]
            assert step.evidence['cache_output_length'] == length+offset
            assert tuple(step.arrays['raw_z'].shape) == (1, 1, 286)
            if offset:
                assert step.evidence['pulse_fired'] == 0
        assert history[31].arrays == {}
    for row in record['cached_full_parity']:
        arm, offset = row['arm'], row['offset']
        a = arrays[f'{arm}_o{offset}_residual_18'].astype(np.float64)
        b = arrays[f'{arm}_full_o{offset}_residual_18'].astype(np.float64)
        assert row['B18_L2_error'] == pytest.approx(float(np.linalg.norm(a-b)), rel=1e-12, abs=1e-15)
        assert row['logits_allclose'] is True
        assert row['passed'] is True and row['arrays_allclose'] is True and row['kv_allclose'] is True
        assert row['array_keys_equal'] is True and row['cache_layer_count_equal'] is True
        assert len(row['kv_parity']) == 28
        assert all(value['native_fp32'] and value['finite'] and value['passed']
                   for value in row['arrays_parity'].values())
        assert all(value[kind]['native_fp32'] and value[kind]['finite'] and value[kind]['passed']
                   for value in row['kv_parity'] for kind in ('keys', 'values'))
        if arm == 'pulse':
            assert row['full']['forward']['pulse_absolute_positions'] == [length-1]
    assert len(neutral) == 32
    assert all(torch.isfinite(step.logits).all() for history in histories.values() for step in history.values())
    assert [row['offset'] for row in record['null_causal_checks']] == [0, 1, 8, 32]
    assert all(row['passed'] and all(check['passed'] for check in row['checks'].values())
               for row in record['null_causal_checks'])
    assert record['null_causal_checks'][0]['checks']['bridge_positions_layer15']['applicable'] is False
    assert all(row['checks']['bridge_positions_layer15']['applicable'] for row in record['null_causal_checks'][1:])


@pytest.mark.parametrize('mutation', ['capture', 'cache', 'nonfinite'])
def test_full_parity_rejects_state_drift_even_when_logits_stay_identical(native_engine, baselines, mutation):
    R = runtime()
    torch = native_engine[0]
    original = baselines[2]['clean'][8]
    changed = copy.copy(original)
    changed.arrays = {key: value.clone() for key, value in original.arrays.items()}
    changed.cache = R.cache_api().clone_cache(original.cache)
    with torch.no_grad():
        if mutation == 'cache':
            changed.cache.layers[17].values[..., -1, 0] += .25
        elif mutation == 'capture':
            changed.arrays['raw_z'][0, 0, 0] += .25
        else:
            changed.arrays['residual_18'][0, 0, 0] = float('inf')
    report = R._parity_snapshot(original, changed)
    assert report['logits']['passed'] is True
    assert report['passed'] is False
    assert report['kv_allclose'] is (mutation != 'cache')
    assert report['arrays_allclose'] is (mutation == 'cache')


@pytest.mark.parametrize('region,layer,kind,relative_position', [
    ('earlier_prompt_all_layers', 17, 'keys', -8),
    ('all_positions_layers0_through14', 14, 'values', 7),
    ('bridge_positions_layer15', 15, 'keys', 2),
])
def test_causal_nulls_detect_a_single_changed_cache_scalar(native_engine, baselines,
                                                         region, layer, kind, relative_position):
    R = runtime()
    torch = native_engine[0]
    length = len(native_engine[-1][8]['prompt_ids'])
    clean = baselines[2]['clean'][8]
    changed = copy.copy(baselines[2]['pulse'][8])
    changed.cache = R.cache_api().clone_cache(changed.cache)
    with torch.no_grad():
        getattr(changed.cache.layers[layer], kind)[..., length+relative_position, 0] += .25
    checks = R._null_causal_snapshots(clean, changed, prompt_length=length, offset=8)
    assert checks['passed'] is False and checks['checks'][region]['passed'] is False
    assert all(value['passed'] for label, value in checks['checks'].items() if label != region)
    assert any(row['changed_scalars'] > 0 for row in checks['checks'][region]['details'])


def test_full_candidate_likelihood_uses_first_endpoint_then_candidate_prefix(native_engine, baselines, monkeypatch):
    R = runtime()
    _, engine, reference, stats, items = native_engine
    step = baselines[2]['clean'][32]
    tc = R.cache_api()
    original = tc.forward_capture
    seen = []
    before = R.manifest_sha(step.cache)
    def capture(*args, **kwargs):
        seen.append((list(args[1]), kwargs.get('capture'), list(kwargs['capture_positions'])))
        return original(*args, **kwargs)
    monkeypatch.setattr(tc, 'forward_capture', capture)
    result = R.score_candidates(engine, step, items[8], reference, stats)
    start = tc.cache_length(step.cache)
    assert seen == [([3], False, [start]), ([5, 6], False, [start, start+1])]
    # A zero LM head supplies exactly uniform token probabilities: unequal
    # answer lengths alter the sum margin, while the per-token margin is zero.
    assert result['gold']['token_logprobs'] == pytest.approx([-np.log(41)]*2)
    assert result['foil']['token_logprobs'] == pytest.approx([-np.log(41)]*3)
    assert result['full_sum_margin'] == pytest.approx(np.log(41))
    assert result['first_token_margin'] == pytest.approx(0.)
    assert result['mean_per_token_margin'] == pytest.approx(0.)
    assert result['primary'] == 'full_sum_margin'
    assert result['gold']['eos_added'] is False
    assert result['gold']['cache_output_length'] == start+1
    assert result['foil']['cache_output_length'] == start+2
    assert R.manifest_sha(step.cache) == before and R.hook_count(engine['model']) == 0


def test_nonuniform_native_token_scores_match_full_teacher_forcing(native_engine, baselines, monkeypatch):
    """Unlike uniform logits, this oracle detects a one-token scoring shift."""
    R = runtime()
    torch, engine, reference, stats, items = native_engine
    with torch.random.fork_rng():
        torch.manual_seed(4133)
        head = torch.nn.Linear(16, 41, bias=False).float().eval().requires_grad_(False)
    monkeypatch.setattr(engine['model'], 'lm_head', head)
    prefix = items[8]['prompt_ids'] + baselines[3]
    step = R.cache_api().forward_capture(engine['model'], prefix, reference.coordinates, stats,
                                        capture=False)
    observed = R.score_candidates(engine, step, items[8], reference, stats)
    expected = {}
    for label in ('gold', 'foil'):
        tokens = items[8][label+'_token_ids']
        sequence = torch.tensor([prefix+tokens], dtype=torch.long)
        with torch.no_grad(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            logits = engine['model'](input_ids=sequence, attention_mask=torch.ones_like(sequence),
                                     use_cache=False).logits[0]
        selected = logits[len(prefix)-1:len(prefix)+len(tokens)-1].double().log_softmax(-1)
        expected[label] = selected[torch.arange(len(tokens)), torch.tensor(tokens)].numpy()
        assert float(selected.std()) > 1e-3
        np.testing.assert_allclose(observed[label]['token_logprobs'], expected[label], rtol=0, atol=3e-6)
        assert observed[label]['sum_logprob'] == pytest.approx(float(expected[label].sum()), abs=6e-6)
    assert observed['first_token_margin'] == pytest.approx(expected['gold'][0]-expected['foil'][0], abs=6e-6)
    assert observed['full_sum_margin'] == pytest.approx(expected['gold'].sum()-expected['foil'].sum(), abs=1e-5)
    assert R.hook_count(engine['model']) == 0


@pytest.mark.parametrize('tokens', [[], [True], [-1], [41], [0]])
def test_invalid_candidate_token_sequences_fail_before_a_forward(tokens, native_engine, baselines, monkeypatch):
    R = runtime()
    _, engine, reference, stats, _ = native_engine
    monkeypatch.setattr(R.cache_api(), 'forward_capture', lambda *_a, **_k: pytest.fail('invalid tokens reached model'))
    with pytest.raises(ValueError):
        R.score_sequence(engine, baselines[2]['clean'][32], tokens, reference, stats)


def test_native_greedy_eos_comes_from_endpoint_and_leaves_cache_unchanged(native_engine, baselines, monkeypatch):
    R = runtime()
    _, engine, reference, stats, items = native_engine
    step = baselines[2]['clean'][32]
    before = R.manifest_sha(step.cache)
    monkeypatch.setattr(R.cache_api(), 'forward_capture', lambda *_a, **_k: pytest.fail('EOS must not be fed'))
    generated = R.greedy_answer(engine, step, items[8], reference, stats)
    assert generated['selected_token_ids'] == [0] and generated['generated_token_ids'] == []
    assert generated['num_steps'] == 1 and generated['num_generated_tokens'] == 0
    assert generated['stop_reason'] == 'eos' and generated['truncated'] is False
    assert generated['steps'][0]['phase'] == 'endpoint'
    assert generated['final_cache_length'] == R.cache_api().cache_length(step.cache)
    assert generated['endpoint_cache_sha256'] == before and generated['endpoint_cache_unchanged'] is True
    assert R.manifest_sha(step.cache) == before


def test_cached_greedy_cap_never_feeds_terminal_token_or_adds_hooks(native_engine, baselines, monkeypatch):
    R = runtime()
    torch, engine, reference, stats, items = native_engine
    step = copy.copy(baselines[2]['clean'][32])
    step.logits = torch.zeros_like(step.logits)
    step.logits[..., 1] = 1.
    start = R.cache_api().cache_length(step.cache)
    calls = []
    def advance(_model, token_ids, _coords, _stats, **kwargs):
        assert token_ids == [1] and kwargs.get('capture') is False
        assert 'pulse' not in kwargs and 'residual_patch' not in kwargs
        calls.append(list(token_ids))
        return SimpleNamespace(
            cache=SimpleNamespace(length=start+len(calls)), logits=step.logits,
            evidence={'pulse_fired':0, 'residual_patch':None, 'capture_fired':0})
    original_length = R.cache_api().cache_length
    monkeypatch.setattr(R.cache_api(), 'cache_length', lambda cache: cache.length if isinstance(cache, SimpleNamespace) else original_length(cache))
    monkeypatch.setattr(R.cache_api(), 'forward_capture', advance)
    generated = R.greedy_answer(engine, step, items[8], reference, stats)
    assert generated['selected_token_ids'] == generated['generated_token_ids'] == [1]*64
    assert len(calls) == 63
    assert generated['stop_reason'] == 'max_new_tokens' and generated['truncated'] is True
    assert generated['final_cache_length'] == start+63
    assert generated['steps'][0]['phase'] == 'endpoint'
    assert all(row['phase'] == 'unhooked_cached_decode' for row in generated['steps'][1:])


def test_signed_calibration_vector_and_invalid_gate_are_explicit():
    from attention_probe.temporal_candidate import verify_candidate
    valid, invalid = fitted_candidate(), fitted_candidate(invalid=True)
    assert verify_candidate(valid) is True and verify_candidate(invalid) is True
    assert valid['valid'] is True and valid['amplitude'] < 0
    np.testing.assert_array_equal(np.asarray(valid['direction']), np.eye(16)[0])
    assert valid['vector'][0] < 0
    assert invalid['valid'] is False and len(invalid['invalid_reasons']) == 2
    assert invalid['diagnostics']['invalid_policy'].startswith('do not apply')


def test_numeric_gate_uses_separate_offset8_and_offset32_cache_floors():
    # The late floor deliberately exceeds the early signal. The frozen rule
    # compares each signal to its corresponding offset's noise floor.
    valid = fitted_candidate(early_value=1e-4, late_value=-1e-2,
                             cache_error8=0., cache_error32=2e-4)
    assert valid['diagnostics']['floor8'] == 1e-6
    assert valid['diagnostics']['floor32'] == pytest.approx(1.6e-3)
    assert valid['diagnostics']['mean_delta8_norm'] < valid['diagnostics']['floor32']
    assert valid['valid'] is True
    invalid = fitted_candidate(early_value=1e-4, late_value=-1e-2,
                               cache_error8=2e-4, cache_error32=0.)
    assert invalid['valid'] is False
    assert invalid['invalid_reasons'] == ['mean_offset8_signal_below_prespecified_numeric_floor']


def copy_store(tmp_path, frozen_store):
    source = frozen_store[0]
    destination = tmp_path/'output'
    shutil.copytree(source, destination)
    return destination


def test_native_valid_and_invalid_candidates_keep_all_fixed_slots(native_cases):
    R = runtime()
    for candidate, record, arrays in native_cases[1:]:
        assert record['state'] == 'complete' and record['checks'] == dict.fromkeys(R.PROOF_KEYS, True)
        assert [row['name'] for row in record['endpoints']] == list(R.ENDPOINT_ARMS)
        complete = [row for row in record['endpoints'] if row['state'] == 'complete']
        skipped = [row for row in record['endpoints'] if row['state'] == 'not_run']
        assert len(complete) == (23 if candidate['valid'] else 5)
        assert len(skipped) == (0 if candidate['valid'] else 18)
        assert [row['name'] for row in record['generations']] == list(R.MAIN_ARMS)
        assert sum(row['state']=='complete' for row in record['generations']) == (4 if candidate['valid'] else 2)
        for row in record['generations']:
            if row['state'] == 'complete':
                assert row['selected_token_ids'] == [0] and row['generated_token_ids'] == []
        for row in complete:
            snapshot = row['snapshot']['forward']
            assert snapshot['cache_output_length'] == len(record['item']['prompt_ids'])+32
            assert snapshot['absolute_capture_positions'] == [len(record['item']['prompt_ids'])+31]
        restored = next(row for row in complete if row['name'] == R.CACHE_ARMS[2])
        assert restored['cache_swap']['both_cache_exact_clean_o8'] is True
        assert restored['cache_swap']['both_endpoint_exact_clean_o32'] is True
        R.validate_record(bound_record(record), 'a'*64, record['item'], candidate)
        R.validate_arrays(bound_record(record), arrays)
    candidate, _, arrays = native_cases[1]
    vector = np.asarray(candidate['vector'], dtype=np.float32)
    np.testing.assert_array_equal(arrays['endpoint_clean_add_shared_requested_delta'], vector)
    np.testing.assert_array_equal(arrays['endpoint_pulse_remove_shared_requested_delta'], -vector)
    assert vector[0] < 0
    for name, receiver, delta in [('clean_add_shared','clean',vector), ('pulse_remove_shared','pulse',-vector)]:
        np.testing.assert_array_equal(arrays['endpoint_'+name+'_residual_18'], arrays[receiver+'_o32_residual_18']+delta)
    invalid_arrays = native_cases[2][2]
    assert not any(key.startswith('endpoint_clean_add_shared_') for key in invalid_arrays)


def test_split_and_candidate_rejection_precede_model_work(native_engine, monkeypatch):
    R = runtime()
    _, engine, reference, stats, items = native_engine
    neutral = [11+i % 20 for i in range(32)]
    monkeypatch.setattr(R, 'run_baselines', lambda *_a, **_k: pytest.fail('invalid split/candidate reached model'))
    with pytest.raises(ValueError, match='calibration'):
        R.compute_calibration(engine, reference, stats, items[8], neutral)
    with pytest.raises(ValueError, match='test'):
        R.compute_test(engine, reference, stats, items[0], neutral, fitted_candidate())
    candidate = fitted_candidate()
    candidate['vector'][0] += 1.
    with pytest.raises(ValueError, match='hash'):
        R.compute_test(engine, reference, stats, items[8], neutral, candidate)


def test_candidate_freeze_refits_exact_calibration_only_without_reading_test(tmp_path, frozen_store, monkeypatch):
    R = runtime()
    output = copy_store(tmp_path, frozen_store)
    _, document, split, _, plan_hash, expected, _ = frozen_store
    calibration = document['items'][:8]
    original = R.verified
    seen = []
    def verify(path, item, digest, candidate=None):
        assert item['split'] == 'calibration'
        seen.append(item['id'])
        return original(path, item, digest, candidate)
    monkeypatch.setattr(R, 'verified', verify)
    # This deliberately corrupts a held-out artifact. Reconstructing the
    # calibration-only candidate must not read it or derive anything from it.
    (R.checkpoint_path(output, split['test_ids'][0])/'result.json').write_text('HELDOUT SENTINEL')
    rebuilt = R.freeze_candidate(output, calibration, plan_hash)
    assert seen == split['calibration_ids'] and rebuilt == expected
    assert np.asarray(rebuilt['vector'], np.float32).tobytes() == np.asarray(expected['vector'], np.float32).tobytes()
    freeze = json.loads((output/'candidate/freeze.json').read_text())
    assert freeze['candidate_file_sha256'] == R.base.sha_file(output/'candidate/candidate.json')


def test_self_consistent_candidate_forgery_is_rejected_against_real_calibration(tmp_path, frozen_store):
    from attention_probe.temporal_candidate import candidate_hash, verify_candidate
    R = runtime()
    output = copy_store(tmp_path, frozen_store)
    _, document, _, _, plan_hash, original, _ = frozen_store
    forged = copy.deepcopy(original)
    forged['source_binding']['injected_test_choice'] = 'forbidden'
    forged['candidate_sha256'] = candidate_hash(forged)
    assert verify_candidate(forged) is True  # Its self-hash is internally sound.
    R.base.atomic_json(output/'candidate/candidate.json', forged)
    freeze = json.loads((output/'candidate/freeze.json').read_text())
    freeze.update(candidate_sha256=forged['candidate_sha256'],
                  candidate_file_sha256=R.base.sha_file(output/'candidate/candidate.json'))
    R.base.atomic_json(output/'candidate/freeze.json', freeze)
    with pytest.raises(ValueError, match='actual validated calibration fit'):
        R.freeze_candidate(output, document['items'][:8], plan_hash)


def test_checkpoint_atomic_nooverwrite_and_full_archive_resume(tmp_path, frozen_store):
    R = runtime()
    output = copy_store(tmp_path, frozen_store)
    _, document, scientific, _, plan_hash, candidate, saved = frozen_store
    record = R.verified(output, document['items'][8], plan_hash, candidate)
    with np.load(R.checkpoint_path(output, document['items'][8]['id'])/'arrays.npz', allow_pickle=False) as values:
        arrays = {key: values[key] for key in values.files}
    with pytest.raises(ValueError, match='overwrite'):
        R.write_checkpoint(output, record, arrays, plan_hash, {}, candidate)
    archive = R.archive(output)
    destination = tmp_path/'restored'
    R.restore_resume(archive, destination, plan_hash, document, scientific)
    assert R.freeze_candidate(destination, document['items'][:8], plan_hash) == candidate
    assert R.verified(destination, document['items'][8], plan_hash, candidate) == saved
    R.restore_resume(archive, destination, plan_hash, document, scientific)
    path = R.checkpoint_path(destination, document['items'][8]['id'])/'arrays.npz'
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='NPZ hash'):
        R.verified(destination, document['items'][8], plan_hash, candidate)


@pytest.mark.parametrize('mutation', ['compact', 'incomplete_pair', 'test_before_calibration', 'wrong_plan'])
def test_resume_rejects_invalid_or_precalibration_evidence(tmp_path, frozen_store, mutation):
    R = runtime()
    output, document, scientific, plan, plan_hash, _, _ = frozen_store
    names = {'run_plan.json':R.base.canonical(plan)+b'\n'}
    if mutation == 'compact':
        names['COMPACT_PACKAGE.json'] = b'{}'
    elif mutation == 'incomplete_pair':
        path = R.checkpoint_path(output, document['items'][0]['id'])/'result.json'
        names[str(path.relative_to(output))] = path.read_bytes()
    elif mutation == 'test_before_calibration':
        path = R.checkpoint_path(output, document['items'][8]['id'])
        for file in path.iterdir():
            names[str(file.relative_to(output))] = file.read_bytes()
    else:
        names['run_plan.json'] = R.base.canonical({**plan,'changed':True})
    archive = tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as bundle:
        for name, data in names.items():
            bundle.writestr(name, data)
    with pytest.raises(ValueError):
        R.restore_resume(archive, tmp_path/'destination', plan_hash, document, scientific)


def test_prepare_is_model_free_and_resets_stale_metrics(tmp_path, native_engine, monkeypatch):
    R = runtime()
    _, _, reference, stats, items = native_engine
    document, scientific = document_and_split(items)
    plan = {'schema':R.SCHEMA,'synthetic':'offline prepare'}
    monkeypatch.setattr(R, 'load_inputs', lambda *_: (reference,stats,document,scientific))
    monkeypatch.setattr(R, 'plan_document', lambda *_: plan)
    monkeypatch.setattr(R, 'initialize_engine', lambda *_: pytest.fail('prepare must not acquire a model/GPU'))
    monkeypatch.setattr(R, 'compute_calibration', lambda *_: pytest.fail('prepare must not calibrate'))
    R.base.atomic_json(tmp_path/'metrics.json', {'weights_loaded':True,'pretrained_model_run':True,'stale':True})
    status = R.run('hn','policy','refw','items','plan',tmp_path,offline_plan_only=True)
    assert status['state'] == 'complete' and status['stage'] == 'offline_plan_only'
    assert status['no_model_executed'] is True
    metrics = json.loads((tmp_path/'metrics.json').read_text())
    assert metrics['weights_loaded'] is False and metrics['pretrained_model_run'] is False
    assert metrics['generation_count'] == 0 and 'stale' not in metrics
    assert not (tmp_path/'candidate').exists() and not (tmp_path/'requests').exists()


def test_first_test_forward_requires_reconstructed_frozen_candidate(tmp_path, frozen_store, native_engine, monkeypatch):
    import attention_probe.temporal_candidate as candidate_module
    R = runtime()
    output = copy_store(tmp_path, frozen_store)
    _, document, scientific, plan, plan_hash, expected, _ = frozen_store
    for item in document['items'][8:]:
        shutil.rmtree(R.checkpoint_path(output,item['id']), ignore_errors=True)
    _, native, reference, stats, _ = native_engine
    monkeypatch.setattr(R,'load_inputs',lambda *_:(reference,stats,document,scientific))
    monkeypatch.setattr(R,'plan_document',lambda *_:plan)
    monkeypatch.setattr(R,'compute_calibration',lambda *_:pytest.fail('validated calibration must not rerun'))
    engine = {'model':SimpleNamespace(),'hardware':{'device':'synthetic'},'tokenizer':native['tokenizer']}
    def initialize(_ref,_cache,on_loaded):
        on_loaded(engine['hardware'])
        return engine
    monkeypatch.setattr(R,'initialize_engine',initialize)
    fits = []
    original_fit = candidate_module.fit_candidate
    def fit(rows, **kwargs):
        assert all(row['split']=='calibration' for row in rows)
        assert [row['id'] for row in rows] == scientific['calibration_ids']
        fits.append([row['id'] for row in rows])
        return original_fit(rows, **kwargs)
    monkeypatch.setattr(candidate_module,'fit_candidate',fit)
    calls = []
    def compute(_engine,_ref,_stats,item,_neutral,candidate,**_kwargs):
        assert fits and candidate == expected
        saved = json.loads((output/'candidate/candidate.json').read_text())
        freeze = json.loads((output/'candidate/freeze.json').read_text())
        assert saved == candidate and freeze['candidate_file_sha256'] == R.base.sha_file(output/'candidate/candidate.json')
        calls.append(item['id'])
        return {'state':'partial','split':'test','item':item,
                'stop_reason':'budget_exhausted_between_endpoint_arms'}, {}
    monkeypatch.setattr(R,'compute_test',compute)
    status = R.run('hn','policy','refw','items','plan',output)
    assert status['state'] == 'paused' and status['stage'] == 'partial_test_checkpoint'
    assert status['completed_calibration'] == 8 and status['completed_test'] == 0
    assert calls == [scientific['test_ids'][0]]
    assert len(fits) >= 2  # Resume reconstruction and before-forward recheck.
    assert R.verified(output,document['items'][8],plan_hash,expected) is None
    assert json.loads((output/status['partial_path']/'result.json').read_text())['resume_eligible'] is False


def test_complete_resume_is_cpu_only_preserves_proofs_and_fresh_metrics(tmp_path, frozen_store, native_engine, monkeypatch):
    R = runtime()
    output = copy_store(tmp_path,frozen_store)
    _,document,scientific,plan,plan_hash,candidate,saved = frozen_store
    source = R.checkpoint_path(output,document['items'][8]['id'])
    with np.load(source/'arrays.npz',allow_pickle=False) as values:
        arrays = {key:values[key] for key in values.files}
    for item in document['items'][9:]:
        record = copy.deepcopy(saved)
        record['item'] = item
        R.write_checkpoint(output,record,arrays,plan_hash,{'device':'cpu/synthetic'},candidate)
    _,_,reference,stats,_ = native_engine
    monkeypatch.setattr(R,'load_inputs',lambda *_:(reference,stats,document,scientific))
    monkeypatch.setattr(R,'plan_document',lambda *_:plan)
    monkeypatch.setattr(R,'initialize_engine',lambda *_:pytest.fail('completed resume must not load GPU/model'))
    R.base.atomic_json(output/'metrics.json',{'weights_loaded':True,'pretrained_model_run':True,'generation_count':99,'stale':True})
    status = R.run('hn','policy','refw','items','plan',output)
    assert status['state']=='complete' and status['resumed_without_gpu'] is True
    assert status['completed_inputs']==16 and status['completed_calibration']==status['completed_test']==8
    metrics = json.loads((output/'metrics.json').read_text())
    assert metrics['weights_loaded'] is False and metrics['pretrained_model_run'] is False
    assert metrics['retained_pretrained_model_run'] is True and len(metrics['retained_technical_checks'])==16
    assert 'stale' not in metrics
    assert metrics['generation_count']==8*sum(row['state']=='complete' for row in saved['generations'])
