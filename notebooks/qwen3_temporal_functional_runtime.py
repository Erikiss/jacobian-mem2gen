"""Frozen-calibration temporal candidate/cache experiment on original Qwen3.

Eight calibration cases precede candidate publication and every test forward.
Only the final neutral token32 is patched; earlier cache swaps occur after8 and
therefore affect later forwards. No test-dependent fitting or arm selection.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import tempfile
import time
import traceback
import zipfile
from collections import defaultdict
from pathlib import Path

import qwen3_h286_runtime as base

SCHEMA = 'qwen3_temporal_functional_v1'
ITEMS_SHA256 = '997f59b8ab3fae7190a609958def98227bcad2546666e1cc66bcdf89778e225e'
PLAN_SHA256 = 'af42159b61e639da19cdb99c140e048cd5e88ef2c82bd467b5217f88dfb40b62'
SEED = 20261118
OFFSETS = (0, 1, 8, 32)
RESIDUAL_LAYERS = (13, 14, 15, 17, 18, 19)
GREEDY_CAP = 64
BUDGET_SECONDS = 1200
MAIN_ARMS = ('clean', 'pulse', 'clean_add_shared', 'pulse_remove_shared')
CACHE_ARMS = ('pulse_restore_prompt_cache', 'pulse_restore_continuation_cache', 'pulse_restore_both_cache')
ENDPOINT_ARMS = (*MAIN_ARMS, *[name for k in range(8) for name in (f'clean_add_noise_{k}', f'pulse_remove_noise_{k}')], *CACHE_ARMS)
SOURCE_NAMES = ('qwen3_h286_runtime.py', 'attention_probe/reference.py',
                'attention_probe/fixed_refw.py', 'attention_probe/interventions.py',
                'attention_probe/answer_validation.py', 'attention_probe/temporal_cache.py',
                'attention_probe/temporal_candidate.py', 'attention_probe/temporal_validation.py')
PROOF_KEYS = ('native_capture_exact', 'beta_one_sham_exact', 'cached_full_parity',
              'source_tokens_exact', 'cache_donors_unchanged', 'frozen_parameters',
              'global_rng', 'hook_cleanup', 'finite_arrays', 'null_causal_checks')


def cache_api():
    from attention_probe import temporal_cache
    return temporal_cache


def hook_count(model):
    return sum(len(module._forward_hooks)+len(module._forward_pre_hooks) for module in model.modules())


def tensor_sha(value):
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def token_sha(values):
    return hashlib.sha256(base.canonical(list(values))).hexdigest()


def manifest_sha(cache):
    return hashlib.sha256(base.canonical(cache_api().cache_manifest(cache))).hexdigest()


def _last_logits(step):
    if step.logits.ndim != 3 or tuple(step.logits.shape[:2]) != (1, 1):
        raise ValueError('Endpoint must carry exactly one last-position logit vector')
    return step.logits[0, 0]


def score_sequence(engine, step, token_ids, reference, stats):
    """Teacher-force one fixed candidate without EOS or donor-cache mutation."""
    tc = cache_api()
    model = engine['model']
    ids = list(token_ids)
    if not ids or any(type(token) is not int or not 0 <= token < model.config.vocab_size for token in ids):
        raise ValueError('Scoring requires fixed nonempty vocabulary token IDs')
    from attention_probe.answer_validation import _eos_ids
    eos, _ = _eos_ids(model, model.config.vocab_size)
    if any(token in eos for token in ids):
        raise ValueError('Fixed gold/foil scoring candidates must exclude EOS')
    start = tc.cache_length(step.cache)
    first = _last_logits(step).double().log_softmax(-1)
    logprobs = [float(first[ids[0]])]
    incoming_hash = manifest_sha(step.cache)
    if len(ids) > 1:
        positions = list(range(start, start+len(ids)-1))
        scored = tc.forward_capture(model, ids[:-1], reference.coordinates, stats,
            cache=step.cache, capture_positions=positions, capture=False,
            weight_norms=engine.get('weight_norms'))
        matrix = scored.logits[0].double().log_softmax(-1)
        if tuple(matrix.shape) != (len(ids)-1, model.config.vocab_size):
            raise ValueError('Teacher-forced logits do not match candidate positions')
        logprobs.extend(float(matrix[index, token]) for index, token in enumerate(ids[1:]))
        output_length = tc.cache_length(scored.cache)
    else:
        output_length = start
    if incoming_hash != manifest_sha(step.cache):
        raise ValueError('Teacher forcing mutated the endpoint donor cache')
    return {'token_ids': ids, 'token_logprobs': logprobs, 'first_logprob': logprobs[0],
            'sum_logprob': sum(logprobs), 'mean_logprob': sum(logprobs)/len(logprobs),
            'token_count': len(ids), 'metric_dtype': 'float64', 'eos_added': False,
            'cache_input_length': start, 'cache_output_length': output_length,
            'endpoint_cache_sha256': incoming_hash, 'endpoint_cache_unchanged': True,
            'scoring': 'first token from endpoint logits; remaining tokens from a fresh clone fed candidate[:-1]'}


def score_candidates(engine, step, item, reference, stats):
    gold = score_sequence(engine, step, item['gold_token_ids'], reference, stats)
    foil = score_sequence(engine, step, item['foil_token_ids'], reference, stats)
    return {'gold': gold, 'foil': foil,
            'first_token_margin': gold['first_logprob']-foil['first_logprob'],
            'full_sum_margin': gold['sum_logprob']-foil['sum_logprob'],
            'mean_per_token_margin': gold['mean_logprob']-foil['mean_logprob'],
            'primary': 'full_sum_margin', 'metric_dtype': 'float64',
            'interpretation': 'Fixed gold-versus-foil likelihood contrast; not generated semantic accuracy'}


def greedy_answer(engine, step, item, reference, stats, *, max_new_tokens=GREEDY_CAP, top_k=10):
    """Greedy continuation from a cloned endpoint, with no decode interventions."""
    import torch

    from attention_probe.answer_validation import _eos_ids, classify_answer
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= GREEDY_CAP:
        raise ValueError('Greedy cap must be in1..64 selected tokens including EOS')
    tc = cache_api()
    model, tokenizer = engine['model'], engine['tokenizer']
    eos, eos_source = _eos_ids(model, model.config.vocab_size)
    cache = tc.clone_cache(step.cache)
    logits = _last_logits(step)
    original_hash = manifest_sha(step.cache)
    original_hooks = hook_count(model)
    emitted, selected, records = [], [], []
    start = time.monotonic()
    for index in range(max_new_tokens):
        lp = logits.double().log_softmax(-1)
        token = int(torch.argmax(logits))
        selected.append(token)
        values, indices = lp.topk(min(top_k, lp.numel()))
        is_eos = token in eos
        records.append({'step': index, 'selected_token_id': token,
            'selected_token_text': tokenizer.decode([token], skip_special_tokens=False, clean_up_tokenization_spaces=False),
            'selected_token_logprob': float(lp[token]), 'is_eos': is_eos,
            'phase': 'endpoint' if index == 0 else 'unhooked_cached_decode',
            'top_tokens': [{'token_id': int(tok), 'logprob': float(value),
                            'text': tokenizer.decode([int(tok)], skip_special_tokens=False, clean_up_tokenization_spaces=False)}
                           for tok, value in zip(indices, values, strict=True)]})
        if is_eos:
            break
        emitted.append(token)
        if index+1 < max_new_tokens:
            next_step = tc.forward_capture(model, [token], reference.coordinates, stats,
                cache=cache, capture=False, weight_norms=engine.get('weight_norms'))
            if next_step.evidence['pulse_fired'] or next_step.evidence['residual_patch'] is not None or next_step.evidence['capture_fired']:
                raise ValueError('Greedy cached decode installed an intervention/capture')
            cache, logits = next_step.cache, _last_logits(next_step)
    stopped = bool(records[-1]['is_eos'])
    text = tokenizer.decode(emitted, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if manifest_sha(step.cache) != original_hash or hook_count(model) != original_hooks:
        raise ValueError('Greedy continuation mutated donor cache or leaked hooks')
    return {'state': 'complete', 'max_new_tokens': max_new_tokens, 'top_k': top_k,
            'generated_text': text, 'generated_token_ids': emitted, 'selected_token_ids': selected,
            'steps': records, 'stop_reason': 'eos' if stopped else 'max_new_tokens', 'truncated': not stopped,
            'classification': classify_answer(text, item['gold_text'], truncated=not stopped),
            'eos_ids': list(eos), 'eos_source': eos_source,
            'num_generated_tokens': len(emitted), 'num_steps': len(selected),
            'endpoint_cache_sha256': original_hash, 'endpoint_cache_unchanged': True,
            'final_cache_length': tc.cache_length(cache),
            'cache_convention': 'selected terminal token is not fed; EOS excluded from emitted IDs/text',
            'elapsed_seconds': time.monotonic()-start,
            'scope': 'four prespecified main arms only; no generated-answer selectivity test versus noise'}


def save_step(arrays, prefix, step):
    arrays[prefix+'_logits'] = _last_logits(step).numpy()
    arrays.update({prefix+'_'+key: value.numpy() for key, value in step.arrays.items()})


def step_record(step, token_prefix):
    tc = cache_api()
    return {'forward': step.evidence, 'token_prefix_sha256': token_sha(token_prefix),
            'position_ids_sha256': token_sha(range(len(token_prefix))),
            'cache': tc.cache_manifest(step.cache), 'logits_sha256': tensor_sha(_last_logits(step))}


def _parity_tensor(cached, rebuilt, *, rtol=1e-3, atol=1e-4):
    """Measure native FP32 drift using FP64 reductions, never infer from argmax."""
    import torch
    same_shape = tuple(cached.shape) == tuple(rebuilt.shape)
    same_dtype = cached.dtype == rebuilt.dtype == torch.float32
    row = {'cached_shape': list(cached.shape), 'full_shape': list(rebuilt.shape),
           'cached_dtype': str(cached.dtype), 'full_dtype': str(rebuilt.dtype),
           'same_shape': same_shape, 'native_fp32': same_dtype,
           'rtol': rtol, 'atol': atol, 'passed': False}
    if not same_shape:
        return row
    left = cached.detach().double()
    right = rebuilt.detach().to(device=left.device, dtype=torch.float64)
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    row['finite'] = finite
    if finite:
        difference = left-right
        row.update(max_abs_error=float(difference.abs().max()) if difference.numel() else 0.,
                   L2_error=float(difference.norm()), full_L2_norm=float(right.norm()),
                   bitwise_equal=bool(torch.equal(cached, rebuilt.to(cached.device))),
                   passed=bool(same_dtype and torch.allclose(left, right, rtol=rtol, atol=atol)))
    return row


def _parity_snapshot(cached, rebuilt):
    """Full readout and every native K/V layer must pass the declared tolerance."""
    logits = _parity_tensor(_last_logits(cached), _last_logits(rebuilt))
    same_keys = set(cached.arrays) == set(rebuilt.arrays)
    arrays = {key: _parity_tensor(cached.arrays[key], rebuilt.arrays[key])
              for key in sorted(set(cached.arrays) & set(rebuilt.arrays))}
    same_layers = len(cached.cache.layers) == len(rebuilt.cache.layers)
    cache = []
    for layer, (left, right) in enumerate(zip(cached.cache.layers, rebuilt.cache.layers)):
        cache.append({'layer': layer,
                      'keys': _parity_tensor(left.keys, right.keys),
                      'values': _parity_tensor(left.values, right.values)})
    arrays_passed = bool(same_keys and arrays and all(row['passed'] for row in arrays.values()))
    kv_passed = bool(same_layers and cache and
                     all(row[kind]['passed'] for row in cache for kind in ('keys', 'values')))
    return {'logits': logits, 'array_keys_equal': same_keys, 'arrays_parity': arrays,
            'arrays_allclose': arrays_passed, 'cache_layer_count_equal': same_layers,
            'kv_parity': cache, 'kv_allclose': kv_passed,
            'passed': bool(logits['passed'] and arrays_passed and kv_passed)}


def _null_causal_snapshots(clean, pulse, *, prompt_length, offset):
    """Actual-run causal nulls; these are exact comparisons, not tolerance gates."""
    import torch
    if (len(clean.cache.layers) != len(pulse.cache.layers) or
            cache_api().cache_length(clean.cache) != prompt_length+offset or
            cache_api().cache_length(pulse.cache) != prompt_length+offset):
        raise ValueError('Causal null snapshots have incompatible cache shapes or lengths')
    regions = {
        'earlier_prompt_all_layers': (range(len(clean.cache.layers)), 0, prompt_length-1),
        'all_positions_layers0_through14': (range(min(15, len(clean.cache.layers))), 0, prompt_length+offset),
        'bridge_positions_layer15': ([15] if len(clean.cache.layers)>15 else [], prompt_length, prompt_length+offset),
    }
    checks = {}
    for label, (layers, start, end) in regions.items():
        layers = list(layers)
        details = []
        for layer in layers:
            for kind in ('keys', 'values'):
                left = getattr(clean.cache.layers[layer], kind)[..., start:end, :]
                right = getattr(pulse.cache.layers[layer], kind)[..., start:end, :]
                same_shape = left.shape == right.shape
                exact = bool(same_shape and torch.equal(left, right))
                difference = left.double()-right.double() if same_shape else None
                details.append({'layer': layer, 'kind': kind, 'bitwise_equal': exact,
                                'L2_difference': float(difference.norm()) if difference is not None else None,
                                'changed_scalars': int(torch.count_nonzero(difference)) if difference is not None else None})
        checks[label] = {'layer_indices': layers, 'absolute_start': start, 'absolute_stop_exclusive': end,
                         'applicable': bool(layers and end>start), 'details': details,
                         'passed': all(row['bitwise_equal'] for row in details)}
    return {'offset': offset, 'cache_length': prompt_length+offset,
            'checks': checks, 'passed': all(row['passed'] for row in checks.values()),
            'scope': 'Empty position ranges are explicitly inapplicable; no signal is inferred from them.'}


def run_baselines(engine, reference, stats, item, neutral_ids):
    """Paired trajectories; keep offset31 internally solely to replay token32."""
    import torch
    tc = cache_api()
    model = engine['model']
    prompt = list(item['prompt_ids'])
    neutral = list(neutral_ids)
    if len(neutral) != 32:
        raise ValueError('Exactly the same thirty-two frozen neutral tokens are required')
    started = time.monotonic()
    arrays, snapshots, histories, parity = {}, {}, {}, []
    for arm in ('clean', 'pulse'):
        pulse = None if arm == 'clean' else {'block': 14, 'beta': 1/1.08, 'mode': 'scale',
                                            'positions': [len(prompt)-1], 'seed': SEED}
        first = tc.forward_capture(model, prompt, reference.coordinates, stats, pulse=pulse,
                                    weight_norms=engine.get('weight_norms'))
        if arm == 'clean' and engine.get('on_model_run'):
            engine['on_model_run']()
        history = {0: first}
        for previous, offset in ((0, 1), (1, 8), (8, 31), (31, 32)):
            history[offset] = tc.forward_capture(model, neutral[previous:offset], reference.coordinates, stats,
                cache=history[previous].cache, capture=offset != 31, weight_norms=engine.get('weight_norms'))
        histories[arm] = history
        for offset in OFFSETS:
            prefix = f'{arm}_o{offset}'
            save_step(arrays, prefix, history[offset])
            snapshots[prefix] = step_record(history[offset], prompt+neutral[:offset])
        for offset in (8, 32):
            rebuilt = tc.forward_capture(model, prompt+neutral[:offset], reference.coordinates, stats,
                capture_positions=[len(prompt)+offset-1], pulse=pulse,
                weight_norms=engine.get('weight_norms'))
            save_step(arrays, f'{arm}_full_o{offset}', rebuilt)
            cached = history[offset]
            proof = _parity_snapshot(cached, rebuilt)
            residual_error = float((cached.arrays['residual_18'].double()-rebuilt.arrays['residual_18'].double()).norm())
            passed = proof['passed']
            parity.append({'arm': arm, 'offset': offset, 'logits_allclose': proof['logits']['passed'],
                           'rtol': 1e-3, 'atol': 1e-4, 'logits_max_abs_error': proof['logits'].get('max_abs_error'),
                           'B18_L2_error': residual_error,
                           **proof,
                           'cached': snapshots[f'{arm}_o{offset}'],
                           'full': step_record(rebuilt, prompt+neutral[:offset])})
            if not passed:
                raise ValueError('Cached/full-prefix parity failed: '+json.dumps(parity[-1], allow_nan=False))
    null_checks = [_null_causal_snapshots(histories['clean'][offset], histories['pulse'][offset],
                   prompt_length=len(prompt), offset=offset) for offset in OFFSETS]
    if not all(row['passed'] for row in null_checks):
        raise ValueError('Measured clean/pulse causal cache null failed: '+json.dumps(null_checks, allow_nan=False))
    # Direct native/passive-capture and beta1 sham checks bind actual KV state,
    # not just the first argmax of two instrumented forwards.
    plain = tc.forward_capture(model, prompt, reference.coordinates, stats, capture=False)
    sham = tc.forward_capture(model, prompt, reference.coordinates, stats, capture=False,
        pulse={'block': 14, 'beta': 1., 'mode': 'scale', 'positions': [len(prompt)-1], 'seed': SEED})
    clean = histories['clean'][0]
    clean_cache = manifest_sha(clean.cache)
    native_exact = bool(torch.equal(plain.logits, clean.logits) and manifest_sha(plain.cache) == clean_cache)
    sham_exact = bool(torch.equal(sham.logits, clean.logits) and manifest_sha(sham.cache) == clean_cache)
    if not native_exact or not sham_exact:
        raise ValueError('Native/passive-capture/beta-one sham prefill cache/logit identity failed')
    return {'snapshots': snapshots, 'cached_full_parity': parity, 'null_causal_checks': null_checks,
            'native_capture_exact': native_exact, 'beta_one_sham_exact': sham_exact,
            'elapsed_seconds': time.monotonic()-started}, arrays, histories


def _proof_start(model):
    import torch
    device = next(model.parameters()).device
    return {'parameters': base.parameter_evidence(model), 'hooks': hook_count(model),
            'cpu_rng': torch.random.get_rng_state().clone(),
            'cuda_rng': torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None}


def _proof_finish(model, before, baseline, arrays, donors_unchanged=True):
    import numpy as np
    import torch
    device = next(model.parameters()).device
    checks = {'native_capture_exact': baseline['native_capture_exact'],
              'beta_one_sham_exact': baseline['beta_one_sham_exact'],
              'cached_full_parity': all(row['passed'] for row in baseline['cached_full_parity']),
              'null_causal_checks': all(row['passed'] for row in baseline.get('null_causal_checks', [])),
              'source_tokens_exact': True, 'cache_donors_unchanged': donors_unchanged,
              'frozen_parameters': base.parameter_evidence(model) == before['parameters'],
              'global_rng': bool(torch.equal(torch.random.get_rng_state(), before['cpu_rng']) and
                                  (before['cuda_rng'] is None or torch.equal(torch.cuda.get_rng_state(device), before['cuda_rng']))),
              'hook_cleanup': hook_count(model) == before['hooks'],
              'finite_arrays': all(value.dtype == np.float32 and np.isfinite(value).all() for value in arrays.values())}
    if not all(checks.values()):
        raise ValueError('Temporal technical checks failed: '+str(checks))
    return checks


def _contract(engine, reference):
    return {'coordinates': len(reference.coordinates), 'hidden_size': engine['model'].config.hidden_size,
            'vocab_size': engine['model'].config.vocab_size}


def compute_calibration(engine, reference, stats, item, neutral_ids):
    import numpy as np
    if item['split'] != 'calibration':
        raise ValueError('Calibration computation accepts designated calibration cases only')
    started = time.monotonic()
    before = _proof_start(engine['model'])
    baseline, arrays, _histories = run_baselines(engine, reference, stats, item, neutral_ids)
    fit = {}
    for offset in (8, 32):
        delta = arrays[f'pulse_o{offset}_residual_18'][0, 0]-arrays[f'clean_o{offset}_residual_18'][0, 0]
        arrays[f'calibration_delta{offset}'] = delta.astype(np.float32)
        fit[f'cache_error{offset}'] = max(row['B18_L2_error'] for row in baseline['cached_full_parity'] if row['offset'] == offset)
    checks = _proof_finish(engine['model'], before, baseline, arrays)
    return {'state': 'complete', 'split': 'calibration', 'item': item,
            'neutral_token_ids': list(neutral_ids),
            'baseline': baseline, 'fit_record': fit, 'checks': checks,
            'array_contract': _contract(engine, reference),
            'timing': {'baseline_seconds': baseline['elapsed_seconds'], 'total_seconds': time.monotonic()-started}}, arrays


def endpoint_readout(step, clean, pulse, item):
    import torch
    current = _last_logits(step).double().log_softmax(-1)
    values, ids = current.topk(min(10, current.numel()))
    refs = {}
    for name, donor in (('clean', clean), ('pulse', pulse)):
        ref = _last_logits(donor).double().log_softmax(-1)
        refs[name] = {'reference_to_arm_KL': float((ref.exp()*(ref-current)).sum()),
                      'arm_to_reference_KL': float((current.exp()*(current-ref)).sum())}
    return {'logits_sha256': tensor_sha(_last_logits(step)), 'metric_dtype': 'float64',
            'top10': [{'token_id': int(token), 'logprob': float(value)} for token, value in zip(ids, values, strict=True)],
            'references': refs, 'gold_first_token_id': item['gold_token_ids'][0],
            'foil_first_token_id': item['foil_token_ids'][0],
            'gold_first_token_logprob': float(current[item['gold_token_ids'][0]]),
            'foil_first_token_logprob': float(current[item['foil_token_ids'][0]]),
            'argmax_id': int(torch.argmax(current))}


def compute_test(engine, reference, stats, item, neutral_ids, candidate, *, stop_requested=None):
    import torch

    from attention_probe.temporal_candidate import verify_candidate
    if item['split'] != 'test':
        raise ValueError('Held-out computation accepts test cases only')
    verify_candidate(candidate)
    stop = stop_requested or (lambda: False)
    started = time.monotonic()
    before = _proof_start(engine['model'])
    baseline, arrays, histories = run_baselines(engine, reference, stats, item, neutral_ids)
    tc = cache_api()
    endpoints = {'clean': histories['clean'][32], 'pulse': histories['pulse'][32]}
    endpoint_records, generations = [], []
    prompt_length = len(item['prompt_ids'])
    model = engine['model']
    device = next(model.parameters()).device
    vectors = {'shared': torch.tensor(candidate['vector'], dtype=torch.float32, device=device)}
    vectors.update({f'noise_{k}': torch.tensor(row, dtype=torch.float32, device=device)
                    for k, row in enumerate(candidate['noise_vectors'])})
    candidate_binding = {'candidate_sha256': candidate['candidate_sha256'], 'valid': candidate['valid'],
        'vector_sha256': tensor_sha(vectors['shared']),
        'noise_vector_sha256': [tensor_sha(vectors[f'noise_{k}']) for k in range(8)],
        'native_norm': float(vectors['shared'].double().norm())}
    cache_evidence = {}
    source_prefix = list(item['prompt_ids'])+list(neutral_ids[:8])
    assert baseline['snapshots']['clean_o8']['token_prefix_sha256'] == token_sha(source_prefix)
    assert baseline['snapshots']['pulse_o8']['token_prefix_sha256'] == token_sha(source_prefix)
    for name in ENDPOINT_ARMS:
        if stop():
            return {'state': 'partial', 'split': 'test', 'item': item,
                    'neutral_token_ids': list(neutral_ids), 'candidate_binding': candidate_binding,
                    'candidate_sha256': candidate['candidate_sha256'], 'baseline': baseline,
                    'endpoints': endpoint_records, 'generations': generations,
                    'stop_reason': 'budget_exhausted_between_endpoint_arms',
                    'array_contract': _contract(engine, reference)}, arrays
        if name in ('clean', 'pulse'):
            step = endpoints[name]
        elif name in CACHE_ARMS:
            positions = ([prompt_length-1] if name == CACHE_ARMS[0] else
                         list(range(prompt_length, prompt_length+8)) if name == CACHE_ARMS[1] else
                         list(range(prompt_length-1, prompt_length+8)))
            changed, evidence = tc.swap_cache_positions(histories['pulse'][8].cache, histories['clean'][8].cache,
                layer_indices=list(range(15, len(model.model.layers))), absolute_positions=positions)
            evidence.update(token_prefix_sha256=token_sha(source_prefix),
                            position_ids_sha256=token_sha(range(len(source_prefix))),
                            existing_offset8_logits_unchanged=True)
            if name == CACHE_ARMS[2]:
                evidence['both_cache_exact_clean_o8'] = manifest_sha(changed) == manifest_sha(histories['clean'][8].cache)
                if not evidence['both_cache_exact_clean_o8']:
                    raise ValueError('Both-slot technical cache cut did not restore the complete clean cache')
            # A swap changes the stored history, never the already computed o8
            # logits. Its first possible effect is the forward for neutral token9.
            through31 = tc.forward_capture(model, neutral_ids[8:31], reference.coordinates, stats,
                cache=changed, capture=False, weight_norms=engine.get('weight_norms'))
            step = tc.forward_capture(model, neutral_ids[31:32], reference.coordinates, stats,
                cache=through31.cache, weight_norms=engine.get('weight_norms'))
            if name == CACHE_ARMS[2]:
                clean32 = histories['clean'][32]
                exact = bool(torch.equal(step.logits, clean32.logits) and
                             manifest_sha(step.cache) == manifest_sha(clean32.cache) and
                             set(step.arrays) == set(clean32.arrays) and
                             all(torch.equal(step.arrays[key], clean32.arrays[key]) for key in step.arrays))
                evidence['both_endpoint_exact_clean_o32'] = exact
                if not exact:
                    raise ValueError('Both-slot technical cache cut failed to reproduce the clean offset32 endpoint')
            cache_evidence[name] = evidence
            endpoints[name] = step
        else:
            if not candidate['valid']:
                endpoint_records.append({'name': name, 'state': 'not_run',
                    'reason': 'frozen_candidate_failed_prespecified_numerical_gate',
                    'candidate_sha256': candidate['candidate_sha256']})
                continue
            receiver = 'clean' if name.startswith('clean_add_') else 'pulse'
            key = name.split('_add_', 1)[1] if receiver == 'clean' else name.split('_remove_', 1)[1]
            delta = vectors[key] if receiver == 'clean' else -vectors[key]
            step = tc.forward_capture(model, neutral_ids[31:32], reference.coordinates, stats,
                cache=histories[receiver][31].cache,
                residual_patch={'block': 18, 'delta': delta}, weight_norms=engine.get('weight_norms'))
            endpoints[name] = step
            arrays['endpoint_'+name+'_requested_delta'] = delta.detach().cpu().numpy()
        save_step(arrays, 'endpoint_'+name, step)
        scored = score_candidates(engine, step, item, reference, stats)
        endpoint_records.append({'name': name, 'state': 'complete',
            'candidate_sha256': candidate['candidate_sha256'],
            'snapshot': step_record(step, list(item['prompt_ids'])+list(neutral_ids)),
            'readout': endpoint_readout(step, endpoints['clean'], endpoints['pulse'], item),
            'scores': scored, 'cache_swap': cache_evidence.get(name)})
    for name in MAIN_ARMS:
        if name not in endpoints:
            generations.append({'name': name, 'state': 'not_run', 'reason': 'invalid_frozen_candidate'})
            continue
        if stop():
            return {'state': 'partial', 'split': 'test', 'item': item,
                    'neutral_token_ids': list(neutral_ids), 'candidate_binding': candidate_binding,
                    'candidate_sha256': candidate['candidate_sha256'], 'baseline': baseline,
                    'endpoints': endpoint_records, 'generations': generations,
                    'stop_reason': 'budget_exhausted_between_greedy_arms',
                    'array_contract': _contract(engine, reference)}, arrays
        generations.append({'name': name, **greedy_answer(engine, endpoints[name], item, reference, stats)})
    if len(endpoint_records) != 23:
        raise ValueError('All twenty-three fixed endpoint slots must be explicit')
    donors_ok = all(row['cache_swap']['donor_unchanged'] and row['cache_swap']['receiver_unchanged']
                    for row in endpoint_records if row.get('cache_swap'))
    checks = _proof_finish(model, before, baseline, arrays, donors_ok)
    return {'state': 'complete', 'split': 'test', 'item': item,
            'neutral_token_ids': list(neutral_ids), 'candidate_binding': candidate_binding,
            'candidate_sha256': candidate['candidate_sha256'], 'candidate_valid': candidate['valid'],
            'baseline': baseline, 'endpoints': endpoint_records, 'generations': generations,
            'array_contract': _contract(engine, reference), 'checks': checks,
            'timing': {'baseline_seconds': baseline['elapsed_seconds'], 'total_seconds': time.monotonic()-started}}, arrays


def _pinned_json(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Exact pinned temporal input required: '+Path(path).name)
    return json.loads(raw)


def load_inputs(hn, policy, refw, items_path, plan_path):
    from attention_probe.fixed_refw import load_fixed_refw
    from attention_probe.reference import load_reference
    reference = load_reference(hn, policy)
    stats = load_fixed_refw(refw, reference)
    document = _pinned_json(items_path, ITEMS_SHA256)
    plan = _pinned_json(plan_path, PLAN_SHA256)
    rows = document.get('items', [])
    neutral = document.get('neutral_token_ids', [])
    if (len(rows) != 16 or len({row['id'] for row in rows}) != 16 or len(neutral) != 32
            or any(type(token) is not int or not 0 <= token < 151936 or token in (151643, 151645) for token in neutral)):
        raise ValueError('Expected sixteen source-bound cases and thirty-two fixed scaffold tokens')
    for row in rows:
        for key in ('prompt_ids', 'gold_token_ids', 'foil_token_ids'):
            ids = row.get(key, [])
            limit = 64 if key == 'prompt_ids' else 32
            if (not 1 <= len(ids) <= limit or any(type(token) is not int or not 0 <= token < 151936 for token in ids)
                    or (key != 'prompt_ids' and any(token in (151643, 151645) for token in ids))):
                raise ValueError('Source or fixed-candidate token budget/EOS rule changed')
        if row.get('prompt_ids_sha256') != token_sha(row['prompt_ids']):
            raise ValueError('Original source prompt token hash mismatch')
        if row['gold_token_ids'] == row['foil_token_ids']:
            raise ValueError('Gold and foil must be distinct frozen continuations')
    expected = {'schema': 'temporal_functional_plan_v1', 'offsets': list(OFFSETS), 'patch_offset': 32,
                'pulse_block': 14, 'target_block': 18, 'pulse_beta': 1/1.08, 'candidate_seed': SEED,
                'noise_count': 8, 'greedy_max_new_tokens': GREEDY_CAP,
                'seconds_after_model_load': BUDGET_SECONDS, 'cache_layers': list(range(15, 28)),
                'parity_rtol': 1e-3, 'parity_atol': 1e-4}
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError('Temporal plan changes fixed experiment rules')
    for split, field in (('calibration', 'calibration_ids'), ('test', 'test_ids')):
        chosen = [row for row in rows if row['split'] == split]
        if (len(chosen) != 8 or [row['id'] for row in chosen] != plan.get(field)
                or {family: sum(row['family'] == family for row in chosen) for family in ('s3', 'adet')} != {'s3': 4, 'adet': 4}):
            raise ValueError('The fixed disjoint eight-case calibration/test split changed')
    return reference, stats, document, plan


def source_path(name):
    folder = Path(__file__).parent
    candidate = folder/name
    return candidate if candidate.exists() else folder.parent/name


def plan_document(reference, stats, document, scientific_plan):
    return {'schema': SCHEMA, 'helper_sha256': base.sha_file(__file__),
        'source_sha256': {name: base.sha_file(source_path(name)) for name in SOURCE_NAMES},
        'items_sha256': ITEMS_SHA256, 'scientific_plan_sha256': PLAN_SHA256,
        'scientific_plan': scientific_plan, 'input_document': document,
        'model': reference.model.to_dict(), 'coordinates': [list(value) for value in reference.coordinates],
        'hn_sha256': reference.hn_sha256, 'policy_sha256': reference.policy_sha256,
        'refw_sha256': base.REFW_SHA256, 'refw_source_sha256': stats['source_hn_meta_sha256'],
        'calibration_ids': scientific_plan['calibration_ids'], 'test_ids': scientific_plan['test_ids'],
        'dtype': 'float32', 'tf32': False, 'attention': 'SDPA MATH',
        'transformers': base.TRANSFORMERS, 'accelerate': base.ACCELERATE,
        'candidate_seed': SEED, 'endpoints': list(ENDPOINT_ARMS), 'generation_arms': list(MAIN_ARMS),
        'generation_cap': GREEDY_CAP,
        'scope': 'fixed forced scaffold, calibration-only shared vector, eight exploratory held-out-from-fit contexts'}


def validate_record(record, plan_hash, item=None, candidate=None):
    from attention_probe.temporal_validation import validate_record as validate
    return validate(record, plan_hash, item, candidate)


def validate_arrays(record, arrays):
    from attention_probe.temporal_validation import validate_arrays as validate
    return validate(record, arrays)


def checkpoint_path(output, item_id):
    if not isinstance(item_id, str) or not item_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in item_id):
        raise ValueError('Unsafe checkpoint item identifier')
    return Path(output)/'requests'/item_id


def write_checkpoint(output, record, arrays, plan_hash, hardware, candidate=None):
    import numpy as np
    destination = checkpoint_path(output, record['item']['id'])
    if destination.exists():
        raise ValueError('A complete case already exists; no checkpoint overwrite')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+destination.name+'-', dir=destination.parent))
    try:
        np.savez_compressed(staging/'arrays.npz', **arrays)
        final = dict(record, schema=SCHEMA, state='complete', plan_sha256=plan_hash,
                     hardware=hardware, completed_utc=dt.datetime.now(dt.UTC).isoformat(),
                     npz_sha256=base.sha_file(staging/'arrays.npz'))
        final['technical_checks'] = {'state': 'passed', 'schema': SCHEMA,
                                    'plan_sha256': plan_hash, 'checks': final['checks']}
        validate_record(final, plan_hash, record['item'], candidate)
        validate_arrays(final, arrays)
        base.atomic_json(staging/'result.json', final)
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return final


def verified(output, item, plan_hash, candidate=None):
    import numpy as np
    folder = checkpoint_path(output, item['id'])
    if not folder.exists():
        return None
    record = json.loads((folder/'result.json').read_text())
    validate_record(record, plan_hash, item, candidate)
    if base.sha_file(folder/'arrays.npz') != record.get('npz_sha256'):
        raise ValueError('Case NPZ hash changed')
    with np.load(folder/'arrays.npz', allow_pickle=False) as saved:
        validate_arrays(record, {name: saved[name] for name in saved.files})
    return record


def freeze_candidate(output, calibration_items, plan_hash):
    """Reconstruct from validated calibration bytes even when resuming.

    A self-consistent candidate hash alone is insufficient. The saved artifact
    must exactly equal this deterministic fit of the eight actual checkpoints.
    """
    import numpy as np

    from attention_probe.temporal_candidate import fit_candidate, verify_candidate
    if len(calibration_items) != 8 or any(item['split'] != 'calibration' for item in calibration_items):
        raise ValueError('Candidate freeze requires exactly eight calibration items')
    calibration_records, hashes = [], {}
    for item in calibration_items:
        record = verified(output, item, plan_hash)
        if record is None:
            raise ValueError('Cannot fit a candidate from incomplete calibration')
        folder = checkpoint_path(output, item['id'])
        digest = base.sha_file(folder/'result.json')
        hashes[item['id']] = digest
        with np.load(folder/'arrays.npz', allow_pickle=False) as saved:
            calibration_records.append({'id': item['id'], 'split': 'calibration',
                'delta8': saved['calibration_delta8'].tolist(), 'delta32': saved['calibration_delta32'].tolist(),
                'cache_error8': record['fit_record']['cache_error8'],
                'cache_error32': record['fit_record']['cache_error32'], 'checkpoint_sha256': digest})
    ids = [item['id'] for item in calibration_items]
    binding = {'runtime_plan_sha256': plan_hash, 'items_sha256': ITEMS_SHA256,
               'scientific_plan_sha256': PLAN_SHA256, 'calibration_checkpoints': hashes}
    fitted = fit_candidate(calibration_records, calibration_ids=ids, source_binding=binding, seed=SEED)
    verify_candidate(fitted, expected_calibration_ids=ids, expected_source_binding=binding)
    destination = Path(output)/'candidate'
    candidate_bytes = base.canonical(fitted)+b'\n'
    freeze = {'state': 'frozen', 'schema': SCHEMA, 'plan_sha256': plan_hash,
              'candidate_sha256': fitted['candidate_sha256'],
              'candidate_file_sha256': hashlib.sha256(candidate_bytes).hexdigest(),
              'calibration_checkpoints': hashes}
    if destination.exists():
        if ((destination/'candidate.json').read_bytes() != candidate_bytes
                or json.loads((destination/'freeze.json').read_text()) != freeze):
            raise ValueError('Frozen candidate differs from the actual validated calibration fit')
    else:
        staging = Path(tempfile.mkdtemp(prefix='.candidate-', dir=Path(output)))
        try:
            base.atomic_json(staging/'candidate.json', fitted)
            base.atomic_json(staging/'freeze.json', freeze)
            os.rename(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    # Readback precedes every possible test forward.
    saved = json.loads((destination/'candidate.json').read_text())
    verify_candidate(saved, expected_calibration_ids=ids, expected_source_binding=binding)
    if base.sha_file(destination/'candidate.json') != freeze['candidate_file_sha256']:
        raise ValueError('Candidate freeze readback hash mismatch')
    return saved


def write_partial(output, record, arrays, plan_hash):
    import numpy as np
    parent = Path(output)/'partial'/record['item']['id']
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent/str(time.time_ns())
    staging = Path(tempfile.mkdtemp(prefix='.partial-', dir=parent))
    try:
        np.savez_compressed(staging/'arrays.npz', **arrays)
        base.atomic_json(staging/'result.json', dict(record, schema=SCHEMA, state='partial',
            plan_sha256=plan_hash, npz_sha256=base.sha_file(staging/'arrays.npz'), resume_eligible=False))
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def _copy_atomic(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        source_files = {path.relative_to(source): path.read_bytes() for path in source.rglob('*') if path.is_file()}
        destination_files = {path.relative_to(destination): path.read_bytes() for path in destination.rglob('*') if path.is_file()}
        if source_files != destination_files:
            raise ValueError('Resume would overwrite a different existing complete artifact')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+destination.name+'-copy-', dir=destination.parent))
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def restore_resume(resume_zip, output, plan_hash, document, scientific_plan):
    if not resume_zip:
        return
    items = {item['id']: item for item in document['items']}
    calibration = [items[item_id] for item_id in scientific_plan['calibration_ids']]
    with zipfile.ZipFile(resume_zip) as bundle, tempfile.TemporaryDirectory(prefix='temporal-resume-') as temporary:
        names = bundle.namelist()
        if (len(names) != len(set(names)) or 'COMPACT_PACKAGE.json' in names or bundle.testzip() is not None):
            raise ValueError('Only intact full temporal archives can resume, never compact analysis summaries')
        if hashlib.sha256(base.canonical(json.loads(bundle.read('run_plan.json')))).hexdigest() != plan_hash:
            raise ValueError('Resume scientific/source plan changed')
        allowed = {f'requests/{item_id}/{name}' for item_id in items for name in ('result.json', 'arrays.npz')}
        selected = [name for name in names if name.startswith('requests/') and not name.endswith('/')]
        if set(selected)-allowed:
            raise ValueError('Unexpected resume checkpoint path')
        staging = Path(temporary)
        for item in document['items']:
            pair = [f'requests/{item["id"]}/{name}' for name in ('result.json', 'arrays.npz')]
            present = [name in names for name in pair]
            if any(present) and not all(present):
                raise ValueError('Incomplete pair in complete resume namespace')
            if all(present):
                target = checkpoint_path(staging, item['id'])
                target.mkdir(parents=True)
                for name in pair:
                    (target/Path(name).name).write_bytes(bundle.read(name))
        candidate_names = ('candidate/candidate.json', 'candidate/freeze.json')
        if any(name in names for name in candidate_names):
            if not all(name in names for name in candidate_names):
                raise ValueError('Incomplete candidate freeze pair')
            (staging/'candidate').mkdir()
            for name in candidate_names:
                (staging/name).write_bytes(bundle.read(name))
        complete_cal = []
        for item in calibration:
            record = verified(staging, item, plan_hash)
            if record is not None:
                complete_cal.append(item)
        test_present = any(checkpoint_path(staging, item_id).exists() for item_id in scientific_plan['test_ids'])
        if len(complete_cal) == 8:
            candidate = freeze_candidate(staging, calibration, plan_hash)
        else:
            candidate = None
            if (staging/'candidate').exists() or test_present:
                raise ValueError('Candidate/test checkpoints cannot precede complete calibration')
        for item in document['items']:
            if checkpoint_path(staging, item['id']).exists():
                verified(staging, item, plan_hash, candidate if item['split'] == 'test' else None)
                _copy_atomic(checkpoint_path(staging, item['id']), checkpoint_path(output, item['id']))
        if candidate is not None:
            _copy_atomic(staging/'candidate', Path(output)/'candidate')


def archive(output):
    output = Path(output)
    destination = output/'temporal_functional_results.zip'
    temporary = output/'.temporal_functional_results.zip.tmp'
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path not in (destination, temporary) and not any(part.startswith('.') for part in path.relative_to(output).parts):
                bundle.write(path, path.relative_to(output))
    with zipfile.ZipFile(temporary) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('Final temporal archive CRC failure')
    os.replace(temporary, destination)
    return destination


def initialize_engine(reference, cache, on_loaded):
    from importlib.metadata import version

    import torch
    import transformers
    if transformers.__version__ != base.TRANSFORMERS or version('accelerate') != base.ACCELERATE:
        raise ValueError('Exact original Transformers5.22.2 and accelerate1.16.0 required')
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(base.SEED)
    transformers.utils.logging.disable_progress_bar()
    hardware = base.hardware_preflight(torch, cache)
    print('Stage model: Original Qwen3-1.7B FP32 laden; native SDPA MATH.', flush=True)
    tokenizer, model = base.load_model(reference, cache)
    hardware.update(torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version(),
                    transformers=transformers.__version__, dtype='float32', tf32=False,
                    attention='SDPA MATH', compute_capability=list(torch.cuda.get_device_capability()))
    on_loaded(hardware)
    groups = defaultdict(list)
    for block, channel in reference.coordinates:
        groups[block].append(channel)
    with torch.no_grad():
        norms = {block: model.model.layers[block].mlp.down_proj.weight.norm(dim=0)[columns]
                 for block, columns in groups.items()}
    return {'model': model, 'tokenizer': tokenizer, 'hardware': hardware, 'weight_norms': norms}


def run(hn, policy, refw, items_path, plan_path, out_dir, cache_dir=None, resume_zip=None, offline_plan_only=False):
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    nonce = os.environ.get('H286_WRAPPER_ATTEMPT')
    status = {'state': 'running', 'stage': 'reference', 'schema': SCHEMA,
              'completed_calibration': 0, 'completed_test': 0, 'completed_inputs': 0,
              'wrapper_attempt_id': nonce, 'confirmatory_eligible': False}
    metrics = {'weights_loaded': False, 'pretrained_model_run': False, 'no_training': True,
               'generation_count': 0, 'wrapper_attempt_id': nonce}
    started = time.monotonic()
    def save():
        base.atomic_json(output/'status.json', status)
        base.atomic_json(output/'metrics.json', metrics)
    save()
    try:
        reference, stats, document, scientific_plan = load_inputs(hn, policy, refw, items_path, plan_path)
        plan = plan_document(reference, stats, document, scientific_plan)
        plan_hash = hashlib.sha256(base.canonical(plan)).hexdigest()
        existing = output/'run_plan.json'
        if existing.exists() and existing.read_bytes() != base.canonical(plan)+b'\n':
            raise ValueError('Output directory belongs to another exact temporal plan')
        if not existing.exists():
            base.atomic_json(existing, plan)
        if offline_plan_only:
            status.update(state='complete', stage='offline_plan_only', plan_sha256=plan_hash, no_model_executed=True)
            return status
        restore_resume(resume_zip, output, plan_hash, document, scientific_plan)
        by_id = {item['id']: item for item in document['items']}
        calibration = [by_id[item_id] for item_id in scientific_plan['calibration_ids']]
        test = [by_id[item_id] for item_id in scientific_plan['test_ids']]
        cal_done = {item['id']: verified(output, item, plan_hash) for item in calibration}
        if (output/'candidate').exists() and not all(cal_done.values()):
            raise ValueError('Candidate exists without eight complete valid calibration checkpoints')
        candidate = freeze_candidate(output, calibration, plan_hash) if all(cal_done.values()) else None
        test_done = {item['id']: (verified(output, item, plan_hash, candidate) if candidate is not None else None) for item in test}
        if candidate is None and any(checkpoint_path(output, item['id']).exists() for item in test):
            raise ValueError('Test artifacts cannot exist before a validated frozen candidate')
        def progress():
            status['completed_calibration'] = sum(record is not None for record in cal_done.values())
            status['completed_test'] = sum(record is not None for record in test_done.values())
            status['completed_inputs'] = status['completed_calibration']+status['completed_test']
            records = [record for record in test_done.values() if record is not None]
            metrics['generation_count'] = sum(row['state'] == 'complete' for record in records for row in record['generations'])
            metrics['endpoint_complete_count'] = sum(row['state'] == 'complete' for record in records for row in record['endpoints'])
            metrics['endpoint_not_run_count'] = sum(row['state'] == 'not_run' for record in records for row in record['endpoints'])
            metrics['completed_inputs'] = status['completed_inputs']
        progress()
        if all(cal_done.values()) and all(test_done.values()):
            status.update(state='complete', stage='complete', resumed_without_gpu=True,
                          candidate_sha256=candidate['candidate_sha256'], candidate_valid=candidate['valid'])
            metrics.update(retained_pretrained_model_run=True,
                           retained_technical_checks=[record['technical_checks'] for record in [*cal_done.values(), *test_done.values()]])
            return status
        status['stage'] = 'pinned_model_load'
        save()
        def loaded(hardware):
            metrics.update(weights_loaded=True, hardware=hardware)
            save()
        engine = initialize_engine(reference, Path(cache_dir or '/content/h286_model_cache'), loaded)
        def model_ran():
            if not metrics['pretrained_model_run']:
                metrics['pretrained_model_run'] = True
                save()
        engine['on_model_run'] = model_ran
        compute_started = time.monotonic()
        deadline = compute_started+BUDGET_SECONDS
        stop = lambda: time.monotonic() >= deadline
        durations = []
        test_durations = []
        for item in calibration:
            if cal_done[item['id']] is not None:
                continue
            if stop():
                status.update(state='paused', stage='safe_calibration_checkpoint', reason='20-minute soft compute budget reached')
                break
            status.update(stage='calibration', current_input=item['id'])
            save()
            print(f'Stage calibration {status["completed_calibration"]+1}/8: {item["id"]}', flush=True)
            metrics['model_forward_attempted'] = True
            record, arrays = compute_calibration(engine, reference, stats, item, document['neutral_token_ids'])
            metrics['pretrained_model_run'] = True
            cal_done[item['id']] = write_checkpoint(output, record, arrays, plan_hash, engine['hardware'])
            durations.append(record['timing']['total_seconds'])
            progress()
            save()
            print(f'Gespeichert calibration {status["completed_calibration"]}/8; letzte Rechnung {durations[-1]:.2f}s.', flush=True)
        if not all(cal_done.values()):
            metrics['compute_elapsed_seconds'] = time.monotonic()-compute_started
            return status
        # Refit uses only validated calibration bytes, and publication/readback
        # occurs before the first possible held-out forward in this attempt.
        candidate = freeze_candidate(output, calibration, plan_hash)
        status.update(candidate_sha256=candidate['candidate_sha256'], candidate_valid=candidate['valid'])
        save()
        print('Stage candidate: eingefroren; gültig='+str(candidate['valid'])+'; SHA256 '+candidate['candidate_sha256'][:16], flush=True)
        for item in test:
            if test_done[item['id']] is not None:
                continue
            if stop():
                status.update(state='paused', stage='safe_test_checkpoint', reason='20-minute soft compute budget reached')
                break
            status.update(stage='test', current_input=item['id'])
            save()
            print(f'Stage test {status["completed_test"]+1}/8: {item["id"]}; eingefrorene23Endpunkte.', flush=True)
            case_started = time.monotonic()
            # An exact reconstruction/readback precedes each case, including a
            # resumed case, without inspecting any current test target states.
            checked_candidate = freeze_candidate(output, calibration, plan_hash)
            metrics['model_forward_attempted'] = True
            record, arrays = compute_test(engine, reference, stats, item, document['neutral_token_ids'],
                                         checked_candidate, stop_requested=stop)
            metrics['pretrained_model_run'] = True
            if record['state'] != 'complete':
                partial = write_partial(output, record, arrays, plan_hash)
                status.update(state='paused', stage='partial_test_checkpoint',
                              reason=record['stop_reason'], partial_path=str(partial.relative_to(output)))
                break
            test_done[item['id']] = write_checkpoint(output, record, arrays, plan_hash, engine['hardware'], checked_candidate)
            test_durations.append(time.monotonic()-case_started)
            progress()
            status.update(compute_elapsed_seconds=time.monotonic()-compute_started,
                          estimated_remaining_seconds=sum(test_durations)/len(test_durations)*(8-status['completed_test']),
                          eta_scope='remaining test-case wall including candidate verification/checkpoint; excludes final archive/install/download')
            save()
            print(f'Gespeichert test {status["completed_test"]}/8; {metrics["generation_count"]} Antworten; '
                  f'letzter Fall {test_durations[-1]:.2f}s; Rechenrestschätzung {status["estimated_remaining_seconds"]/60:.1f}min.', flush=True)
        else:
            status.update(state='complete', stage='complete')
        metrics['compute_elapsed_seconds'] = time.monotonic()-compute_started
    except BaseException as exc:  # noqa: BLE001 — preserve model/interrupt diagnostics.
        status.update(state='failed', error_type=type(exc).__name__, error=str(exc))
        (output/'traceback.txt').write_text(traceback.format_exc())
        print('HARD FAILURE: '+str(exc), flush=True)
    finally:
        metrics['elapsed_seconds'] = time.monotonic()-started
        save()
        archive(output)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('hn-set', 'policy', 'refw', 'items', 'plan', 'out-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--cache-dir')
    parser.add_argument('--resume-zip')
    parser.add_argument('--offline-plan-only', action='store_true')
    options = parser.parse_args()
    terminal = run(options.hn_set, options.policy, options.refw, options.items, options.plan,
                   options.out_dir, options.cache_dir, options.resume_zip, options.offline_plan_only)
    raise SystemExit(0 if terminal['state'] in ('complete', 'paused') else 1)
