"""Prespecified local B14→B18 inverse refinement; no free generations.

All four same-lambda stages precede the two fixed damping ablations. Original
weights, H286, source tokens and the old runtime remain unchanged. Checkpoints
bind one case and one damping factor; a complete run requires all twelve.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

import qwen3_b14_b18_runtime as old

base = old.base
SCHEMA = 'qwen3_b14_b18_refinement_v1'
INDICES = (0, 1, 12, 13)
FACTORS = (1., .3, .1)
MAX_ITER = 512
RTOL = 1e-4
NOISE_COUNT = 8
NOISE_SEED_BASE = base.SEED + 14000
BUDGET_SECONDS = 1200
ANCHOR_RTOL = 1e-3
ANCHOR_ATOL = 1e-5
ANCHORS_SHA256 = 'd56873915a388bcf9c42e606a64f512d02098bf1747e45306196938e9284b2c7'
REFINE_PLAN_SHA256 = 'ff4cd84d3ff1e01e87074d5c2938d79224c83634fa441a095a70c7d1d9b78afb'
OLD_RUNTIME_SHA256 = 'c04586073a54fed7a257db7e27fed48209abab8b151914a74439d154e1fad6af'
SOURCE_NAMES = ('qwen3_b14_b18_runtime.py', *old.SOURCE_NAMES,
                'attention_probe/transport_refinement.py')
VECTOR_KEYS = ('baseline_source', 'baseline_target', 'source_delta',
               'target_delta', 'jvp', 'old_delta_source')
PROOF_KEYS = ('native_sham_exact', 'mapped_baseline', 'mapped_pulse',
              'anchor_continuity', 'derivative_calibration', 'frozen_parameters',
              'global_rng', 'hook_cleanup', 'finite_arrays', 'unchanged_positions',
              'upstream_causal')


def factor_name(factor):
    if type(factor) not in (int, float) or factor not in FACTORS:
        raise ValueError('Only the three prespecified damping factors are allowed')
    return {1.: 'f1', .3: 'f0p3', .1: 'f0p1'}[factor]


def stage_order():
    return [(index, factor) for factor in FACTORS for index in INDICES]


def arm_names(factor):
    factor_name(factor)
    return (['recorded_old_ridge', 'raw_target_copy'] if factor == 1. else []) + [
        'refined_ridge', 'norm_matched_copy', *[f'norm_matched_noise_{i}' for i in range(NOISE_COUNT)]]


def _sha(path):
    return base.sha_file(path)


def _json_file(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Exact pinned refinement input bytes required: ' + Path(path).name)
    return json.loads(raw)


def load_anchors(path, items, *, expected_sha256=None, expected_width=2048):
    document = _json_file(path, expected_sha256 or ANCHORS_SHA256)
    cases = document.get('cases', [])
    if (document.get('schema') != 'b14_b18_refine_anchors_v1'
            or document.get('source_runtime_sha256') != OLD_RUNTIME_SHA256
            or [case.get('index') for case in cases] != list(INDICES)):
        raise ValueError('Anchors must bind the four original cases and original runtime')
    for case in cases:
        index = case['index']
        if (case.get('request_id') != f'b14b18_{index:03d}'
                or case.get('item_sha256') != hashlib.sha256(base.canonical(items[index])).hexdigest()
                or type(case.get('lambda')) not in (float, int)
                or not math.isfinite(case['lambda']) or case['lambda'] <= 0):
            raise ValueError('Anchor context, item or original damping changed')
        for key in VECTOR_KEYS:
            values = case.get(key)
            if (not isinstance(values, list) or len(values) != expected_width
                    or any(type(value) not in (float, int) or not math.isfinite(value)
                           or abs(value) > 3.4028234663852886e38 for value in values)):
                raise ValueError('Anchor vectors must have finite native hidden width')
    return document


def load_inputs(hn, policy, refw, items_path, plan_path, refine_plan_path, anchors_path):
    reference, stats, items, original_plan = old.load_inputs(hn, policy, refw, items_path, plan_path)
    refinement = _json_file(refine_plan_path, REFINE_PLAN_SHA256)
    expected = {'schema': 'b14_b18_refine_plan_v1', 'indices': list(INDICES),
                'lambda_factors': list(FACTORS), 'max_iter': MAX_ITER, 'rtol': RTOL,
                'noise_count': NOISE_COUNT, 'noise_seed_base': NOISE_SEED_BASE,
                'true_residual_interval': 16,
                'seconds_after_model_load': BUDGET_SECONDS,
                'anchor_rtol': ANCHOR_RTOL, 'anchor_atol': ANCHOR_ATOL}
    if any(refinement.get(key) != value for key, value in expected.items()):
        raise ValueError('Refinement plan changes prespecified numerical rules')
    anchors = load_anchors(anchors_path, items)
    if _sha(old.__file__) != OLD_RUNTIME_SHA256:
        raise ValueError('The frozen original runtime source has changed')
    parent = old.plan_document(reference, stats, items, original_plan)
    parent_raw = base.canonical(parent)
    if (hashlib.sha256(parent_raw).hexdigest() != anchors.get('source_plan_sha256')
            or hashlib.sha256(parent_raw + b'\n').hexdigest() != anchors.get('source_plan_file_sha256')):
        raise ValueError('Original parent plan/source binding cannot be reconstructed')
    return reference, stats, items, original_plan, refinement, anchors


def plan_document(reference, stats, items, original_plan, refinement, anchors):
    return {'schema': SCHEMA, 'helper_sha256': _sha(__file__),
        'source_sha256': {name: _sha(old.source_path(name)) for name in SOURCE_NAMES},
        'parent_plan_sha256': anchors['source_plan_sha256'],
        'parent_plan_file_sha256': anchors['source_plan_file_sha256'],
        'source_zip_sha256': anchors['source_zip_sha256'],
        'refine_plan_sha256': REFINE_PLAN_SHA256, 'anchors_sha256': ANCHORS_SHA256,
        'refinement': refinement, 'original_plan': original_plan,
        'items_sha256': old.ITEMS_SHA256, 'original_input_plan_sha256': old.INPUT_PLAN_SHA256,
        'items': items, 'indices': list(INDICES), 'stage_order': [list(row) for row in stage_order()],
        'model': reference.model.to_dict(), 'coordinates': [list(c) for c in reference.coordinates],
        'hn_sha256': reference.hn_sha256, 'policy_sha256': reference.policy_sha256,
        'refw_sha256': base.REFW_SHA256, 'refw_source_sha256': stats['source_hn_meta_sha256'],
        'seed': base.SEED, 'dtype': 'float32', 'tf32': False,
        'transformers': base.TRANSFORMERS, 'accelerate': base.ACCELERATE,
        'attention': 'sdpa/MATH; native local derivatives and nonlinear endpoint probes',
        'max_iter': MAX_ITER, 'rtol': RTOL, 'lambda_factors': list(FACTORS),
        'lambda_rule': 'recorded original lambda times fixed factor; no gain re-estimation or outcome selection',
        'controls': {'noise_seeds': {str(i): [NOISE_SEED_BASE+i+k for k in range(NOISE_COUNT)] for i in INDICES},
                     'copy': 'target_delta direction normalized to the refined source norm after native FP32 rounding',
                     'paired_noise': 'same eight fixed directions across damping factors',
                     'old_baselines': 'recorded sixteen-CG delta and original unmatched raw target-copy once per case'},
        'anchor_gate': {'metric': 'L2(current-reference) <= atol + rtol*L2(reference)',
                        'rtol': ANCHOR_RTOL, 'atol': ANCHOR_ATOL,
                        'logit_SHA': 'recorded evidence, not a cross-hardware exact gate'},
        'budget': {'compute_seconds': BUDGET_SECONDS, 'boundary': 'before stages, solver iterations and endpoint arms',
                   'cooperative': 'an in-flight GPU call cannot be preempted; solver may finish its final true-residual evidence',
                   'partial': 'partial stages retained separately; only full stage checkpoints are resumable'},
        'free_generations': 0, 'claim': 'exploratory numerical local inverse refinement; no training or global inverse proof'}


def anchor_comparison(current, values):
    import torch
    reference = torch.tensor(values, dtype=current.dtype, device=current.device)
    if (reference.shape != current.shape or not bool(torch.isfinite(current).all())
            or not bool(torch.isfinite(reference).all())):
        raise ValueError('Current anchor is nonfinite or has changed shape')
    norm = float(reference.double().norm())
    error = float((current.double()-reference.double()).norm())
    limit = ANCHOR_ATOL + ANCHOR_RTOL*norm
    return {'reference_norm': norm, 'current_norm': float(current.double().norm()),
            'error_norm': error, 'relative_error': error/norm if norm else None,
            'limit': limit, 'passed': error <= limit, 'bitwise_equal': bool(torch.equal(current, reference))}


def control_vectors(target, solution, index):
    """Fixed directions; local CPU generators do not alter global RNG state."""
    import torch
    norm, target_norm = float(solution.double().norm()), float(target.double().norm())
    if target_norm == 0:
        copy = torch.zeros_like(target)
    else:
        copy = (target.double()*(norm/target_norm)).to(dtype=target.dtype)
    return [('refined_ridge', solution), ('norm_matched_copy', copy)] + [
        (f'norm_matched_noise_{k}', old.fixed_noise_like(solution, NOISE_SEED_BASE+index+k))
        for k in range(NOISE_COUNT)]


def endpoint_probe(model, request, coordinates, stats, arm, *, weight_norms=None, patch=None):
    """Observe all earlier prompt positions at the six fixed residual layers."""
    earlier = {}
    observers = []
    calls = 0
    def remove_observers(*_args):
        while observers:
            observers.pop().remove()
    def install_observers(_model, _args):
        nonlocal calls
        calls += 1
        if calls != 1:
            raise ValueError('Endpoint observer expected exactly one native full forward')
        # old.forward_probe has now installed its patch/capture hooks. Appending
        # these output observers here places them after the actual source patch.
        for block in old.RESIDUAL_BLOCKS:
            def observe(_module, _args, output, block=block):
                value = output[0] if isinstance(output, tuple) else output
                earlier[str(block)] = old.digest_tensor(value[:, :-1, :])
            observers.append(model.model.layers[block].register_forward_hook(observe))
    with contextlib.ExitStack() as contexts:
        contexts.callback(remove_observers)
        contexts.callback(model.register_forward_pre_hook(install_observers).remove)
        contexts.callback(model.register_forward_hook(remove_observers, always_call=True).remove)
        result = old.forward_probe(model, request, 0, coordinates, stats, arm,
                                   backend='MATH', weight_norms=weight_norms, patch=patch)
    if calls != 1 or set(earlier) != {str(block) for block in old.RESIDUAL_BLOCKS}:
        raise ValueError('Missing native post-patch earlier-position observers')
    result['earlier_prompt_residual_sha256'] = earlier
    return result


def _calibrate(mapping, source_delta, index, arrays):
    import torch
    predicted = mapping.jvp(source_delta)
    differences = []
    for step in old.FD_STEPS:
        finite = mapping.central_fd(source_delta, step)
        row = old.vector_metrics(finite, predicted)
        floor = max(1e-7, 64*torch.finfo(source_delta.dtype).eps*float(mapping.baseline_target.double().norm())/step)
        row.update(step=step, estimated_FP32_noise_floor=floor,
                   informative=row['actual_norm'] > floor)
        row['passed'] = bool(row['informative'] and row['relative_error'] <= .05
                             and row['cosine'] is not None and row['cosine'] >= .99)
        differences.append(row)
        arrays[f'calibration_fd_{str(step).replace(".", "_")}'] = finite.cpu().numpy()
    direction = old.fixed_noise_like(torch.ones_like(source_delta), base.SEED+index+51000)
    cotangent = old.fixed_noise_like(torch.ones_like(source_delta), base.SEED+index+52000)
    image, transpose = mapping.jvp(direction), mapping.vjp(cotangent)
    left = float(torch.dot(image.double(), cotangent.double()))
    right = float(torch.dot(direction.double(), transpose.double()))
    scale = float(image.double().norm()*cotangent.double().norm()+direction.double().norm()*transpose.double().norm())
    error = abs(left-right)
    adjoint = {'left': left, 'right': right, 'absolute_error': error,
               'normalized_error': error/scale if scale else 0., 'tolerance': 1e-5,
               'passed': error <= 1e-5*scale+1e-8}
    passed = any(row['passed'] for row in differences) and adjoint['passed']
    if not passed:
        raise ValueError('Native finite-difference/adjoint gate failed before inverse refinement: '
                         +json.dumps({'finite_differences': differences, 'adjoint': adjoint}, allow_nan=False))
    return {'state': 'passed', 'finite_differences': differences, 'adjoint': adjoint}


def compute_stage(engine, reference, stats, request, anchor, factor, *, stop_requested=None):
    import numpy as np
    import torch

    from attention_probe.native_jacobian import build_residual_map
    from attention_probe.transport_refinement import refine_transport

    factor_name(factor)
    stop = stop_requested or (lambda: False)
    model = engine['model']
    device = next(model.parameters()).device
    started = time.monotonic()
    before_parameters, before_hooks = base.parameter_evidence(model), old.hook_count(model)
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device.type == 'cuda' else None
    arrays = {}
    record = {'request': request, 'lambda_factor': factor, 'original_lambda': anchor['lambda'],
              'lambda': anchor['lambda']*factor, 'state': 'partial', 'arms': [],
              'array_contract': {'coordinates': len(reference.coordinates),
                                 'coordinate_layers': [block for block, _ in reference.coordinates],
                                 'hidden_size': model.config.hidden_size,
                                 'vocab_size': model.config.vocab_size},
              'timing': {}, 'checks': {}}
    if stop():
        record.update(stop_reason='budget_exhausted_before_stage')
        return record, arrays
    clean = endpoint_probe(model, request, reference.coordinates, stats, old.ARMS[0],
                           weight_norms=engine.get('weight_norms'))
    if engine.get('on_model_run') is not None:
        engine['on_model_run']()
    pulse = endpoint_probe(model, request, reference.coordinates, stats, old.ARMS[4],
                           weight_norms=engine.get('weight_norms'))
    for name, probe in (('math_clean', clean), ('math_pulse', pulse)):
        old.save_probe(arrays, name, probe)
        arrays[name+'_logits'] = probe['logits'].numpy()
    ids = torch.tensor([request['prompt_ids']], dtype=torch.long, device=device)
    with torch.no_grad(), old.attention_context(torch, 'MATH'):
        native = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1].detach().float().cpu()
    native_exact = bool(torch.equal(native, clean['logits']))
    if not native_exact:
        raise ValueError('Uninstrumented MATH clean differs from passive-capture sham')
    mapping = build_residual_map(model, ids, 14, 18, attention_mask=torch.ones_like(ids))
    if (not torch.equal(mapping.baseline_source, old.tensor_residual(clean, 14, device))
            or not torch.equal(mapping.baseline_target, old.tensor_residual(clean, 18, device))):
        raise ValueError('Native suffix baseline differs from the captured MATH clean source/target')
    source_delta = old.tensor_residual(pulse, 14, device)-mapping.baseline_source
    target_delta = old.tensor_residual(pulse, 18, device)-mapping.baseline_target
    prediction = mapping.jvp(source_delta)
    vectors = {'baseline_source': mapping.baseline_source, 'baseline_target': mapping.baseline_target,
               'source_delta': source_delta, 'target_delta': target_delta, 'jvp': prediction}
    comparisons = {key: anchor_comparison(value, anchor[key]) for key, value in vectors.items()}
    if not all(row['passed'] for row in comparisons.values()):
        raise ValueError('Prior native-state/JVP continuity gate failed: '+str(comparisons))
    recovered = mapping(mapping.baseline_source).detach()
    actual = mapping(old.tensor_residual(pulse, 14, device)).detach()-mapping.baseline_target
    baseline_ok = bool(torch.allclose(recovered, mapping.baseline_target, rtol=1e-5, atol=1e-5))
    pulse_ok = bool(torch.allclose(actual, target_delta, rtol=1e-4, atol=1e-5))
    if not baseline_ok or not pulse_ok:
        raise ValueError('Native suffix primal continuity failed')
    arrays.update({f'jac_{key}': value.detach().cpu().numpy() for key, value in vectors.items()})
    old_delta = torch.tensor(anchor['old_delta_source'], dtype=torch.float32, device=device)
    arrays['recorded_old_delta_source'] = old_delta.cpu().numpy()
    record['anchor_continuity'] = {'vectors': comparisons,
        'earlier_prompt_residual_sha256': clean['earlier_prompt_residual_sha256'],
        'pulse_earlier_positions_equal_clean': pulse['earlier_prompt_residual_sha256'] == clean['earlier_prompt_residual_sha256'],
        'pulse_upstream_raw_z_unchanged': bool(np.array_equal(
            pulse['arrays']['raw_z'][..., [i for i, (block, _) in enumerate(reference.coordinates) if block < 14]],
            clean['arrays']['raw_z'][..., [i for i, (block, _) in enumerate(reference.coordinates) if block < 14]])),
        'pulse_upstream_resid13_unchanged': bool(np.array_equal(pulse['arrays']['residual_13'], clean['arrays']['residual_13'])),
        'math_clean_logits_sha256': old.digest_tensor(clean['logits']),
        'original_math_clean_logits_sha256': anchor['math_clean_logits_sha256'],
        'math_pulse_logits_sha256': old.digest_tensor(pulse['logits']),
        'original_math_pulse_logits_sha256': anchor['math_pulse_logits_sha256'],
        'claim': 'bounded vector continuity, not proof of an exactly identical operator across hardware'}
    record['calibration'] = (_calibrate(mapping, source_delta, request['index'], arrays) if factor == 1.
                              else {'state': 'not_run', 'reason': 'all four primary f1 stages perform the fixed gates'})
    record['timing']['setup_seconds'] = time.monotonic()-started
    solve_started = time.monotonic()
    solved = refine_transport(mapping.jvp, mapping.vjp, target_delta, record['lambda'],
                              max_iter=MAX_ITER, rtol=RTOL, stop_requested=stop,
                              true_residual_interval=16)
    record['solver'] = solved.diagnostics
    record['timing']['solver_seconds'] = time.monotonic()-solve_started
    arrays['solver_delta_source'] = solved.delta_source.detach().cpu().numpy()
    arrays['solver_dual_solution'] = solved.dual_solution.detach().cpu().numpy()
    if solved.diagnostics.get('stop_reason') == 'budget_exhausted' or stop():
        record.update(stop_reason='budget_exhausted_during_solver')
        record['timing']['total_seconds'] = time.monotonic()-started
        return record, arrays
    controls = control_vectors(target_delta, solved.delta_source, request['index'])
    if factor == 1.:
        original_target = torch.tensor(anchor['target_delta'], dtype=torch.float32, device=device)
        controls = [('recorded_old_ridge', old_delta), ('raw_target_copy', original_target), *controls]
    evaluation_started = time.monotonic()
    for name, delta in controls:
        if stop():
            record.update(stop_reason='budget_exhausted_between_endpoint_arms')
            break
        arm_started = time.monotonic()
        probe = endpoint_probe(model, request, reference.coordinates, stats, old.ARMS[0],
                               weight_norms=engine.get('weight_norms'),
                               patch={'block': 14, 'delta': delta})
        achieved = old.tensor_residual(probe, 18, device)-mapping.baseline_target
        realized = old.tensor_residual(probe, 14, device)-mapping.baseline_source
        linear = mapping.jvp(delta)
        prefix = 'endpoint_'+name
        old.save_probe(arrays, prefix, probe)
        arrays[prefix+'_logits'] = probe['logits'].numpy()
        arrays[prefix+'_requested_delta'] = delta.detach().cpu().numpy()
        arrays[prefix+'_realized_source_delta'] = realized.detach().cpu().numpy()
        arrays[prefix+'_target_actual'] = achieved.detach().cpu().numpy()
        arrays[prefix+'_target_predicted'] = linear.detach().cpu().numpy()
        source_norm = float(delta.double().norm())
        refined_norm = float(solved.delta_source.double().norm())
        row = {'name': name, **probe['evidence'],
               'earlier_prompt_positions_unchanged': probe['earlier_prompt_residual_sha256'] == clean['earlier_prompt_residual_sha256'],
               'earlier_prompt_residual_sha256': probe['earlier_prompt_residual_sha256'],
               'upstream_raw_z_unchanged': bool(np.array_equal(
                   probe['arrays']['raw_z'][..., [i for i, (block, _) in enumerate(reference.coordinates) if block <= 14]],
                   clean['arrays']['raw_z'][..., [i for i, (block, _) in enumerate(reference.coordinates) if block <= 14]])),
               'upstream_resid13_unchanged': bool(np.array_equal(probe['arrays']['residual_13'], clean['arrays']['residual_13'])),
               'requested_source_norm': source_norm, 'realized_source_norm': float(realized.double().norm()),
               'source_rounding_error_norm': float((realized.double()-delta.double()).norm()),
               'refined_source_norm': refined_norm,
               'norm_ratio_to_refined': source_norm/refined_norm if refined_norm else None,
               'norm_match_error': abs(source_norm-refined_norm),
               'noise_seed': NOISE_SEED_BASE+request['index']+int(name.rsplit('_', 1)[-1]) if name.startswith('norm_matched_noise_') else None,
               'linear_target_prediction': old.vector_metrics(target_delta, linear),
               'nonlinear_target_prediction': old.vector_metrics(target_delta, achieved),
               'linear_vs_nonlinear': old.vector_metrics(achieved, linear),
               'logit_references': {'math_clean': old.logits_metrics(probe['logits'], clean['logits'], request['continuation_ids'][0]),
                                    'math_pulse': old.logits_metrics(probe['logits'], pulse['logits'], request['continuation_ids'][0])},
               'elapsed_seconds': time.monotonic()-arm_started}
        if name.startswith('norm_matched_') and abs(source_norm-refined_norm) > 2e-6*max(1., refined_norm):
            raise ValueError('Native FP32 control norm matching failed')
        record['arms'].append(row)
    record['timing']['endpoint_seconds'] = time.monotonic()-evaluation_started
    record['timing']['total_seconds'] = time.monotonic()-started
    if len(record['arms']) != len(arm_names(factor)):
        return record, arrays
    checks = {'native_sham_exact': native_exact, 'mapped_baseline': baseline_ok,
              'mapped_pulse': pulse_ok, 'anchor_continuity': all(row['passed'] for row in comparisons.values()),
              'derivative_calibration': record['calibration']['state'] in ('passed', 'not_run'),
              'frozen_parameters': before_parameters == base.parameter_evidence(model),
              'global_rng': bool(torch.equal(cpu_rng, torch.random.get_rng_state()) and
                                 (cuda_rng is None or torch.equal(cuda_rng, torch.cuda.get_rng_state()))),
              'hook_cleanup': old.hook_count(model) == before_hooks,
              'unchanged_positions': bool(record['anchor_continuity']['pulse_earlier_positions_equal_clean'] and
                                           all(row['earlier_prompt_positions_unchanged'] for row in record['arms'])),
              'upstream_causal': bool(record['anchor_continuity']['pulse_upstream_raw_z_unchanged'] and
                                      record['anchor_continuity']['pulse_upstream_resid13_unchanged'] and
                                      all(row['upstream_raw_z_unchanged'] and row['upstream_resid13_unchanged'] for row in record['arms'])),
              'finite_arrays': all(value.dtype == np.float32 and np.isfinite(value).all() for value in arrays.values())}
    if not all(checks.values()):
        raise ValueError('Native/frozen/RNG/cleanup evidence failed: '+str(checks))
    record.update(state='complete', checks=checks)
    return record, arrays


def array_shapes(record):
    width = record['array_contract']['hidden_size']
    n = record['array_contract']['coordinates']
    vocab = record['array_contract']['vocab_size']
    shapes = {}
    def probe(prefix):
        shapes.update({prefix+'_'+key: (1, 1, n) for key in ('raw_z', 'cett', 'zcett', 'down_output_norm_raw')})
        shapes.update({prefix+f'_residual_{block}': (1, 1, width) for block in old.RESIDUAL_BLOCKS})
        shapes[prefix+'_logits'] = (vocab,)
    probe('math_clean')
    probe('math_pulse')
    shapes.update({f'jac_{key}': (width,) for key in VECTOR_KEYS if key != 'old_delta_source'})
    shapes.update({key: (width,) for key in ('recorded_old_delta_source', 'solver_delta_source', 'solver_dual_solution')})
    if record['lambda_factor'] == 1.:
        shapes.update({f'calibration_fd_{str(step).replace(".", "_")}': (width,) for step in old.FD_STEPS})
    for name in arm_names(record['lambda_factor']):
        prefix = 'endpoint_'+name
        probe(prefix)
        shapes.update({prefix+'_'+key: (width,) for key in ('requested_delta', 'realized_source_delta', 'target_actual', 'target_predicted')})
    return shapes


def validate_record(record, plan_hash, item=None, factor=None):
    request = record.get('request', {})
    current_factor = record.get('lambda_factor')
    if (record.get('schema') != SCHEMA or record.get('state') != 'complete'
            or record.get('plan_sha256') != plan_hash or type(request.get('index')) is not int
            or request.get('index') not in INDICES
            or request.get('request_id') != f'b14b18_{request["index"]:03d}'
            or (item is not None and request.get('item') != item)
            or type(current_factor) not in (int, float) or current_factor not in FACTORS
            or (factor is not None and current_factor != factor)
            or record.get('lambda') != record.get('original_lambda', -1)*current_factor
            or not math.isfinite(record['lambda']) or record['lambda'] <= 0):
        raise ValueError('Complete stage changes its bound schema/context/plan/damping')
    if (request.get('prompt_ids') != request['item']['prompt_ids']
            or request.get('pulse_positions') != [len(request['prompt_ids'])-1]):
        raise ValueError('Stage changed original source tokens or pulse position')
    checks = record.get('checks', {})
    proof = record.get('technical_checks', {})
    if (set(checks) != set(PROOF_KEYS) or not all(value is True for value in checks.values())
            or proof != {'state': 'passed', 'schema': SCHEMA, 'plan_sha256': plan_hash, 'checks': checks}):
        raise ValueError('Complete stage requires its own full technical proof')
    if not all(row.get('passed') is True for row in record.get('anchor_continuity', {}).get('vectors', {}).values()) or set(record.get('anchor_continuity', {}).get('vectors', {})) != set(VECTOR_KEYS)-{'old_delta_source'}:
        raise ValueError('Native prior-anchor continuity proof missing')
    calibration = record.get('calibration', {})
    if (current_factor == 1. and calibration.get('state') != 'passed') or (current_factor != 1. and calibration.get('state') != 'not_run'):
        raise ValueError('Fixed derivative-calibration phase changed')
    solver = record.get('solver', {})
    if (solver.get('max_iter') != MAX_ITER or solver.get('rtol') != RTOL
            or solver.get('true_residual_interval') != 16
            or solver.get('lambda') != record['lambda']
            or solver.get('stop_reason') == 'budget_exhausted'
            or not math.isfinite(solver.get('true_dual_relative_residual', math.nan))
            or type(solver.get('iterations')) is not int or not 0 <= solver['iterations'] <= MAX_ITER):
        raise ValueError('Missing bounded finite solver evidence')
    relative = solver['true_dual_relative_residual']
    breakdown = solver.get('breakdown')
    expected_converged = relative <= RTOL and breakdown is None
    expected_status = 'breakdown' if breakdown else 'converged' if expected_converged else 'not_converged'
    if (relative < 0 or type(solver.get('converged')) is not bool
            or solver['converged'] != expected_converged or solver.get('status') != expected_status):
        raise ValueError('Solver convergence/status contradict its measured true residual')
    if [row.get('name') for row in record.get('arms', [])] != arm_names(current_factor):
        raise ValueError('Missing/changed fixed endpoint controls')
    for row in record['arms']:
        if (row.get('pulse_fired') != 1 or row.get('capture_fired') != 1
                or row.get('backend') != 'MATH' or row.get('pulse_positions') != request['pulse_positions']
                or row.get('capture_positions') != request['pulse_positions']
                or row.get('patch', {}).get('fired') != 1):
            raise ValueError('Native endpoint patch/capture evidence missing')
        if not all(row.get(key) is True for key in ('earlier_prompt_positions_unchanged', 'upstream_raw_z_unchanged', 'upstream_resid13_unchanged')):
            raise ValueError('Earlier-position/upstream causal invariance evidence failed')
        earlier = record['anchor_continuity'].get('earlier_prompt_residual_sha256', {})
        if (set(earlier) != {str(block) for block in old.RESIDUAL_BLOCKS}
                or row.get('earlier_prompt_residual_sha256') != earlier):
            raise ValueError('Stored earlier-position hashes contradict the causal invariance claim')
        if row['name'].startswith('norm_matched_') and row.get('norm_match_error', math.inf) > 2e-6*max(1., row['refined_source_norm']):
            raise ValueError('Control source norms no longer match')
        if row['name'].startswith('norm_matched_noise_') and row.get('noise_seed') != NOISE_SEED_BASE+request['index']+int(row['name'].rsplit('_', 1)[-1]):
            raise ValueError('Fixed paired noise seed changed')
        for comparison in ('linear_target_prediction', 'nonlinear_target_prediction'):
            if not math.isfinite(row.get(comparison, {}).get('error_norm', math.nan)):
                raise ValueError('Missing finite linear/nonlinear endpoint result')


def validate_arrays(record, arrays):
    import numpy as np
    import torch
    expected = array_shapes(record)
    if set(arrays) != set(expected):
        raise ValueError('Stage arrays have missing or unrecognized keys')
    for name, shape in expected.items():
        value = arrays[name]
        if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Stage array changed shape/dtype/finiteness: '+name)
    clean = torch.from_numpy(arrays['math_clean_logits'])
    pulse = torch.from_numpy(arrays['math_pulse_logits'])
    layers = record['array_contract']['coordinate_layers']
    if len(layers) != record['array_contract']['coordinates'] or any(type(block) is not int or not 0 <= block < 28 for block in layers):
        raise ValueError('Invalid fixed coordinate-layer layout')
    before_source = [i for i, block in enumerate(layers) if block < 14]
    through_source = [i for i, block in enumerate(layers) if block <= 14]
    if (not np.array_equal(arrays['math_clean_raw_z'][..., before_source], arrays['math_pulse_raw_z'][..., before_source])
            or not np.array_equal(arrays['math_clean_residual_13'], arrays['math_pulse_residual_13'])):
        raise ValueError('Saved pulse changed upstream measurements')
    for row in record['arms']:
        prefix = 'endpoint_'+row['name']
        if (not np.array_equal(arrays[prefix+'_raw_z'][..., through_source], arrays['math_clean_raw_z'][..., through_source])
                or not np.array_equal(arrays[prefix+'_residual_13'], arrays['math_clean_residual_13'])):
            raise ValueError('Saved endpoint changed upstream measurements')
        logits = torch.from_numpy(arrays[prefix+'_logits'])
        for label, baseline in (('math_clean', clean), ('math_pulse', pulse)):
            if old.logits_metrics(logits, baseline, record['request']['continuation_ids'][0]) != row['logit_references'][label]:
                raise ValueError('Full-logit comparator/hash mismatch')
        for field, key in (('nonlinear_target_prediction', 'target_actual'), ('linear_target_prediction', 'target_predicted')):
            if old.vector_metrics(torch.from_numpy(arrays['jac_target_delta']), torch.from_numpy(arrays[prefix+'_'+key])) != row[field]:
                raise ValueError('Saved endpoint vector no longer agrees with numerical metrics')
        norm = float(np.linalg.norm(arrays[prefix+'_requested_delta'].astype(np.float64)))
        realized_norm = float(np.linalg.norm(arrays[prefix+'_realized_source_delta'].astype(np.float64)))
        refined_norm = float(np.linalg.norm(arrays['solver_delta_source'].astype(np.float64)))
        if (not math.isclose(norm, row['requested_source_norm'], rel_tol=1e-12, abs_tol=1e-12)
                or not math.isclose(realized_norm, row['realized_source_norm'], rel_tol=1e-12, abs_tol=1e-12)
                or not math.isclose(refined_norm, row['refined_source_norm'], rel_tol=1e-12, abs_tol=1e-12)
                or not math.isclose(abs(norm-refined_norm), row['norm_match_error'], rel_tol=1e-10, abs_tol=1e-12)):
            raise ValueError('Saved intervention vectors disagree with measured source norms')
    reconstruction = arrays['endpoint_refined_ridge_target_predicted'].astype(np.float64)
    damping = record['solver']['lambda_squared_native']
    target = arrays['jac_target_delta'].astype(np.float64)
    residual = target-reconstruction-damping*arrays['solver_dual_solution'].astype(np.float64)
    norm = float(np.linalg.norm(target))
    relative = float(np.linalg.norm(residual))/norm if norm else 0.
    if not math.isclose(relative, record['solver']['true_dual_relative_residual'], rel_tol=1e-7, abs_tol=1e-11):
        raise ValueError('Returned native dual/source do not reproduce the saved true residual')
    if not np.array_equal(arrays['solver_delta_source'], arrays['endpoint_refined_ridge_requested_delta']):
        raise ValueError('Endpoint did not use the exact returned refined source vector')


def stage_path(output, index, factor):
    return Path(output)/'requests'/f'b14b18_{index:03d}'/factor_name(factor)


def write_checkpoint(output, record, arrays, plan_hash, hardware):
    import numpy as np
    target = stage_path(output, record['request']['index'], record['lambda_factor'])
    if target.exists():
        raise ValueError('Complete stage already exists; no overwrite')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.'+target.name+'-', dir=target.parent))
    try:
        np.savez_compressed(temporary/'arrays.npz', **arrays)
        record = dict(record, schema=SCHEMA, state='complete', plan_sha256=plan_hash,
                      hardware=hardware, completed_utc=dt.datetime.now(dt.UTC).isoformat(),
                      npz_sha256=_sha(temporary/'arrays.npz'))
        record['technical_checks'] = {'state': 'passed', 'schema': SCHEMA,
                                     'plan_sha256': plan_hash, 'checks': record['checks']}
        validate_record(record, plan_hash)
        validate_arrays(record, arrays)
        base.atomic_json(temporary/'result.json', record)
        os.rename(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return record


def verified(output, index, factor, plan_hash, item):
    import numpy as np
    folder = stage_path(output, index, factor)
    if not folder.exists():
        return None
    record = json.loads((folder/'result.json').read_text())
    validate_record(record, plan_hash, item, factor)
    if _sha(folder/'arrays.npz') != record.get('npz_sha256'):
        raise ValueError('Checkpoint NPZ hash mismatch')
    with np.load(folder/'arrays.npz', allow_pickle=False) as saved:
        validate_arrays(record, {key: saved[key] for key in saved.files})
    return record


def write_partial(output, record, arrays, plan_hash):
    import numpy as np
    parent = Path(output)/'partial'/record['request']['request_id']/factor_name(record['lambda_factor'])
    parent.mkdir(parents=True, exist_ok=True)
    target = parent/str(time.time_ns())
    temporary = Path(tempfile.mkdtemp(prefix='.partial-', dir=parent))
    try:
        np.savez_compressed(temporary/'arrays.npz', **arrays)
        base.atomic_json(temporary/'result.json', dict(record, schema=SCHEMA, state='partial',
                         plan_sha256=plan_hash, npz_sha256=_sha(temporary/'arrays.npz'),
                         resume_eligible=False, interpretation='Incomplete stage; not full evidence or a complete case'))
        os.rename(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target


def restore_resume(resume_zip, output, plan_hash, items):
    if not resume_zip:
        return
    with zipfile.ZipFile(resume_zip) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('Resume CRC failure')
        if ('COMPACT_PACKAGE.json' in bundle.namelist()
                or json.loads(bundle.read('run_plan.json')).get('schema') != SCHEMA):
            raise ValueError('Only a full matching refinement archive can resume')
        if hashlib.sha256(base.canonical(json.loads(bundle.read('run_plan.json')))).hexdigest() != plan_hash:
            raise ValueError('Resume plan differs')
        allowed = {f'requests/b14b18_{index:03d}/{factor_name(factor)}/{name}'
                   for index, factor in stage_order() for name in ('result.json', 'arrays.npz')}
        request_names = [name for name in bundle.namelist() if name.startswith('requests/') and not name.endswith('/')]
        if len(request_names) != len(set(request_names)) or set(request_names)-allowed:
            raise ValueError('Unrecognized resume request path')
        for index, factor in stage_order():
            folder = f'requests/b14b18_{index:03d}/{factor_name(factor)}'
            names = [folder+'/result.json', folder+'/arrays.npz']
            present = [name in request_names for name in names]
            if not any(present):
                continue
            if not all(present):
                raise ValueError('Partial file pair inside complete resume namespace')
            existing = verified(output, index, factor, plan_hash, items[index])
            if existing is not None:
                if any((Path(output)/name).read_bytes() != bundle.read(name) for name in names):
                    raise ValueError('Resume conflicts with current complete checkpoint')
                continue
            with tempfile.TemporaryDirectory(prefix='refine-resume-') as directory:
                staging = Path(directory)
                target = stage_path(staging, index, factor)
                target.mkdir(parents=True)
                for name in names:
                    (target/Path(name).name).write_bytes(bundle.read(name))
                verified(staging, index, factor, plan_hash, items[index])
                destination = stage_path(output, index, factor)
                destination.parent.mkdir(parents=True, exist_ok=True)
                copying = Path(tempfile.mkdtemp(prefix='.'+destination.name+'-resume-', dir=destination.parent))
                try:
                    shutil.copytree(target, copying, dirs_exist_ok=True)
                    os.rename(copying, destination)
                finally:
                    if copying.exists():
                        shutil.rmtree(copying)


def archive(output):
    output = Path(output)
    target, temporary = output/'b14_b18_refine_results.zip', output/'.b14_b18_refine_results.zip.tmp'
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path not in (target, temporary) and not any(part.startswith('.') for part in path.relative_to(output).parts):
                bundle.write(path, path.relative_to(output))
    with zipfile.ZipFile(temporary) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('Final archive CRC failure')
    os.replace(temporary, target)
    return target


def run(hn, policy, refw, items_path, plan_path, refine_plan_path, anchors_path,
        out_dir, cache_dir=None, resume_zip=None, offline_plan_only=False):
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    nonce = os.environ.get('H286_WRAPPER_ATTEMPT')
    status = {'state': 'running', 'stage': 'reference', 'schema': SCHEMA, 'completed_inputs': 0,
              'completed_stages': 0, 'wrapper_attempt_id': nonce, 'confirmatory_eligible': False}
    metrics = {'weights_loaded': False, 'pretrained_model_run': False, 'no_training': True,
               'wrapper_attempt_id': nonce, 'generation_count': 0}
    started = time.monotonic()
    def save():
        base.atomic_json(output/'status.json', status)
        base.atomic_json(output/'metrics.json', metrics)
    save()
    try:
        reference, stats, items, original, refinement, anchors = load_inputs(
            hn, policy, refw, items_path, plan_path, refine_plan_path, anchors_path)
        plan = plan_document(reference, stats, items, original, refinement, anchors)
        plan_hash = hashlib.sha256(base.canonical(plan)).hexdigest()
        existing = output/'run_plan.json'
        if existing.exists() and existing.read_bytes() != base.canonical(plan)+b'\n':
            raise ValueError('Different bound refinement plan; no output overwrite')
        if not existing.exists():
            base.atomic_json(existing, plan)
        if offline_plan_only:
            status.update(state='complete', stage='offline_plan_only', plan_sha256=plan_hash, no_model_executed=True)
            return status
        restore_resume(resume_zip, output, plan_hash, items)
        done = {(index, factor): verified(output, index, factor, plan_hash, items[index])
                for index, factor in stage_order()}
        def progress():
            status['completed_stages'] = sum(value is not None for value in done.values())
            status['completed_inputs'] = sum(all(done[index, factor] is not None for factor in FACTORS) for index in INDICES)
            status['completed_primary_stages'] = sum(done[index, 1.] is not None for index in INDICES)
            retained = [record for record in done.values() if record is not None]
            metrics['completed_stages'] = len(retained)
            metrics['completed_inputs'] = status['completed_inputs']
            metrics['solver_count'] = len(retained)
            metrics['converged_solver_count'] = sum(record['solver']['converged'] for record in retained)
            metrics['all_solvers_converged'] = len(retained) == 12 and metrics['converged_solver_count'] == 12
        progress()
        if all(value is not None for value in done.values()):
            status.update(state='complete', stage='complete', resumed_without_gpu=True)
            metrics['retained_pretrained_model_run'] = True
            metrics['retained_technical_checks'] = [record['technical_checks'] for record in done.values()]
            return status
        status['stage'] = 'pinned_model_load'
        save()
        def loaded(hardware):
            hardware = dict(hardware, attention='sdpa/MATH native local derivatives and endpoint measurements')
            metrics.update(weights_loaded=True, hardware=hardware)
            save()
        engine = old.initialize_engine(reference, Path(cache_dir or '/content/h286_model_cache'), loaded)
        engine['hardware'] = dict(engine['hardware'], attention='sdpa/MATH native local derivatives and endpoint measurements')
        def model_ran():
            metrics['pretrained_model_run'] = True
            save()
        engine['on_model_run'] = model_ran
        compute_started = time.monotonic()
        deadline = compute_started+BUDGET_SECONDS
        stop = lambda: time.monotonic() >= deadline
        anchor_by_index = {case['index']: case for case in anchors['cases']}
        measured = []
        for index, factor in stage_order():
            if done[index, factor] is not None:
                continue
            if stop():
                status.update(state='paused', stage='safe_stage_checkpoint', reason='20-minute soft compute budget reached')
                break
            status.update(stage='same_lambda_convergence' if factor == 1. else 'fixed_damping_ablation',
                          current_input=f'b14b18_{index:03d}', current_factor=factor)
            save()
            print(f'Stage {status["completed_stages"]+1}/12: Fall {index}, Lambda-Faktor {factor:g}; feste Kontrollen.', flush=True)
            request = old.prepare_request(engine['tokenizer'], index, items[index], engine['model'].config.vocab_size)
            metrics['model_forward_attempted'] = True
            save()
            record, arrays = compute_stage(engine, reference, stats, request, anchor_by_index[index], factor, stop_requested=stop)
            metrics['pretrained_model_run'] = True
            if record['state'] != 'complete':
                partial = write_partial(output, record, arrays, plan_hash)
                status.update(state='paused', stage='partial_stage_checkpoint',
                              reason=record.get('stop_reason', 'incomplete stage'), partial_path=str(partial.relative_to(output)))
                break
            done[index, factor] = write_checkpoint(output, record, arrays, plan_hash, engine['hardware'])
            measured.append(record['timing']['total_seconds'])
            progress()
            remaining = 12-status['completed_stages']
            status.update(compute_elapsed_seconds=time.monotonic()-compute_started,
                          estimated_remaining_seconds=sum(measured)/len(measured)*remaining)
            save()
            print(f'Gespeichert {status["completed_stages"]}/12 Stufen; {status["completed_inputs"]}/4 Fälle vollständig; '
                  f'letzte Rechnung {measured[-1]:.2f}s; Restschätzung {status["estimated_remaining_seconds"]/60:.1f}min.', flush=True)
        else:
            status.update(state='complete', stage='complete')
        metrics['compute_elapsed_seconds'] = time.monotonic()-compute_started
        metrics['completed_stages'] = status['completed_stages']
        metrics['completed_inputs'] = status['completed_inputs']
    except BaseException as exc:  # noqa: BLE001 — preserve interruption/OOM diagnostics.
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
    for flag in ('hn-set', 'policy', 'refw', 'items', 'plan', 'refine-plan', 'anchors', 'out-dir'):
        parser.add_argument('--'+flag, required=True)
    parser.add_argument('--cache-dir')
    parser.add_argument('--resume-zip')
    parser.add_argument('--offline-plan-only', action='store_true')
    arguments = parser.parse_args()
    terminal = run(arguments.hn_set, arguments.policy, arguments.refw, arguments.items,
                   arguments.plan, arguments.refine_plan, arguments.anchors, arguments.out_dir,
                   arguments.cache_dir, arguments.resume_zip, arguments.offline_plan_only)
    raise SystemExit(0 if terminal['state'] in ('complete', 'paused') else 1)
