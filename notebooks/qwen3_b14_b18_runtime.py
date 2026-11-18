"""Bounded native Qwen3 B14→B18 local-Jacobian/transport pilot.

Original H286/REF-W and weights are read-only. This is a local, prompt-specific
Jacobian experiment, not a fitted/global JLens or a confirmatory benchmark.
Source-bound prefixes are teacher forced; separate short free answers are saved
for later blind interpretation, with no online semantic correctness evaluator.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
import time
import traceback
import types
import zipfile
from collections import defaultdict
from pathlib import Path

import qwen3_h286_runtime as base

SCHEMA = 'qwen3_b14_b18_local_transport_v1'
ITEMS_SHA256 = 'cc482e87fc49dcea8fa77b0ccb4bf63ed8df41e5f75c5c47c434decd4e3d1a5e'
INPUT_PLAN_SHA256 = '76dcc81a06142b2fb662383ec82a6dc17834e9612543163036d2a7c278643bbf'
OFFSETS = (0, 1, 4, 8)
RESIDUAL_BLOCKS = (13, 14, 15, 17, 18, 19)
CALIBRATION_INDICES = (0, 12)
TRANSPORT_INDICES = (0, 1, 12, 13)
FD_STEPS = (0.25, 0.5)
RIDGE_RELATIVE = 0.1
CG_MAX_ITER = 16
CG_RTOL = 1e-4
REQUEST_LIMIT_SECONDS = 45 * 60
SOURCE_NAMES = ('qwen3_h286_runtime.py', 'attention_probe/reference.py',
                'attention_probe/fixed_refw.py', 'attention_probe/interventions.py',
                'attention_probe/answer_validation.py', 'attention_probe/native_jacobian.py',
                'attention_probe/local_transport.py')
ARMS = ({'name': 'clean', 'block': 14, 'beta': 1., 'mode': 'scale'},) + tuple(
    {'name': f'b{block}_soft_{level}', 'block': block, 'beta': 1 / divisor, 'mode': 'scale'}
    for block in (13, 14, 15) for level, divisor in (('103', 1.03), ('108', 1.08)))
PROOF_KEYS = ('sham_exact', 'pulse_once', 'captures_complete', 'upstream_causal',
              'frozen_parameters', 'global_rng', 'hook_cleanup', 'jacobian_finite',
              'generation_prefill_exact')
GENERATION_ARMS = tuple(arm for arm in ARMS if arm['name'] == 'clean' or arm['name'].endswith('108'))
GENERATION_CAP = 64


def source_path(name):
    folder = Path(__file__).parent
    candidate = folder / name
    return candidate if candidate.exists() else folder.parent / name


def load_inputs(hn, policy, refw, items, plan_path):
    from attention_probe.fixed_refw import load_fixed_refw
    from attention_probe.reference import load_reference
    reference = load_reference(hn, policy)
    stats = load_fixed_refw(refw, reference)
    raw = Path(items).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ITEMS_SHA256:
        raise ValueError('Exact source-bound sixteen-item private input file required')
    selected = json.loads(raw)
    if (not isinstance(selected, list) or len(selected) != 16
            or len({row['id'] for row in selected}) != 16
            or [row['family'] for row in selected] != ['s3'] * 12 + ['adet'] * 4):
        raise ValueError('Expected fixed 12 S3 plus 4 A-det source-bound inputs')
    for row in selected:
        ids = row.get('prompt_ids')
        if (not isinstance(ids, list) or not 1 <= len(ids) <= 64
                or any(type(token) is not int or token < 0 for token in ids)
                or not isinstance(row.get('fixed_continuation_text'), str)
                or not row['fixed_continuation_text']):
            raise ValueError('Malformed source prompt or fixed continuation; no truncation/padding')
    raw_plan = Path(plan_path).read_bytes()
    if hashlib.sha256(raw_plan).hexdigest() != INPUT_PLAN_SHA256:
        raise ValueError('Exact prespecified private scientific plan required')
    scientific_plan = json.loads(raw_plan)
    return reference, stats, selected, scientific_plan


def plan_document(reference, stats, items, scientific_plan):
    return {'schema': SCHEMA, 'helper_sha256': base.sha_file(__file__),
        'source_sha256': {name: base.sha_file(source_path(name)) for name in SOURCE_NAMES},
        'items_sha256': ITEMS_SHA256, 'items': items, 'model': reference.model.to_dict(),
        'input_plan_sha256': INPUT_PLAN_SHA256, 'scientific_plan': scientific_plan,
        'hn_sha256': reference.hn_sha256, 'policy_sha256': reference.policy_sha256,
        'refw_sha256': base.REFW_SHA256, 'refw_source_sha256': stats['source_hn_meta_sha256'],
        'coordinates': [list(pair) for pair in reference.coordinates], 'seed': base.SEED,
        'dtype': 'float32', 'tf32': False, 'transformers': base.TRANSFORMERS,
        'accelerate': base.ACCELERATE, 'measurement_attention': 'sdpa/EFFICIENT_ATTENTION',
        'derivative_attention': 'sdpa/MATH', 'arms': list(ARMS), 'offsets': list(OFFSETS),
        'prefix': 'same source-bound GOLD continuation per arm; unavailable offsets not_run; no padding',
        'pulse_position': 'original final prompt token only, even on extended prefixes',
        'capture_position': 'final token of each matched prompt+gold-prefix',
        'residuals': {'kind': 'native decoder resid_post, zero-based', 'blocks': list(RESIDUAL_BLOCKS)},
        'jacobian': {'source': 14, 'target': 18, 'position': 'final original prompt token',
                     'arm': 'b14_soft_108', 'indices': list(range(16)), 'full_fitted_jlens': False},
        'bounded_counts': {'matched_prefix_measurement_slots_upper_bound':448,
                           'secondary_free_generations':64,'jvp_requests':16,'transport_requests':4,
                           'derivative_calls':'native suffix evaluations; not counted as full model prefills'},
        'calibration': {'indices': list(CALIBRATION_INDICES), 'fd_relative_steps': list(FD_STEPS),
                        'math_efficient_rtol': 1e-3, 'math_efficient_atol': 1e-4},
        'transport': {'indices': list(TRANSPORT_INDICES), 'lambda_relative_to_power_gain_estimate': RIDGE_RELATIVE,
                      'gain_estimator': 'four fixed-seed J^T J power iterations',
                      'max_iter': CG_MAX_ITER, 'rtol': CG_RTOL,
                      'equation': 'delta14=J^T (J J^T + lambda^2 I)^-1 delta18',
                      'controls': ['direct_delta18_at14', 'source_norm_matched_fixed_noise_at14']},
        'b18_controls': ['jvp_direction_projection_replay', 'jvp_direction_projection_removal',
                         'projection_norm_matched_fixed_noise', 'full_delta18_replay_cut_control',
                         'full_delta18_removal_cut_control'],
        'free_generation': {'arm_names': [arm['name'] for arm in GENERATION_ARMS],
                            'max_selected_tokens_including_EOS': GENERATION_CAP, 'planned_count': 64,
                            'scoring': 'blind later; no online semantic evaluator',
                            'pulse': 'original prompt prefill only; fresh native KV cache per arm'},
        'budget': {'seconds_after_model_load': REQUEST_LIMIT_SECONDS,
                   'soft_boundary': 'between complete input checkpoints; atomic input may overrun',
                   'pause_if_next_input_estimate_exceeds_remaining_budget': True,
                   'next_input_estimate':'maximum of last three complete-input durations; not total remaining ETA'},
        'claim': 'exploratory native local delta transport; no training, no observer proof, no global JLens'}


def hook_count(model):
    return sum(len(module._forward_hooks) + len(module._forward_pre_hooks) for module in model.modules())


def attention_context(torch, backend):
    return torch.nn.attention.sdpa_kernel(getattr(torch.nn.attention.SDPBackend, backend))


def digest_tensor(tensor):
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def prepare_request(tokenizer, index, item, vocab_size):
    prompt = list(item['prompt_ids'])
    continuation = tokenizer(item['fixed_continuation_text'], add_special_tokens=False)['input_ids']
    if not continuation or len(continuation) > 32:
        raise ValueError('Fixed continuation must have 1–32 tokens; no silent truncation')
    if any(type(token) is not int or not 0 <= token < vocab_size for token in prompt + continuation):
        raise ValueError('Source/continuation token outside the pinned vocabulary')
    return {'request_id': f'b14b18_{index:03d}', 'index': index, 'item': item,
        'prompt_ids': prompt, 'continuation_ids': continuation,
        'continuation_text': item['fixed_continuation_text'],
        'pulse_positions': [len(prompt) - 1],
        'available_offsets': [offset for offset in OFFSETS if offset <= len(continuation)],
        'missing_offsets': [offset for offset in OFFSETS if offset > len(continuation)],
        'thinking_enabled': False, 'tokenization': 'original source prompt IDs unchanged; fixed gold continuation only'}


@contextlib.contextmanager
def residual_patch(model, block, position, delta):
    """One full-forward additive resid_post patch; other tokens remain exact."""
    import torch
    if not isinstance(delta, torch.Tensor) or tuple(delta.shape) != (model.config.hidden_size,):
        raise ValueError('Residual delta must have the exact [hidden_size] shape')
    if not bool(torch.isfinite(delta).all()):
        raise ValueError('Nonfinite residual delta')
    state = {'fired': 0}
    def patch(_module, _args, output):
        value = output[0] if isinstance(output, tuple) else output
        if state['fired'] or value.ndim != 3 or value.shape[0] != 1 or not 0 <= position < value.shape[1]:
            raise ValueError('Residual patch must fire exactly once on one complete sequence')
        before = value[0, position].detach().clone()
        changed = value.clone()
        changed[0, position] = value[0, position] + delta.to(device=value.device, dtype=value.dtype)
        state.update(fired=1, requested_delta_norm=float(delta.detach().double().norm()),
                     actual_delta_norm=float((changed[0, position] - before).detach().double().norm()))
        return (changed, *output[1:]) if isinstance(output, tuple) else changed
    handle = model.model.layers[block].register_forward_hook(patch)
    try:
        yield state
        if state['fired'] != 1:
            raise ValueError('Residual patch did not fire')
    finally:
        handle.remove()


def vector_metrics(actual, predicted):
    import torch
    a, p = actual.detach().double().cpu(), predicted.detach().double().cpu()
    an, pn = float(a.norm()), float(p.norm())
    return {'actual_norm': an, 'predicted_norm': pn, 'error_norm': float((a-p).norm()),
            'relative_error': float((a-p).norm()) / an if an else None,
            'cosine': float(torch.dot(a, p) / (an*pn)) if an and pn else None}


def fixed_noise_like(vector, seed):
    import torch
    norm = float(vector.detach().double().norm())
    generator = torch.Generator(device='cpu').manual_seed(seed)
    direction = torch.randn(vector.shape, dtype=torch.float64, generator=generator)
    direction = direction / direction.norm() * norm if norm else torch.zeros_like(direction)
    return direction.to(device=vector.device, dtype=vector.dtype)


def power_gain(mapping, width, device, seed):
    """Fixed four-step directional spectral estimate, independent of outcomes."""
    import torch
    direction = fixed_noise_like(torch.ones(width, device=device, dtype=torch.float32), seed)
    direction /= direction.norm()
    iterations = []
    for _ in range(4):
        image = mapping.jvp(direction)
        transpose = mapping.vjp(image)
        gain, norm = float(image.double().norm()), float(transpose.double().norm())
        iterations.append({'gain': gain, 'transpose_norm': norm})
        if not math.isfinite(gain) or not math.isfinite(norm) or gain == 0 or norm == 0:
            return 0., iterations
        direction = transpose / norm
    return gain, iterations


def generation_runner():
    # Isolated globals retain the old decoder byte-for-byte. Only its explicit
    # cap guard changes; the original module still rejects >48 tokens.
    from attention_probe import answer_validation
    namespace = dict(vars(answer_validation))
    namespace['MAX_NEW_TOKENS'] = GENERATION_CAP
    original = answer_validation.run_generation_arm
    isolated = types.FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    isolated.__kwdefaults__ = dict(original.__kwdefaults__ or {})
    return isolated


def logits_metrics(logits, baseline, next_token=None):
    current, clean = logits.detach().double().cpu().log_softmax(-1), baseline.detach().double().cpu().log_softmax(-1)
    values, indices = current.topk(min(10, current.numel()))
    return {'logits_sha256': digest_tensor(logits), 'metric_dtype': 'float64',
        'top10': [{'token_id': int(i), 'logprob': float(v)} for i, v in zip(indices, values, strict=True)],
        'clean_to_arm_KL': float((clean.exp()*(clean-current)).sum()),
        'arm_to_clean_KL': float((current.exp()*(current-clean)).sum()),
        'next_fixed_token_id': next_token,
        'next_fixed_token_logprob': float(current[next_token]) if next_token is not None else None,
        'next_fixed_token_scope': 'source-bound gold next token; unavailable after final supplied token'}


def forward_probe(model, request, offset, coordinates, stats, arm, *, backend='EFFICIENT_ATTENTION',
                  weight_norms=None, patch=None, capture_blocks=RESIDUAL_BLOCKS):
    import torch

    from attention_probe.interventions import AttentionPulse, CoordinateCapture
    prompt = request['prompt_ids'] + request['continuation_ids'][:offset]
    if patch is not None and offset != 0:
        raise ValueError('Local residual-map patches are defined at original prompt offset zero only')
    position = len(prompt)-1
    ids = torch.tensor([prompt], dtype=torch.long, device=next(model.parameters()).device)
    before = hook_count(model)
    with (torch.no_grad(), attention_context(torch, backend), contextlib.ExitStack() as contexts):
        patched = contexts.enter_context(residual_patch(model, patch['block'], position, patch['delta'])) if patch else None
        capture = contexts.enter_context(CoordinateCapture(model, coordinates, [position],
            residual_layers=tuple(capture_blocks), expected_count=len(coordinates)))
        cett = contexts.enter_context(base.cett_tap(model, coordinates, [position], weight_norms))
        pulse = contexts.enter_context(AttentionPulse(model, arm['block'], request['pulse_positions'], arm['beta'],
            arm['mode'], seed=base.SEED+request['index']))
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        logits = output.logits[0, -1].detach().float().cpu()
        mu = torch.tensor(stats['refw']['c_mu'], dtype=torch.float32)
        sd = torch.tensor(stats['refw']['c_sd'], dtype=torch.float32).clamp_min(1e-6)
        arrays = {'raw_z': capture.values.numpy(), 'cett': cett['cett'].numpy(),
                  'zcett': ((cett['cett']-mu)/sd).numpy(), 'down_output_norm_raw': cett['denominator'].numpy()}
        arrays.update({f'residual_{block}': value.numpy() for block,value in capture.residuals.items()})
        evidence = {'pulse_fired': pulse.fired_count, 'capture_fired': capture.fired_count,
            'query_requested_delta_norms': pulse.requested_delta_norms.tolist(),
            'query_actual_delta_norms': pulse.actual_delta_norms.tolist(),
            'pulse_positions': request['pulse_positions'], 'capture_positions': [position],
            'input_ids_sha256': hashlib.sha256(base.canonical(prompt)).hexdigest(), 'backend': backend,
            'patch': dict(patched) if patched else None}
    if hook_count(model) != before:
        raise ValueError('Forward hook cleanup failed')
    if not bool(torch.isfinite(logits).all()) or evidence['pulse_fired'] != 1 or evidence['capture_fired'] != 1:
        raise ValueError('Incomplete/nonfinite pulse or capture')
    import numpy as np
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Nonfinite fixed-coordinate or residual capture')
    return {'logits': logits, 'arrays': arrays, 'evidence': evidence}


def tensor_residual(probe, block, device):
    import torch
    return torch.from_numpy(probe['arrays'][f'residual_{block}'][0,0]).to(device)


def save_probe(arrays, prefix, probe):
    arrays.update({prefix+'_'+key: value for key,value in probe['arrays'].items()})


def compute_request(engine, reference, stats, request, *, backend='EFFICIENT_ATTENTION'):
    import numpy as np
    import torch

    from attention_probe.local_transport import ridge_transport
    from attention_probe.native_jacobian import build_residual_map
    model = engine['model']; device = next(model.parameters()).device
    started = time.monotonic()
    norms = engine.get('weight_norms')
    parameters, hooks = base.parameter_evidence(model), hook_count(model)
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device.type == 'cuda' else None
    arrays, measurements, probes = {}, [], {}
    upstream_equal = True
    for offset in OFFSETS:
        if offset not in request['available_offsets']:
            measurements.append({'offset': offset, 'state': 'not_run', 'reason': 'source continuation shorter than offset'})
            continue
        baseline = None
        for arm in ARMS:
            probe = forward_probe(model, request, offset, reference.coordinates, stats, arm,
                                  backend=backend, weight_norms=norms)
            baseline = probe if baseline is None else baseline
            if arm['name'] != 'clean':
                upstream = [i for i,(block,_) in enumerate(reference.coordinates) if block < arm['block']]
                upstream_equal &= np.array_equal(probe['arrays']['raw_z'][...,upstream], baseline['arrays']['raw_z'][...,upstream])
            prefix = f'o{offset}_{arm["name"]}'
            save_probe(arrays, prefix, probe)
            arrays[prefix+'_logits'] = probe['logits'].numpy()
            next_id = request['continuation_ids'][offset] if offset < len(request['continuation_ids']) else None
            measurements.append({'offset': offset, 'arm': arm['name'], 'state': 'complete',
                **logits_metrics(probe['logits'], baseline['logits'], next_id), **probe['evidence']})
            if offset == 0:
                probes[arm['name']] = probe
    # A genuine uninstrumented forward checks that passive captures + beta=1
    # preserve the original native forward, rather than comparing two hook runs.
    ids = torch.tensor([request['prompt_ids']], dtype=torch.long, device=device)
    with torch.no_grad(), attention_context(torch, backend):
        native = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0,-1].detach().float().cpu()
    if not torch.equal(native, probes['clean']['logits']):
        raise ValueError('Uninstrumented/native clean and sham differ')
    primary_elapsed = time.monotonic()-started
    print(f"Stage {request['index']+1}/16: lokaler JVP 14→18", flush=True)
    # Map internals always use MATH; calculate its actual pulse in MATH too.
    math_clean = forward_probe(model, request, 0, reference.coordinates, stats, ARMS[0], backend='MATH', weight_norms=norms)
    math_pulse = forward_probe(model, request, 0, reference.coordinates, stats, ARMS[4], backend='MATH', weight_norms=norms)
    arrays['math_clean_logits'] = math_clean['logits'].numpy()
    arrays['math_pulse_logits'] = math_pulse['logits'].numpy()
    save_probe(arrays,'math_clean',math_clean)
    save_probe(arrays,'math_pulse',math_pulse)
    mapping = build_residual_map(model, ids, 14, 18, attention_mask=torch.ones_like(ids))
    if not (torch.equal(mapping.baseline_source,tensor_residual(math_clean,14,device))
            and torch.equal(mapping.baseline_target,tensor_residual(math_clean,18,device))):
        raise ValueError('Residual-map baseline differs from the captured native MATH clean measurement')
    recovered = mapping(mapping.baseline_source).detach()
    if not torch.allclose(recovered,mapping.baseline_target,rtol=1e-5,atol=1e-5):
        raise ValueError('Native suffix primal baseline parity failed')
    source_delta = tensor_residual(math_pulse,14,device)-mapping.baseline_source
    actual_delta = tensor_residual(math_pulse,18,device)-mapping.baseline_target
    predicted = mapping.jvp(source_delta)
    nonlinear = mapping(tensor_residual(math_pulse,14,device)).detach()-mapping.baseline_target
    if not torch.allclose(nonlinear,actual_delta,rtol=1e-4,atol=1e-5):
        raise ValueError('Native suffix actual-pulse primal parity failed')
    jacobian = {'source':14, 'target':18, 'backend':'MATH', 'arm':'b14_soft_108',
        'prediction':vector_metrics(actual_delta,predicted),
        'last_prompt_only':True, 'map_source_shape':list(mapping.baseline_source.shape),
        'primal_baseline_parity':vector_metrics(mapping.baseline_target,recovered),
        'primal_baseline_allclose':True,
        'nonlinear_actual_pulse_parity':vector_metrics(actual_delta,nonlinear),
        'b15_source14_unchanged':bool(np.array_equal(probes['clean']['arrays']['residual_14'],
                                                   probes['b15_soft_108']['arrays']['residual_14']))}
    arrays.update(jac_source_delta=source_delta.detach().cpu().numpy(),
                  jac_target_actual=actual_delta.detach().cpu().numpy(), jac_target_predicted=predicted.detach().cpu().numpy(),
                  jac_baseline_source=mapping.baseline_source.cpu().numpy(), jac_baseline_target=mapping.baseline_target.cpu().numpy(),
                  jac_target_nonlinear=nonlinear.cpu().numpy())
    calibration = {'state':'not_run','reason':'fixed calibration indices only'}
    if request['index'] in CALIBRATION_INDICES:
        efficient_clean = probes['clean']['logits']
        delta = efficient_clean-math_clean['logits']
        parity = bool(torch.allclose(efficient_clean,math_clean['logits'],rtol=1e-3,atol=1e-4))
        calibration = {'state':'complete','math_efficient_logits_allclose':parity,
            'math_efficient_max_abs':float(delta.abs().max()),
            'math_efficient_resid18':vector_metrics(tensor_residual(probes['clean'],18,device),tensor_residual(math_clean,18,device)),
            'finite_differences':[], 'fd_gate':{'minimum_cosine':.99,'maximum_relative_error':.05,
                'noise_floor_factor':64,'minimum_absolute_floor':1e-7}}
        if not parity:
            raise ValueError('MATH/EFFICIENT baseline parity failed before local transport')
        for step in FD_STEPS:
            fd = mapping.central_fd(source_delta,step)
            comparison = vector_metrics(fd,predicted)
            floor = max(1e-7,64*torch.finfo(torch.float32).eps*float(mapping.baseline_target.double().norm())/step)
            informative = comparison['actual_norm'] > floor
            passed = informative and comparison['relative_error'] <= .05 and comparison['cosine'] is not None and comparison['cosine'] >= .99
            calibration['finite_differences'].append({'relative_step':step, **comparison,
                'estimated_FP32_noise_floor':floor,'informative':informative,'agreement_passed':passed})
            arrays[f'jac_fd_{str(step).replace(".","_")}'] = fd.detach().cpu().numpy()
        informative = [row for row in calibration['finite_differences'] if row['informative']]
        calibration['fd_gate_state'] = ('passed' if any(row['agreement_passed'] for row in informative)
                                        else 'failed' if informative else 'degraded_below_noise_floor')
        if engine.get('diagnostics_dir') is not None:
            base.atomic_json(Path(engine['diagnostics_dir'])/(request['request_id']+'_calibration.json'),calibration)
        if calibration['fd_gate_state'] != 'passed':
            raise ValueError('Local JVP finite-difference calibration '+calibration['fd_gate_state'])
    derivative_elapsed = time.monotonic()-started-primary_elapsed
    # Directional projection is the mechanistic diagnostic. The complete delta
    # replay/removal is retained only as a trivial downstream cut control.
    direction_norm = float(predicted.detach().double().norm())
    projection = predicted * (float(torch.dot(actual_delta.double(),predicted.double()))/(direction_norm**2)) if direction_norm else torch.zeros_like(predicted)
    noise = fixed_noise_like(projection,base.SEED+request['index']+18000)
    patch_specs = [('projection_replay',ARMS[0],projection),('projection_removal',ARMS[4],-projection),
                   ('projection_noise',ARMS[0],noise),('full_replay_cut_control',ARMS[0],actual_delta),
                   ('full_removal_cut_control',ARMS[4],-actual_delta)]
    patch_rows = []
    for name,receiver,delta in patch_specs:
        probe = forward_probe(model,request,0,reference.coordinates,stats,receiver,backend='MATH',weight_norms=norms,
                              patch={'block':18,'delta':delta})
        target = tensor_residual(probe,18,device)-mapping.baseline_target
        patch_rows.append({'name':name,'receiver':receiver['name'],**probe['evidence'],
            'target_vs_actual':vector_metrics(actual_delta,target),
            'logit_references':{'math_clean':logits_metrics(probe['logits'],math_clean['logits']),
                                'math_pulse':logits_metrics(probe['logits'],math_pulse['logits'])}})
        save_probe(arrays,'patch18_'+name,probe)
        arrays['patch18_'+name+'_logits'] = probe['logits'].numpy()
    controls_elapsed = time.monotonic()-started-primary_elapsed-derivative_elapsed
    transport = {'state':'not_run','reason':'fixed transport indices only'}
    if request['index'] in TRANSPORT_INDICES:
        gain, gain_iterations = power_gain(mapping,model.config.hidden_size,device,base.SEED+request['index']+40000)
        if gain <= 0 or not math.isfinite(gain):
            transport = {'state':'invalid','reason':'zero/nonfinite fixed directional gain; no resampling'}
        else:
            regularizer = RIDGE_RELATIVE*gain
            solved = ridge_transport(mapping.jvp,mapping.vjp,actual_delta,regularizer,max_iter=CG_MAX_ITER,rtol=CG_RTOL)
            transported = solved.delta_source
            transport = {'state':'complete','lambda':regularizer,'lambda_relative':RIDGE_RELATIVE,
                         'power_gain_estimate':gain,'power_iterations':gain_iterations,
                         'power_seed':base.SEED+request['index']+40000,'solver':solved.diagnostics,'arms':[]}
            controls = [('ridge_transport',transported),('direct_delta18_at14',actual_delta),
                        ('source_norm_matched_noise',fixed_noise_like(transported,base.SEED+request['index']+14000))]
            for name,delta in controls:
                probe = forward_probe(model,request,0,reference.coordinates,stats,ARMS[0],backend='MATH',weight_norms=norms,
                                      patch={'block':14,'delta':delta})
                achieved = tensor_residual(probe,18,device)-mapping.baseline_target
                transport['arms'].append({'name':name,**probe['evidence'],
                    'amplitude_control':('same raw source L2 as ridge' if name=='source_norm_matched_noise'
                                         else 'direct target delta copied without source norm matching' if name=='direct_delta18_at14'
                                         else 'regularized inverse solution'),
                    'target_prediction':vector_metrics(actual_delta,achieved),
                    'logit_references':{'math_clean':logits_metrics(probe['logits'],math_clean['logits']),
                                        'math_pulse':logits_metrics(probe['logits'],math_pulse['logits'])}})
                save_probe(arrays,'transport_'+name,probe)
                arrays['transport_'+name+'_logits'] = probe['logits'].numpy()
                arrays['transport_delta_'+name] = delta.detach().cpu().numpy()
    transport_elapsed = time.monotonic()-started-primary_elapsed-derivative_elapsed-controls_elapsed
    generation_started = time.monotonic()
    generations = []
    generation_prefill_exact = True
    generator = generation_runner()
    generation_request = {'request_id':f'r{request["index"]:03d}',
                          'prompt_ids':request['prompt_ids'], 'answer':request['continuation_text']}
    for arm in GENERATION_ARMS:
        with attention_context(torch,backend):
            record, captured = generator(model,engine['tokenizer'],generation_request,reference.coordinates,stats,arm,
                efficient_gpu=device.type=='cuda' and backend=='EFFICIENT_ATTENTION',weight_norms=norms,
                max_new_tokens=GENERATION_CAP,top_k=10)
        exact = record['first_step_logits_sha256'] == digest_tensor(probes[arm['name']]['logits'])
        generation_prefill_exact &= exact
        generations.append({'arm':arm['name'],'prefill_equals_primary_measurement':exact,'record':record,
                            'evaluation':'unscored free text for later blind analysis'})
        arrays.update({f'generation_{arm["name"]}_{key}':value for key,value in captured.items()})
    checks = {'sham_exact':True,'pulse_once':all(row.get('pulse_fired')==1 for row in measurements if row['state']=='complete'),
        'captures_complete':all(row.get('capture_fired')==1 for row in measurements if row['state']=='complete'),
        'upstream_causal':bool(upstream_equal),'frozen_parameters':base.parameter_evidence(model)==parameters,
        'global_rng':bool(torch.equal(cpu_rng,torch.random.get_rng_state()) and
                           (cuda_rng is None or torch.equal(cuda_rng,torch.cuda.get_rng_state()))),
        'hook_cleanup':hook_count(model)==hooks,'jacobian_finite':bool(torch.isfinite(predicted).all()),
        'generation_prefill_exact':bool(generation_prefill_exact)}
    if not all(checks.values()):
        raise ValueError('Bound native/causal/RNG/parameter technical checks failed: '+str(checks))
    return {'request':request,'measurements':measurements,'jacobian':jacobian,'calibration':calibration,
            'b18_controls':patch_rows,'transport':transport,'generations':generations,'checks':checks,
            'timing':{'primary_measurements_seconds':primary_elapsed,
                      'jacobian_seconds':derivative_elapsed,'b18_controls_seconds':controls_elapsed,
                      'transport_seconds':transport_elapsed,'generation_seconds':time.monotonic()-generation_started,
                      'total_seconds':time.monotonic()-started,
                      'scope':'synchronized generation timers plus CPU capture/solver/logging; not energy telemetry'},
            'array_contract':{'coordinates':len(reference.coordinates),'hidden_size':model.config.hidden_size,
                              'vocab_size':model.config.vocab_size}},arrays


def array_shapes(record):
    """Derive the entire checkpoint contract without trusting uploaded shapes."""
    n, hidden = record['array_contract']['coordinates'], record['array_contract']['hidden_size']
    expected = {}
    def probe(prefix):
        expected.update({prefix+'_'+key:(1,1,n) for key in ('raw_z','cett','zcett','down_output_norm_raw')})
        expected.update({prefix+f'_residual_{block}':(1,1,hidden) for block in RESIDUAL_BLOCKS})
    for offset in record['request']['available_offsets']:
        for arm in ARMS:
            probe(f'o{offset}_{arm["name"]}')
            expected[f'o{offset}_{arm["name"]}_logits'] = (record['array_contract']['vocab_size'],)
    expected.update({name:(record['array_contract']['vocab_size'],) for name in ('math_clean_logits','math_pulse_logits')})
    probe('math_clean'); probe('math_pulse')
    for key in ('jac_source_delta','jac_target_actual','jac_target_predicted','jac_baseline_source','jac_baseline_target','jac_target_nonlinear'):
        expected[key] = (hidden,)
    if record['request']['index'] in CALIBRATION_INDICES:
        expected.update({f'jac_fd_{str(step).replace(".","_")}':(hidden,) for step in FD_STEPS})
    for name in ('projection_replay','projection_removal','projection_noise','full_replay_cut_control','full_removal_cut_control'):
        probe('patch18_'+name)
        expected['patch18_'+name+'_logits'] = (record['array_contract']['vocab_size'],)
    if record['transport']['state'] == 'complete':
        for name in ('ridge_transport','direct_delta18_at14','source_norm_matched_noise'):
            probe('transport_'+name)
            expected['transport_'+name+'_logits'] = (record['array_contract']['vocab_size'],)
            expected['transport_delta_'+name] = (hidden,)
    positions = min(4,len(record['request']['prompt_ids']))
    for arm in GENERATION_ARMS:
        prefix = f'generation_{arm["name"]}_'
        expected.update({prefix+key:(1,positions,n) for key in ('raw_z','cett','zcett','down_output_norm_raw')})
        expected.update({prefix+f'residual_layer_{block}':(1,positions,hidden) for block in (3,7,14,27)})
    return expected


def validate_record(record, plan_hash, item=None):
    request = record.get('request', {})
    if (record.get('state') != 'complete' or record.get('schema') != SCHEMA
            or record.get('plan_sha256') != plan_hash or type(request.get('index')) is not int
            or request.get('request_id') != f'b14b18_{request["index"]:03d}'
            or not 0 <= request['index'] < 16 or (item is not None and request.get('item') != item)):
        raise ValueError('Complete checkpoint has a different bound request/plan/schema')
    if (request.get('prompt_ids') != request['item']['prompt_ids']
            or request.get('pulse_positions') != [len(request['prompt_ids'])-1]
            or request.get('available_offsets') != [v for v in OFFSETS if v <= len(request['continuation_ids'])]
            or request.get('missing_offsets') != [v for v in OFFSETS if v > len(request['continuation_ids'])]):
        raise ValueError('Checkpoint changes source tokens, pulse position or continuation offsets')
    proof = record.get('technical_checks', {})
    if (proof.get('plan_sha256') != plan_hash or proof.get('state') != 'passed'
            or set(proof.get('checks', {})) != set(PROOF_KEYS)
            or not all(value is True for value in proof['checks'].values())
            or record.get('checks') != proof['checks']):
        raise ValueError('Bound complete technical proof is required')
    wanted = [(offset,arm['name']) for offset in request['available_offsets'] for arm in ARMS]
    rows = [row for row in record.get('measurements',[]) if row.get('state') == 'complete']
    missing = [row for row in record.get('measurements',[]) if row.get('state') == 'not_run']
    if ([(row.get('offset'),row.get('arm')) for row in rows] != wanted
            or [row.get('offset') for row in missing] != request['missing_offsets']
            or len(record['measurements']) != len(rows)+len(missing)):
        raise ValueError('Missing, duplicated or altered fixed measurement slots')
    for row in rows:
        offset = row['offset']
        if (row.get('pulse_positions') != request['pulse_positions']
                or row.get('capture_positions') != [len(request['prompt_ids'])+offset-1]
                or row.get('pulse_fired') != 1 or row.get('capture_fired') != 1
                or not math.isfinite(row.get('clean_to_arm_KL',math.nan))
                or not math.isfinite(row.get('arm_to_clean_KL',math.nan))
                or len(row.get('top10',[])) != min(10,record['array_contract']['vocab_size'])):
            raise ValueError('Incomplete finite full-KL/readout/pulse measurement')
    jacobian = record.get('jacobian',{})
    if (jacobian.get('source') != 14 or jacobian.get('target') != 18 or jacobian.get('arm') != 'b14_soft_108'
            or jacobian.get('backend') != 'MATH' or jacobian.get('last_prompt_only') is not True
            or jacobian.get('primal_baseline_allclose') is not True or jacobian.get('b15_source14_unchanged') is not True):
        raise ValueError('Missing fixed native local Jacobian identity/parity')
    calibration = record.get('calibration',{})
    if request['index'] in CALIBRATION_INDICES:
        if (calibration.get('state') != 'complete' or calibration.get('fd_gate_state') != 'passed'
                or calibration.get('math_efficient_logits_allclose') is not True
                or [row.get('relative_step') for row in calibration.get('finite_differences',[])] != list(FD_STEPS)
                or not any(row.get('agreement_passed') is True for row in calibration['finite_differences'])):
            raise ValueError('Calibration input requires passed FD and MATH/EFFICIENT gates')
    elif calibration.get('state') != 'not_run':
        raise ValueError('Calibration differs from its fixed subset')
    names = ['projection_replay','projection_removal','projection_noise','full_replay_cut_control','full_removal_cut_control']
    controls = record.get('b18_controls',[])
    if ([row.get('name') for row in controls] != names
            or [row.get('receiver') for row in controls] != ['clean','b14_soft_108','clean','clean','b14_soft_108']):
        raise ValueError('Exactly five fixed B18 controls and receivers required')
    for row in controls+record.get('transport',{}).get('arms',[]):
        if (row.get('pulse_positions') != request['pulse_positions'] or row.get('capture_positions') != request['pulse_positions']
                or row.get('pulse_fired') != 1 or row.get('capture_fired') != 1
                or row.get('patch',{}).get('fired') != 1 or set(row.get('logit_references',{})) != {'math_clean','math_pulse'}):
            raise ValueError('Fixed offset-zero residual patch/evidence/references required')
    if record.get('transport',{}).get('state') not in ('complete','invalid','not_run'):
        raise ValueError('Missing explicit local transport status')
    if (request['index'] in TRANSPORT_INDICES) == (record['transport']['state']=='not_run'):
        raise ValueError('Transport execution differs from the fixed subset')
    if record['transport']['state']=='complete':
        transport=record['transport']
        if ([row.get('name') for row in transport.get('arms',[])] !=
                ['ridge_transport','direct_delta18_at14','source_norm_matched_noise']
                or transport.get('lambda_relative') != RIDGE_RELATIVE or len(transport.get('power_iterations',[])) != 4
                or transport.get('solver',{}).get('max_iter') != CG_MAX_ITER
                or transport.get('solver',{}).get('rtol') != CG_RTOL):
            raise ValueError('Transport solver/controls differ from the prespecified bounded rule')
    gens = record.get('generations',[])
    if [row.get('arm') for row in gens] != [arm['name'] for arm in GENERATION_ARMS]:
        raise ValueError('Exactly four fixed secondary free generations required')
    for row,arm in zip(gens,GENERATION_ARMS,strict=True):
        gen = row.get('record',{})
        selected, emitted, steps = gen.get('selected_token_ids',[]),gen.get('generated_token_ids',[]),gen.get('steps',[])
        eos = gen.get('stop_reason') == 'eos'
        if (gen.get('arm') != arm or gen.get('request_id') != f'r{request["index"]:03d}'
                or gen.get('seed') != base.SEED+request['index'] or gen.get('max_new_tokens') != GENERATION_CAP
                or not 1 <= len(selected) <= GENERATION_CAP or len(steps) != len(selected)
                or selected != [step.get('selected_token_id') for step in steps]
                or emitted != (selected[:-1] if eos else selected)
                or gen.get('decode_forward_count') != len(selected)-1 or gen.get('prefill_forward_count') != 1
                or gen.get('stop_reason') not in ('eos','max_new_tokens')
                or gen.get('truncated') is not (not eos) or (not eos and len(selected)!=GENERATION_CAP)
                or any(step.get('is_eos') for step in steps[:-1])
                or bool(steps[-1].get('is_eos')) is not eos
                or gen.get('prompt_length') != len(request['prompt_ids'])
                or gen.get('final_cache_length') != len(request['prompt_ids'])+len(selected)-1
                or gen.get('pulse_positions') != request['pulse_positions']
                or gen.get('prefill_capture_positions') != list(range(max(0,len(request['prompt_ids'])-4),len(request['prompt_ids'])))
                or row.get('prefill_equals_primary_measurement') is not True):
            raise ValueError('Generation cap/EOS/cache/prefill/prompt seed contract violated')


def validate_arrays(record, arrays):
    import numpy as np
    expected = array_shapes(record)
    if set(arrays) != set(expected):
        raise ValueError('Checkpoint has missing or unexpected activation arrays')
    for key,shape in expected.items():
        value = np.asarray(arrays[key])
        if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Checkpoint array must have exact finite FP32 shape: '+key)
    for row in record['measurements']:
        if row.get('state')=='complete':
            digest=hashlib.sha256(np.asarray(arrays[f'o{row["offset"]}_'+row['arm']+'_logits']).tobytes()).hexdigest()
            if digest != row.get('logits_sha256'):
                raise ValueError('Full-vocabulary measurement hash disagrees with its record')
    for row in record['generations']:
        digest=hashlib.sha256(np.asarray(arrays['o0_'+row['arm']+'_logits']).tobytes()).hexdigest()
        if row['record'].get('first_step_logits_sha256') != digest:
            raise ValueError('Generation prefill hash differs from the saved primary full logits')
    for prefix,rows in (('patch18_',record['b18_controls']),('transport_',record['transport'].get('arms',[]))):
        for row in rows:
            digest=hashlib.sha256(np.asarray(arrays[prefix+row['name']+'_logits']).tobytes()).hexdigest()
            if any(ref.get('logits_sha256') != digest for ref in row['logit_references'].values()):
                raise ValueError('Control/transport full logits disagree with the recorded comparator')


def should_pause(elapsed, measured, remaining):
    return bool(remaining and (elapsed>=REQUEST_LIMIT_SECONDS
                              or elapsed+max(measured[-3:])>REQUEST_LIMIT_SECONDS))


def write_checkpoint(output, record, arrays, plan_hash, hardware):
    import numpy as np
    record = {**record,'schema':SCHEMA,'state':'complete','plan_sha256':plan_hash,'hardware':hardware,
        'technical_checks':{'state':'passed','plan_sha256':plan_hash,'checks':record['checks']},
        'completed_utc':dt.datetime.now(dt.UTC).isoformat()}
    validate_record(record,plan_hash)
    validate_arrays(record,arrays)
    destination = output/'requests'/record['request']['request_id']
    if destination.exists():
        raise ValueError('Refusing to overwrite a completed or conflicting input checkpoint')
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.'+destination.name+'_',dir=destination.parent))
    try:
        np.savez_compressed(temporary/'arrays.npz',**arrays)
        with (temporary/'arrays.npz').open('rb') as stream:
            os.fsync(stream.fileno())
        record['npz_sha256'] = base.sha_file(temporary/'arrays.npz')
        base.atomic_json(temporary/'result.json',record)
        os.rename(temporary,destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return record


def verified(output, index, plan_hash, item):
    import numpy as np
    folder = Path(output)/'requests'/f'b14b18_{index:03d}'
    if not folder.exists():
        return None
    if not (folder/'result.json').is_file() or not (folder/'arrays.npz').is_file():
        raise ValueError('Incomplete/conflicting input checkpoint; no automatic deletion')
    record = json.loads((folder/'result.json').read_text())
    if base.sha_file(folder/'arrays.npz') != record.get('npz_sha256'):
        raise ValueError('Input checkpoint arrays SHA mismatch')
    validate_record(record,plan_hash,item)
    with np.load(folder/'arrays.npz',allow_pickle=False) as arrays:
        validate_arrays(record,arrays)
    return record


def restore_resume(resume_zip, output, plan_hash, items):
    if not resume_zip:
        return
    with zipfile.ZipFile(resume_zip) as bundle:
        plan = json.loads(bundle.read('run_plan.json'))
        if hashlib.sha256(base.canonical(plan)).hexdigest() != plan_hash:
            raise ValueError('Resume ZIP belongs to another exact source/plan/reference binding')
        if len(bundle.namelist()) != len(set(bundle.namelist())) or bundle.testzip() is not None:
            raise ValueError('Corrupt/duplicate-member resume ZIP')
        for index,item in enumerate(items):
            folder = f'requests/b14b18_{index:03d}'
            names = [folder+'/'+name for name in ('result.json','arrays.npz')]
            present = [name in bundle.namelist() for name in names]
            if not any(present):
                continue
            if not all(present) or any(bundle.getinfo(name).file_size > 64*(1<<20) for name in names):
                raise ValueError('Incomplete or oversized resume checkpoint')
            existing = verified(output,index,plan_hash,item)
            target = output/folder
            if existing is not None:
                if any((target/Path(name).name).read_bytes()!=bundle.read(name) for name in names):
                    raise ValueError('Resume conflicts with existing complete input; no overwrite')
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            temp = Path(tempfile.mkdtemp(prefix='.'+target.name,dir=target.parent))
            try:
                for name in names:
                    (temp/Path(name).name).write_bytes(bundle.read(name))
                # Validate before publish using the same reader under a private root.
                staging = temp/'requests'/target.name
                staging.mkdir(parents=True)
                for name in ('result.json','arrays.npz'):
                    os.rename(temp/name,staging/name)
                verified(temp,index,plan_hash,item)
                os.rename(staging,target)
            finally:
                shutil.rmtree(temp)


def archive(output):
    target = output/'b14_b18_results.zip'
    temporary = output/'.b14_b18_results.zip.tmp'
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path not in (target,temporary) and not any(part.startswith('.') for part in path.relative_to(output).parts):
                bundle.write(path,path.relative_to(output))
    os.replace(temporary,target)
    return target


def initialize_engine(reference, cache, on_loaded):
    from importlib.metadata import version

    import torch
    import transformers
    if transformers.__version__ != base.TRANSFORMERS or version('accelerate') != base.ACCELERATE:
        raise ValueError('Exact original transformers5.22.2 and accelerate1.16.0 required')
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(base.SEED)
    transformers.utils.logging.disable_progress_bar()
    hardware = base.hardware_preflight(torch,cache)
    os.environ['HF_DEACTIVATE_ASYNC_LOAD'] = '1'
    print('Stage model: Original Qwen3-1.7B FP32 laden und Dateien prüfen.',flush=True)
    tokenizer,model = base.load_model(reference,cache)
    hardware.update(torch=torch.__version__,cuda=torch.version.cuda,python=platform.python_version(),
        transformers=transformers.__version__,attention='sdpa/EFFICIENT_ATTENTION + explicit derivative MATH',
        dtype='float32',tf32=False,compute_capability=list(torch.cuda.get_device_capability()))
    on_loaded(hardware)
    groups = defaultdict(list)
    for layer,neuron in reference.coordinates:
        groups[layer].append(neuron)
    with torch.no_grad():
        norms = {layer:model.model.layers[layer].mlp.down_proj.weight.float().norm(dim=0)[columns]
                 for layer,columns in groups.items()}
    return {'model':model,'tokenizer':tokenizer,'hardware':hardware,'weight_norms':norms}


def run(hn,policy,refw,items_path,plan_path,out_dir,cache_dir=None,resume_zip=None,offline_plan_only=False):
    output = Path(out_dir); output.mkdir(parents=True,exist_ok=True)
    nonce = os.environ.get('H286_WRAPPER_ATTEMPT')
    status = {'state':'running','stage':'reference','completed_inputs':0,'scope':SCHEMA,
              'confirmatory_eligible':False,'wrapper_attempt_id':nonce}
    metrics = {'weights_loaded':False,'pretrained_model_run':False,'no_training':True,
               'wrapper_attempt_id':nonce,'generation_count':0}
    started = time.monotonic()
    def save():
        base.atomic_json(output/'status.json',status); base.atomic_json(output/'metrics.json',metrics)
    save()
    try:
        reference,stats,items,scientific_plan = load_inputs(hn,policy,refw,items_path,plan_path)
        plan = plan_document(reference,stats,items,scientific_plan)
        plan_hash = hashlib.sha256(base.canonical(plan)).hexdigest()
        existing = output/'run_plan.json'
        if existing.exists() and existing.read_bytes() != base.canonical(plan)+b'\n':
            raise ValueError('Output directory belongs to another exact plan; no overwrite')
        if not existing.exists():
            base.atomic_json(existing,plan)
        if offline_plan_only:
            status.update(state='complete',stage='offline_plan_only',plan_sha256=plan_hash,no_model_executed=True)
            return status
        restore_resume(resume_zip,output,plan_hash,items)
        done = [verified(output,index,plan_hash,item) for index,item in enumerate(items)]
        status['completed_inputs'] = sum(row is not None for row in done)
        metrics['generation_count'] = 4*status['completed_inputs']
        if all(row is not None for row in done):
            status.update(state='complete',stage='complete',resumed_without_gpu=True)
            metrics['retained_technical_checks'] = [row['technical_checks'] for row in done]
            return status
        status['stage'] = 'pinned_model_load'; save()
        def loaded(hardware):
            metrics.update(weights_loaded=True,hardware=hardware); save()
        engine = initialize_engine(reference,Path(cache_dir or '/content/h286_model_cache'),loaded)
        engine['diagnostics_dir'] = output/'technical'
        compute_start = time.monotonic(); measured = []
        import torch
        with torch.no_grad(),attention_context(torch,'EFFICIENT_ATTENTION'):
            ids = torch.tensor([items[next(i for i,row in enumerate(done) if row is None)]['prompt_ids']],
                               dtype=torch.long,device=next(engine['model'].parameters()).device)
            warm = engine['model'](input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
            if not bool(torch.isfinite(warm.logits).all()):
                raise ValueError('Nonfinite original-model warmup logits')
        metrics.update(pretrained_model_run=True,warmup_forward_count=1); save()
        metrics['soft_budget_seconds_after_load'] = REQUEST_LIMIT_SECONDS
        for index,item in enumerate(items):
            if done[index] is not None:
                continue
            status.update(stage='request',current_input=f'b14b18_{index:03d}'); save()
            print(f'Request {index+1}/16: sourcegebundene Präfixe, lokaler Jacobian, vier freie Antworten.',flush=True)
            request = prepare_request(engine['tokenizer'],index,item,engine['model'].config.vocab_size)
            t0 = time.monotonic()
            result,arrays = compute_request(engine,reference,stats,request)
            metrics['pretrained_model_run'] = True
            done[index] = write_checkpoint(output,result,arrays,plan_hash,engine['hardware'])
            measured.append(time.monotonic()-t0)
            status['completed_inputs'] = sum(row is not None for row in done)
            metrics['generation_count'] = 4*status['completed_inputs']
            remaining = len(items)-status['completed_inputs']
            eta = sum(measured)/len(measured)*remaining
            elapsed = time.monotonic()-compute_start
            status.update(estimated_remaining_seconds=eta,compute_elapsed_seconds=elapsed)
            save()
            print(f'Gespeichert {status["completed_inputs"]}/16; Restschätzung {eta/60:.1f} min.',flush=True)
            if should_pause(elapsed,measured,remaining):
                status.update(state='paused',stage='safe_input_checkpoint',
                    reason='Soft 45-minute compute budget/ETA reached; complete inputs preserved in resume ZIP')
                break
        else:
            status.update(state='complete',stage='complete')
        metrics['compute_elapsed_seconds'] = time.monotonic()-compute_start
    except BaseException as exc:  # noqa: BLE001 — preserve diagnostics on interruption/OOM.
        status.update(state='failed',error_type=type(exc).__name__,error=str(exc))
        (output/'traceback.txt').write_text(traceback.format_exc())
        print('HARD FAILURE: '+str(exc),flush=True)
    finally:
        metrics['elapsed_seconds'] = time.monotonic()-started
        save(); archive(output)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hn-set',required=True)
    parser.add_argument('--policy',required=True)
    parser.add_argument('--refw',required=True)
    parser.add_argument('--items',required=True)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--out-dir',required=True)
    parser.add_argument('--cache-dir')
    parser.add_argument('--resume-zip')
    parser.add_argument('--offline-plan-only',action='store_true')
    args = parser.parse_args()
    state = run(args.hn_set,args.policy,args.refw,args.items,args.plan,args.out_dir,
                args.cache_dir,args.resume_zip,args.offline_plan_only)
    print(json.dumps(state),flush=True)
    raise SystemExit(0 if state['state'] in ('complete','paused') else 1)
