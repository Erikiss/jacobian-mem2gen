"""Prespecified local H286 restoration/transfer probe; no training or adaptive arms.

Three fresh same-prompt baselines provide donors. Every input then receives the
same 32 local group tests before any prespecified generation is selected. Only
three full-vocabulary baseline vectors are saved; all other KLs use those full
vectors online. Completion/format remain separate from semantic correctness.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

import qwen3_h286_completion_128_runtime as completion

previous = completion.previous
base = previous.base
SCHEMA = 'qwen3_h286_group_restoration_v1'
CONTINUITY_SHA256 = '111d7213f76374a459c38ca17f4701eb9228693708ee1657a7e25f987c65ddc1'
GROUP_ITEMS_SHA256 = '0bf76a995436092728cacc46e1be78c92b8e5416b66402324687b2a796b8ec88'
BLOCKS = (7, 14)
COUNTS = {7: 9, 14: 25}
ORIGINAL_IDS = ('r012', 'r013', 'r014', 'r015')
REQUEST_IDS = ORIGINAL_IDS + tuple(f'r{i:03d}' for i in range(100, 106))
BASELINE_ARMS = (base.ARMS[0], base.ARMS[2], base.ARMS[4])
BASELINE_NAMES = tuple(arm['name'] for arm in BASELINE_ARMS)
PULSE_NAME = {7: 'other_block_108', 14: 'soft_108'}
ARRAY_CONTRACT = {'vocab_size': 151936, 'hidden_size': 2048, 'mlp_width': 6144, 'n_coordinates': 286}
PROOF_CHECKS = ('native_clean_logits', 'identity_logits', 'prefill_counts', 'hook_cleanup',
                'generation_prefill_logits', 'frozen_parameters', 'global_rng', 'fresh_same_prompt_donors')
MAIN_GROUP_NAMES = tuple(f'b{block}_{direction}' for block in BLOCKS for direction in ('restore', 'transfer'))


def source_path(name):
    root = Path(base.__file__).parent
    path = root/name
    if not path.exists():
        path = root.parent/name
    return path


def load_inputs(hn, policy, refw, items_path, continuity_path):
    reference, stats, original_items = base.load_inputs(hn, policy, refw, source_path('attention_probe/pilot_items.json'))
    raw = Path(items_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != GROUP_ITEMS_SHA256:
        raise ValueError('Exact prespecified ten group-probe inputs required')
    items = json.loads(raw)
    if len(items) != 10 or tuple(item['request_id'] for item in items) != REQUEST_IDS:
        raise ValueError('Group inputs must keep original request IDs and fixed variant IDs')
    for item, old in zip(items[:4], original_items[12:16], strict=True):
        if any(item[key] != old[key] for key in ('id', 'pair_id', 'prompt', 'target', 'category')):
            raise ValueError('Original group-probe prompt/target changed')
    raw = Path(continuity_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONTINUITY_SHA256:
        raise ValueError('Exact private 128-token continuity reference SHA mismatch')
    continuity = json.loads(raw)
    if (continuity.get('schema') != 'h286_group_continuity_v1'
            or set(continuity.get('requests', {})) != set(ORIGINAL_IDS)
            or continuity.get('model') != reference.model.to_dict()):
        raise ValueError('Continuity model/request identity mismatch')
    for name, expected in continuity['source_module_sha256'].items():
        if base.sha_file(source_path(name)) != expected:
            raise ValueError('Original scientific source changed: '+name)
    return reference, stats, items, continuity


def make_controls(coordinates, width):
    """One fixed mask/direction plan, independent of prompts, deltas and outcomes."""
    result = {}
    for block in BLOCKS:
        group = [channel for layer, channel in coordinates if layer == block]
        if len(group) != COUNTS[block] or len(set(group)) != len(group):
            raise ValueError('Expected fixed H286 group cardinality at block '+str(block))
        outside = sorted(set(range(width))-set(group))
        if len(outside) < len(group):
            raise ValueError('Not enough outside-H channels for fixed size-matched masks')
        masks, within_units, mask_seeds, direction_seeds = [], [], [], []
        for index in range(3):
            mask_seed = base.SEED + 10000*block + 100*index
            mask = sorted(random.Random(mask_seed).sample(outside, len(group)))
            if mask in masks:
                raise ValueError('Duplicate fixed outside-H mask; no resampling')
            masks.append(mask); mask_seeds.append(mask_seed)
            direction_seed = mask_seed+53; direction_seeds.append(direction_seed)
            rng = random.Random(direction_seed)
            direction = [rng.gauss(0., 1.) for _ in group]
            norm = math.sqrt(sum(value*value for value in direction))
            within_units.append([value/norm for value in direction] if norm > 0 else None)
        result[block] = {'group': group, 'outside_masks': masks, 'within_units': within_units,
                         'outside_mask_seeds': mask_seeds, 'within_direction_seeds': direction_seeds,
                         'outside_overlaps': [{'a': a, 'b': b, 'count': len(set(masks[a]) & set(masks[b]))}
                                              for a in range(3) for b in range(a+1,3)]}
    return result


def arm_specs():
    specs = [{'name': arm['name'], 'kind': 'baseline', 'receiver': arm['name'], 'block': None}
             for arm in BASELINE_ARMS]
    for block in BLOCKS:
        for direction in ('restore', 'transfer'):
            receiver = PULSE_NAME[block] if direction == 'restore' else 'sham'
            specs.append({'name': f'b{block}_{direction}', 'kind': 'target', 'block': block,
                          'direction': direction, 'receiver': receiver})
        for receiver, suffix in (('sham', 'sham'), (PULSE_NAME[block], 'pulse')):
            specs.append({'name': f'b{block}_identity_{suffix}', 'kind': 'identity', 'block': block, 'receiver': receiver})
        for kind in ('outside', 'within'):
            for direction in ('restore', 'transfer'):
                for index in range(3):
                    specs.append({'name': f'b{block}_{kind}_{direction}_{index}', 'kind': kind,
                        'block': block, 'direction': direction, 'receiver': PULSE_NAME[block] if direction == 'restore' else 'sham',
                        'control_index': index})
    if len(specs) != 35:
        raise AssertionError('Expected exactly 35 prespecified prefill slots')
    return specs


def generated_names(request_id):
    return set(BASELINE_NAMES + MAIN_GROUP_NAMES) if request_id in ORIGINAL_IDS else set(BASELINE_NAMES)


def plan_document(reference, stats, items, continuity, controls):
    names = ('attention_probe/group_restoration.py', 'attention_probe/group_probe_items.json')
    return {'schema': SCHEMA, 'purpose': 'same-prompt local H286 restoration/transfer; exploratory follow-up',
        'helper_sha256': base.sha_file(__file__),
        'new_source_sha256': {name: base.sha_file(source_path(name)) for name in names},
        'unchanged_source_sha256': continuity['source_module_sha256'],
        'continuity_sha256': CONTINUITY_SHA256, 'prior_archive_sha256': continuity['source_archive_sha256'],
        'prior_plan_sha256': continuity['source_plan_sha256'], 'model': reference.model.to_dict(),
        'hn_sha256': reference.hn_sha256, 'policy_sha256': reference.policy_sha256,
        'refw_sha256': base.REFW_SHA256, 'refw_source_sha256': stats['source_hn_meta_sha256'],
        'coordinates': [list(pair) for pair in reference.coordinates], 'items': items,
        'group_items_sha256': GROUP_ITEMS_SHA256, 'seed': base.SEED,
        'controls': {str(key):value for key,value in controls.items()}, 'array_contract': ARRAY_CONTRACT, 'prefill_arms': arm_specs(), 'baseline_arms': list(BASELINE_ARMS),
        'generation': {'max_new_tokens': 128, 'original_names': sorted(generated_names('r012')),
                       'variant_names': sorted(generated_names('r100')), 'planned_count': 46,
                       'hooks': 'prefill only; fresh cache and original request seed per arm'},
        'prefill_slots': 350, 'native_uninstrumented_prefills': 10, 'generation_replay_prefills': 46,
        'planned_total_prefill_forwards': 406, 'dtype': 'float32', 'tf32': False, 'attention': 'sdpa/EFFICIENT_ATTENTION',
        'transformers': base.TRANSFORMERS, 'donors': 'fresh current-session clean/pulse states of identical prompt only',
        'normalization': 'match raw delta-z norm; report projected Wdown norm/cosine and actual addition rounding',
        'degeneracy': 'zero target gives explicit zero no-op; nonzero target with degenerate donor/random direction invalid; no resampling',
        'outside_direction': 'actual same-prompt pulseZ_C-cleanZ_C, raw-norm scaled; not Gaussian',
        'numeric_guards': {'zero_target':'exactly zero','normalizer_min_norm':1e-12,
                           'raw_norm_match_rtol':1e-3,'raw_norm_match_atol':0.,'cosine_denominator_min':1e-24},
        'full_vocabulary_KL': 'all vocabulary entries against each of three same-prompt baselines, both directions',
        'semantic_metric': 'correct-vs-foil FIRST TOKEN only, spelling/format-dependent; no semantic correctness verdict',
        'queue_budget': {'wallclock_limit_seconds': None, 'completion_guaranteed': False},
        'claim': 'No training; post-result follow-up, not confirmatory; output format and semantics distinct'}


def prepare_request(tokenizer, index, item):
    original = base.prepare_request(tokenizer, int(item['request_id'][1:]), item)
    keys = ('request_id','item_id','pair_id','category','question','answer','formatted_prompt',
            'formatted_prompt_sha256','prompt_ids','answer_ids','thinking_enabled')
    request = {key: original[key] for key in keys}
    position = len(request['prompt_ids'])-1
    request.update(group_item=dict(item), validation_schema=SCHEMA, pulse_positions=[position],
                   prefill_capture_positions=[position], prefill_capture_scope='last prompt token only; no teacher-forced gold',
                   generation_capture_positions=list(range(max(0,position-3),position+1)))
    if request['request_id'] != item['request_id']:
        raise ValueError('Original/random-control request seed identity changed')
    return request


def tensor_sha(value):
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def receiver_arm(name):
    return next(arm for arm in BASELINE_ARMS if arm['name'] == name)


def make_patch(spec, donors, controls):
    import torch
    block = spec['block']
    setting = controls[block]
    group = setting['group']
    clean = donors['sham']['z'][block][:, :, group]
    pulse = donors[PULSE_NAME[block]]['z'][block][:, :, group]
    delta_transfer = pulse-clean
    raw_target = float(delta_transfer.double().norm())
    sign = -1 if spec.get('direction') == 'restore' else 1
    target_delta = sign*delta_transfer
    record = {'raw_target_delta_norm': raw_target, 'target_delta_degenerate': raw_target == 0,
              'same_prompt_donor': True, 'donor_clean_sha256': tensor_sha(donors['sham']['z'][block]),
              'donor_pulse_sha256': tensor_sha(donors[PULSE_NAME[block]]['z'][block])}
    if spec['kind'] == 'target':
        donor = clean if spec['direction'] == 'restore' else pulse
        return {'channels': group, 'donor_values': donor}, target_delta, record
    if spec['kind'] == 'identity':
        donor = donors[spec['receiver']]['z'][block][:, :, group]
        return {'channels': group, 'donor_values': donor}, torch.zeros_like(delta_transfer), record
    index = spec['control_index']
    channels = setting['outside_masks'][index] if spec['kind'] == 'outside' else group
    if not math.isfinite(raw_target):
        return None, target_delta, dict(record, invalid_reason='nonfinite target delta norm; no resampling')
    if raw_target == 0:
        record['zero_target_noop'] = True
        return {'channels': channels, 'delta': torch.zeros((1,1,len(channels)),dtype=torch.float32)}, target_delta, record
    if spec['kind'] == 'outside':
        direction = (donors[PULSE_NAME[block]]['z'][block][:,:,channels]-donors['sham']['z'][block][:,:,channels]).double().reshape(-1)
        record['direction_source'] = 'same-prompt actual pulseZ_C-cleanZ_C'
    else:
        unit = setting['within_units'][index]
        if unit is None:
            return None, target_delta, dict(record, invalid_reason='degenerate fixed random direction; no resampling')
        direction = torch.tensor(unit,dtype=torch.float64)
        record['direction_source'] = 'fixed within-G Gaussian unit vector'
    norm = float(direction.norm()); record['unscaled_direction_raw_norm'] = norm
    if not math.isfinite(norm) or norm <= 1e-12:
        return None, target_delta, dict(record, invalid_reason='degenerate normalizing donor/random direction; no resampling')
    delta = (direction*(sign*raw_target/norm)).float()[None,None,:]
    record['scale_requested'] = sign*raw_target/norm
    return {'channels': channels, 'delta': delta}, target_delta, record


def jsonable(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def patch_summary(patch, model, spec, target_delta, controls, calibration):
    if patch is None:
        return None
    record = dict(patch.record)
    vector = record.pop('projected_delta_vector').double().reshape(-1)
    group = controls[spec['block']]['group']
    weight = model.model.layers[spec['block']].mlp.down_proj.weight
    desired = (weight[:, group] @ target_delta.to(weight.device).reshape(-1)).detach().double().cpu()
    denominator = float(vector.norm()*desired.norm())
    cosine = float(vector.dot(desired)/denominator) if denominator > 1e-24 else None
    actual = float(record['raw_delta_norm'].reshape(-1)[0])
    target = calibration['raw_target_delta_norm']
    matched = (spec['kind'] not in ('outside','within') or (target == 0 and actual == 0)
               or (target > 0 and actual > 0 and abs(actual-target) <= 1e-3*target))
    return {**jsonable(record), **calibration, 'projected_cosine_to_target_delta': cosine,
            'projected_cosine_valid': cosine is not None, 'target_projected_delta_norm':float(desired.norm()),
            'actual_direction_scale_norm':actual/calibration['unscaled_direction_raw_norm']
                if calibration.get('unscaled_direction_raw_norm',0)>0 else None, 'raw_norm_match_passed': matched,
            'actual_to_target_raw_norm_ratio': actual/target if target > 0 else None,
            'fired_count': patch.fired_count, 'completed_count': patch.completed_count,
            'removed_before_decode': patch.removed_before_decode}


def run_prefill(model, request, coordinates, stats, arm, *, patch_args=None, efficient_gpu=True, weight_norms=None):
    import torch

    from attention_probe.group_restoration import GroupZPatch, ProbeCollector
    from attention_probe.interventions import AttentionPulse, CoordinateCapture
    ids = torch.tensor([request['prompt_ids']], dtype=torch.long, device=next(model.parameters()).device)
    positions = [len(request['prompt_ids'])-1]
    patch = None
    with torch.inference_mode(), previous.attention_context(torch, efficient_gpu), contextlib.ExitStack() as stack:
        # Patch PRE-hook precedes the readout PRE-hooks: all recorded z is post-patch.
        if patch_args is not None:
            patch = stack.enter_context(GroupZPatch(model, patch_args['block'], positions,
                patch_args['channels'], **{k:v for k,v in patch_args.items() if k not in ('block', 'channels')}))
        probe = stack.enter_context(ProbeCollector(model, blocks=BLOCKS, positions=positions))
        capture = stack.enter_context(CoordinateCapture(model, coordinates, positions, residual_layers=(), expected_count=len(coordinates)))
        cett = stack.enter_context(base.cett_tap(model, coordinates, positions, weight_norms))
        pulse = stack.enter_context(AttentionPulse(model, arm['block'], positions, arm['beta'], arm['mode'],
                                                   seed=base.SEED+int(request['request_id'][1:])))
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
        del output  # Never reuse this prefill cache as a donor or a generation cache.
        raw = capture.values.clone()
        contribution = cett['cett'].clone()
        mu = torch.tensor(stats['refw']['c_mu'], dtype=torch.float32)
        sd = torch.tensor(stats['refw']['c_sd'], dtype=torch.float32).clamp_min(1e-6)
        arrays = {'raw_z': raw.numpy(), 'cett': contribution.numpy(), 'zcett': ((contribution-mu)/sd).numpy(),
                  'down_output_norm_raw': cett['denominator'].numpy().copy()}
    if (pulse.fired_count!=1 or capture.fired_count!=1 or probe.full_passes!=1 or probe.completed_count!=1
        or (patch is not None and (patch.fired_count!=1 or patch.completed_count!=1))):
        raise ValueError('Prefill pulse/readout/patch did not fire exactly once')
    logits = probe.first_logits.reshape(-1).detach().float().cpu()
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('Nonfinite full-vocabulary prefill logits')
    return {'logits': logits, 'z': probe.z_by_block, 'arrays': arrays, 'patch': patch,
            'pulse_fired': pulse.fired_count, 'capture_fired': capture.fired_count}


def logit_metrics(logits, baselines, tokenizer, request):
    import torch
    logq = logits.double().log_softmax(-1)
    top = torch.topk(logq, min(10, len(logq)))
    fixed = int(baselines['sham']['logits'].argmax())
    gold_ids = request['answer_ids']
    foil_ids = tokenizer(request['group_item']['semantic_alternative'], add_special_tokens=False)['input_ids']
    if not gold_ids or not foil_ids:
        raise ValueError('Correct/foil strings must tokenize without silent replacement')
    gold, foil = int(gold_ids[0]), int(foil_ids[0])
    kl = {}
    for name, baseline in baselines.items():
        logp = baseline['logits'].double().log_softmax(-1)
        kl[name] = {'baseline_to_arm': float((logp.exp()*(logp-logq)).sum()),
                    'arm_to_baseline': float((logq.exp()*(logq-logp)).sum())}
    return {'logits_sha256': tensor_sha(logits), 'full_vocab_kl': kl,
            'fixed_clean_top1_token_id': fixed, 'fixed_clean_top1_logprob': float(logq[fixed]),
            'first_token_correct_id': gold, 'first_token_foil_id': foil,
            'correct_token_ids': gold_ids, 'foil_token_ids': foil_ids,
            'correct_text': request['answer'], 'foil_text': request['group_item']['semantic_alternative'],
            'metric_dtype': 'float64 full-vocabulary log_softmax/KL; generation logger remains original float32',
            'first_token_correct_minus_foil_logprob': float(logq[gold]-logq[foil]),
            'first_token_format_dependent': True, 'first_token_contrast_informative': gold != foil,
            'top10': [{'token_id': int(index), 'text': tokenizer.decode([int(index)], skip_special_tokens=False,
                       clean_up_tokenization_spaces=False), 'logprob': float(value)}
                      for index,value in zip(top.indices.tolist(),top.values.tolist(),strict=True)]}


def run_generation(model, tokenizer, request, coordinates, stats, spec, patch_args, *, efficient_gpu=True, weight_norms=None):
    import torch

    from attention_probe.completion_128 import run_generation_arm
    from attention_probe.group_restoration import GroupZPatch
    patch = None
    with previous.attention_context(torch,efficient_gpu), contextlib.ExitStack() as stack:
        if patch_args is not None:
            patch = stack.enter_context(GroupZPatch(model, patch_args['block'], request['pulse_positions'],
                patch_args['channels'], **{k:v for k,v in patch_args.items() if k not in ('block','channels')}))
        # The unchanged decoder enters CoordinateCapture AFTER the patch and removes
        # its pulse/readout hooks before cached decode. Outer patch auto-removes too.
        record, arrays = run_generation_arm(model, tokenizer, request, coordinates, stats,
            receiver_arm(spec['receiver']), efficient_gpu=efficient_gpu, weight_norms=weight_norms, max_new_tokens=128)
    if patch is not None and not patch.removed_before_decode:
        raise ValueError('Group patch was not removed before cached decoding')
    return record, arrays


def continuity_checks(request, rows, continuity, arrays):
    if request['request_id'] not in ORIGINAL_IDS:
        return {'scope': 'new prespecified variants; no prior output reference'}
    old = continuity['requests'][request['request_id']]
    completion.check_original_request(request, old)
    checks = {}
    for name in BASELINE_NAMES:
        prior = old['arms'][name]
        generation = rows[name]['generation']['record']
        old_gen = prior['generation']
        protocol = all(generation[key] == old_gen[key] for key in
                       ('seed','prompt_length','eos_token_ids','eos_source','stop_reason','selected_token_ids'))
        import numpy as np
        prefix = 'generation_'+name+'_'
        saved = {'gen_'+key[len(prefix):]:value for key,value in arrays.items() if key.startswith(prefix)}
        hashes = {key:hashlib.sha256(np.asarray(value).tobytes(order='C')).hexdigest() for key,value in saved.items()}
        checks[name] = {'tokens_and_protocol_equal': protocol,
            'prefill_arrays_exact': hashes == prior['gen_array_sha256'],
            'prefill_shapes_exact': {key:list(value.shape) for key,value in saved.items()} == prior['gen_array_shapes'],
            'prefill_array_sha256': hashes,
            'first_logits_exact': generation['first_step_logits_sha256'] == old_gen['first_step_logits_sha256'],
            'steps_exact': hashlib.sha256(base.canonical(generation['steps'])).hexdigest() == prior['steps_sha256']}
    return checks


def compute_input(engine, reference, stats, request, controls, continuity, *, efficient_gpu=True):
    import numpy as np
    import torch
    model, tokenizer = engine['model'], engine['tokenizer']
    specs = arm_specs()
    donors, records, arrays, patches = {}, {}, {}, {}
    initial_hooks = previous.hook_count(model)
    if initial_hooks != 0: raise ValueError('Native clean control requires genuinely uninstrumented model')
    ids = torch.tensor([request['prompt_ids']],dtype=torch.long,device=next(model.parameters()).device)
    with torch.inference_mode(), previous.attention_context(torch,efficient_gpu):
        native_output = model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=True)
        native_logits = native_output.logits[0,-1].detach().float().cpu().clone()
        del native_output
    native_hash = tensor_sha(native_logits)
    for spec in specs[:3]:
        name = spec['name']
        result = run_prefill(model, request, reference.coordinates, stats, receiver_arm(name),
                             efficient_gpu=efficient_gpu, weight_norms=engine.get('weight_norms'))
        donors[name] = result
        records[name] = {'name': name, 'spec': spec, 'state': 'complete', 'prefill_executed': True, 'patch': None}
        arrays[f'baseline_{name}_logits'] = result['logits'].numpy()
        for block in BLOCKS:
            arrays[f'baseline_{name}_z_block_{block}'] = result['z'][block].numpy()
        for key,value in result['arrays'].items(): arrays[f'prefill_{name}_{key}'] = value
    if native_hash != tensor_sha(donors['sham']['logits']): raise ValueError('Uninstrumented clean prefill differs from passive sham')
    print(f'Stage {request["request_id"]}: frische Donors; feste Kontrollkalibrierung.',flush=True)
    # All masks/directions are already fixed in the plan. All 32 raw-norm
    # calibrations are constructed before any control outcome or free generation.
    calibration = {}
    for spec in specs[3:]:
        args, target, record = make_patch(spec, donors, controls)
        if args is not None: args = dict(args, block=spec['block'])
        calibration[spec['name']] = (args, target, record)
    for name in BASELINE_NAMES:
        records[name].update(logit_metrics(donors[name]['logits'], donors, tokenizer, request))
    order = list(range(3,35)); random.Random(base.SEED+int(request['request_id'][1:])).shuffle(order)
    for index in order:
        spec=specs[index];name=spec['name'];args,target,cal=calibration[name]
        patches[name]=args
        if args is None:
            records[name]={'name':name,'spec':spec,'state':'invalid','prefill_executed':False,'reason':cal['invalid_reason'],'calibration':cal}
            continue
        result=run_prefill(model,request,reference.coordinates,stats,receiver_arm(spec['receiver']),
            patch_args=args,efficient_gpu=efficient_gpu,weight_norms=engine.get('weight_norms'))
        summary=patch_summary(result['patch'],model,spec,target,controls,cal)
        records[name]={'name':name,'spec':spec,'state':'complete' if summary['raw_norm_match_passed'] else 'invalid',
                       'prefill_executed':True,'patch':summary,
                       **logit_metrics(result['logits'],donors,tokenizer,request)}
        for key,value in result['arrays'].items():arrays[f'prefill_{name}_{key}']=value
        if spec['kind']=='identity' and records[name]['logits_sha256']!=records[spec['receiver']]['logits_sha256']:
            raise ValueError('Identity group patch changed logits')
    print(f'Stage {request["request_id"]}: alle 35 Prefill-Slots abgeschlossen; vorgeplante Generierungen.',flush=True)
    generation_names=generated_names(request['request_id'])
    for spec in specs:
        name=spec['name'];row=records[name]
        if name not in generation_names:
            row['generation']={'state':'not_run','reason':'not selected by fixed generation plan'}
            continue
        if row['state']!='complete':raise ValueError('A main generation arm is invalid; no adaptive replacement')
        gen,gen_arrays=run_generation(model,tokenizer,request,reference.coordinates,stats,spec,patches.get(name),
            efficient_gpu=efficient_gpu,weight_norms=engine.get('weight_norms'))
        if gen['first_step_logits_sha256']!=row['logits_sha256']:
            raise ValueError('Generation first logits differ from corresponding measured prefill arm: '+name)
        row['generation']={'state':'complete','record':gen}
        for key,value in gen_arrays.items():arrays[f'generation_{name}_{key}']=value
    check=continuity_checks(request,records,continuity,arrays)
    if any(row.get('tokens_and_protocol_equal') is False for row in check.values() if isinstance(row,dict)):
        raise ValueError('Fresh original baseline token/protocol differs from the bound 128-token run')
    for value in arrays.values():
        if np.asarray(value).dtype!=np.float32 or not np.isfinite(value).all():raise ValueError('Nonfinite/non-FP32 group readout')
    if previous.hook_count(model) != initial_hooks: raise ValueError('Group probe left model hooks installed')
    return {'technical_checks': {'state':'passed','checks':{'native_clean_logits':True,'identity_logits':True,
            'prefill_counts':True,'hook_cleanup':True,'generation_prefill_logits':True,'fresh_same_prompt_donors':True},
            'native_clean_first_logits_sha256':native_hash},'request':{k:v for k,v in request.items() if k!='ids'},'arms':[records[s['name']] for s in specs],
            'prefill_execution_order':list(BASELINE_NAMES)+[specs[i]['name'] for i in order],
            'continuity':check,'planned_prefill_slots':35,'executed_prefills':sum(r['prefill_executed'] for r in records.values()),
            'generation_count':len(generation_names),'array_contract':{'vocab_size':int(model.config.vocab_size),
            'hidden_size':int(model.config.hidden_size),'mlp_width':int(model.config.intermediate_size),
            'n_coordinates':len(reference.coordinates)}},arrays


def frozen_items():
    path=source_path('attention_probe/group_probe_items.json')
    if base.sha_file(path)!=GROUP_ITEMS_SHA256:raise ValueError('Frozen group item source changed')
    return {item['request_id']:item for item in json.loads(path.read_text())}


def validate_record(record, plan_hash, request_id):
    if (record.get('schema')!=SCHEMA or record.get('state')!='complete' or record.get('plan_sha256')!=plan_hash
        or record.get('request_id')!=request_id or record.get('planned_prefill_slots')!=35
        or record.get('generation_count')!=len(generated_names(request_id))
        or record.get('array_contract')!=ARRAY_CONTRACT):
        raise ValueError('Incomplete or differently bound group input checkpoint')
    request=record.get('request',{});item=frozen_items()[request_id]
    if (request.get('request_id')!=request_id or request.get('group_item')!=item
        or request.get('item_id')!=item['id'] or request.get('question')!=item['prompt']
        or request.get('answer')!=item['target'] or request.get('validation_schema')!=SCHEMA):
        raise ValueError('Checkpoint request identity/frozen item changed')
    ids=request.get('prompt_ids')
    if not isinstance(ids,list) or not ids or any(type(value) is not int or not 0<=value<ARRAY_CONTRACT['vocab_size'] for value in ids):
        raise ValueError('Missing or invalid original prompt token IDs')
    positions=[len(ids)-1]
    if (request.get('pulse_positions')!=positions or request.get('prefill_capture_positions')!=positions
        or request.get('generation_capture_positions')!=list(range(max(0,len(ids)-4),len(ids)))
        or hashlib.sha256(request['formatted_prompt'].encode()).hexdigest()!=request.get('formatted_prompt_sha256')):
        raise ValueError('Prompt-only prefill/generation positions or template hash changed')
    proof=record.get('technical_checks',{})
    if (proof.get('state')!='passed' or proof.get('plan_sha256')!=plan_hash
        or any(proof.get('checks',{}).get(name) is not True for name in PROOF_CHECKS)):
        raise ValueError('Missing plan-bound group technical checks')
    rows=record.get('arms',[])
    if [row.get('spec') for row in rows]!=arm_specs():raise ValueError('Group arm plan changed')
    for row,spec in zip(rows,arm_specs(),strict=True):
        if row.get('name')!=spec['name'] or type(row.get('prefill_executed')) is not bool:
            raise ValueError('Group arm identity/prefill execution marker changed')
        expected='complete' if row['name'] in generated_names(request_id) else 'not_run'
        if row.get('generation',{}).get('state')!=expected:raise ValueError('Group generation/not_run contract changed')
        if row.get('state') not in ('complete','invalid') or (row['state']=='complete' and not row['prefill_executed']):
            raise ValueError('Unfinished prefill slot')
        if row['state']=='invalid' and spec['kind'] not in ('outside','within'):
            raise ValueError('A fixed main/identity arm cannot be replaced by an invalid control')
        if row['state']=='complete' and spec['kind'] in ('outside','within') and row.get('patch',{}).get('raw_norm_match_passed') is not True:
            raise ValueError('An unmatched control cannot enter the valid comparison')
        if row['prefill_executed']:
            if (not isinstance(row.get('logits_sha256'),str) or len(row['logits_sha256'])!=64
                or set(row.get('full_vocab_kl',{}))!=set(BASELINE_NAMES)):
                raise ValueError('Missing full-vocabulary KL endpoints/hash')
            for value in row['full_vocab_kl'].values():
                if set(value)!= {'baseline_to_arm','arm_to_baseline'} or any(not math.isfinite(x) for x in value.values()):
                    raise ValueError('Nonfinite/incomplete full-vocabulary KL')
        if expected=='complete':
            generation=row['generation']['record']
            completion.FUNCTIONS['validate_generation'](generation,request,receiver_arm(spec['receiver']))
            if generation.get('seed')!=base.SEED+int(request_id[1:]):
                raise ValueError('Original per-request generation seed changed')
            if generation.get('first_step_logits_sha256')!=row['logits_sha256']:
                raise ValueError('Generation/prefill logit identity changed')
    if record.get('executed_prefills')!=sum(row['prefill_executed'] for row in rows):
        raise ValueError('Executed prefill count changed')


def validate_arrays(record, arrays):
    import numpy as np
    contract=ARRAY_CONTRACT;expected={};request=record['request']
    for name in BASELINE_NAMES:
        expected[f'baseline_{name}_logits']=(contract['vocab_size'],)
        for block in BLOCKS:expected[f'baseline_{name}_z_block_{block}']=(1,1,contract['mlp_width'])
    for row in record['arms']:
        name=row['name']
        if row['prefill_executed']:
            for key in ('raw_z','cett','zcett','down_output_norm_raw'):
                expected[f'prefill_{name}_{key}']=(1,1,contract['n_coordinates'])
        if row['generation']['state']=='complete':
            length=min(4,len(request['prompt_ids']))
            for key in ('raw_z','cett','zcett','down_output_norm_raw'):
                expected[f'generation_{name}_{key}']=(1,length,contract['n_coordinates'])
            for block in (3,7,14,27):expected[f'generation_{name}_residual_layer_{block}']=(1,length,contract['hidden_size'])
    keys=set(arrays.files) if hasattr(arrays,'files') else set(arrays)
    if keys!=set(expected):raise ValueError('Missing or unexpected group NPZ measurement keys')
    for key,shape in expected.items():
        value=np.asarray(arrays[key])
        if value.shape!=shape or value.dtype!=np.float32 or not np.isfinite(value).all():
            raise ValueError('Group array shape/dtype/finiteness mismatch: '+key)
    rows={row['name']:row for row in record['arms']}
    for name in BASELINE_NAMES:
        value=np.asarray(arrays[f'baseline_{name}_logits'])
        if hashlib.sha256(value.tobytes(order='C')).hexdigest()!=rows[name]['logits_sha256']:
            raise ValueError('Saved baseline full-vocabulary endpoint hash changed')


def verified_input(output, request_id, plan_hash, continuity=None):
    import numpy as np
    directory=Path(output)/'requests'/request_id
    if not directory.exists():return None
    record=json.loads((directory/'result.json').read_text());validate_record(record,plan_hash,request_id)
    if base.sha_file(directory/'arrays.npz')!=record['npz_sha256']:raise ValueError('Group checkpoint NPZ SHA mismatch')
    with np.load(directory/'arrays.npz',allow_pickle=False) as saved:
        validate_arrays(record,saved)
        if continuity is not None:
            actual=continuity_checks(record['request'],{row['name']:row for row in record['arms']},continuity,saved)
            if actual!=record['continuity']:raise ValueError('Stored original continuity evidence changed')
    return record


def write_input(output, request_id, result, arrays, plan_hash, hardware, proof):
    import numpy as np
    parent=Path(output)/'requests';parent.mkdir(parents=True,exist_ok=True)
    destination=parent/request_id
    if destination.exists():raise ValueError('Refusing to overwrite completed group input')
    temporary=Path(tempfile.mkdtemp(prefix='.'+request_id+'_',dir=parent))
    try:
        np.savez_compressed(temporary/'arrays.npz',**arrays)
        record=dict(result,schema=SCHEMA,state='complete',request_id=request_id,plan_sha256=plan_hash,
                    npz_sha256=base.sha_file(temporary/'arrays.npz'),hardware=hardware,technical_checks=proof)
        validate_record(record,plan_hash,request_id)
        validate_arrays(record,arrays)
        base.atomic_json(temporary/'result.json',record)
        os.rename(temporary,destination)
    finally:
        if temporary.exists():shutil.rmtree(temporary)
    return record


def restore_resume(path,output,plan_hash,continuity=None):
    if not path:return
    with zipfile.ZipFile(path) as packed:
        plan=json.loads(packed.read('run_plan.json'))
        if plan.get('schema')!=SCHEMA or hashlib.sha256(base.canonical(plan)).hexdigest()!=plan_hash:
            raise ValueError('Resume has another group/source/reference plan')
        for rid in REQUEST_IDS:
            prefix='requests/'+rid+'/'
            names=[prefix+'result.json',prefix+'arrays.npz'];present=[n in packed.namelist() for n in names]
            if not any(present):continue
            if not all(present):raise ValueError('Incomplete group input in resume')
            parent=Path(output)/'requests';parent.mkdir(parents=True,exist_ok=True);dest=parent/rid
            if dest.exists():
                verified_input(output,rid,plan_hash,continuity)
                if any((dest/Path(n).name).read_bytes()!=packed.read(n) for n in names):raise ValueError('Resume conflicts; no overwrite')
                continue
            temporary=Path(tempfile.mkdtemp(prefix='.'+rid+'_',dir=parent))
            try:
                for name in names:
                    if packed.getinfo(name).file_size>64*(1<<20):raise ValueError('Oversized group checkpoint')
                    (temporary/Path(name).name).write_bytes(packed.read(name))
                rec=json.loads((temporary/'result.json').read_text());validate_record(rec,plan_hash,rid)
                if base.sha_file(temporary/'arrays.npz')!=rec['npz_sha256']:raise ValueError('Resume group arrays SHA mismatch')
                import numpy as np
                with np.load(temporary/'arrays.npz',allow_pickle=False) as arrays:validate_arrays(rec,arrays)
                os.rename(temporary,dest);verified_input(output,rid,plan_hash,continuity)
            finally:
                if temporary.exists():shutil.rmtree(temporary)


def archive(output):
    target=Path(output)/'h286_group_restore_results.zip';temporary=target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as packed:
        for path in sorted(Path(output).rglob('*')):
            if path.is_file() and path.suffix not in ('.zip','.tmp') and not any(p.startswith('.') for p in path.relative_to(output).parts):
                packed.write(path,path.relative_to(output))
    os.replace(temporary,target)


def run(hn,policy,refw,items_path,continuity_path,out_dir,cache_dir=None,resume_zip=None):
    output=Path(out_dir);output.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    status={'state':'running','stage':'reference','scope':SCHEMA,'completed_inputs':0,'confirmatory_eligible':False}
    metrics={'weights_loaded':False,'pretrained_model_run':False,'wrapper_attempt_id':os.environ.get('H286_WRAPPER_ATTEMPT'),
             'no_training':True,'no_confirmatory_claim':True}
    def save():base.atomic_json(output/'status.json',status);base.atomic_json(output/'metrics.json',metrics)
    save()
    try:
        reference,stats,items,continuity=load_inputs(hn,policy,refw,items_path,continuity_path)
        controls=make_controls(reference.coordinates,reference.model.mlp_width)
        plan=plan_document(reference,stats,items,continuity,controls);digest=hashlib.sha256(base.canonical(plan)).hexdigest()
        path=output/'run_plan.json'
        if path.exists() and path.read_bytes()!=base.canonical(plan)+b'\n':raise ValueError('Different group plan; do not mix outputs')
        if not path.exists():base.atomic_json(path,plan)
        restore_resume(resume_zip,output,digest,continuity)
        done={rid:verified_input(output,rid,digest,continuity) for rid in REQUEST_IDS}
        if not all(done.values()):
            status['stage']='hardware_and_original_model_load';save()
            def loaded(hardware):metrics.update(weights_loaded=True,hardware=hardware);save()
            engine=previous.initialize_engine(reference,Path(cache_dir or Path.home()/'.cache/h286_pilot'),on_loaded=loaded)
            import torch
            model=engine['model']
            actual={'vocab_size':int(model.config.vocab_size),'hidden_size':int(model.config.hidden_size),
                    'mlp_width':int(model.config.intermediate_size),'n_coordinates':len(reference.coordinates)}
            if actual!=ARRAY_CONTRACT:raise ValueError('Original fixed model/array architecture changed')
            frozen=base.parameter_evidence(engine['model']);measured=[]
            for index,item in enumerate(items):
                rid=item['request_id']
                if done[rid] is not None:continue
                request=prepare_request(engine['tokenizer'],index,item)
                status.update(stage='prefills_and_prespecified_generations',current_request=rid);save()
                print(f'Request {index+1}/10 {rid}: 35 Prefill-Slots, feste Gruppen vor Generierung.',flush=True)
                cpu_rng=torch.random.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone();t0=time.monotonic()
                result,arrays=compute_input(engine,reference,stats,request,controls,continuity)
                if base.parameter_evidence(engine['model'])!=frozen or not torch.equal(cpu_rng,torch.random.get_rng_state()) or not torch.equal(cuda_rng,torch.cuda.get_rng_state()):
                    raise ValueError('Frozen parameters/global RNG changed during group input')
                metrics['pretrained_model_run']=True
                proof=dict(result['technical_checks'],plan_sha256=digest)
                proof['checks'].update(frozen_parameters=True,global_rng=True)
                done[rid]=write_input(output,rid,result,arrays,digest,engine['hardware'],proof)
                measured.append(time.monotonic()-t0);remaining=sum(v is None for v in done.values())
                status.update(completed_inputs=10-remaining,estimated_remaining_seconds=sum(measured)/len(measured)*remaining);save()
                print(f'Gespeichert {10-remaining}/10; Restschätzung {status["estimated_remaining_seconds"]/60:.1f} min.',flush=True)
        metrics['generation_count']=sum(row['generation_count'] for row in done.values())
        metrics['prefill_slots']=sum(row['planned_prefill_slots'] for row in done.values())
        metrics['native_uninstrumented_prefills']=10
        metrics['generation_replay_prefills']=46
        metrics['executed_measurement_prefills']=sum(row['executed_prefills'] for row in done.values())
        metrics['total_prefill_forwards']=metrics['executed_measurement_prefills']+10+46
        if metrics['generation_count']!=46 or metrics['prefill_slots']!=350:raise ValueError('Fixed group-run cardinality changed')
        status.update(state='complete',stage='complete',completed_inputs=10)
    except BaseException as error:  # noqa: BLE001 -- keep diagnostic ZIP on interruption.
        status.update(state='failed',error_type=type(error).__name__,error=str(error));(output/'traceback.txt').write_text(traceback.format_exc())
        print('HARD FAILURE: '+str(error),flush=True)
    finally:
        metrics['elapsed_seconds']=time.monotonic()-started;save();archive(output)
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('hn-set','policy','refw','items','continuity','out-dir'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--cache-dir');parser.add_argument('--resume-zip');args=parser.parse_args()
    result=run(args.hn_set,args.policy,args.refw,args.items,args.continuity,args.out_dir,args.cache_dir,args.resume_zip)
    print(json.dumps(result),flush=True);raise SystemExit(0 if result['state']=='complete' else 1)
