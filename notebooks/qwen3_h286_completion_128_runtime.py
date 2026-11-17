"""Four prespecified EOS-completion checks; keep the completed 48-token run intact."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
import types
import zipfile
from pathlib import Path

import qwen3_h286_validation_runtime as previous

base = previous.base
SCHEMA = 'qwen3_h286_completion_128_v1'
PROOF_SCHEMA = 'qwen3_h286_completion_128_checks_v1'
INDICES = (12, 13, 14, 15)
REQUEST_IDS = tuple(f'r{i:03d}' for i in INDICES)
BASELINE_SHA256 = '930d772f999b91931023cc1dbbc4bbd98ad68eb27717d6b6fdc6356d1f2e1bd0'
SOURCE_ARCHIVE_SHA256 = '004eae9d743a529aa8e6661c11e892871a41fc6e43745310274a6ae524d432e2'
SOURCE_PLAN_SHA256 = '179f4a37cfd2822a3f4011b7e18c1050cd017bb96484e4dfa270919dcfa2ce4e'
MAX_NEW_TOKENS = 128


def generation_module():
    from attention_probe import completion_128
    return completion_128


def validation_functions():
    """Reuse validated code with consistent private globals, never patch old modules."""
    scope = dict(previous.__dict__, SCHEMA=SCHEMA, PROOF_SCHEMA=PROOF_SCHEMA,
                 MAX_NEW_TOKENS=MAX_NEW_TOKENS, generation_module=generation_module)
    for name in ('verify_proof', 'technical_checks', 'validate_generation', 'validate_payload',
                 'compute_request', 'verified_request', 'restore_resume'):
        old = getattr(previous, name)
        new = types.FunctionType(old.__code__, scope, name, old.__defaults__, old.__closure__)
        new.__kwdefaults__ = dict(old.__kwdefaults__ or {})
        scope[name] = new
    return scope


FUNCTIONS = validation_functions()


def load_baseline(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASELINE_SHA256:
        raise ValueError('Exact private 48-token baseline SHA mismatch')
    value = json.loads(raw)
    if (value.get('schema') != 'h286_completion_128_baseline_v1'
            or value.get('source_archive_sha256') != SOURCE_ARCHIVE_SHA256
            or value.get('source_plan_sha256') != SOURCE_PLAN_SHA256
            or set(value.get('requests', {})) != set(REQUEST_IDS)
            or hashlib.sha256(base.canonical(value['old_plan'])).hexdigest() != SOURCE_PLAN_SHA256):
        raise ValueError('Baseline source/plan/four original request identities mismatch')
    return value


def plan_document(reference, stats, items, baseline):
    original_plan = previous.plan_document(reference, stats, items)
    if hashlib.sha256(base.canonical(original_plan)).hexdigest() != SOURCE_PLAN_SHA256:
        raise ValueError('Original scientific sources/references differ from the completed 48-token run')
    result = dict(original_plan)
    result.update(schema=SCHEMA, purpose='post-result-selected four-input EOS continuation only',
        previous_validation_plan_sha256=SOURCE_PLAN_SHA256, baseline_sha256=BASELINE_SHA256,
        source_archive_sha256=SOURCE_ARCHIVE_SHA256, request_indices=list(INDICES),
        helper_sha256=base.sha_file(__file__), completion_module_sha256=base.sha_file(generation_module().__file__),
        technical_proof_schema=PROOF_SCHEMA,
        unchanged_scientific_source_sha256=baseline['embedded_source'],
        queue_budget={'scope': 'four requests times five arms times total 128 selected tokens',
                      'wallclock_limit_seconds': None, 'wallclock_finish_guaranteed': False},
        decoder_adapter={'source': 'unchanged original decoder code object',
                         'isolated_global_change': {'MAX_NEW_TOKENS': [48, 128]}},
        generation=dict(original_plan['generation'], max_new_tokens=128,
            compatibility='fresh generation from original prompt; exact old selected-token prefix including old EOS',
            numerical_diagnostics='logit/top-k/logprob/prefill hashes reported separately; not hard token-prefix conditions'),
        inference_scope='diagnostic follow-up selected after old results; no independent confirmation',
        semantic_review='separate human review after EOS; surface classification is not semantic correctness')
    result['module_sha256'] = dict(original_plan['module_sha256'],
                                  **{'completion_128.py': result['completion_module_sha256']})
    return result


def check_original_request(request, prior):
    old = prior['request']
    for key in ('request_id', 'item_id', 'pair_id', 'prompt_ids', 'formatted_prompt_sha256',
                'answer_ids', 'pulse_positions'):
        if request.get(key) != old.get(key):
            raise ValueError('Original request/tokenization changed: ' + key)


def compare_prefix(record, prior, arrays):
    """Token/EOS compatibility is hard; numerical bitwise diagnostics are separate."""
    import numpy as np
    old = prior['generation']
    if record.get('arm') != prior['arm'] or record.get('seed') != old['seed']:
        raise ValueError('Arm or original per-request random-control seed changed')
    selected = old['selected_token_ids']
    token_equal = record['selected_token_ids'][:len(selected)] == selected
    if old['stop_reason'] == 'eos':
        eos_equal = record['stop_reason'] == 'eos' and record['selected_token_ids'] == selected
    else:
        eos_equal = len(selected) == 48 and len(record['selected_token_ids']) >= 48
    old_steps, new_steps = old['steps'], record['steps'][:len(selected)]
    steps_equal = old_steps == new_steps
    logprob_differences = [abs(a['selected_token_logprob'] - b['selected_token_logprob'])
                           for a, b in zip(old_steps, new_steps)]
    hashes = {key: hashlib.sha256(np.asarray(value).tobytes(order='C')).hexdigest()
              for key, value in arrays.items() if key.startswith('gen_')}
    expected_hashes = prior['gen_array_sha256']
    eos_protocol_equal = (record.get('eos_token_ids') == old.get('eos_token_ids')
                          and record.get('eos_source') == old.get('eos_source'))
    return {'compatible': bool(token_equal and eos_equal and eos_protocol_equal),
            'eos_protocol_equal': eos_protocol_equal, 'tokens_equal': token_equal,
            'old_eos_or_cap_preserved': eos_equal, 'old_selected_tokens': len(selected),
            'old_stop_reason': old['stop_reason'], 'first_logits_exact': record['first_step_logits_sha256'] == old['first_step_logits_sha256'],
            'prefix_steps_logprobs_topk_exact': steps_equal,
            'max_abs_selected_logprob_difference': max(logprob_differences, default=0.),
            'prefill_arrays_exact': hashes == expected_hashes,
            'prefill_array_shapes_exact': {key: list(value.shape) for key, value in arrays.items()
                                           if key.startswith('gen_')} == prior['gen_array_shapes'],
            'prefill_array_sha256': hashes,
            'requested_delta_norms_exact': record['requested_delta_norms'] == old['requested_delta_norms'],
            'actual_delta_norms_exact': record['actual_delta_norms'] == old['actual_delta_norms'],
            'numerical_diagnostics_are_additional_hard_conditions': False}


def compare_result(request, rows, arrays, baseline):
    prior = baseline['requests'][request['request_id']]
    check_original_request(request, prior)
    checks = []
    for row, old, values in zip(rows, prior['arms'], arrays, strict=True):
        check = compare_prefix(row['generation'], old, values)
        row['continuation_48_check'] = check
        row['completion'] = {'eos_finished': row['generation']['stop_reason'] == 'eos',
                             'truncated_at_128': row['generation']['truncated']}
        row['format'] = row['generation']['classification_details']
        row['semantic_answer'] = {'status': 'requires_separate_review',
            'target': request['answer'], 'reviewable_complete_output': not row['generation']['truncated'],
            'note': 'Do not infer correctness from truncation, gold mentions, or noncanonical formatting.'}
        checks.append(check)
    return checks


def verified_request(output, request_id, plan_hash, count, baseline):
    import numpy as np
    record = FUNCTIONS['verified_request'](output, request_id, plan_hash, count)
    if record is None:
        return None
    with np.load(Path(output)/'requests'/request_id/'arrays.npz', allow_pickle=False) as saved:
        arrays = [{key: saved[key][i] for key in saved.files} for i in range(5)]
    original_checks = [row.get('continuation_48_check') for row in record['arms']]
    checks = compare_result(record['request'], record['arms'], arrays, baseline)
    if checks != original_checks:
        raise ValueError('Stored prefix compatibility diagnostics changed')
    return record


def archive(output):
    target = Path(output)/'h286_completion_128_results.zip'
    temporary = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as packed:
        for path in sorted(Path(output).rglob('*')):
            if path.is_file() and path.suffix not in ('.zip', '.tmp') and not any(
                    part.startswith('.') for part in path.relative_to(output).parts):
                packed.write(path, path.relative_to(output))
    os.replace(temporary, target)


def run(hn_set, policy, refw, items_path, baseline_path, out_dir, cache_dir=None, resume_zip=None):
    output = Path(out_dir); output.mkdir(parents=True, exist_ok=True)
    status = {'state': 'running', 'stage': 'reference', 'scope': SCHEMA, 'completed_requests': 0,
              'confirmatory_eligible': False}
    metrics = {'weights_loaded': False, 'pretrained_model_run': False,
               'wrapper_attempt_id': os.environ.get('H286_WRAPPER_ATTEMPT'), 'checks': {},
               'semantic_scoring': 'separate review required; no automatic semantic-correctness rescue'}
    started = time.monotonic()
    def save(extra=None):
        if extra: status.update(extra)
        base.atomic_json(output/'status.json', status); base.atomic_json(output/'metrics.json', metrics)
    save()
    try:
        reference, stats, items = base.load_inputs(hn_set, policy, refw, items_path)
        baseline = load_baseline(baseline_path)
        plan = plan_document(reference, stats, items, baseline)
        digest = hashlib.sha256(base.canonical(plan)).hexdigest()
        path = output/'run_plan.json'
        if path.exists() and path.read_bytes() != base.canonical(plan)+b'\n':
            raise ValueError('Existing completion output has another plan; no mixing or overwrite')
        if not path.exists(): base.atomic_json(path, plan)
        base.atomic_json(output/'prior_48_provenance.json', {'baseline_sha256': BASELINE_SHA256,
            'source_archive_sha256': SOURCE_ARCHIVE_SHA256, 'source_plan_sha256': SOURCE_PLAN_SHA256,
            'requests': {rid: {key: value[key] for key in ('source_result_sha256', 'source_arrays_sha256')}
                         for rid, value in baseline['requests'].items()}})
        if resume_zip:
            with zipfile.ZipFile(resume_zip) as packed:
                ids = {Path(name).parts[1] for name in packed.namelist() if name.startswith('requests/') and len(Path(name).parts)>2}
                if not ids.issubset(REQUEST_IDS): raise ValueError('Resume contains requests outside the four-input continuation')
            FUNCTIONS['restore_resume'](resume_zip, output, digest, len(reference.coordinates))
        done = {rid: verified_request(output, rid, digest, len(reference.coordinates), baseline) for rid in REQUEST_IDS}
        status['completed_requests'] = sum(row is not None for row in done.values())
        if not all(done.values()):
            status['stage'] = 'hardware_and_original_model_load'; save()
            def loaded(hardware):
                metrics.update(weights_loaded=True, hardware=hardware); save()
            engine = previous.initialize_engine(reference, Path(cache_dir or Path.home()/'.cache/h286_pilot'), on_loaded=loaded)
            model, tokenizer = engine['model'], engine['tokenizer']
            import torch
            first = next(i for i in INDICES if done[f'r{i:03d}'] is None)
            request = base.prepare_request(tokenizer, first, items[first])
            check_original_request(request, baseline['requests'][request['request_id']])
            with torch.inference_mode(), previous.attention_context(torch, True):
                ids = request['ids'].to(next(model.parameters()).device)
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            metrics['pretrained_model_run'] = True
            status['stage'] = 'unchanged_controls_with_cap128'; save()
            proof = FUNCTIONS['technical_checks'](model, tokenizer, request, reference, stats, digest,
                engine['hardware'], weight_norms=engine['weight_norms'])
            metrics['checks'] = proof['checks']
            frozen = base.parameter_evidence(model)
            for index in INDICES:
                rid = f'r{index:03d}'
                if done[rid] is not None: continue
                request = base.prepare_request(tokenizer, index, items[index])
                check_original_request(request, baseline['requests'][rid])
                status.update(stage='four_input_completion', current_request=rid); save()
                print(f'Request {rid}: originaler Prompt, fünf Arme, Gesamtkappe128.', flush=True)
                cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
                rows, arrays = FUNCTIONS['compute_request'](model, tokenizer, request, reference, stats,
                    weight_norms=engine['weight_norms'])
                checks = compare_result(request, rows, arrays, baseline)
                if (base.parameter_evidence(model) != frozen or not torch.equal(cpu_rng, torch.random.get_rng_state())
                        or not torch.equal(cuda_rng, torch.cuda.get_rng_state())):
                    raise ValueError('Original frozen parameter/global RNG evidence changed')
                done[rid] = base.write_request(output, request, rows, arrays, digest, engine['hardware'], proof)
                status['completed_requests'] = sum(row is not None for row in done.values()); save()
                print(f'Gespeichert {status["completed_requests"]}/4; Präfix passend: {sum(c["compatible"] for c in checks)}/5.', flush=True)
        all_checks = [arm['continuation_48_check'] for row in done.values() for arm in row['arms']]
        metrics['prefix_compatibility_passed'] = all(check['compatible'] for check in all_checks)
        metrics['prefix_checks'] = all_checks
        metrics['retained_request_proofs'] = [row['technical_checks'] for row in done.values()]
        status.update(state='complete' if metrics['prefix_compatibility_passed'] else 'failed',
                      stage='complete' if metrics['prefix_compatibility_passed'] else 'prefix_compatibility_failure')
        if not metrics['prefix_compatibility_passed']:
            status['error'] = 'Old selected-token prefix/EOS differs; longer text is not an established compatible continuation.'
    except BaseException as error:  # noqa: BLE001 -- retain diagnostics after interruption.
        status.update(state='failed', error_type=type(error).__name__, error=str(error))
        (output/'traceback.txt').write_text(traceback.format_exc())
        print('HARD FAILURE: '+str(error), flush=True)
    finally:
        metrics['elapsed_seconds'] = time.monotonic()-started
        save(); archive(output)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('hn-set', 'policy', 'refw', 'items', 'baseline', 'out-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--cache-dir'); parser.add_argument('--resume-zip')
    args = parser.parse_args()
    result = run(args.hn_set, args.policy, args.refw, args.items, args.baseline, args.out_dir, args.cache_dir, args.resume_zip)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['state']=='complete' else 1)
