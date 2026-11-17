"""Short exploratory ORIGINAL Qwen3/H286 pilot; embedded notebook entry point.

No re-selection, fitting, generation, training, or cross-model coordinate mapping.
Original FP32, TF32-off, fixed REF-W and efficient SDPA are retained. The two
technical checks precede a fixed 16-request/five-arm exploratory run. Results do
not constitute a held-out confirmatory test or a proven attention bottleneck.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import platform
import random
import shutil
import tempfile
import time
import traceback
import zipfile
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

MODEL = 'Qwen/Qwen3-1.7B'
REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
TRANSFORMERS = '5.22.2'
ACCELERATE = '1.16.0'
REFW_SHA256 = '68f96129d8ab00f3c7e6a68a9b4a0bdb58798dfac4d1d405f6cce5ccca0ae5cc'
META_SHA256 = '9c61bc34e804719e5bf5e72b644e8a38a94ad0f6c1708efaef75386acdae49ff'
SEED = 20261117
ARMS = (
    {'name': 'sham', 'block': 14, 'beta': 1., 'mode': 'scale'},
    {'name': 'soft_103', 'block': 14, 'beta': 1 / 1.03, 'mode': 'scale'},
    {'name': 'soft_108', 'block': 14, 'beta': 1 / 1.08, 'mode': 'scale'},
    {'name': 'random_matched_108', 'block': 14, 'beta': 1 / 1.08, 'mode': 'random'},
    {'name': 'other_block_108', 'block': 7, 'beta': 1 / 1.08, 'mode': 'scale'},
)
# Declared before any new model result. These are technical/exploratory items,
# not the frozen DSSE evaluation sample or a benchmark accuracy test.
ITEMS_SHA256 = '12a3cba862185b9f25ca39ea3ea49bd3bb6fc5f248c12dbbe02bcc8836663fb3'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(canonical(value) + b'\n')
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def archive(output):
    target = output / 'qwen3_h286_results.zip'
    temporary = output / '.qwen3_h286_results.zip.tmp'
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for item in sorted(output.rglob('*')):
            if item.is_file() and item not in (target, temporary) and not any(part.startswith('.') for part in item.relative_to(output).parts):
                bundle.write(item, item.relative_to(output))
    os.replace(temporary, target)
    return target


def load_inputs(hn_path, policy_path, refw_path, items_path):
    from attention_probe.fixed_refw import load_fixed_refw
    from attention_probe.reference import load_reference
    reference = load_reference(hn_path, policy_path)
    stats = load_fixed_refw(refw_path, reference)
    raw = Path(items_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ITEMS_SHA256:
        raise ValueError('The exact prespecified 16-item file is required; no replacement or post-result selection')
    items = json.loads(raw)
    if len(items) != 16 or len({item['id'] for item in items}) != 16:
        raise ValueError('Exactly 16 unique fixed requests are required')
    return reference, stats, items


def hardware_preflight(torch, cache_dir):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU erforderlich; keine Warteschleife. A100 empfohlen, L4 ausreichend.')
    free, total = torch.cuda.mem_get_info(0)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(cache_dir).free
    if free < 10 * (1 << 30):
        raise RuntimeError('Mindestens 10 GiB freie GPU-VRAM für unverändertes FP32 erforderlich.')
    if disk < 8 * (1 << 30):
        raise RuntimeError('Mindestens 8 GiB freier Cache-Speicher erforderlich; keine Gewichte heruntergeladen.')
    available = None
    meminfo = Path('/proc/meminfo')
    if meminfo.exists():
        fields = dict(line.split(':', 1) for line in meminfo.read_text().splitlines())
        available = int(fields['MemAvailable'].split()[0]) * 1024
        if available < 10 * (1 << 30):
            raise RuntimeError('Mindestens 10 GiB verfügbarer Host-RAM erforderlich.')
    return {'gpu': torch.cuda.get_device_name(0), 'free_vram': free, 'total_vram': total,
            'free_disk': disk, 'available_host_ram': available, 'device': 'cuda:0'}


def model_load_kwargs(torch):
    return {'revision': REVISION, 'dtype': torch.float32, 'device_map': {'': 0},
            'attn_implementation': 'sdpa', 'trust_remote_code': False, 'use_safetensors': True,
            'local_files_only': True}


def load_model(reference, cache_dir):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    snapshot = Path(snapshot_download(MODEL, revision=REVISION, cache_dir=str(cache_dir),
                                     allow_patterns=['*.json', '*.safetensors', '*.jinja']))
    if snapshot.name != REVISION:
        raise ValueError('HF snapshot path does not identify the exact pinned revision')
    for filename, expected in reference.model.sha256:
        if sha_file(snapshot / filename) != expected:
            raise ValueError('Original model/tokenizer SHA mismatch: ' + filename)
    config = AutoConfig.from_pretrained(str(snapshot), revision=REVISION, trust_remote_code=False,
                                       local_files_only=True)
    if config.model_type != 'qwen3' or config.num_hidden_layers != 28 or config.intermediate_size != 6144:
        raise ValueError('Original Qwen3-1.7B architecture mismatch')
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), revision=REVISION, trust_remote_code=False,
                                             local_files_only=True)
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), config=config, **model_load_kwargs(torch))
    model.eval().requires_grad_(False)
    if any(parameter.device.type != 'cuda' or parameter.device.index != 0 or parameter.dtype != torch.float32
           for parameter in model.parameters()):
        raise ValueError('All model parameters must remain full FP32 on cuda:0; no offload/quantization')
    return tokenizer, model


def prepare_request(tokenizer, index, item):
    import torch
    category, question, answer = item['category'], item['prompt'], item['target']
    formatted = tokenizer.apply_chat_template([{'role': 'user', 'content': question}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
    prompt = tokenizer(formatted, add_special_tokens=False)['input_ids']
    answer_ids = tokenizer(answer, add_special_tokens=False)['input_ids']
    if not prompt or not answer_ids or len(prompt) + len(answer_ids) > 128:
        raise ValueError('Unexpected empty or long fixed teacher-forcing request; no silent truncation')
    # Token at p predicts token p+1. The last answer activation is recorded but
    # cannot cause its own prediction. Answer likelihood uses preceding positions.
    ids = prompt + answer_ids
    positions = list(range(max(0, len(ids) - 4), len(ids)))
    return {'request_id': f'r{index:03d}', 'item_id': item['id'], 'pair_id': item['pair_id'], 'category': category, 'question': question, 'answer': answer,
            'formatted_prompt': formatted, 'formatted_prompt_sha256': hashlib.sha256(formatted.encode()).hexdigest(), 'prompt_ids': prompt, 'answer_ids': answer_ids,
            'input_ids': ids, 'capture_positions': positions, 'pulse_positions': [len(prompt) - 1],
            'answer_logit_positions': list(range(len(prompt) - 1, len(ids) - 1)),
            'thinking_enabled': False, 'ids': torch.tensor([ids], dtype=torch.long)}


@contextlib.contextmanager
def cett_tap(model, coordinates, positions, weight_norms=None):
    import torch
    groups = defaultdict(list)
    for order, (layer, neuron) in enumerate(coordinates):
        groups[layer].append((order, neuron))
    result = {'cett': torch.empty((1, len(positions), len(coordinates)), dtype=torch.float32),
              'denominator': torch.empty((1, len(positions), len(coordinates)), dtype=torch.float32)}
    handles, seen = [], set()
    try:
        for layer, pairs in groups.items():
            down = model.model.layers[layer].mlp.down_proj
            columns = [neuron for _, neuron in pairs]
            weight_norm = (weight_norms[layer] if weight_norms is not None
                           else down.weight.float().norm(dim=0)[columns])

            def hook(_module, args, output, layer=layer, pairs=pairs, columns=columns, weight_norm=weight_norm):
                if layer in seen:
                    raise ValueError('CETT tap fired repeatedly')
                z = args[0][:, positions][:, :, columns].abs().float()
                den = output.norm(dim=-1)[:, positions].float()
                contribution = z * weight_norm / den.clamp_min(1e-9)[..., None]
                order = [index for index, _ in pairs]
                result['cett'][:, :, order] = contribution.detach().cpu()
                result['denominator'][:, :, order] = den[..., None].expand_as(contribution).detach().cpu()
                seen.add(layer)
            handles.append(down.register_forward_hook(hook))
        yield result
        if seen != set(groups):
            raise ValueError('CETT capture incomplete')
    finally:
        for handle in reversed(handles):
            handle.remove()


def parameter_evidence(model):
    return tuple((name, id(value), value.data_ptr(), value._version, str(value.dtype), str(value.device))
                 for name, value in model.named_parameters())


def run_arm(model, request, coordinates, stats, arm, *, efficient_gpu=True, include_logits=False, weight_norms=None):
    import torch

    from attention_probe.interventions import AttentionPulse, CoordinateCapture
    device = next(model.parameters()).device
    ids = request['ids'].to(device)
    attention = (torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION)
                 if efficient_gpu else contextlib.nullcontext())
    started = time.monotonic()
    with (torch.inference_mode(), attention,
          CoordinateCapture(model, coordinates, request['capture_positions'],
                            residual_layers=tuple(layer for layer in (3, 7, 14, 27) if layer < len(model.model.layers)),
                            expected_count=len(coordinates)) as capture,
          cett_tap(model, coordinates, request['capture_positions'], weight_norms) as contributions,
          AttentionPulse(model, arm['block'], request['pulse_positions'], arm['beta'], arm['mode'],
                         seed=SEED + int(request['request_id'][1:])) as pulse):
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        if not bool(torch.isfinite(output.logits).all()):
            raise ValueError('Nonfinite model logits')
        selected_logits = output.logits[:, request['answer_logit_positions']].float()
        targets = torch.tensor(request['answer_ids'], device=device)[None, :, None]
        logprob = selected_logits.log_softmax(-1).gather(-1, targets).squeeze(-1).cpu()
        raw = capture.values
        cett = contributions['cett']
        mu = torch.tensor(stats['refw']['c_mu'], dtype=torch.float32)
        sd = torch.tensor(stats['refw']['c_sd'], dtype=torch.float32).clamp_min(1e-6)
        zcett = (cett - mu) / sd
        result = {'arm': arm, 'answer_token_logprobs': logprob.flatten().tolist(),
                  'answer_logprob_sum': float(logprob.sum()), 'answer_logprob_mean': float(logprob.mean()),
                  'pulse_fired': pulse.fired_count, 'capture_fired': capture.fired_count,
                  'requested_delta_norms': pulse.requested_delta_norms.tolist(),
                  'actual_delta_norms': pulse.actual_delta_norms.tolist(),
                  'elapsed_seconds': time.monotonic() - started}
        arrays = {'raw_z': raw.numpy(), 'cett': cett.numpy(), 'zcett': zcett.numpy(),
                  'down_output_norm_raw': contributions['denominator'].numpy(),
                  'gold_token_logprob': logprob.numpy()}
        for layer, value in capture.residuals.items():
            arrays[f'residual_layer_{layer}'] = value.numpy()
        if include_logits:
            arrays['technical_logits'] = selected_logits.detach().cpu().numpy()
    return result, arrays


def plan_document(reference, stats, items):
    import attention_probe.reference as reference_module
    module_root = Path(reference_module.__file__).parent
    module_hashes = {name: sha_file(module_root / name) for name in ('reference.py', 'interventions.py', 'fixed_refw.py')}
    return {'schema': 'qwen3_h286_short_pilot_v1', 'model': reference.model.to_dict(),
            'hn_sha256': reference.hn_sha256, 'policy_sha256': reference.policy_sha256,
            'refw_sha256': REFW_SHA256, 'refw_source_sha256': stats['source_hn_meta_sha256'],
            'coordinates': [list(pair) for pair in reference.coordinates], 'seed': SEED,
            'arms': list(ARMS), 'items': items, 'items_sha256': ITEMS_SHA256,
            'helper_sha256': sha_file(__file__), 'module_sha256': module_hashes,
            'dtype': 'float32', 'tf32': False, 'attention': 'sdpa/EFFICIENT_ATTENTION',
            'transformers': TRANSFORMERS, 'thinking_enabled': False,
            'claim': 'externally supplied draft-checking proxy; not own prior output or proof of self-monitoring; no held-out confirmatory test',
            'candidate_block': '14 zero-based, prespecified candidate; not a proven bottleneck',
            'cett': 'abs(z)*norm(Wdown_col)/max(norm(down(z)),1e-9)',
            'zcett': '(CETT-fixed_REF_W_c_mu)/max(fixed_REF_W_c_sd,1e-6)'}


def verified_request(output, request_id, plan_hash):
    directory = output / 'requests' / request_id
    if not directory.exists():
        return None
    record = json.loads((directory / 'result.json').read_text())
    if (record.get('state') != 'complete' or record.get('plan_sha256') != plan_hash
            or record.get('request_id') != request_id or len(record.get('arms', [])) != 5
            or [row['arm'] for row in record['arms']] != list(ARMS)):
        raise ValueError('Existing request does not match fixed all-five-arm plan: ' + request_id)
    proof = record.get('technical_checks')
    if not isinstance(proof, dict) or proof.get('state') != 'passed' or proof.get('plan_sha256') != plan_hash:
        raise ValueError('Request lacks bound technical-check evidence: ' + request_id)
    if sha_file(directory / 'arrays.npz') != record['npz_sha256']:
        raise ValueError('Existing request array SHA mismatch: ' + request_id)
    return record


def restore_resume(resume_zip, output, plan_hash):
    if not resume_zip:
        return
    with zipfile.ZipFile(resume_zip) as bundle:
        source_plan = json.loads(bundle.read('run_plan.json'))
        if hashlib.sha256(canonical(source_plan)).hexdigest() != plan_hash:
            raise ValueError('Resume ZIP belongs to a different immutable plan/reference')
        for index in range(16):
            request_id = f'r{index:03d}'
            prefix = f'requests/{request_id}/'
            names = [prefix + 'result.json', prefix + 'arrays.npz']
            present = [name in bundle.namelist() for name in names]
            if not any(present):
                continue
            if not all(present):
                raise ValueError('Resume ZIP contains an incomplete request chunk')
            destination = output / 'requests' / request_id
            if destination.exists():
                verified_request(output, request_id, plan_hash)
                if any((destination / Path(name).name).read_bytes() != bundle.read(name) for name in names):
                    raise ValueError('Resume ZIP conflicts with existing complete request; no overwrite')
                continue
            parent = destination.parent
            parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix='.' + request_id + '_', dir=parent))
            try:
                for name in names:
                    if bundle.getinfo(name).file_size > 128 * (1 << 20):
                        raise ValueError('Oversized request chunk in resume ZIP')
                    (temporary / Path(name).name).write_bytes(bundle.read(name))
                record = json.loads((temporary / 'result.json').read_text())
                if record.get('plan_sha256') != plan_hash or sha_file(temporary / 'arrays.npz') != record.get('npz_sha256'):
                    raise ValueError('Resume request failed hash/plan verification')
                os.rename(temporary, destination)
                verified_request(output, request_id, plan_hash)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)


def write_request(output, request, results, arrays, plan_hash, hardware, technical_checks=None):
    import numpy as np
    destination = output / 'requests' / request['request_id']
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError('Refusing to overwrite existing request outputs')
    temporary = Path(tempfile.mkdtemp(dir=destination.parent, prefix='.' + destination.name + '_'))
    try:
        np.savez_compressed(temporary / 'arrays.npz',
                            **{key: np.stack([row[key] for row in arrays]) for key in arrays[0]})
        with (temporary / 'arrays.npz').open('rb') as stream:
            os.fsync(stream.fileno())
        public_request = {key: value for key, value in request.items() if key != 'ids'}
        record = {'state': 'complete', 'request_id': request['request_id'], 'request': public_request,
                  'plan_sha256': plan_hash, 'npz_sha256': sha_file(temporary / 'arrays.npz'), 'arms': results,
                  'hardware': hardware, 'technical_checks': technical_checks, 'completed_utc': dt.datetime.now(dt.UTC).isoformat()}
        atomic_json(temporary / 'result.json', record)
        os.rename(temporary, destination)  # Both files become visible as one paired-request checkpoint.
        return record
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def run(hn_set, policy, refw, items_path, out_dir, cache_dir=None, resume_zip=None):
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    status = {'state': 'running', 'stage': 'reference', 'completed_requests': 0,
              'confirmatory_eligible': False, 'scope': 'original_fixed_H286_exploratory_pilot'}
    metrics = {'pretrained_model_run': False, 'checks': {}, 'no_training': True}
    started = time.monotonic()
    def save():
        atomic_json(output / 'status.json', status)
        atomic_json(output / 'metrics.json', metrics)
    save()
    try:
        reference, stats, items = load_inputs(hn_set, policy, refw, items_path)
        plan = plan_document(reference, stats, items)
        plan_hash = hashlib.sha256(canonical(plan)).hexdigest()
        existing = output / 'run_plan.json'
        if existing.exists() and existing.read_bytes() != canonical(plan) + b'\n':
            raise ValueError('Existing output uses a different plan; choose a fresh directory')
        if not existing.exists():
            atomic_json(existing, plan)
        restore_resume(resume_zip, output, plan_hash)
        done = [verified_request(output, f'r{index:03d}', plan_hash) for index in range(len(items))]
        status['completed_requests'] = sum(item is not None for item in done)
        if all(item is not None for item in done):
            metrics['technical_checks_from_completed_requests'] = [row.get('technical_checks') for row in done]
            metrics['checks_executed_this_attempt'] = False
            if any(not isinstance(row.get('technical_checks'), dict) or row['technical_checks'].get('state') != 'passed'
                   or row['technical_checks'].get('plan_sha256') != plan_hash for row in done):
                raise ValueError('Completed resume lacks the bound technical-check evidence')
            status.update(state='complete', stage='complete', resumed_without_gpu=True)
            return status
        from importlib.metadata import version

        import torch
        import transformers
        if transformers.__version__ != TRANSFORMERS or version('accelerate') != ACCELERATE:
            raise ValueError('Exact transformers5.22.2 and accelerate1.16.0 are required')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        torch.manual_seed(SEED)
        disable_bar = getattr(transformers.utils.logging, 'disable_progress_bar', None)
        if disable_bar is not None:
            disable_bar()
        metrics.update(torch=torch.__version__, transformers=transformers.__version__, python=platform.python_version())
        status['stage'] = 'hardware_preflight'
        cache = Path(cache_dir or Path.home() / '.cache' / 'h286_pilot')
        hardware = hardware_preflight(torch, cache)
        metrics['hardware'] = hardware
        status['stage'] = 'pinned_download_and_load'
        save()
        os.environ['HF_DEACTIVATE_ASYNC_LOAD'] = '1'
        print('Status: Original Qwen3-1.7B laden, exakte Dateien prüfen; FP32/SDPA.', flush=True)
        tokenizer, model = load_model(reference, cache)
        metrics['weights_loaded'] = True
        hardware.update(torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version(), compute_capability=list(torch.cuda.get_device_capability(0)), transformers=transformers.__version__, attention='sdpa/EFFICIENT_ATTENTION', dtype='float32', tf32=False)
        groups = defaultdict(list)
        for layer, neuron in reference.coordinates:
            groups[layer].append(neuron)
        with torch.inference_mode():
            weight_norms = {layer: model.model.layers[layer].mlp.down_proj.weight.float().norm(dim=0)[columns]
                            for layer, columns in groups.items()}
        evidence = parameter_evidence(model)
        first_index = next(index for index, result in enumerate(done) if result is None)
        warm = prepare_request(tokenizer, first_index, items[first_index])
        ids = warm['ids'].to('cuda:0')
        status['stage'] = 'warmup'
        save()
        with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
            model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)  # untimed warmup
            baseline = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits.detach().cpu()
        metrics['pretrained_model_run'] = True
        status['stage'] = 'technical_checks'
        print('Status: Zwei technische Prüfungen vor der festen 16-Request-Auswertung.', flush=True)
        _sham, sham_arrays = run_arm(model, warm, reference.coordinates, stats, ARMS[0], include_logits=True, weight_norms=weight_norms)
        selected_baseline = baseline[:, warm['answer_logit_positions']].float()
        observed = torch.from_numpy(sham_arrays['technical_logits'])
        delta = (selected_baseline - observed).abs()
        metrics['checks']['sham'] = {'bitwise_equal': torch.equal(selected_baseline, observed),
                                     'max_abs_logit_difference': float(delta.max()), 'rtol': 1e-5, 'atol': 1e-6}
        if not torch.allclose(selected_baseline, observed, rtol=1e-5, atol=1e-6):
            raise ValueError('Technical sham logit check failed; exploratory queue not started')
        pulse_result, pulse_arrays = run_arm(model, warm, reference.coordinates, stats, ARMS[1], weight_norms=weight_norms)
        import numpy as np
        upstream = [index for index, (layer, _) in enumerate(reference.coordinates) if layer < 14]
        earlier = [index for index, position in enumerate(warm['capture_positions']) if position < warm['pulse_positions'][0]]
        upstream_equal = np.array_equal(sham_arrays['raw_z'][..., upstream], pulse_arrays['raw_z'][..., upstream])
        earlier_equal = np.array_equal(sham_arrays['raw_z'][:, earlier], pulse_arrays['raw_z'][:, earlier])
        metrics['checks']['causality'] = {'upstream_coordinates': len(upstream), 'upstream_raw_equal': upstream_equal,
                                           'earlier_positions': len(earlier), 'earlier_raw_equal': earlier_equal}
        if not upstream_equal or not earlier_equal:
            raise ValueError('Technical causal guard failed; exploratory queue not started')
        metrics['checks']['pulse_and_capture'] = {'pulse_once': pulse_result['pulse_fired'] == 1,
                                                  'capture_once': pulse_result['capture_fired'] == 1,
                                                  'frozen_parameter_evidence': parameter_evidence(model) == evidence}
        if parameter_evidence(model) != evidence:
            raise ValueError('Model parameter identity/storage/version counters changed')
        metrics['checks_executed_this_attempt'] = True
        technical_checks = {'state': 'passed', 'plan_sha256': plan_hash, 'checks': metrics['checks'],
                            'hardware': hardware, 'executed_utc': dt.datetime.now(dt.UTC).isoformat()}
        measured = []
        for index, item in enumerate(items):
            if done[index] is not None:
                continue
            request = prepare_request(tokenizer, index, item)
            status.update(stage='exploratory_requests', current_request=request['request_id'])
            save()
            print(f"Request {index + 1}/16: {request['request_id']} — fünf feste Arme", flush=True)
            t0 = time.monotonic()
            order = list(range(len(ARMS)))
            random.Random(SEED + index).shuffle(order)
            request['execution_order'] = [ARMS[arm_index]['name'] for arm_index in order]
            results, array_rows = [None] * len(ARMS), [None] * len(ARMS)
            cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            for arm_index in order:
                result, arrays = run_arm(model, request, reference.coordinates, stats, ARMS[arm_index], weight_norms=weight_norms)
                results[arm_index], array_rows[arm_index] = result, arrays
            if not torch.equal(cpu_rng, torch.random.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state()):
                raise ValueError('Global RNG changed during fixed five-arm request')
            if parameter_evidence(model) != evidence:
                raise ValueError('Frozen parameter evidence changed; request is not published complete')
            record = write_request(output, request, results, array_rows, plan_hash, hardware, technical_checks)
            done[index] = record
            measured.append(time.monotonic() - t0)
            status['completed_requests'] = sum(row is not None for row in done)
            remaining = len(items) - status['completed_requests']
            eta_seconds = sum(measured) / len(measured) * remaining
            finish = dt.datetime.now(ZoneInfo('Europe/Berlin')) + dt.timedelta(seconds=eta_seconds)
            status.update(estimated_remaining_seconds=eta_seconds, estimated_finish_berlin=finish.isoformat())
            save()
            print(f"Gespeichert {status['completed_requests']}/16; Restschätzung {eta_seconds / 60:.1f} min; Berlin {finish:%H:%M}", flush=True)
            if remaining and (eta_seconds > 3600 or finish.date() > dt.datetime.now(ZoneInfo('Europe/Berlin')).date()):
                status.update(state='paused', stage='safe_request_checkpoint',
                              reason='Runtime estimate exceeds one hour or today; resume ZIP keeps paired requests together')
                break
        else:
            status.update(state='complete', stage='complete')
        metrics['weights_unchanged_evidence'] = 'same parameter identities, storage pointers, version counters; no training; no full tensor rehash'
        metrics['hardware_comparison_warning'] = 'Each complete paired request stays on one GPU; resumed chunks can differ by GPU/software.'
    except BaseException as exc:  # noqa: BLE001 — preserve diagnostics also on interruption
        status.update(state='failed', error_type=type(exc).__name__, error=str(exc))
        (output / 'traceback.txt').write_text(traceback.format_exc())
        print('HARD FAILURE: ' + str(exc), flush=True)
    finally:
        metrics['elapsed_seconds'] = time.monotonic() - started
        save()
        archive(output)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hn-set', required=True)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--refw', required=True)
    parser.add_argument('--items', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--cache-dir')
    parser.add_argument('--resume-zip')
    args = parser.parse_args()
    result = run(args.hn_set, args.policy, args.refw, args.items, args.out_dir, args.cache_dir, args.resume_zip)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['state'] in ('complete', 'paused') else 1)
