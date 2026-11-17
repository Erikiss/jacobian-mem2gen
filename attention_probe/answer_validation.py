"""Prompt-only greedy answer validation; the attention pulse ends after prefill.

The old teacher-forcing runner is unchanged. This module retains its fixed REF-W
CETT definition but never supplies request['ids'], input_ids or answer_ids to
generation. All saved activation arrays contain prompt positions only.
"""
from __future__ import annotations

import contextlib
import hashlib
import re
import time
from numbers import Integral

import torch

from .interventions import AttentionPulse, CoordinateCapture, _cache_length

SEED = 20261117
MAX_NEW_TOKENS = 48
TERMINAL_PUNCTUATION = ".!?,;:"


def normalize_answer(text: str) -> str:
    """Only case, outer whitespace and one simple terminal punctuation mark."""
    if not isinstance(text, str):
        raise TypeError('Answer text must be a string')
    result = text.strip().casefold()
    if result and result[-1] in TERMINAL_PUNCTUATION:
        result = result[:-1].rstrip()
    return result


def classify_answer(generated: str, expected: str, *, truncated: bool = False) -> dict:
    """Strict surface comparison, never extract an answer from an explanation.

    Numeric values are not parsed or rounded: '15.0', '+15' and '015' do not
    match '15'. A capped output is always classified as truncated, even if its
    current surface matches. Noncanonical is not a semantic-incorrect verdict.
    """
    normalized, gold = normalize_answer(generated), normalize_answer(expected)
    if not gold:
        raise ValueError('Expected answer must not be empty')
    surface = ('exact' if generated == expected else
               'format_equivalent' if normalized == gold else 'noncanonical')
    return {"classification": 'truncated' if truncated else surface,
                "surface_classification": surface, "normalized_generated": normalized,
                "normalized_expected": gold, "truncated": bool(truncated),
                "normalization": 'casefold; outer whitespace; at most one final . ! ? , ; :',
                "interpretation": 'Surface comparison only; noncanonical is not a semantic-wrong label.'}


def _decode(tokenizer, ids):
    return tokenizer.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _eos_ids(model, vocab_size):
    configured = getattr(getattr(model, 'generation_config', None), 'eos_token_id', None)
    source = 'generation_config'
    if configured is None:
        configured = getattr(model.config, 'eos_token_id', None)
        source = 'config'
    if configured is None:
        return (), 'none'
    values = [configured] if isinstance(configured, Integral) else list(configured)
    if any(isinstance(v, bool) or not isinstance(v, Integral) or not 0 <= v < vocab_size for v in values):
        raise ValueError('EOS IDs must be integer vocabulary IDs')
    return tuple(sorted({int(v) for v in values})), source


def _cett_context(model, coordinates, positions, weight_norms):
    # The source bundle installs the old runner at its root. A repository checkout
    # can instead resolve it as the notebooks namespace; neither path changes it.
    try:
        from qwen3_h286_runtime import cett_tap
    except ModuleNotFoundError as error:
        if error.name != 'qwen3_h286_runtime':
            raise
        from notebooks.qwen3_h286_runtime import cett_tap
    return cett_tap(model, coordinates, positions, weight_norms)


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run_generation_arm(model, tokenizer, request, coordinates, stats, arm, *,
                       efficient_gpu=True, weight_norms=None, max_new_tokens=48,
                       top_k=10):
    """Return a JSON-compatible record and fixed-shape prefill numpy arrays.

    arm=None disables the pulse; passive prefill capture still runs. Only that
    prefill has hooks. Decode uses its returned cache and newly selected tokens.
    Selected EOS is logged in steps/selected_token_ids but excluded from emitted
    generated_token_ids and generated_text. The 48-step maximum includes EOS.
    """
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, Integral) or not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
        raise ValueError('max_new_tokens must be an integer from 1 through 48')
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or not 1 <= top_k <= 10:
        raise ValueError('top_k must be an integer from 1 through 10')
    prompt = list(request['prompt_ids'])  # Sole source of generation input IDs.
    vocab_size = int(model.config.vocab_size)
    if not prompt or any(isinstance(v, bool) or not isinstance(v, Integral) or not 0 <= v < vocab_size for v in prompt):
        raise ValueError('prompt_ids must be a nonempty list of integer vocabulary IDs')
    request_id = request['request_id']
    if not isinstance(request_id, str) or re.fullmatch(r'r[0-9]+', request_id) is None:
        raise ValueError('request_id must have the original rNNN form')
    seed = SEED + int(request_id[1:])
    coordinates = tuple(tuple(pair) for pair in coordinates)
    device = next(model.parameters()).device
    if any(parameter.is_floating_point() and (parameter.dtype != torch.float32 or parameter.device != device)
           for parameter in model.parameters()):
        raise ValueError('Generation validation requires all floating parameters in FP32 on one device')
    if efficient_gpu:
        if device.type != 'cuda':
            raise ValueError('efficient_gpu=True requires a CUDA model')
        if getattr(model.config, '_attn_implementation', None) != 'sdpa':
            raise ValueError('The GPU validation requires the sdpa attention implementation')
        if torch.backends.cuda.matmul.allow_tf32:
            raise ValueError('Disable TF32 before full-FP32 generation validation')
    model.eval()
    prompt_length = len(prompt)
    positions = list(range(max(0, prompt_length-4), prompt_length))
    pulse_positions = [prompt_length-1]
    mu = torch.as_tensor(stats['refw']['c_mu'], dtype=torch.float32)
    sd = torch.as_tensor(stats['refw']['c_sd'], dtype=torch.float32)
    if mu.shape != (len(coordinates),) or sd.shape != mu.shape or not bool(torch.isfinite(mu).all() and torch.isfinite(sd).all()) or bool((sd < 0).any()):
        raise ValueError('Fixed REF-W means/SDs must match the finite ordered coordinates')
    sd = sd.clamp_min(1e-6)
    eos_ids, eos_source = _eos_ids(model, vocab_size)
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    attention = (torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION)
                 if efficient_gpu else contextlib.nullcontext())
    pulse = None
    steps, emitted, selected_ids = [], [], []
    decode_forwards = 0
    _sync(device)
    started = time.monotonic()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False), attention:
        with contextlib.ExitStack() as hooks:
            capture = hooks.enter_context(CoordinateCapture(model, coordinates, positions,
                residual_layers=tuple(l for l in (3, 7, 14, 27) if l < len(model.model.layers)),
                expected_count=len(coordinates)))
            contributions = hooks.enter_context(_cett_context(model, coordinates, positions, weight_norms))
            if arm is not None:
                pulse = hooks.enter_context(AttentionPulse(model, arm['block'], pulse_positions,
                    arm['beta'], arm['mode'], seed=seed))
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
            cache = output.past_key_values
            if cache is None:
                raise RuntimeError('Prefill did not return a cache')
            # Capture arrays are copied to CPU by the existing read-only hooks.
            raw = capture.values
            cett = contributions['cett']
            zcett = (cett-mu)/sd
            if not all(bool(torch.isfinite(v).all()) for v in (raw, cett, zcett, contributions['denominator'])):
                raise ValueError('Nonfinite prefill raw/CETT/reference-normalized capture')
            arrays = {"raw_z": raw.numpy(), "cett": cett.numpy(), "zcett": zcett.numpy(),
                          "down_output_norm_raw": contributions['denominator'].numpy()}
            arrays.update({f'residual_layer_{l}': value.numpy() for l, value in capture.residuals.items()})
            logits = output.logits[0, -1].float()
        # ExitStack has removed every pulse/capture hook before any decode call.
        _sync(device)
        prefill_seconds = time.monotonic()-started
        if not bool(torch.isfinite(logits).all()):
            raise ValueError('Nonfinite prefill logits')
        first_logits = logits.detach().cpu().clone()
        first_logits_sha = hashlib.sha256(first_logits.numpy().tobytes()).hexdigest()
        stop_reason = 'max_new_tokens'
        for step in range(int(max_new_tokens)):
            if not bool(torch.isfinite(logits).all()):
                raise ValueError('Nonfinite generation logits')
            logprobs = logits.log_softmax(-1)
            chosen = int(torch.argmax(logits))
            values, indices = torch.topk(logprobs, k=min(int(top_k), vocab_size))
            top_tokens = [{"token_id": int(t), "text": _decode(tokenizer, [int(t)]), "logprob": float(v)}
                          for t, v in zip(indices.detach().cpu().tolist(), values.detach().cpu().tolist())]
            is_eos = chosen in eos_ids
            steps.append({"step": step, "phase": 'prefill' if step == 0 else 'decode',
                "logit_position": prompt_length-1+step, "selected_token_id": chosen,
                "selected_token_text": _decode(tokenizer, [chosen]), "selected_token_logprob": float(logprobs[chosen]),
                "is_eos": is_eos, "top_tokens": top_tokens})
            selected_ids.append(chosen)
            if is_eos:
                stop_reason = 'eos'
                break
            emitted.append(chosen)
            if step+1 == max_new_tokens:
                break
            # Only the model's own last selected token is passed to cached decode.
            current = torch.tensor([[chosen]], dtype=torch.long, device=device)
            mask = torch.ones((1, prompt_length+len(emitted)), dtype=torch.long, device=device)
            output = model(input_ids=current, attention_mask=mask, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            if cache is None:
                raise RuntimeError('Decode lost its cache')
            logits = output.logits[0, -1].float()
            decode_forwards += 1
        _sync(device)
        elapsed = time.monotonic()-started
    generated_text = _decode(tokenizer, emitted)
    truncated = stop_reason == 'max_new_tokens'
    # Gold is consulted only after generation has ended, solely for diagnostics.
    answer = request['answer']
    classification = classify_answer(generated_text, answer, truncated=truncated)
    gold_ids = tokenizer(answer, add_special_tokens=False)['input_ids']
    if not gold_ids or not isinstance(gold_ids[0], Integral) or not 0 <= gold_ids[0] < vocab_size:
        raise ValueError('Expected answer must tokenize to a valid first vocabulary ID')
    gold_first = int(gold_ids[0])
    first_logprobs = first_logits.log_softmax(-1)
    first_gold = {"token_id": gold_first, "text": _decode(tokenizer, [gold_first]),
        "rank": int((first_logits > first_logits[gold_first]).sum())+1,
        "rank_rule": 'one plus number of strictly higher logits; ties can share rank',
        "logprob": float(first_logprobs[gold_first]), "is_top1": int(torch.argmax(first_logits)) == gold_first}
    record = {"schema": 'h286_prompt_only_generation_v1', "request_id": request_id,
        "arm": dict(arm) if arm is not None else None, "seed": seed, "prompt_length": prompt_length,
        "generation_input_source": 'request.prompt_ids only; no teacher-forced answer input',
        "prefill_capture_positions": positions, "pulse_positions": pulse_positions if pulse is not None else [],
        "max_new_tokens": int(max_new_tokens), "top_k": int(top_k), "eos_token_ids": list(eos_ids), "eos_source": eos_source,
        "generated_token_ids": emitted, "generated_text": generated_text, "selected_token_ids": selected_ids,
        "eos_convention": 'selected EOS is recorded in steps/selected_token_ids but not emitted text/IDs',
        "steps": steps, "num_steps": len(steps), "num_generated_tokens": len(emitted), "stop_reason": stop_reason,
        "truncated": truncated, "classification": classification['classification'], "classification_details": classification,
        "first_token_gold": first_gold, "first_step_logits_sha256": first_logits_sha,
        "pulse_fired": 0 if pulse is None else pulse.fired_count, "capture_fired": capture.fired_count,
        "prefill_forward_count": 1, "decode_forward_count": decode_forwards, "final_cache_length": _cache_length(cache),
        "requested_delta_norms": None if pulse is None else pulse.requested_delta_norms.tolist(),
        "actual_delta_norms": None if pulse is None else pulse.actual_delta_norms.tolist(),
        "elapsed_seconds": elapsed, "prefill_seconds": prefill_seconds, "decode_and_logging_seconds": elapsed-prefill_seconds,
        "timing_note": 'Synchronized wall time includes capture and top-k logging; not energy telemetry.',
        "dtype": 'float32', "attention": 'sdpa/EFFICIENT_ATTENTION' if efficient_gpu else str(getattr(model.config, '_attn_implementation', None)),
        "capture_scope": 'last min(4,prompt_length) PROMPT positions; no generated-token activation capture'}
    record['baseline_kind'] = 'no_pulse_with_passive_capture' if arm is None else 'prefill_pulse_with_passive_capture'
    return record, arrays
