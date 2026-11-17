"""Fixed H286 answer-format validation: original teacher forcing plus greedy KV decode.

This is a separate plan/output format. Original reference files, FP32 numerics,
teacher-forced measurements and five arms are reused without changing the old
runtime. The 15-minute budget covers the experimental queue, checked only after
complete all-five-arm request checkpoints; model loading/checks are additional.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import time
import traceback
import zipfile
from pathlib import Path

import qwen3_h286_runtime as base

SCHEMA = "qwen3_h286_answer_validation_v1"
PROOF_SCHEMA = "qwen3_h286_answer_validation_checks_v1"
MAX_NEW_TOKENS, TOP_K, QUEUE_BUDGET_SECONDS = 48, 10, 900
PROOF_CHECKS = ("teacher_forced_sham", "teacher_forced_causality", "generation_uninstrumented_prefill",
                "generation_no_pulse_vs_sham", "pulse_capture_counts", "parameters_and_hook_cleanup")


def generation_module():
    from attention_probe import answer_validation
    return answer_validation


def plan_document(reference, stats, items):
    previous = base.plan_document(reference, stats, items)
    plan = dict(previous)
    plan.update(schema=SCHEMA, purpose="original_fixed_H286_teacher_forcing_and_answer_format_validation",
        previous_plan_sha256=hashlib.sha256(base.canonical(previous)).hexdigest(),
        base_runtime_sha256=base.sha_file(base.__file__), helper_sha256=base.sha_file(__file__),
        generation_module_sha256=base.sha_file(generation_module().__file__),
        generation={"max_new_tokens": MAX_NEW_TOKENS, "top_k": TOP_K, "decoding": "greedy argmax",
            "input": "prompt_ids only; never the teacher-forced target input",
            "cache": "fresh cache per arm; one prefill pulse; hooks removed before cached decode",
            "eos": "selected EOS counts toward cap; EOS is logged but not emitted",
            "classification": "exact / format_equivalent / noncanonical / truncated; not semantic accuracy"},
        array_contract={"tf": "unchanged original teacher-forced arrays","gen": "prompt-prefill activation arrays only",
                            "variable_generation": "JSON only; no object/ragged NPZ arrays"},
        queue_budget={"seconds": QUEUE_BUDGET_SECONDS, "scope": "experimental paired-request queue only",
            "checkpoint": "complete all-five-arm request", "pause": "elapsed >= budget or remaining ETA > budget",
            "limitation": "boundary checked; a slow complete request can exceed the budget; setup is additional"},
        technical_proof_schema=PROOF_SCHEMA)
    plan["module_sha256"] = dict(previous["module_sha256"],
        **{"answer_validation.py": plan["generation_module_sha256"]})
    return plan


def attention_context(torch, efficient_gpu):
    backend = torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION if efficient_gpu else torch.nn.attention.SDPBackend.MATH
    return torch.nn.attention.sdpa_kernel(backend)


def hook_count(model):
    return sum(len(module._forward_hooks) + len(module._forward_pre_hooks) for module in model.modules())


def verify_proof(proof, plan_hash):
    if (not isinstance(proof, dict) or proof.get("schema") != PROOF_SCHEMA
            or proof.get("state") != "passed" or proof.get("plan_sha256") != plan_hash
            or any(proof.get("checks", {}).get(name, {}).get("passed") is not True for name in PROOF_CHECKS)):
        raise ValueError("Missing or incomplete bound answer-validation technical evidence")


def technical_checks(model, tokenizer, request, reference, stats, plan_hash, hardware, *,
                     weight_norms=None, efficient_gpu=True):
    """Native TF/prefill controls; full generation comparison is capture-instrumented no-pulse vs sham."""
    import numpy as np
    import torch
    generation = generation_module().run_generation_arm
    evidence, hooks = base.parameter_evidence(model), hook_count(model)
    checks = {}
    device = next(model.parameters()).device
    with torch.inference_mode(), attention_context(torch, efficient_gpu):
        full_ids = request["ids"].to(device)
        native = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids), use_cache=False).logits
        native_tf = native[:, request["answer_logit_positions"]].detach().float().cpu()
        del native
        tf_sham, a_sham = base.run_arm(model, request, reference.coordinates, stats, base.ARMS[0],
            efficient_gpu=efficient_gpu, include_logits=True, weight_norms=weight_norms)
        observed = torch.from_numpy(a_sham["technical_logits"])
        checks["teacher_forced_sham"] = {"passed": bool(torch.equal(native_tf, observed)),
            "bitwise_equal": bool(torch.equal(native_tf, observed)), "max_abs_logit_difference": float((native_tf-observed).abs().max())}
        tf_pulse, a_pulse = base.run_arm(model, request, reference.coordinates, stats, base.ARMS[1],
            efficient_gpu=efficient_gpu, weight_norms=weight_norms)
        upstream = [i for i, (layer, _) in enumerate(reference.coordinates) if layer < 14]
        earlier = [i for i, position in enumerate(request["capture_positions"]) if position < request["pulse_positions"][0]]
        up_equal = np.array_equal(a_sham["raw_z"][..., upstream], a_pulse["raw_z"][..., upstream])
        early_equal = np.array_equal(a_sham["raw_z"][:, earlier], a_pulse["raw_z"][:, earlier])
        checks["teacher_forced_causality"] = {"passed": bool(up_equal and early_equal),
            "upstream_coordinates": len(upstream), "upstream_raw_equal": up_equal, "earlier_positions": len(earlier), "earlier_raw_equal": early_equal}
        prompt = torch.tensor([request["prompt_ids"]], dtype=torch.long, device=device)
        plain_output = model(input_ids=prompt, attention_mask=torch.ones_like(prompt), use_cache=True)
        plain_logits = plain_output.logits[0, -1].detach().float().cpu().contiguous()
        plain_hash = hashlib.sha256(plain_logits.numpy().tobytes()).hexdigest()
        del plain_output, plain_logits
        no_pulse, no_pulse_arrays = generation(model, tokenizer, request, reference.coordinates, stats, None,
            efficient_gpu=efficient_gpu, weight_norms=weight_norms, max_new_tokens=MAX_NEW_TOKENS, top_k=TOP_K)
        sham, sham_arrays = generation(model, tokenizer, request, reference.coordinates, stats, base.ARMS[0],
            efficient_gpu=efficient_gpu, weight_norms=weight_norms, max_new_tokens=MAX_NEW_TOKENS, top_k=TOP_K)
    prefix_equal = plain_hash == no_pulse["first_step_logits_sha256"] == sham["first_step_logits_sha256"]
    checks["generation_uninstrumented_prefill"] = {"passed": prefix_equal,
        "plain_vs_captured_first_logits_bitwise_equal": prefix_equal, "first_step_logits_sha256": plain_hash}
    validate_generation(sham, request, base.ARMS[0])
    sequence_keys = ("generated_token_ids", "selected_token_ids", "generated_text", "steps", "stop_reason", "truncated", "first_token_gold",
                     "prompt_length", "prefill_forward_count", "decode_forward_count", "final_cache_length")
    seq_equal = all(no_pulse[key] == sham[key] for key in sequence_keys)
    arrays_equal = set(no_pulse_arrays) == set(sham_arrays) and all(
        np.array_equal(no_pulse_arrays[key], sham_arrays[key]) for key in no_pulse_arrays)
    checks["generation_no_pulse_vs_sham"] = {"passed": bool(seq_equal and arrays_equal),
        "tokens_and_step_logprobs_equal": seq_equal, "prefill_arrays_equal": arrays_equal,
        "control_scope": "Both full-decode controls have passive prefill capture; only the separate native prefill is fully unhooked."}
    counts_ok = (tf_sham["pulse_fired"] == tf_sham["capture_fired"] == tf_pulse["pulse_fired"] == tf_pulse["capture_fired"] == 1
        and no_pulse["pulse_fired"] == 0 and sham["pulse_fired"] == 1
        and no_pulse["capture_fired"] == sham["capture_fired"] == 1
        and no_pulse["prefill_forward_count"] == sham["prefill_forward_count"] == 1)
    finite = all(np.isfinite(value).all() for recorded in (a_sham, a_pulse, no_pulse_arrays, sham_arrays)
                 for value in recorded.values())
    checks["pulse_capture_counts"] = {"passed": bool(counts_ok and finite), "all_recordings_finite": bool(finite)}
    frozen, clean = base.parameter_evidence(model) == evidence, hook_count(model) == hooks
    checks["parameters_and_hook_cleanup"] = {"passed": bool(frozen and clean),
        "frozen_parameter_evidence": frozen, "hook_count_before": hooks, "hook_count_after": hook_count(model)}
    proof = {"schema": PROOF_SCHEMA, "state": "passed", "plan_sha256": plan_hash, "checks": checks,
                 "hardware": hardware, "executed_utc": base.dt.datetime.now(base.dt.UTC).isoformat()}
    verify_proof(proof, plan_hash)
    return proof


def validate_generation(record, request, arm):
    if (record.get("schema") != "h286_prompt_only_generation_v1" or record.get("arm") != arm
        or record.get("request_id") != request["request_id"] or record.get("max_new_tokens") != MAX_NEW_TOKENS
        or record.get("top_k") != TOP_K or record.get("prompt_length") != len(request["prompt_ids"])
        or record.get("prefill_forward_count") != 1 or record.get("pulse_fired") != 1 or record.get("capture_fired") != 1):
        raise ValueError("Generation record does not match the fixed prefill/arm protocol")
    positions = list(range(max(0, len(request["prompt_ids"])-4), len(request["prompt_ids"])))
    if record.get("prefill_capture_positions") != positions or record.get("pulse_positions") != [len(request["prompt_ids"])-1]:
        raise ValueError("Generation capture must use prompt positions only")
    selected, generated, steps = (record.get(key) for key in ("selected_token_ids", "generated_token_ids", "steps"))
    if (not isinstance(selected, list) or not 1 <= len(selected) <= MAX_NEW_TOKENS
        or not isinstance(generated, list) or not isinstance(steps, list) or len(steps) != len(selected)
        or [step.get("selected_token_id") for step in steps] != selected
        or record.get("decode_forward_count") != len(selected)-1
        or record.get("final_cache_length") != len(request["prompt_ids"])+len(selected)-1):
        raise ValueError("Generation token/cache counts are inconsistent")
    eos = record.get("stop_reason") == "eos"
    capped = record.get("stop_reason") == "max_new_tokens"
    if (not (eos or capped) or record.get("truncated") is not capped
        or generated != (selected[:-1] if eos else selected)
        or (capped and len(selected) != MAX_NEW_TOKENS)
        or (eos and not steps[-1].get("is_eos"))
        or (capped and any(step.get("is_eos") for step in steps))):
        raise ValueError("EOS/cap/truncation convention is inconsistent")
    if record.get("classification") not in ("exact", "format_equivalent", "noncanonical", "truncated"):
        raise ValueError("Unknown surface classification")
    if capped and record["classification"] != "truncated":
        raise ValueError("A capped output must be classified as truncated")
    for step in steps:
        if (not math.isfinite(step["selected_token_logprob"]) or not isinstance(step.get("top_tokens"), list)
                or not 1 <= len(step["top_tokens"]) <= TOP_K
                or any(not math.isfinite(token["logprob"]) for token in step["top_tokens"])):
            raise ValueError("Missing or nonfinite generated-token/top-k log probabilities")


def validate_payload(request, results, arrays, coordinate_count):
    import numpy as np
    if len(results) != len(base.ARMS) or len(arrays) != len(base.ARMS):
        raise ValueError("A checkpoint needs all five arms")
    expected_keys = None
    expected_shapes = None
    for arm, result, values in zip(base.ARMS, results, arrays):
        if result.get("arm") != arm or not isinstance(result.get("teacher_forced"), dict) or not isinstance(result.get("generation"), dict):
            raise ValueError("Both teacher-forced and generation branches are required for every arm")
        tf = result["teacher_forced"]
        if tf.get("arm") != arm or tf.get("pulse_fired") != 1 or tf.get("capture_fired") != 1:
            raise ValueError("Teacher-forced pulse/capture evidence is incomplete")
        probabilities = tf.get("answer_token_logprobs", [])
        if len(probabilities) != len(request["answer_ids"]) or not probabilities or any(not math.isfinite(p) for p in probabilities):
            raise ValueError("Teacher-forced answer probabilities are missing/nonfinite")
        validate_generation(result["generation"], request, arm)
        required = {"tf_raw_z", "tf_cett", "tf_zcett", "tf_down_output_norm_raw", "tf_gold_token_logprob",
                    "gen_raw_z", "gen_cett", "gen_zcett", "gen_down_output_norm_raw"}
        required.update(f"{prefix}residual_layer_{layer}" for prefix in ("tf_", "gen_") for layer in (3, 7, 14, 27))
        if not required.issubset(values) or any(not key.startswith(("tf_", "gen_")) for key in values):
            raise ValueError("Missing or unprefixed validation activation arrays")
        shapes = {key: tuple(np.asarray(value).shape) for key, value in values.items()}
        if expected_keys is not None and (set(values) != expected_keys or shapes != expected_shapes):
            raise ValueError("All five arms need identical fixed NPZ keys/shapes")
        expected_keys, expected_shapes = set(values), shapes
        for value in values.values():
            a = np.asarray(value)
            if a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError("All published activation arrays must be finite FP32")
        for prefix, positions in (("tf_", request["capture_positions"]), ("gen_", result["generation"]["prefill_capture_positions"])):
            expected = (1, len(positions), coordinate_count)
            if any(shapes[prefix+key] != expected for key in ("raw_z", "cett", "zcett", "down_output_norm_raw")):
                raise ValueError("Activation shapes do not match the ordered fixed H286 coordinates")
            width = shapes["tf_residual_layer_3"][-1]
            if width < 1 or any(shapes[f"{prefix}residual_layer_{layer}"] != (1, len(positions), width) for layer in (3, 7, 14, 27)):
                raise ValueError("Residual arrays must share hidden width and the correct branch positions")
        if (shapes["tf_gold_token_logprob"] != (1, len(request["answer_ids"]))
            or not np.array_equal(values["tf_gold_token_logprob"], np.asarray(probabilities, dtype=np.float32)[None, :])):
            raise ValueError("Teacher-forced saved token probabilities disagree with JSON")


def compute_request(model, tokenizer, request, reference, stats, *, weight_norms=None, efficient_gpu=True):
    import torch
    runner = generation_module().run_generation_arm
    results, arrays = [None]*5, [None]*5
    order = list(range(5)); random.Random(base.SEED + int(request["request_id"][1:])).shuffle(order)
    request["execution_order"] = [base.ARMS[i]["name"] for i in order]
    request["validation_schema"] = SCHEMA
    for i in order:
        arm = base.ARMS[i]
        with attention_context(torch, efficient_gpu):
            tf, tf_arrays = base.run_arm(model, request, reference.coordinates, stats, arm,
                                         efficient_gpu=efficient_gpu, weight_norms=weight_norms)
            gen, gen_arrays = runner(model, tokenizer, request, reference.coordinates, stats, arm,
                efficient_gpu=efficient_gpu, weight_norms=weight_norms, max_new_tokens=MAX_NEW_TOKENS, top_k=TOP_K)
        results[i] = {"arm": arm, "teacher_forced": tf, "generation": gen}
        arrays[i] = {**{"tf_"+key: value for key, value in tf_arrays.items()},
                     **{"gen_"+key: value for key, value in gen_arrays.items()}}
    validate_payload(request, results, arrays, len(reference.coordinates))
    return results, arrays


def verified_request(output, request_id, plan_hash, coordinate_count):
    import numpy as np
    record = base.verified_request(output, request_id, plan_hash)
    if record is None:
        return None
    request = record["request"]
    if request.get("validation_schema") != SCHEMA:
        raise ValueError("Old teacher-forcing chunks cannot be resumed as answer validation")
    verify_proof(record["technical_checks"], plan_hash)
    with np.load(Path(output)/"requests"/request_id/"arrays.npz", allow_pickle=False) as packed:
        if any(packed[key].shape[0] != 5 for key in packed.files):
            raise ValueError("Saved NPZ lacks the five-arm dimension")
        arrays = [{key: packed[key][i] for key in packed.files} for i in range(5)]
    validate_payload(request, record["arms"], arrays, coordinate_count)
    return record


def restore_resume(path, output, plan_hash, coordinate_count):
    if not path:
        return
    with zipfile.ZipFile(path) as packed:
        document = json.loads(packed.read("run_plan.json"))
        if document.get("schema") != SCHEMA or hashlib.sha256(base.canonical(document)).hexdigest() != plan_hash:
            raise ValueError("Resume ZIP has another validation plan/schema")
    # The old importer copies paired chunks atomically and verifies the bound NPZ
    # digest. Strengthen every imported row before it can be counted as complete.
    base.restore_resume(path, output, plan_hash)
    for i in range(16):
        verified_request(output, f"r{i:03d}", plan_hash, coordinate_count)


def archive(output):
    output = Path(output); target = output/"qwen3_h286_validation_results.zip"
    temporary = output/".qwen3_h286_validation_results.zip.tmp"
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as packed:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path not in (target, temporary) and path.suffix != ".zip" and not any(
                    part.startswith(".") for part in path.relative_to(output).parts):
                packed.write(path, path.relative_to(output))
    os.replace(temporary, target)
    return target


def initialize_engine(reference, cache, *, on_loaded=None):
    from importlib.metadata import version

    import torch
    import transformers
    if transformers.__version__ != base.TRANSFORMERS or version("accelerate") != base.ACCELERATE:
        raise ValueError("Exact transformers5.22.2 and accelerate1.16.0 are required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest"); torch.manual_seed(base.SEED)
    transformers.utils.logging.disable_progress_bar()
    hardware = base.hardware_preflight(torch, cache)
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    print("Status: Original FP32 Qwen3-1.7B laden; feste Referenz und neue Antwortvalidierung.", flush=True)
    tokenizer, model = base.load_model(reference, cache)
    hardware.update(torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version(),
        compute_capability=list(torch.cuda.get_device_capability(0)), transformers=transformers.__version__,
        dtype="float32", tf32=False, attention="sdpa/EFFICIENT_ATTENTION")
    if on_loaded is not None:
        on_loaded(hardware)
    groups = {}
    for layer, channel in reference.coordinates:
        groups.setdefault(layer, []).append(channel)
    with torch.inference_mode():
        norms = {layer:model.model.layers[layer].mlp.down_proj.weight.float().norm(dim=0)[channels]
                 for layer, channels in groups.items()}
    return {"model": model, "tokenizer": tokenizer, "hardware": hardware, "weight_norms": norms}


def run_queue(engine, reference, stats, items, output, plan_hash, proof, done, save, *, clock=None, efficient_gpu=True):
    import torch
    clock = clock or time.monotonic
    started = clock(); measured = []
    model, tokenizer = engine["model"], engine["tokenizer"]
    hardware = engine["hardware"]; evidence = base.parameter_evidence(model)
    verify_proof(proof, plan_hash)
    status = {"state": "running", "stage": "answer_validation_requests", "completed_requests": sum(r is not None for r in done)}
    for i, item in enumerate(items):
        if done[i] is not None:
            continue
        request = base.prepare_request(tokenizer, i, item)
        status.update(current_request=request["request_id"]); save(status)
        print(f"Request {i+1}/16: {request['request_id']} — fünf Arme, je Teacher-Forcing plus freie Greedy-Antwort", flush=True)
        request_started = clock()
        cpu_rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone() if efficient_gpu else None
        result, arrays = compute_request(model, tokenizer, request, reference, stats,
            weight_norms=engine.get("weight_norms"), efficient_gpu=efficient_gpu)
        validate_payload(request, result, arrays, len(reference.coordinates))
        if not torch.equal(cpu_rng, torch.random.get_rng_state()) or (cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state())):
            raise ValueError("Global RNG changed during a paired request")
        if base.parameter_evidence(model) != evidence:
            raise ValueError("Frozen parameter evidence changed; request not published")
        record = base.write_request(output, request, result, arrays, plan_hash, hardware, proof)
        done[i] = record
        measured.append(clock()-request_started)
        remaining = sum(row is None for row in done)
        eta = sum(measured)/len(measured)*remaining
        elapsed = clock()-started
        status.update(completed_requests=len(done)-remaining, experimental_queue_seconds=elapsed,
                      estimated_remaining_seconds=eta)
        save(status)
        print(f"Gespeichert {status['completed_requests']}/16; Restschätzung {eta/60:.1f} min; Queue {elapsed/60:.1f} min", flush=True)
        if remaining and (elapsed >= QUEUE_BUDGET_SECONDS or eta > QUEUE_BUDGET_SECONDS):
            status.update(state="paused",stage="safe_request_checkpoint",
                reason="Experimental queue elapsed or remaining ETA reached 15-minute boundary; all five arms saved together")
            save(status)
            return status
    status.update(state="complete", stage="complete"); save(status)
    return status


def run(hn_set, policy, refw, items_path, out_dir, cache_dir=None, resume_zip=None):
    output = Path(out_dir); output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    status = {"state": "running", "stage": "reference", "completed_requests": 0,
                  "confirmatory_eligible": False, "scope": SCHEMA}
    metrics = {"weights_loaded": False, "weights_loaded_this_attempt": False, "pretrained_model_run": False,
                   "wrapper_attempt_id": os.environ.get("H286_WRAPPER_ATTEMPT"),
                   "checks_executed_this_attempt": False, "checks": {}, "no_training": True,
                   "purpose": "teacher-forcing comparison and prompt-only answer-format validation"}
    def save(extra=None):
        if extra:status.update(extra)
        base.atomic_json(output/"status.json",status); base.atomic_json(output/"metrics.json",metrics)
    save()
    try:
        reference, stats, items = base.load_inputs(hn_set, policy, refw, items_path)
        plan = plan_document(reference, stats, items); plan_hash = hashlib.sha256(base.canonical(plan)).hexdigest()
        path = output/"run_plan.json"
        if path.exists() and path.read_bytes() != base.canonical(plan)+b"\n":
            raise ValueError("Existing output has another plan; old pilot chunks cannot be mixed")
        if not path.exists():base.atomic_json(path, plan)
        restore_resume(resume_zip, output, plan_hash, len(reference.coordinates))
        done = [verified_request(output,f"r{i:03d}",plan_hash,len(reference.coordinates)) for i in range(16)]
        status["completed_requests"] = sum(r is not None for r in done)
        if all(r is not None for r in done):
            metrics.update(retained_request_proofs=[r["technical_checks"] for r in done],
                           previous_complete_requests_verified=True)
            status.update(state="complete",stage="complete",resumed_without_gpu=True)
            return status
        status.update(stage="hardware_preflight_and_pinned_model_load"); save()
        def mark_loaded(hardware):
            metrics.update(weights_loaded=True, weights_loaded_this_attempt=True, hardware=hardware)
            save()
        engine = initialize_engine(reference, Path(cache_dir or Path.home()/".cache/h286_pilot"), on_loaded=mark_loaded)
        metrics.update(weights_loaded=True, weights_loaded_this_attempt=True, hardware=engine["hardware"])
        request = base.prepare_request(engine["tokenizer"],next(i for i,r in enumerate(done) if r is None),
                                       items[next(i for i,r in enumerate(done) if r is None)])
        import torch
        status.update(stage="warmup"); save()
        ids = request["ids"].to(next(engine["model"].parameters()).device)
        with torch.inference_mode(), attention_context(torch, True):
            engine["model"](input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
        metrics["pretrained_model_run"] = True
        status.update(stage="technical_checks"); save()
        print("Status: Teacher-Forcing-, Prompt-Prefill- und Greedy-Sham-Prüfungen.",flush=True)
        proof = technical_checks(engine["model"],engine["tokenizer"],request,reference,stats,plan_hash,engine["hardware"],
                                 weight_norms=engine["weight_norms"])
        metrics.update(checks=proof["checks"],checks_executed_this_attempt=True)
        status.update(run_queue(engine,reference,stats,items,output,plan_hash,proof,done,save))
        metrics["weights_unchanged_evidence"] = "parameter identities, storage pointers, version counters; no training; not a full tensor rehash"
        metrics["generation_scope"] = "prompt-only greedy continuation with fresh cache per arm; pulse/capture only on prefill"
    except BaseException as error:  # noqa: BLE001 — interrupt paths must retain the diagnostic ZIP.
        status.update(state="failed",error_type=type(error).__name__,error=str(error))
        (output/"traceback.txt").write_text(traceback.format_exc())
        print("HARD FAILURE: "+str(error),flush=True)
    finally:
        metrics["elapsed_seconds"] = time.monotonic()-started
        save(); archive(output)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("hn-set","policy","refw","items","out-dir"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--cache-dir");parser.add_argument("--resume-zip")
    args = parser.parse_args()
    result = run(args.hn_set,args.policy,args.refw,args.items,args.out_dir,args.cache_dir,args.resume_zip)
    print(json.dumps(result),flush=True)
    raise SystemExit(0 if result["state"] in ("complete","paused") else 1)
