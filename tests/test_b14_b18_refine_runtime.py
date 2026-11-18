"""Model-free checkpoint tests and synthetic native CPU refinement checks.

Every identity below is synthetic.  No pretrained weights or private input files
are read, and these tests never download or initialize a production engine.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))


def runtime():
    """The owner announces the runtime API before these tests are executed."""
    return importlib.import_module("qwen3_b14_b18_refine_runtime")


@pytest.fixture(scope="module")
def native_engine():
    """Small randomly initialized native architecture; no from_pretrained call."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(4002)
        config = Qwen3Config(
            vocab_size=41, hidden_size=16, intermediate_size=64,
            num_hidden_layers=28, num_attention_heads=2,
            num_key_value_heads=1, head_dim=8, eos_token_id=0,
            bos_token_id=1, pad_token_id=0,
        )
        config._attn_implementation = "sdpa"
        model = Qwen3ForCausalLM(config).float().eval().requires_grad_(False)

    class Tokenizer:
        def __call__(self, _text, **_kwargs):
            return {"input_ids": [3, 4, 5, 6, 7, 8, 9, 10]}

        def decode(self, values, **_kwargs):
            return " ".join(str(int(value)) for value in values)

    coordinates = tuple((layer, channel) for layer in range(28) for channel in range(10))
    coordinates += tuple((layer, 10) for layer in range(6))
    assert len(coordinates) == 286
    from attention_probe.reference import ModelPins, Reference
    reference = Reference(coordinates, "1" * 64, "2" * 64,
                          ModelPins("Qwen/Qwen3-1.7B",
                                    "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                                    (("synthetic.safetensors", "3" * 64),)))
    stats = {"refw": {"c_mu": [0.1] * 286, "c_sd": [0.2] * 286},
             "source_hn_meta_sha256": "4" * 64}
    items = [
        {"id": f"synthetic-{index}",
         "request_id": f"{'s3' if index < 12 else 'adet'}:synthetic-{index}",
         "family": "s3" if index < 12 else "adet",
         "prompt_ids": [1, 5, 7, 9, 2, 3, 4, 8],
         "fixed_continuation_text": f"synthetic fixed gold {index}",
         "prompt": f"synthetic prompt {index}"}
        for index in range(16)
    ]
    engine = {"model": model, "tokenizer": Tokenizer(),
              "hardware": {"device": "cpu", "scope": "synthetic native test"}}
    yield torch, engine, reference, stats, items
    torch.set_num_threads(old_threads)


def json_write(path, value):
    raw = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def anchor_document(items, *, width=16):
    R = runtime()
    return {
        "schema": "b14_b18_refine_anchors_v1",
        "source_runtime_sha256": R.OLD_RUNTIME_SHA256,
        "source_plan_sha256": "5" * 64,
        "source_plan_file_sha256": "6" * 64,
        "source_zip_sha256": "7" * 64,
        "cases": [
            {"index": index, "request_id": f"b14b18_{index:03d}",
             "item_sha256": hashlib.sha256(R.base.canonical(items[index])).hexdigest(),
             "lambda": 0.25,
             **{key: [0.01] * width for key in R.VECTOR_KEYS}}
            for index in R.INDICES
        ],
    }


def bound_record(record, plan_hash="synthetic-plan"):
    R = runtime()
    result = copy.deepcopy(record)
    result.update(schema=R.SCHEMA, state="complete", plan_sha256=plan_hash)
    result["technical_checks"] = {
        "state": "passed", "schema": R.SCHEMA,
        "plan_sha256": plan_hash, "checks": result["checks"],
    }
    return result


@pytest.fixture(scope="module")
def stages(native_engine):
    """Exercise the real native derivatives/solver/endpoint code once per dose."""
    R = runtime()
    torch, engine, reference, stats, items = native_engine
    from attention_probe.local_transport import ridge_transport
    from attention_probe.native_jacobian import build_residual_map

    request = R.old.prepare_request(engine["tokenizer"], 0, items[0], 41)
    clean = R.old.forward_probe(engine["model"], request, 0, reference.coordinates,
                                stats, R.old.ARMS[0], backend="MATH")
    pulse = R.old.forward_probe(engine["model"], request, 0, reference.coordinates,
                                stats, R.old.ARMS[4], backend="MATH")
    ids = torch.tensor([request["prompt_ids"]], dtype=torch.long)
    mapping = build_residual_map(engine["model"], ids, 14, 18)
    source_delta = R.old.tensor_residual(pulse, 14, "cpu") - mapping.baseline_source
    target_delta = R.old.tensor_residual(pulse, 18, "cpu") - mapping.baseline_target
    vectors = {
        "baseline_source": mapping.baseline_source,
        "baseline_target": mapping.baseline_target,
        "source_delta": source_delta,
        "target_delta": target_delta,
        "jvp": mapping.jvp(source_delta),
        "old_delta_source": ridge_transport(mapping.jvp, mapping.vjp, target_delta,
                                             0.25, max_iter=16, rtol=1e-4).delta_source,
    }
    anchor = {"index": 0, "request_id": "b14b18_000", "lambda": 0.25,
              "math_clean_logits_sha256": R.old.digest_tensor(clean["logits"]),
              "math_pulse_logits_sha256": R.old.digest_tensor(pulse["logits"]),
              **{key: value.cpu().tolist() for key, value in vectors.items()}}
    initial = R.base.parameter_evidence(engine["model"])
    rng = torch.random.get_rng_state().clone()
    output = {factor: R.compute_stage(engine, reference, stats, request, anchor, factor)
              for factor in R.FACTORS}
    assert R.base.parameter_evidence(engine["model"]) == initial
    assert torch.equal(rng, torch.random.get_rng_state())
    assert R.old.hook_count(engine["model"]) == 0
    return request, anchor, output


def test_fixed_stage_order_and_arm_counts_have_no_winner_selection():
    R = runtime()
    assert R.INDICES == (0, 1, 12, 13)
    assert R.FACTORS == (1.0, 0.3, 0.1)
    assert R.stage_order() == [(i, f) for f in (1.0, 0.3, 0.1) for i in (0, 1, 12, 13)]
    assert R.NOISE_COUNT == 8 and R.MAX_ITER == 512 and R.RTOL == 1e-4
    assert len(R.arm_names(1.0)) == 12
    assert len(R.arm_names(0.3)) == len(R.arm_names(0.1)) == 10
    assert R.arm_names(1.0)[:2] == ["recorded_old_ridge", "raw_target_copy"]
    assert len(set(R.arm_names(1.0))) == 12
    for invalid in (0.0, 0.2, 0.5, -1.0, float("nan"), True):
        with pytest.raises((ValueError, TypeError)):
            R.factor_name(invalid)


def test_input_and_anchor_source_bytes_are_fixed(tmp_path, native_engine):
    R = runtime()
    items = native_engine[-1]
    document = anchor_document(items)
    path = tmp_path / "anchors.json"
    digest = json_write(path, document)
    assert R.load_anchors(path, items, expected_sha256=digest, expected_width=16) == document
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="pinned"):
        R.load_anchors(path, items, expected_sha256=digest, expected_width=16)
    with pytest.raises(ValueError, match="pinned"):
        R.load_anchors(path, items, expected_width=16)


def test_load_inputs_binds_refinement_parent_and_exact_source_bytes(tmp_path, monkeypatch, native_engine):
    R = runtime()
    _, _, reference, stats, items = native_engine
    parent = {"schema": R.old.SCHEMA, "synthetic": "original exact parent"}
    anchors = anchor_document(items, width=2048)
    anchors["source_plan_sha256"] = hashlib.sha256(R.base.canonical(parent)).hexdigest()
    anchors["source_plan_file_sha256"] = hashlib.sha256(R.base.canonical(parent) + b"\n").hexdigest()
    refinement = {
        "schema": "b14_b18_refine_plan_v1", "indices": [0, 1, 12, 13],
        "lambda_factors": [1.0, 0.3, 0.1], "max_iter": 512, "rtol": 1e-4,
        "noise_count": 8, "noise_seed_base": R.NOISE_SEED_BASE,
        "true_residual_interval": 16,
        "seconds_after_model_load": R.BUDGET_SECONDS,
        "anchor_rtol": R.ANCHOR_RTOL, "anchor_atol": R.ANCHOR_ATOL,
    }
    anchor_path, refine_path = tmp_path / "anchors.json", tmp_path / "refine.json"
    monkeypatch.setattr(R, "ANCHORS_SHA256", json_write(anchor_path, anchors))
    monkeypatch.setattr(R, "REFINE_PLAN_SHA256", json_write(refine_path, refinement))
    monkeypatch.setattr(R.old, "load_inputs", lambda *_: (reference, stats, items, {"synthetic": "original scientific plan"}))
    monkeypatch.setattr(R.old, "plan_document", lambda *_: parent)
    result = R.load_inputs("hn", "policy", "refw", "items", "original", refine_path, anchor_path)
    assert result[:3] == (reference, stats, items)
    assert result[4:] == (refinement, anchors)
    raw = refine_path.read_bytes()
    refine_path.write_bytes(raw + b"\n")
    with pytest.raises(ValueError, match="pinned"):
        R.load_inputs("hn", "policy", "refw", "items", "original", refine_path, anchor_path)
    refine_path.write_bytes(raw)
    refinement["lambda_factors"] = [1.0, 0.3, 0.2]
    monkeypatch.setattr(R, "REFINE_PLAN_SHA256", json_write(refine_path, refinement))
    with pytest.raises(ValueError, match="numerical rules"):
        R.load_inputs("hn", "policy", "refw", "items", "original", refine_path, anchor_path)
    refinement["lambda_factors"] = [1.0, 0.3, 0.1]
    monkeypatch.setattr(R, "REFINE_PLAN_SHA256", json_write(refine_path, refinement))
    monkeypatch.setattr(R.old, "plan_document", lambda *_: {**parent, "changed": True})
    with pytest.raises(ValueError, match="binding"):
        R.load_inputs("hn", "policy", "refw", "items", "original", refine_path, anchor_path)


def test_plan_binds_new_source_and_twelve_fixed_stages(native_engine):
    R = runtime()
    _, _, reference, stats, items = native_engine
    anchors = anchor_document(items)
    plan = R.plan_document(reference, stats, items, {"synthetic": "old"}, {"synthetic": "refine"}, anchors)
    assert plan["schema"] != R.old.SCHEMA
    assert plan["stage_order"] == [[i, f] for i, f in R.stage_order()]
    assert plan["lambda_factors"] == [1.0, 0.3, 0.1]
    assert plan["free_generations"] == 0
    assert plan["source_zip_sha256"] == anchors["source_zip_sha256"]
    assert plan["parent_plan_sha256"] == anchors["source_plan_sha256"]
    assert plan["helper_sha256"] == R.base.sha_file(R.__file__)
    assert "attention_probe/transport_refinement.py" in plan["source_sha256"]
    assert plan["source_sha256"]["qwen3_b14_b18_runtime.py"] == R.OLD_RUNTIME_SHA256
    assert len(R.REFINE_PLAN_SHA256) == 64


@pytest.mark.parametrize("mutation", [
    "schema", "runtime", "indices", "item", "lambda_zero", "lambda_bool",
    "width", "nonfinite", "boolean_vector", "request_id",
])
def test_anchor_validation_rejects_semantic_changes(tmp_path, native_engine, mutation):
    R = runtime()
    items = native_engine[-1]
    document = anchor_document(items)
    case = document["cases"][0]
    if mutation == "schema": document["schema"] = "old"
    elif mutation == "runtime": document["source_runtime_sha256"] = "0" * 64
    elif mutation == "indices": document["cases"][2]["index"] = 2
    elif mutation == "item": case["item_sha256"] = "0" * 64
    elif mutation == "lambda_zero": case["lambda"] = 0
    elif mutation == "lambda_bool": case["lambda"] = True
    elif mutation == "width": case["jvp"].pop()
    elif mutation == "nonfinite": case["target_delta"][0] = float("inf")
    elif mutation == "boolean_vector": case["source_delta"][0] = True
    else: case["request_id"] = "b14b18_001"
    path = tmp_path / "anchors.json"
    digest = json_write(path, document)
    with pytest.raises(ValueError):
        R.load_anchors(path, items, expected_sha256=digest, expected_width=16)


def test_native_three_factors_preserve_anchors_norms_rng_and_full_logits(stages, native_engine):
    R = runtime()
    torch = native_engine[0]
    _, anchor, outputs = stages
    for factor, (record, arrays) in outputs.items():
        assert record["state"] == "complete"
        assert record["lambda"] == anchor["lambda"] * factor
        assert record["checks"] == dict.fromkeys(R.PROOF_KEYS, True)
        assert [row["name"] for row in record["arms"]] == R.arm_names(factor)
        assert record["solver"]["max_iter"] == 512
        assert record["solver"]["lambda"] == record["lambda"]
        assert record["calibration"]["state"] == ("passed" if factor == 1 else "not_run")
        assert all(row["passed"] is True for row in record["anchor_continuity"]["vectors"].values())
        norm = np.linalg.norm(arrays["solver_delta_source"].astype(np.float64))
        for row in record["arms"]:
            name = row["name"]
            prefix = "endpoint_" + name
            requested = arrays[prefix + "_requested_delta"]
            actual_norm = np.linalg.norm(requested.astype(np.float64))
            assert actual_norm == pytest.approx(row["requested_source_norm"], rel=1e-12, abs=1e-15)
            if name.startswith("norm_matched_"):
                assert actual_norm == pytest.approx(norm, rel=2e-6, abs=1e-12)
            if name.startswith("norm_matched_noise_"):
                expected = R.old.fixed_noise_like(torch.from_numpy(arrays["solver_delta_source"]), row["noise_seed"])
                np.testing.assert_array_equal(requested, expected.numpy())
            logits = torch.from_numpy(arrays[prefix + "_logits"])
            for label in ("math_clean", "math_pulse"):
                baseline = torch.from_numpy(arrays[label + "_logits"])
                assert R.old.logits_metrics(logits, baseline, record['request']['continuation_ids'][0]) == row["logit_references"][label]
        R.validate_record(bound_record(record), "synthetic-plan")
        R.validate_arrays(bound_record(record), arrays)


def test_noise_directions_are_paired_across_all_fixed_damping_factors(stages):
    _, _, outputs = stages
    for k in range(8):
        directions = []
        for _, arrays in outputs.values():
            vector = arrays[f"endpoint_norm_matched_noise_{k}_requested_delta"].astype(np.float64)
            norm = np.linalg.norm(vector)
            assert norm > 0
            directions.append(vector / norm)
        np.testing.assert_allclose(directions[0], directions[1], rtol=2e-6, atol=2e-8)
        np.testing.assert_allclose(directions[0], directions[2], rtol=2e-6, atol=2e-8)


def test_earlier_position_observer_sees_post_patch_source_changes(native_engine, monkeypatch):
    """A faulty patch must not look causal merely because an observer ran first."""
    R = runtime()
    torch, engine, reference, stats, items = native_engine
    request = R.old.prepare_request(engine["tokenizer"], 0, items[0], 41)
    clean = R.endpoint_probe(engine["model"], request, reference.coordinates, stats, R.old.ARMS[0])
    original_patch = R.old.residual_patch

    @contextlib.contextmanager
    def faulty_patch(model, block, position, delta):
        with original_patch(model, block, position, delta) as evidence:
            def also_change_earlier(_module, _args, output):
                value = output[0] if isinstance(output, tuple) else output
                changed = value.clone()
                changed[:, 0, :] += 0.125
                return (changed, *output[1:]) if isinstance(output, tuple) else changed
            handle = model.model.layers[block].register_forward_hook(also_change_earlier)
            try:
                yield evidence
            finally:
                handle.remove()

    monkeypatch.setattr(R.old, "residual_patch", faulty_patch)
    observed = R.endpoint_probe(engine["model"], request, reference.coordinates, stats, R.old.ARMS[0],
                                 patch={"block": 14, "delta": torch.zeros(16)})
    assert clean["earlier_prompt_residual_sha256"]["13"] == observed["earlier_prompt_residual_sha256"]["13"]
    assert clean["earlier_prompt_residual_sha256"]["14"] != observed["earlier_prompt_residual_sha256"]["14"]
    assert clean["earlier_prompt_residual_sha256"]["18"] != observed["earlier_prompt_residual_sha256"]["18"]
    assert R.old.hook_count(engine["model"]) == 0


@pytest.mark.parametrize("mutation", [
    "schema", "proof", "source_ids", "pulse_position", "factor", "lambda",
    "missing_arm", "missing_array", "extra", "shape", "dtype", "nonfinite",
    "logits", "endpoint_vector", "noise_seed", "requested_vector", "solution_vector",
])
def test_checkpoint_validation_rejects_changed_state_and_controls(stages, mutation):
    R = runtime()
    record, values = stages[2][1.0]
    record = bound_record(record)
    arrays = {k: v.copy() for k, v in values.items()}
    if mutation == "schema": record["schema"] = R.old.SCHEMA
    elif mutation == "proof": record["technical_checks"]["checks"].pop("global_rng")
    elif mutation == "source_ids": record["request"]["prompt_ids"][0] += 1
    elif mutation == "pulse_position": record["request"]["pulse_positions"] = [0]
    elif mutation == "factor": record["lambda_factor"] = 0.5
    elif mutation == "lambda": record["lambda"] *= 2
    elif mutation == "missing_arm": record["arms"].pop()
    elif mutation == "missing_array": arrays.pop("math_clean_logits")
    elif mutation == "extra": arrays["unexpected"] = np.zeros(1, np.float32)
    elif mutation == "shape": arrays["math_clean_raw_z"] = np.zeros((2, 1, 286), np.float32)
    elif mutation == "dtype": arrays["math_clean_logits"] = arrays["math_clean_logits"].astype(np.float64)
    elif mutation == "nonfinite": arrays["math_clean_residual_18"][0, 0, 0] = np.inf
    elif mutation == "logits": arrays["endpoint_refined_ridge_logits"][0] += 0.1
    elif mutation == "endpoint_vector": arrays["endpoint_refined_ridge_target_actual"][0] += 1
    elif mutation == "noise_seed": record["arms"][-1]["noise_seed"] += 1
    elif mutation == "requested_vector": arrays["endpoint_norm_matched_copy_requested_delta"] *= 2
    else: arrays["solver_delta_source"] *= 2
    with pytest.raises(ValueError):
        R.validate_record(record, "synthetic-plan")
        R.validate_arrays(record, arrays)


def test_atomic_checkpoint_refuses_overwrite_and_resume_detects_corruption(tmp_path, stages):
    R = runtime()
    request, _, outputs = stages
    record, arrays = outputs[1.0]
    plan = {"schema": R.SCHEMA, "synthetic": True}
    plan_hash = hashlib.sha256(R.base.canonical(plan)).hexdigest()
    R.base.atomic_json(tmp_path / "run_plan.json", plan)
    saved = R.write_checkpoint(tmp_path, record, arrays, plan_hash, {"device": "cpu/synthetic"})
    assert R.verified(tmp_path, 0, 1.0, plan_hash, request["item"]) == saved
    with pytest.raises(ValueError, match="overwrite"):
        R.write_checkpoint(tmp_path, record, arrays, plan_hash, {})
    archive = R.archive(tmp_path)
    destination = tmp_path / "restored"
    destination.mkdir()
    items = [None] * 16
    items[0] = request["item"]
    R.restore_resume(archive, destination, plan_hash, items)
    assert R.verified(destination, 0, 1.0, plan_hash, request["item"]) == saved
    with pytest.raises(ValueError, match="plan"):
        R.restore_resume(archive, tmp_path / "wrong", "wrong", items)
    path = R.stage_path(destination, 0, 1.0) / "arrays.npz"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash"):
        R.verified(destination, 0, 1.0, plan_hash, request["item"])


def test_failed_atomic_publish_leaves_no_complete_checkpoint(tmp_path, monkeypatch, stages):
    R = runtime()
    record, arrays = stages[2][1.0]
    monkeypatch.setattr(R.os, "rename", lambda *_: (_ for _ in ()).throw(OSError("interrupted publish")))
    with pytest.raises(OSError, match="interrupted"):
        R.write_checkpoint(tmp_path, record, arrays, "plan", {})
    assert R.verified(tmp_path, 0, 1.0, "plan", record["request"]["item"]) is None
    assert not list((tmp_path / "requests" / "b14b18_000").iterdir())


def test_partial_stage_keeps_evidence_but_is_not_a_resume_checkpoint(tmp_path, stages):
    R = runtime()
    record, arrays = stages[2][1.0]
    partial = copy.deepcopy(record)
    partial.update(state="partial", stop_reason="budget_exhausted_between_endpoint_arms")
    partial["arms"] = partial["arms"][:3]
    output = R.write_partial(tmp_path, partial, arrays, "plan")
    saved = json.loads((output / "result.json").read_text())
    assert saved["state"] == "partial" and saved["resume_eligible"] is False
    assert saved["stop_reason"] == partial["stop_reason"]
    assert (output / "arrays.npz").is_file()
    assert R.verified(tmp_path, 0, 1.0, "plan", record["request"]["item"]) is None


def test_stop_before_stage_runs_no_forward_or_solver(native_engine, monkeypatch):
    R = runtime()
    _, engine, reference, stats, items = native_engine
    request = R.old.prepare_request(engine["tokenizer"], 0, items[0], 41)
    monkeypatch.setattr(R.old, "forward_probe", lambda *_args, **_kwargs: pytest.fail("unexpected forward"))
    record, arrays = R.compute_stage(engine, reference, stats, request, {"lambda": 0.25}, 1.0,
                                      stop_requested=lambda: True)
    assert record["state"] == "partial" and not arrays
    assert record["stop_reason"] == "budget_exhausted_before_stage"


def test_completed_resume_preserves_proofs_but_has_fresh_attempt_metrics(tmp_path, monkeypatch, native_engine):
    R = runtime()
    _, _, reference, stats, items = native_engine
    plan = {"schema": R.SCHEMA, "synthetic": "complete resume"}
    digest = hashlib.sha256(R.base.canonical(plan)).hexdigest()
    monkeypatch.setattr(R, "load_inputs", lambda *_: (reference, stats, items, {}, {}, {"cases": []}))
    monkeypatch.setattr(R, "plan_document", lambda *_: plan)
    proof = {"state": "passed", "schema": R.SCHEMA, "plan_sha256": digest,
             "checks": dict.fromkeys(R.PROOF_KEYS, True)}
    seen = []
    def verified(_output, index, factor, _plan, _item):
        seen.append((index, factor))
        return {"technical_checks": copy.deepcopy(proof), "solver": {"converged": False}}
    monkeypatch.setattr(R, "verified", verified)
    monkeypatch.setattr(R.old, "initialize_engine", lambda *_: pytest.fail("completed resume must not load model"))
    R.base.atomic_json(tmp_path / "metrics.json", {"weights_loaded": True, "pretrained_model_run": True,
                                                 "generation_count": 99, "stale": True})
    status = R.run("hn", "policy", "refw", "items", "plan", "refine", "anchors", tmp_path)
    assert status["state"] == "complete" and status["resumed_without_gpu"] is True
    assert status["completed_stages"] == 12 and status["completed_inputs"] == 4
    assert seen == R.stage_order()
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["weights_loaded"] is False and metrics["pretrained_model_run"] is False
    assert metrics["generation_count"] == 0 and "stale" not in metrics
    assert metrics["retained_pretrained_model_run"] is True
    assert metrics["retained_technical_checks"] == [proof] * 12
    assert metrics["all_solvers_converged"] is False
    assert metrics["solver_count"] == 12 and metrics["converged_solver_count"] == 0


def test_paused_queue_keeps_complete_proof_and_partial_is_not_complete(tmp_path, monkeypatch, native_engine, stages):
    R = runtime()
    _, native, reference, stats, items = native_engine
    anchors = anchor_document(items)
    plan = {"schema": R.SCHEMA, "synthetic": "paused stage"}
    digest = hashlib.sha256(R.base.canonical(plan)).hexdigest()
    monkeypatch.setattr(R, "load_inputs", lambda *_: (reference, stats, items, {}, {}, anchors))
    monkeypatch.setattr(R, "plan_document", lambda *_: plan)
    engine = {"model": SimpleNamespace(config=SimpleNamespace(vocab_size=41)),
              "tokenizer": native["tokenizer"], "hardware": {"device": "synthetic"}}
    def initialize(_reference, _cache, on_loaded):
        on_loaded(engine["hardware"])
        return engine
    monkeypatch.setattr(R.old, "initialize_engine", initialize)
    calls = []
    real_record, real_arrays = stages[2][1.0]
    def compute(_engine, _reference, _stats, request, _anchor, factor, **_kwargs):
        calls.append((request["index"], factor))
        if len(calls) == 1:
            return real_record, real_arrays
        return {"request": request, "state": "partial", "lambda_factor": factor,
                "stop_reason": "budget_exhausted_during_solver"}, {"synthetic_partial": np.zeros(1, np.float32)}
    monkeypatch.setattr(R, "compute_stage", compute)
    status = R.run("hn", "policy", "refw", "items", "plan", "refine", "anchors", tmp_path)
    assert status["state"] == "paused" and status["stage"] == "partial_stage_checkpoint"
    assert calls == [(0, 1.0), (1, 1.0)]
    assert status["completed_stages"] == 1 and status["completed_inputs"] == 0
    assert status["completed_primary_stages"] == 1
    complete = R.verified(tmp_path, 0, 1.0, digest, items[0])
    assert complete["technical_checks"]["checks"] == dict.fromkeys(R.PROOF_KEYS, True)
    assert R.verified(tmp_path, 1, 1.0, digest, items[1]) is None
    partial = json.loads((tmp_path / status["partial_path"] / "result.json").read_text())
    assert partial["state"] == "partial" and partial["resume_eligible"] is False
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["completed_stages"] == 1 and metrics["all_solvers_converged"] is False
    assert metrics["generation_count"] == 0
    restored = tmp_path / "resume"
    R.restore_resume(tmp_path / "b14_b18_refine_results.zip", restored, digest, items)
    assert R.verified(restored, 0, 1.0, digest, items[0]) == complete
    assert R.verified(restored, 1, 1.0, digest, items[1]) is None


def test_missing_inputs_create_diagnostics_before_loading_a_model(tmp_path, monkeypatch):
    R = runtime()
    monkeypatch.setattr(R.old, "initialize_engine", lambda *_: pytest.fail("missing input must fail before model load"))
    status = R.run("missing", "missing", "missing", "missing", "missing", "missing", "missing", tmp_path)
    assert status["state"] == "failed" and status["stage"] == "reference"
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["weights_loaded"] is False and metrics["pretrained_model_run"] is False
    with zipfile.ZipFile(tmp_path / "b14_b18_refine_results.zip") as bundle:
        assert {"status.json", "metrics.json", "traceback.txt"} <= set(bundle.namelist())
