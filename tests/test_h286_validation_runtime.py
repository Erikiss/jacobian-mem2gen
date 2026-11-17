"""Answer-validation orchestration: synthetic checkpoints and native tiny CPU models only."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
R = importlib.import_module("qwen3_h286_validation_runtime")


def request(index=0):
    return {"request_id": f"r{index:03d}", "input_ids": [1, 2, 3, 4, 5], "prompt_ids": [1, 2, 3, 4],
                "answer_ids": [5], "answer_logit_positions": [3], "capture_positions": [1, 2, 3, 4],
                "pulse_positions": [3], "validation_schema": R.SCHEMA}


def proof(plan_hash):
    return {"schema": R.PROOF_SCHEMA, "state": "passed", "plan_sha256": plan_hash,
                "checks": {name: {"passed": True} for name in R.PROOF_CHECKS}}


def payload(req, count=2):
    records, arrays = [], []
    for arm in R.base.ARMS:
        steps = [{"selected_token_id": token, "selected_token_logprob": -1., "is_eos": token == 0,
                      "top_tokens": [{"token_id": token, "text": str(token), "logprob": -1.}]} for token in (7, 0)]
        gen = {"schema": "h286_prompt_only_generation_v1", "request_id": req["request_id"], "arm": arm,
                   "prompt_length": 4, "prefill_capture_positions": [0, 1, 2, 3], "pulse_positions": [3],
                   "max_new_tokens": 48, "top_k": 10, "prefill_forward_count": 1, "pulse_fired": 1, "capture_fired": 1,
                   "selected_token_ids": [7, 0], "generated_token_ids": [7], "steps": steps, "decode_forward_count": 1,
                   "final_cache_length": 5, "stop_reason": "eos", "truncated": False, "classification": "noncanonical"}
        tf = {"arm": arm, "pulse_fired": 1, "capture_fired": 1, "answer_token_logprobs": [-1.],
                  "answer_logprob_sum": -1., "answer_logprob_mean": -1.}
        records.append({"arm": arm, "teacher_forced": tf, "generation": gen})
        values = {prefix+key: np.ones((1, 4, count), np.float32) for prefix in ("tf_", "gen_")
                  for key in ("raw_z", "cett", "zcett", "down_output_norm_raw")}
        values.update({f"{prefix}residual_layer_{layer}": np.ones((1, 4, 3), np.float32)
                       for prefix in ("tf_", "gen_") for layer in (3, 7, 14, 27)})
        values["tf_gold_token_logprob"] = np.array([[-1.]], np.float32)
        arrays.append(values)
    return records, arrays


def bound_plan():
    plan = {"schema": R.SCHEMA, "test": "synthetic complete five-arm checkpoint"}
    return plan, hashlib.sha256(R.base.canonical(plan)).hexdigest()


def write_chunk(output, index=0, plan_hash=None):
    req = request(index)
    records, arrays = payload(req)
    if plan_hash is None:
        _, plan_hash = bound_plan()
    return R.base.write_request(output, req, records, arrays, plan_hash, {"gpu": "synthetic"}, proof(plan_hash))


def test_plan_binds_new_and_old_sources_and_generation_protocol():
    from attention_probe.reference import ModelPins, Reference
    reference = Reference(((0, 1), (14, 2)), "1"*64, "2"*64,
                          ModelPins(R.base.MODEL, R.base.REVISION, (("test", "3"*64),)))
    plan = R.plan_document(reference, {"source_hn_meta_sha256": "4"*64}, [])
    assert plan["schema"] == R.SCHEMA
    assert plan["base_runtime_sha256"] == R.base.sha_file(R.base.__file__)
    assert plan["helper_sha256"] == R.base.sha_file(R.__file__)
    assert plan["generation_module_sha256"] == R.base.sha_file(R.generation_module().__file__)
    assert plan["module_sha256"]["answer_validation.py"] == plan["generation_module_sha256"]
    assert plan["generation"]["max_new_tokens"] == 48 and plan["generation"]["top_k"] == 10
    assert plan["queue_budget"]["seconds"] == 900


@pytest.mark.parametrize("mutation", ["missing_branch", "missing_residual", "bad_residual", "nonfinite", "gold_mismatch", "gold_shape", "wrong_prompt", "old_proof"])
def test_incomplete_or_corrupt_evidence_cannot_become_valid(mutation):
    req = request()
    rows, values = payload(req)
    if mutation == "missing_branch":
        del rows[0]["generation"]
    elif mutation == "missing_residual":
        del values[0]["gen_residual_layer_27"]
    elif mutation == "bad_residual":
        values[0]["gen_residual_layer_7"] = np.zeros((1, 4, 4), np.float32)
    elif mutation == "nonfinite":
        values[0]["gen_cett"][0, 0, 0] = np.nan
    elif mutation == "gold_mismatch":
        values[0]["tf_gold_token_logprob"][0, 0] = -2.
    elif mutation == "gold_shape":
        values[0]["tf_gold_token_logprob"] = np.array([-1.], np.float32)
    elif mutation == "wrong_prompt":
        rows[0]["generation"]["prefill_capture_positions"] = req["capture_positions"]
    elif mutation == "old_proof":
        with pytest.raises(ValueError, match="technical evidence"):
            R.verify_proof({"state": "passed", "plan_sha256": "p"}, "p")
        return
    with pytest.raises(ValueError):
        R.validate_payload(req, rows, values, 2)


def test_eos_at_cap_is_valid_but_non_eos_cap_must_be_truncated():
    req = request()
    rows, _ = payload(req)
    record = rows[0]["generation"]
    step = record["steps"][0]
    record.update(selected_token_ids=[7]*47+[0], generated_token_ids=[7]*47,
                  steps=[copy.deepcopy(step) for _ in range(47)]+[record["steps"][-1]],
                  decode_forward_count=47, final_cache_length=51)
    R.validate_generation(record, req, R.base.ARMS[0])
    record.update(selected_token_ids=[7]*48, generated_token_ids=[7]*48,
                  steps=[copy.deepcopy(step) for _ in range(48)], stop_reason="max_new_tokens", truncated=True)
    with pytest.raises(ValueError, match="classified as truncated"):
        R.validate_generation(record, req, R.base.ARMS[0])
    record["classification"] = "truncated"
    R.validate_generation(record, req, R.base.ARMS[0])


def test_atomic_five_arm_save_resume_and_sha_tamper_detection(tmp_path):
    output = tmp_path/"first"
    output.mkdir()
    plan, digest = bound_plan()
    R.base.atomic_json(output/"run_plan.json", plan)
    saved = write_chunk(output, plan_hash=digest)
    assert R.verified_request(output, "r000", digest, 2) == saved
    with np.load(output/"requests/r000/arrays.npz", allow_pickle=False) as data:
        assert data["gen_raw_z"].shape == (5, 1, 4, 2)
        assert all(data[key].dtype == np.float32 for key in data.files)
    with pytest.raises(ValueError, match="overwrite"):
        write_chunk(output, plan_hash=digest)
    resumed = tmp_path/"resumed"
    resumed.mkdir()
    R.restore_resume(R.archive(output), resumed, digest, 2)
    assert R.verified_request(resumed, "r000", digest, 2) == saved
    (resumed/"requests/r000/arrays.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA mismatch"):
        R.verified_request(resumed, "r000", digest, 2)


def test_old_plan_and_old_teacher_forcing_chunk_are_not_resumed(tmp_path):
    _plan, digest = bound_plan()
    old_zip = tmp_path/"old.zip"
    with zipfile.ZipFile(old_zip, "w") as packed:
        packed.writestr("run_plan.json", R.base.canonical({"schema": "old"}))
    with pytest.raises(ValueError, match="plan/schema"):
        R.restore_resume(old_zip, tmp_path/"new", digest, 2)
    write_chunk(tmp_path, plan_hash=digest)
    path = tmp_path/"requests/r000/result.json"
    saved = json.loads(path.read_text())
    del saved["request"]["validation_schema"]
    R.base.atomic_json(path, saved)
    with pytest.raises(ValueError, match="Old teacher-forcing"):
        R.verified_request(tmp_path, "r000", digest, 2)


def test_budget_pauses_only_after_whole_five_arm_checkpoint(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    with torch.random.fork_rng():
        model = torch.nn.Linear(2, 2).eval().requires_grad_(False)
    engine = {"model": model, "tokenizer": None, "hardware": {}}
    reference = types.SimpleNamespace(coordinates=((0, 1), (1, 1)))
    monkeypatch.setattr(R.base, "prepare_request", lambda _t, i, _item: request(i))
    monkeypatch.setattr(R, "compute_request", lambda _m, _t, req, *_a, **_k: payload(req))
    times = iter([0., 0., 900., 900.])
    done = [None, None]
    status = R.run_queue(engine, reference, {}, [None]*2, tmp_path, "p", proof("p"), done,
                         lambda _status: None, clock=lambda: next(times), efficient_gpu=False)
    assert status["state"] == "paused" and status["completed_requests"] == 1
    assert R.verified_request(tmp_path, "r000", "p", 2) is not None
    assert R.verified_request(tmp_path, "r001", "p", 2) is None
    status = R.run_queue(engine, reference, {}, [None]*2, tmp_path, "p", proof("p"), done,
                         lambda _status: None, clock=lambda: 0., efficient_gpu=False)
    assert status["state"] == "complete" and status["completed_requests"] == 2


def test_failure_before_full_payload_does_not_publish_request(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2).eval().requires_grad_(False)
    monkeypatch.setattr(R.base, "prepare_request", lambda _t, i, _item: request(i))
    def partial_failure(*_a, **_k):
        raise RuntimeError("failed at fourth arm")
    monkeypatch.setattr(R, "compute_request", partial_failure)
    with pytest.raises(RuntimeError, match="fourth arm"):
        R.run_queue({"model": model, "tokenizer": None, "hardware": {}}, types.SimpleNamespace(coordinates=((0, 1),)),
                    {}, [None], tmp_path, "p", proof("p"), [None], lambda _: None, efficient_gpu=False)
    assert not (tmp_path/"requests/r000").exists()


def test_complete_resume_avoids_gpu_and_keeps_this_attempt_evidence_honest(tmp_path, monkeypatch):
    plan, digest = bound_plan()
    R.base.atomic_json(tmp_path/"run_plan.json", plan)
    for i in range(16):
        write_chunk(tmp_path, i, digest)
    reference = types.SimpleNamespace(coordinates=((0, 1), (1, 1)))
    monkeypatch.setattr(R.base, "load_inputs", lambda *_: (reference, {}, [None]*16))
    monkeypatch.setattr(R, "plan_document", lambda *_: plan)
    monkeypatch.setattr(R, "initialize_engine", lambda *_a, **_k: pytest.fail("completed resume must not load GPU/model"))
    monkeypatch.setenv("H286_WRAPPER_ATTEMPT", "fresh-attempt")
    status = R.run("hn", "policy", "refw", "items", tmp_path)
    metrics = json.loads((tmp_path/"metrics.json").read_text())
    assert status["state"] == "complete" and status["resumed_without_gpu"] is True
    assert metrics["wrapper_attempt_id"] == "fresh-attempt"
    assert metrics["weights_loaded"] is metrics["pretrained_model_run"] is metrics["checks_executed_this_attempt"] is False
    assert metrics["retained_request_proofs"] == [proof(digest)]*16
    with zipfile.ZipFile(tmp_path/"qwen3_h286_validation_results.zip") as packed:
        assert "requests/r015/arrays.npz" in packed.namelist()


def test_missing_reference_yields_failure_zip_and_fresh_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("H286_WRAPPER_ATTEMPT", "failed-fresh-attempt")
    status = R.run(tmp_path/"missing", "policy", "refw", "items", tmp_path/"out")
    assert status["state"] == "failed"
    with zipfile.ZipFile(tmp_path/"out/qwen3_h286_validation_results.zip") as packed:
        metrics = json.loads(packed.read("metrics.json"))
        assert metrics["weights_loaded"] is False and metrics["pretrained_model_run"] is False
        assert metrics["wrapper_attempt_id"] == "failed-fresh-attempt"
        assert "traceback.txt" in packed.namelist()


@pytest.fixture(scope="module")
def tiny():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != R.base.TRANSFORMERS:
        pytest.skip("Native integration needs pinned Transformers5.22.2")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(72)
        config = Qwen3Config(vocab_size=48, hidden_size=16, intermediate_size=16, num_hidden_layers=28,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=8, eos_token_id=0)
        config._attn_implementation = "sdpa"
        model = Qwen3ForCausalLM(config).float().eval().requires_grad_(False)
        with torch.no_grad():
            model.lm_head.weight.zero_()  # Deterministic immediate EOS; no pretrained model.
    class Tokenizer:
        eos_token_id = 0
        def __call__(self, text, **_kwargs):
            return {"input_ids": [5] if text == "gold" else [1, 2, 3, 4]}
        def apply_chat_template(self, *_args, **_kwargs):
            return "prompt"
        def decode(self, ids, **_kwargs):
            return " ".join(map(str, ids))
    yield torch, model, Tokenizer()
    torch.set_num_threads(old_threads)


def test_native_teacher_forcing_unchanged_prompt_only_generation_and_bound_controls(tiny):
    torch, model, tokenizer = tiny
    coordinates = tuple((layer, channel) for layer in range(28) for channel in range(16))[:286]
    reference = types.SimpleNamespace(coordinates=coordinates)
    stats = {"refw": {"c_mu": [0.]*286, "c_sd": [1.]*286}}
    req = R.base.prepare_request(tokenizer, 0, {"id": "tiny", "category": "correct_draft", "prompt": "test", "target": "gold", "pair_id": "p"})
    evidence = R.base.parameter_evidence(model)
    rng = torch.random.get_rng_state().clone()
    controls = R.technical_checks(model, tokenizer, req, reference, stats, "native-plan", {}, efficient_gpu=False)
    assert all(item["passed"] for item in controls["checks"].values())
    assert "passive prefill capture" in controls["checks"]["generation_no_pulse_vs_sham"]["control_scope"]
    results, arrays = R.compute_request(model, tokenizer, req, reference, stats, efficient_gpu=False)
    with R.attention_context(torch, False):
        original, old_arrays = R.base.run_arm(model, req, coordinates, stats, R.base.ARMS[0], efficient_gpu=False)
    assert results[0]["teacher_forced"]["answer_token_logprobs"] == original["answer_token_logprobs"]
    for key in old_arrays:
        np.testing.assert_array_equal(arrays[0]["tf_"+key], old_arrays[key])
    for result, values in zip(results, arrays):
        gen = result["generation"]
        assert gen["prefill_capture_positions"] == [0, 1, 2, 3] != req["capture_positions"]
        assert gen["generated_token_ids"] == [] and gen["selected_token_ids"] == [0]
        assert gen["stop_reason"] == "eos" and gen["truncated"] is False
        assert values["tf_gold_token_logprob"].shape == (1, 1)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert R.base.parameter_evidence(model) == evidence and R.hook_count(model) == 0
