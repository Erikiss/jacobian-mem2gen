"""Offline checkpoint validation for the fixed temporal experiment.

No model is loaded. Stored full-vocabulary logits support independent FP64
checks; later teacher-forced token probabilities can only be checked for
internal consistency because their full logit vectors are not stored.
"""

from __future__ import annotations

import hashlib
import json
import math
from itertools import pairwise

import numpy as np

SCHEMA = "qwen3_temporal_functional_v1"
OFFSETS = (0, 1, 8, 32)
LAYERS = (13, 14, 15, 17, 18, 19)
MAIN = ("clean", "pulse", "clean_add_shared", "pulse_remove_shared")
CACHE = (
    "pulse_restore_prompt_cache",
    "pulse_restore_continuation_cache",
    "pulse_restore_both_cache",
)
ARMS = (
    *MAIN,
    *[n for k in range(8) for n in (f"clean_add_noise_{k}", f"pulse_remove_noise_{k}")],
    *CACHE,
)
PROOF_KEYS = (
    "native_capture_exact",
    "beta_one_sham_exact",
    "cached_full_parity",
    "null_causal_checks",
    "source_tokens_exact",
    "cache_donors_unchanged",
    "frozen_parameters",
    "global_rng",
    "hook_cleanup",
    "finite_arrays",
)
FIELDS = (
    "raw_z",
    "cett",
    "zcett",
    "down_output_norm_raw",
    *[f"residual_{i}" for i in LAYERS],
    "logits",
)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _json_sha(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _array_sha(value):
    return hashlib.sha256(np.asarray(value, dtype="<f4").tobytes(order="C")).hexdigest()


def _sha(value):
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value),
        "Expected lowercase SHA256",
    )


def _finite(value, *, nonnegative=False):
    _require(
        type(value) in (int, float) and math.isfinite(value),
        "Expected finite numeric scalar",
    )
    _require(not nonnegative or value >= 0, "Expected nonnegative scalar")
    return float(value)


def _close(actual, expected, label, *, atol=1e-10, rtol=1e-8):
    _finite(actual)
    _require(
        math.isclose(actual, float(expected), abs_tol=atol, rel_tol=rtol),
        f"{label} differs from recomputation",
    )


def _ids(values, vocab, *, count=None, nonempty=True):
    _require(
        isinstance(values, list) and (bool(values) or not nonempty),
        "Expected token ID list",
    )
    _require(
        all(type(x) is int and 0 <= x < vocab for x in values),
        "Invalid vocabulary token ID",
    )
    _require(count is None or len(values) == count, "Token count differs")


def _cache(value, length, n_layers=None):
    _require(
        value["cache_type"] == "DynamicCache" and value["sequence_length"] == length,
        "Cache type/consumed length differs",
    )
    _require(value["hashes_included"] is True, "Cache hashes missing")
    rows = value["layers"]
    _require(isinstance(rows, list) and len(rows) >= 20, "Cache lacks required layers")
    _require(n_layers is None or len(rows) == n_layers, "Cache layer count differs")
    layouts = set()
    for index, row in enumerate(rows):
        shape = row["shape"]
        _require(
            isinstance(shape, list)
            and len(shape) == 4
            and all(type(x) is int and x > 0 for x in shape),
            "Invalid native cache shape",
        )
        _require(
            row["layer"] == index
            and row["type"] == "DynamicLayer"
            and row["sequence_length"] == length
            and shape[0] == 1
            and shape[2] == length
            and row["dtype"] == "torch.float32"
            and row["is_initialized"] is True,
            "Invalid cache layer metadata",
        )
        _require(
            isinstance(row["device"], str) and bool(row["device"]),
            "Missing cache device",
        )
        layouts.add((shape[1], shape[3], row["device"]))
        _sha(row["keys_sha256"])
        _sha(row["values_sha256"])
    _require(len(layouts) == 1, "Cache layer layouts differ")
    return len(rows)


def _snapshot(
    snapshot, item, neutral, offset, start, pulse, n_layers, *, patched=False
):
    length = len(item["prompt_ids"]) + offset
    forward = snapshot["forward"]
    expected = {
        "cache_input_length": start,
        "consumed_input_tokens": length - start,
        "cache_output_length": length,
        "absolute_capture_positions": [length - 1],
        "local_capture_positions": [length - start - 1],
        "logit_predicts_absolute_positions": [length],
        "capture_fired": 1,
        "pulse_fired": int(pulse),
        "pulse_absolute_positions": [len(item["prompt_ids"]) - 1] if pulse else [],
        "backend": "SDPA MATH",
        "dtype": "float32",
        "use_cache": True,
        "incoming_cache_cloned": start != 0,
    }
    for key, value in expected.items():
        _require(
            forward[key] == value
            and (type(forward[key]) is bool if type(value) is bool else True),
            f"Forward {key} differs",
        )
    prefix = item["prompt_ids"] + neutral[:offset]
    _require(
        snapshot["token_prefix_sha256"] == _json_sha(prefix),
        "Token prefix hash differs",
    )
    _require(
        snapshot["position_ids_sha256"] == _json_sha(list(range(length))),
        "Position hash differs",
    )
    _sha(snapshot["logits_sha256"])
    _cache(snapshot["cache"], length, n_layers)
    patch = forward["residual_patch"]
    if patched:
        _require(
            patch["block"] == 18
            and patch["absolute_positions"] == [length - 1]
            and patch["fired_count"] == 1,
            "Residual patch location/count differs",
        )
        _finite(patch["requested_delta_norm"], nonnegative=True)
        _finite(patch["rounding_error_norm"], nonnegative=True)
        norms = np.asarray(patch["actual_delta_norms"], dtype=np.float64)
        _require(
            norms.shape == (1, 1) and np.isfinite(norms).all() and (norms >= 0).all(),
            "Invalid actual patch norms",
        )
    else:
        _require(patch is None, "Unexpected residual patch")
    for key in ("pulse_requested_delta_norms", "pulse_actual_delta_norms"):
        if pulse:
            norms = np.asarray(forward[key], dtype=np.float64)
            _require(
                norms.ndim == 3
                and norms.shape[:2] == (1, 1)
                and norms.shape[2] > 0
                and np.isfinite(norms).all()
                and (norms >= 0).all(),
                "Invalid pulse norms",
            )
        else:
            _require(forward[key] is None, "Unpulsed forward has pulse norms")


def _scores(scores, item, snapshot, vocab):
    end = snapshot["cache"]["sequence_length"]
    for kind in ("gold", "foil"):
        row = scores[kind]
        ids = row["token_ids"]
        _ids(ids, vocab)
        _require(
            ids == item[f"{kind}_token_ids"], "Scored answer tokens differ from source"
        )
        values = row["token_logprobs"]
        _require(
            isinstance(values, list) and len(values) == len(ids),
            "Token logprob count differs",
        )
        for value in values:
            _finite(value)
            _require(value <= 1e-10, "Positive token logprob")
        _close(row["first_logprob"], values[0], "First token score")
        _close(row["sum_logprob"], math.fsum(values), "Sum score")
        _close(row["mean_logprob"], math.fsum(values) / len(values), "Mean score")
        _require(
            row["token_count"] == len(ids)
            and row["metric_dtype"] == "float64"
            and row["eos_added"] is False
            and row["endpoint_cache_unchanged"] is True,
            "Invalid candidate scoring metadata",
        )
        _require(
            row["cache_input_length"] == end
            and row["cache_output_length"] == end + len(ids) - 1,
            "Candidate teacher-forcing cache positions differ",
        )
        _require(
            row["endpoint_cache_sha256"] == _json_sha(snapshot["cache"]),
            "Scoring donor hash differs",
        )
    for key, field in (
        ("first_token_margin", "first_logprob"),
        ("full_sum_margin", "sum_logprob"),
        ("mean_per_token_margin", "mean_logprob"),
    ):
        _close(scores[key], scores["gold"][field] - scores["foil"][field], key)
    _require(
        scores["primary"] == "full_sum_margin" and scores["metric_dtype"] == "float64",
        "Score definition differs",
    )


def _generation(row, endpoint, item, vocab):
    from .answer_validation import classify_answer

    selected, emitted, steps = (
        row["selected_token_ids"],
        row["generated_token_ids"],
        row["steps"],
    )
    _ids(selected, vocab)
    _ids(emitted, vocab, nonempty=False)
    _ids(row["eos_ids"], vocab, nonempty=False)
    _require(
        row["max_new_tokens"] == 64 and row["top_k"] == 10 and 1 <= len(selected) <= 64,
        "Greedy decoding budget differs",
    )
    _require(
        len(steps) == len(selected) == row["num_steps"]
        and len(emitted) == row["num_generated_tokens"],
        "Greedy token counts differ",
    )
    for index, step in enumerate(steps):
        is_eos = selected[index] in row["eos_ids"]
        _require(
            step["step"] == index
            and step["selected_token_id"] == selected[index]
            and step["is_eos"] is is_eos,
            "Greedy step IDs/EOS differ",
        )
        _require(
            not is_eos or index == len(steps) - 1, "Generation continued after EOS"
        )
        _require(
            step["phase"] == ("endpoint" if index == 0 else "unhooked_cached_decode"),
            "Unexpected greedy decode phase",
        )
        _finite(step["selected_token_logprob"])
        _top_tokens(step["top_tokens"], vocab)
        matching = [x for x in step["top_tokens"] if x["token_id"] == selected[index]]
        # torch.topk may omit argmax's lowest-index token when >k logits tie.
        if matching:
            _close(
                step["selected_token_logprob"],
                matching[0]["logprob"],
                "Selected top-k score",
            )
        _close(
            step["selected_token_logprob"],
            step["top_tokens"][0]["logprob"],
            "Greedy maximal score",
        )
    stopped = selected[-1] in row["eos_ids"]
    _require(
        row["stop_reason"] == ("eos" if stopped else "max_new_tokens")
        and row["truncated"] is (not stopped),
        "Greedy stopping/truncation differs",
    )
    _require(stopped or len(selected) == 64, "Non-EOS generation stopped before cap")
    _require(
        emitted == (selected[:-1] if stopped else selected),
        "Emitted tokens include EOS or omit tokens",
    )
    _require(
        row["endpoint_cache_unchanged"] is True
        and row["endpoint_cache_sha256"] == _json_sha(endpoint["snapshot"]["cache"]),
        "Greedy donor differs",
    )
    _require(
        row["final_cache_length"]
        == endpoint["snapshot"]["cache"]["sequence_length"] + len(selected) - 1,
        "Greedy terminal-token cache convention differs",
    )
    _require(isinstance(row["generated_text"], str), "Generated text missing")
    _require(
        row["classification"]
        == classify_answer(
            row["generated_text"], item["gold_text"], truncated=not stopped
        ),
        "Strict surface classification differs",
    )
    _finite(row["elapsed_seconds"], nonnegative=True)


def _top_tokens(rows, vocab):
    _require(
        isinstance(rows, list) and len(rows) == min(10, vocab), "Top-k count differs"
    )
    ids = [row["token_id"] for row in rows]
    _ids(ids, vocab)
    _require(len(set(ids)) == len(ids), "Duplicate top-k token IDs")
    values = [_finite(row["logprob"]) for row in rows]
    _require(
        all(x <= 1e-10 for x in values) and all(a >= b for a, b in pairwise(values)),
        "Invalid top-k logprobs/order",
    )


def _candidate_binding(record, candidate):
    binding = record["candidate_binding"]
    _sha(record["candidate_sha256"])
    _require(
        type(record["candidate_valid"]) is bool
        and binding["valid"] is record["candidate_valid"]
        and binding["candidate_sha256"] == record["candidate_sha256"],
        "Candidate identity/gate differs",
    )
    _sha(binding["vector_sha256"])
    _require(
        len(binding["noise_vector_sha256"]) == 8, "Eight frozen noise vectors required"
    )
    for value in binding["noise_vector_sha256"]:
        _sha(value)
    _finite(binding["native_norm"], nonnegative=True)
    if candidate is not None:
        from .temporal_candidate import verify_candidate

        verify_candidate(candidate)
        expected = {
            "candidate_sha256": candidate["candidate_sha256"],
            "valid": candidate["valid"],
            "vector_sha256": _array_sha(candidate["vector"]),
            "noise_vector_sha256": [_array_sha(v) for v in candidate["noise_vectors"]],
        }
        for key, value in expected.items():
            _require(binding[key] == value, f"Frozen candidate {key} differs")
        vector = np.asarray(candidate["vector"], dtype=np.float32)
        _require(
            vector.shape == (record["array_contract"]["hidden_size"],),
            "Candidate hidden width differs",
        )
        _close(
            binding["native_norm"],
            np.linalg.norm(vector.astype(np.float64)),
            "Candidate native norm",
        )


def _parity_metric(row, shape, left=None, right=None):
    _require(
        row["cached_shape"] == row["full_shape"] == list(shape)
        and row["cached_dtype"] == row["full_dtype"] == "torch.float32",
        "Parity tensor shape/dtype differs",
    )
    for key in ("same_shape", "native_fp32", "finite", "passed"):
        _require(row[key] is True, f"Parity tensor proof {key} failed")
    _require(
        row["rtol"] == 1e-3
        and row["atol"] == 1e-4
        and type(row["bitwise_equal"]) is bool,
        "Parity tolerance/equality metadata differs",
    )
    for key in ("max_abs_error", "L2_error", "full_L2_norm"):
        _finite(row[key], nonnegative=True)
    _require(
        not row["bitwise_equal"] or row["max_abs_error"] == row["L2_error"] == 0,
        "Bitwise parity contradicts nonzero error",
    )
    if left is not None:
        a, b = left.astype(np.float64), right.astype(np.float64)
        _require(
            np.allclose(a, b, rtol=1e-3, atol=1e-4),
            "Saved capture array fails rebuild parity",
        )
        _require(
            row["bitwise_equal"] is bool(np.array_equal(left, right)),
            "Saved parity equality differs",
        )
        _close(row["max_abs_error"], np.max(np.abs(a - b)), "Saved parity max error")
        _close(row["L2_error"], np.linalg.norm(a - b), "Saved parity L2 error")
        _close(row["full_L2_norm"], np.linalg.norm(b), "Saved parity reference norm")


def _null_checks(baseline, L, n_layers):
    rows = baseline["null_causal_checks"]
    _require(
        [r["offset"] for r in rows] == list(OFFSETS), "Causal null offsets incomplete"
    )
    for row in rows:
        offset = row["offset"]
        _require(
            row["cache_length"] == L + offset and row["passed"] is True,
            "Causal null length/result differs",
        )
        regions = {
            "earlier_prompt_all_layers": (list(range(n_layers)), 0, L - 1),
            "all_positions_layers0_through14": (list(range(15)), 0, L + offset),
            "bridge_positions_layer15": ([15], L, L + offset),
        }
        _require(set(row["checks"]) == set(regions), "Causal null regions incomplete")
        for name, (layers, start, stop) in regions.items():
            region = row["checks"][name]
            _require(
                region["layer_indices"] == layers
                and region["absolute_start"] == start
                and region["absolute_stop_exclusive"] == stop
                and region["applicable"] is (stop > start)
                and region["passed"] is True,
                "Causal null selection/result differs",
            )
            details = region["details"]
            _require(
                [(d["layer"], d["kind"]) for d in details]
                == [(i, k) for i in layers for k in ("keys", "values")],
                "Causal null tensor diagnostics incomplete",
            )
            _require(
                all(
                    d["bitwise_equal"] is True
                    and d["L2_difference"] == 0
                    and type(d["changed_scalars"]) is int
                    and d["changed_scalars"] == 0
                    for d in details
                ),
                "Causal null is not exact",
            )


def _swap(row, record, n_layers):
    evidence = row["cache_swap"]
    L = len(record["item"]["prompt_ids"])
    name = row["name"]
    positions = (
        [L - 1]
        if name == CACHE[0]
        else list(range(L, L + 8))
        if name == CACHE[1]
        else list(range(L - 1, L + 8))
    )
    _require(
        evidence["layer_indices"] == list(range(15, n_layers))
        and evidence["absolute_positions"] == positions,
        "Cache-swap selection differs",
    )
    for key in (
        "donor_unchanged",
        "receiver_unchanged",
        "untouched_slots_exact",
        "existing_offset8_logits_unchanged",
    ):
        _require(evidence[key] is True, f"Cache-swap proof {key} missing")
    snapshots = record["baseline"]["snapshots"]
    _require(
        evidence["receiver_before"] == snapshots["pulse_o8"]["cache"]
        and evidence["donor"] == snapshots["clean_o8"]["cache"],
        "Cache-swap donors differ",
    )
    _require(
        evidence["token_prefix_sha256"] == snapshots["clean_o8"]["token_prefix_sha256"]
        and evidence["position_ids_sha256"]
        == snapshots["clean_o8"]["position_ids_sha256"],
        "Cache-swap prefix binding differs",
    )
    _cache(evidence["result"], L + 8, n_layers)
    for index in range(15):
        _require(
            evidence["result"]["layers"][index]
            == evidence["receiver_before"]["layers"][index],
            "Cache swap changed an upstream layer",
        )
    if name == CACHE[2]:
        _require(
            evidence["both_cache_exact_clean_o8"] is True
            and evidence["both_endpoint_exact_clean_o32"] is True
            and evidence["result"] == snapshots["clean_o8"]["cache"]
            and row["snapshot"]["cache"] == snapshots["clean_o32"]["cache"],
            "Both-slot technical cut does not reproduce clean cache/endpoint",
        )
    changes = evidence["changes"]
    _require(
        [(x["layer"], x["kind"]) for x in changes]
        == [(i, k) for i in range(15, n_layers) for k in ("keys", "values")],
        "Cache-swap diagnostics incomplete",
    )
    for change in changes:
        _require(
            type(change["changed_scalars"]) is int and change["changed_scalars"] >= 0,
            "Invalid change count",
        )
        _finite(change["delta_norm"], nonnegative=True)
        _sha(change["replacement_sha256"])


def validate_record(record, plan_hash, item=None, candidate=None):
    """Reject incomplete/foreign checkpoints; return True, never mutate input."""
    try:
        _sha(plan_hash)
        _require(
            record["schema"] == SCHEMA
            and record["state"] == "complete"
            and record["plan_sha256"] == plan_hash,
            "Checkpoint schema/state/plan differs",
        )
        split = record["split"]
        _require(split in ("calibration", "test"), "Unknown data split")
        technical = record["technical_checks"]
        _require(
            technical["schema"] == SCHEMA
            and technical["state"] == "passed"
            and technical["plan_sha256"] == plan_hash,
            "Technical proof binding differs",
        )
        for checks in (record["checks"], technical["checks"]):
            _require(
                set(checks) == set(PROOF_KEYS)
                and all(checks[k] is True for k in PROOF_KEYS),
                "Technical proofs incomplete or failed",
            )
        source = record["item"]
        _require(
            source["split"] == split
            and isinstance(source["id"], str)
            and bool(source["id"]),
            "Source item/split differs",
        )
        _require(
            isinstance(source["request_id"], str) and bool(source["request_id"]),
            "Source request ID missing",
        )
        if item is not None:
            _require(
                _canonical(source) == _canonical(item), "Frozen source item differs"
            )
        contract = record["array_contract"]
        _require(
            set(contract) == {"coordinates", "hidden_size", "vocab_size"}
            and all(type(x) is int and x > 0 for x in contract.values()),
            "Invalid array contract",
        )
        vocab = contract["vocab_size"]
        for key in ("prompt_ids", "gold_token_ids", "foil_token_ids"):
            _ids(source[key], vocab)
        neutral = record["neutral_token_ids"]
        _ids(neutral, vocab, count=32)
        baseline = record["baseline"]
        _require(
            baseline["native_capture_exact"] is True
            and baseline["beta_one_sham_exact"] is True,
            "Native/sham proofs failed",
        )
        snapshots = baseline["snapshots"]
        _require(
            set(snapshots)
            == {f"{a}_o{o}" for a in ("clean", "pulse") for o in OFFSETS},
            "Baseline snapshots incomplete",
        )
        L = len(source["prompt_ids"])
        n_layers = _cache(snapshots["clean_o0"]["cache"], L)
        _null_checks(baseline, L, n_layers)
        for arm in ("clean", "pulse"):
            for offset, start in ((0, 0), (1, L), (8, L + 1), (32, L + 31)):
                _snapshot(
                    snapshots[f"{arm}_o{offset}"],
                    source,
                    neutral,
                    offset,
                    start,
                    arm == "pulse" and offset == 0,
                    n_layers,
                )
        parity = baseline["cached_full_parity"]
        _require(
            [(x["arm"], x["offset"]) for x in parity]
            == [(a, o) for a in ("clean", "pulse") for o in (8, 32)],
            "Rebuild parity rows incomplete",
        )
        for row in parity:
            _require(
                row["passed"] is True and row["logits_allclose"] is True,
                "Rebuild parity failed",
            )
            _require(
                row["cached"] == snapshots[f"{row['arm']}_o{row['offset']}"],
                "Rebuild cached reference differs",
            )
            _snapshot(
                row["full"],
                source,
                neutral,
                row["offset"],
                0,
                row["arm"] == "pulse",
                n_layers,
            )
            _require(
                row["rtol"] == 1e-3 and row["atol"] == 1e-4, "Rebuild tolerances differ"
            )
            _finite(row["logits_max_abs_error"], nonnegative=True)
            _finite(row["B18_L2_error"], nonnegative=True)
            for key in (
                "array_keys_equal",
                "arrays_allclose",
                "cache_layer_count_equal",
                "kv_allclose",
            ):
                _require(row[key] is True, f"Rebuild {key} failed")
            _parity_metric(row["logits"], (vocab,))
            _require(
                set(row["arrays_parity"]) == set(FIELDS) - {"logits"},
                "Rebuild capture proof inventory differs",
            )
            for key, metric in row["arrays_parity"].items():
                width = (
                    contract["hidden_size"]
                    if key.startswith("residual_")
                    else contract["coordinates"]
                )
                _parity_metric(metric, (1, 1, width))
            _require(
                [r["layer"] for r in row["kv_parity"]] == list(range(n_layers)),
                "Rebuild KV proofs incomplete",
            )
            for index, metric in enumerate(row["kv_parity"]):
                for kind in ("keys", "values"):
                    _parity_metric(
                        metric[kind], row["cached"]["cache"]["layers"][index]["shape"]
                    )
        if split == "calibration":
            _require(
                not any(
                    k in record
                    for k in (
                        "endpoints",
                        "generations",
                        "candidate_sha256",
                        "candidate_binding",
                    )
                ),
                "Calibration record contains test-derived fields",
            )
            for offset in (8, 32):
                _close(
                    record["fit_record"][f"cache_error{offset}"],
                    max(x["B18_L2_error"] for x in parity if x["offset"] == offset),
                    "Calibration numerical floor",
                )
        else:
            _candidate_binding(record, candidate)
            endpoints = record["endpoints"]
            _require(
                [r["name"] for r in endpoints] == list(ARMS),
                "Endpoint arm slots/order differ",
            )
            for row in endpoints:
                name = row["name"]
                patched = name not in ("clean", "pulse", *CACHE)
                _require(
                    row["candidate_sha256"] == record["candidate_sha256"],
                    "Endpoint candidate differs",
                )
                if patched and not record["candidate_valid"]:
                    _require(
                        row["state"] == "not_run"
                        and row["reason"]
                        == "frozen_candidate_failed_prespecified_numerical_gate",
                        "Invalid candidate arm must be explicitly not_run",
                    )
                    _require(
                        not any(k in row for k in ("snapshot", "scores", "readout")),
                        "Not-run arm contains measured data",
                    )
                    continue
                _require(row["state"] == "complete", "Endpoint incomplete")
                _snapshot(
                    row["snapshot"],
                    source,
                    neutral,
                    32,
                    L + 31,
                    False,
                    n_layers,
                    patched=patched,
                )
                _scores(row["scores"], source, row["snapshot"], vocab)
                _top_tokens(row["readout"]["top10"], vocab)
                if name in CACHE:
                    _swap(row, record, n_layers)
                else:
                    _require(row["cache_swap"] is None, "Unexpected cache swap")
                if patched:
                    receiver = "clean" if name.startswith("clean_add_") else "pulse"
                    for index in range(19):
                        _require(
                            row["snapshot"]["cache"]["layers"][index]
                            == snapshots[f"{receiver}_o32"]["cache"]["layers"][index],
                            "Post18 patch changed an upstream/current KV layer",
                        )
            generations = record["generations"]
            _require(
                [r["name"] for r in generations] == list(MAIN),
                "Greedy arm slots/order differ",
            )
            by_name = {r["name"]: r for r in endpoints}
            for row in generations:
                if row["name"] in MAIN[2:] and not record["candidate_valid"]:
                    _require(
                        row["state"] == "not_run"
                        and row["reason"] == "invalid_frozen_candidate",
                        "Invalid candidate generated output",
                    )
                else:
                    _require(row["state"] == "complete", "Greedy record incomplete")
                    _generation(row, by_name[row["name"]], source, vocab)
        return True
    except (KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError(f"Malformed temporal checkpoint: {exc}") from exc


def _lp(logits):
    values = logits.astype(np.float64)
    maximum = values.max()
    return values - (maximum + np.log(np.exp(values - maximum).sum(dtype=np.float64)))


def _logit_checks(array, snapshot):
    _require(_array_sha(array) == snapshot["logits_sha256"], "Saved logits SHA differs")


def validate_arrays(record, arrays):
    """Check exact array inventory, native FP32 geometry and FP64 metric math."""
    try:
        validate_record(record, record["plan_sha256"])
        contract = record["array_contract"]
        H, C, V = (contract[k] for k in ("hidden_size", "coordinates", "vocab_size"))
        prefixes = [f"{a}_o{o}" for a in ("clean", "pulse") for o in OFFSETS]
        prefixes += [f"{a}_full_o{o}" for a in ("clean", "pulse") for o in (8, 32)]
        extra = []
        if record["split"] == "calibration":
            extra = ["calibration_delta8", "calibration_delta32"]
        else:
            for row in record["endpoints"]:
                if row["state"] == "complete":
                    prefixes.append("endpoint_" + row["name"])
                    if row["name"] not in ("clean", "pulse", *CACHE):
                        extra.append("endpoint_" + row["name"] + "_requested_delta")
        expected = {p + "_" + f for p in prefixes for f in FIELDS} | set(extra)
        _require(
            set(arrays) == expected,
            "Saved array inventory differs from completed arm plan",
        )
        for key in expected:
            a = arrays[key]
            shape = (
                (H,)
                if key in extra
                else (V,)
                if key.endswith("_logits")
                else (1, 1, H)
                if "_residual_" in key
                else (1, 1, C)
            )
            _require(
                isinstance(a, np.ndarray)
                and a.dtype == np.dtype("float32")
                and a.shape == shape
                and np.isfinite(a).all(),
                f"Invalid dtype/shape/finite array: {key}",
            )
            if key.endswith(("_cett", "_down_output_norm_raw")):
                _require((a >= 0).all(), f"Negative unsigned measurement: {key}")
        snapshots = record["baseline"]["snapshots"]
        for prefix, snapshot in snapshots.items():
            _logit_checks(arrays[prefix + "_logits"], snapshot)
        for row in record["baseline"]["cached_full_parity"]:
            p, q = (
                f"{row['arm']}_o{row['offset']}",
                f"{row['arm']}_full_o{row['offset']}",
            )
            _logit_checks(arrays[q + "_logits"], row["full"])
            a, b = (
                arrays[p + "_logits"].astype(np.float64),
                arrays[q + "_logits"].astype(np.float64),
            )
            _require(
                np.allclose(a, b, rtol=row["rtol"], atol=row["atol"]),
                "Stored rebuild logits fail parity",
            )
            _close(
                row["logits_max_abs_error"],
                np.max(np.abs(a - b)),
                "Rebuild max logit error",
            )
            difference = arrays[p + "_residual_18"].astype(np.float64) - arrays[
                q + "_residual_18"
            ].astype(np.float64)
            _close(row["B18_L2_error"], np.linalg.norm(difference), "Rebuild B18 error")
            _parity_metric(
                row["logits"], (V,), arrays[p + "_logits"], arrays[q + "_logits"]
            )
            for field, metric in row["arrays_parity"].items():
                _parity_metric(
                    metric,
                    arrays[p + "_" + field].shape,
                    arrays[p + "_" + field],
                    arrays[q + "_" + field],
                )
        if record["split"] == "calibration":
            for offset in (8, 32):
                expected_delta = (
                    arrays[f"pulse_o{offset}_residual_18"][0, 0]
                    - arrays[f"clean_o{offset}_residual_18"][0, 0]
                )
                _require(
                    np.array_equal(
                        arrays[f"calibration_delta{offset}"], expected_delta
                    ),
                    "Calibration delta is not native pulse-minus-clean",
                )
            return True
        refs = {a: _lp(arrays[f"{a}_o32_logits"]) for a in ("clean", "pulse")}
        binding = record["candidate_binding"]
        for row in record["endpoints"]:
            if row["state"] != "complete":
                continue
            name, readout = row["name"], row["readout"]
            prefix = "endpoint_" + name
            logits = arrays[prefix + "_logits"]
            lp = _lp(logits)
            _logit_checks(logits, row["snapshot"])
            _require(
                readout["logits_sha256"] == _array_sha(logits)
                and readout["metric_dtype"] == "float64",
                "Readout logit binding/dtype differs",
            )
            _require(
                readout["argmax_id"] == int(np.argmax(logits)), "Argmax token differs"
            )
            for entry in readout["top10"]:
                _close(
                    entry["logprob"],
                    lp[entry["token_id"]],
                    "Full-vocabulary top-k score",
                )
            threshold = np.partition(lp, -min(10, V))[-min(10, V)]
            _require(
                all(lp[x["token_id"]] >= threshold for x in readout["top10"]),
                "Top-k IDs are not maximal",
            )
            for kind in ("gold", "foil"):
                token = record["item"][f"{kind}_token_ids"][0]
                _require(
                    readout[f"{kind}_first_token_id"] == token,
                    "First target token differs",
                )
                _close(
                    readout[f"{kind}_first_token_logprob"],
                    lp[token],
                    "First-token logprob",
                )
                _close(
                    row["scores"][kind]["first_logprob"],
                    lp[token],
                    "Scoring first-token logprob",
                )
            _require(
                set(readout["references"]) == {"clean", "pulse"}, "KL references differ"
            )
            for ref, ref_lp in refs.items():
                _close(
                    readout["references"][ref]["reference_to_arm_KL"],
                    np.sum(np.exp(ref_lp) * (ref_lp - lp), dtype=np.float64),
                    "KL(reference||arm)",
                )
                _close(
                    readout["references"][ref]["arm_to_reference_KL"],
                    np.sum(np.exp(lp) * (lp - ref_lp), dtype=np.float64),
                    "KL(arm||reference)",
                )
            if name in ("clean", "pulse"):
                for field in FIELDS:
                    _require(
                        np.array_equal(
                            arrays[prefix + "_" + field], arrays[name + "_o32_" + field]
                        ),
                        "Baseline endpoint copy differs",
                    )
            elif name == CACHE[2]:
                for field in FIELDS:
                    _require(
                        np.array_equal(
                            arrays[prefix + "_" + field], arrays["clean_o32_" + field]
                        ),
                        "Both-slot technical endpoint does not reproduce clean arrays",
                    )
            elif name not in CACHE:
                delta = arrays[prefix + "_requested_delta"]
                positive = delta if name.startswith("clean_add_") else -delta
                expected_hash = (
                    binding["vector_sha256"]
                    if name.endswith("_shared")
                    else binding["noise_vector_sha256"][int(name.rsplit("_", 1)[1])]
                )
                _require(
                    _array_sha(positive) == expected_hash,
                    "Applied candidate/noise vector or sign differs",
                )
                _close(
                    np.linalg.norm(delta.astype(np.float64)).item(),
                    binding["native_norm"],
                    "Norm-matched patch",
                    atol=1e-12,
                    rtol=1e-6,
                )
                patch = row["snapshot"]["forward"]["residual_patch"]
                _close(
                    patch["requested_delta_norm"],
                    np.linalg.norm(delta.astype(np.float64)),
                    "Requested patch norm",
                )
                receiver = "clean" if name.startswith("clean_add_") else "pulse"
                before = arrays[receiver + "_o32_residual_18"]
                after = arrays[prefix + "_residual_18"]
                _require(
                    np.array_equal(after, before + delta),
                    "Captured post18 state does not equal native applied patch",
                )
                actual = after.astype(np.float64) - before.astype(np.float64)
                _close(
                    patch["actual_delta_norms"][0][0],
                    np.linalg.norm(actual),
                    "Actual patch norm",
                )
                _close(
                    patch["rounding_error_norm"],
                    np.linalg.norm(actual - delta.astype(np.float64)),
                    "Patch rounding error",
                )
                for layer in (13, 14, 15, 17):
                    _require(
                        np.array_equal(
                            arrays[f"{prefix}_residual_{layer}"],
                            arrays[f"{receiver}_o32_residual_{layer}"],
                        ),
                        "Post18 patch changed an upstream residual",
                    )
        for row in record["generations"]:
            if row["state"] != "complete":
                continue
            lp = _lp(arrays[f"endpoint_{row['name']}_logits"])
            _require(
                row["selected_token_ids"][0] == int(np.argmax(lp)),
                "Greedy first token differs from endpoint",
            )
            for entry in row["steps"][0]["top_tokens"]:
                _close(
                    entry["logprob"],
                    lp[entry["token_id"]],
                    "Greedy initial top-k score",
                )
        return True
    except (KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError(f"Malformed temporal arrays: {exc}") from exc
