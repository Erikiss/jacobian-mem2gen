"""Checkpoint guards exercised on native CPU Qwen3 data and corrupted copies."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from attention_probe.temporal_validation import (
    PROOF_KEYS,
    SCHEMA,
    validate_arrays,
    validate_record,
)

PLAN = "a" * 64


@pytest.fixture(scope="module")
def cases():
    path = Path(__file__).with_name("test_temporal_functional_runtime.py")
    spec = importlib.util.spec_from_file_location("temporal_native_test_support", path)
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    gen = support.native_engine.__wrapped__()
    _, engine, reference, stats, items = next(gen)
    runtime = support.runtime()
    assert set(runtime.PROOF_KEYS) == set(PROOF_KEYS)
    neutral = [11 + i % 20 for i in range(32)]
    result = {}
    try:
        for name in ("calibration", "valid", "invalid"):
            item = items[0 if name == "calibration" else 8]
            candidate = (
                None
                if name == "calibration"
                else support.fitted_candidate(invalid=name == "invalid")
            )
            if candidate is None:
                record, arrays = runtime.compute_calibration(
                    engine, reference, stats, item, neutral
                )
            else:
                record, arrays = runtime.compute_test(
                    engine, reference, stats, item, neutral, candidate
                )
            record.update(
                schema=SCHEMA,
                plan_sha256=PLAN,
                technical_checks={
                    "schema": SCHEMA,
                    "state": "passed",
                    "plan_sha256": PLAN,
                    "checks": dict(record["checks"]),
                },
            )
            result[name] = record, arrays, item, candidate
        yield result
    finally:
        try:
            next(gen)
        except StopIteration:
            pass


@pytest.mark.parametrize("name", ["calibration", "valid", "invalid"])
def test_complete_native_records_and_all_array_checks(cases, name):
    record, arrays, item, candidate = cases[name]
    before = copy.deepcopy(record)
    assert validate_record(record, PLAN, item=item, candidate=candidate) is True
    assert validate_arrays(record, arrays) is True
    assert record == before


def test_invalid_candidate_preserves_all_slots_without_fake_arrays(cases):
    record, arrays, *_ = cases["invalid"]
    assert sum(r["state"] == "not_run" for r in record["endpoints"]) == 18
    assert sum(r["state"] == "complete" for r in record["endpoints"]) == 5
    assert sum(r["state"] == "not_run" for r in record["generations"]) == 2
    assert not any(k.endswith("requested_delta") for k in arrays)


def test_expected_source_and_candidate_bindings_are_required(cases):
    record, _, item, candidate = cases["valid"]
    altered = copy.deepcopy(item)
    altered["request_id"] += "-different"
    with pytest.raises(ValueError, match="source item"):
        validate_record(record, PLAN, item=altered, candidate=candidate)
    with pytest.raises(ValueError, match="plan"):
        validate_record(record, "b" * 64)
    wrong = cases["invalid"][3]
    with pytest.raises(ValueError, match="candidate"):
        validate_record(record, PLAN, candidate=wrong)


def _endpoint(record, name="clean_add_shared"):
    return next(r for r in record["endpoints"] if r["name"] == name)


def _set_nested(obj, path, value):
    for key in path[:-1]:
        obj = obj[key]
    obj[path[-1]] = value


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("state",), "partial"),
        (("checks", "finite_arrays"), 1),
        (("technical_checks", "checks", "frozen_parameters"), False),
        (("neutral_token_ids", 0), 40),
        (
            (
                "baseline",
                "snapshots",
                "pulse_o8",
                "forward",
                "absolute_capture_positions",
            ),
            [7],
        ),
        (
            (
                "baseline",
                "snapshots",
                "clean_o1",
                "cache",
                "layers",
                1,
                "sequence_length",
            ),
            200,
        ),
        (
            (
                "baseline",
                "null_causal_checks",
                0,
                "checks",
                "bridge_positions_layer15",
                "applicable",
            ),
            True,
        ),
        (
            (
                "baseline",
                "null_causal_checks",
                1,
                "checks",
                "all_positions_layers0_through14",
                "details",
                0,
                "changed_scalars",
            ),
            1,
        ),
        (
            ("baseline", "cached_full_parity", 0, "kv_parity", 15, "keys", "passed"),
            False,
        ),
        (
            (
                "baseline",
                "cached_full_parity",
                0,
                "arrays_parity",
                "raw_z",
                "native_fp32",
            ),
            False,
        ),
        (("candidate_binding", "candidate_sha256"), "f" * 64),
        (("candidate_binding", "noise_vector_sha256"), ["f" * 64]),
        (("generations", 0, "final_cache_length"), 999),
        (("generations", 0, "truncated"), True),
    ],
)
def test_record_corruption_is_rejected(cases, path, value):
    record = copy.deepcopy(cases["valid"][0])
    _set_nested(record, path, value)
    with pytest.raises(ValueError):
        validate_record(record, PLAN)


@pytest.mark.parametrize(
    "corruption", ["count", "score", "kv", "both", "slot", "not_run"]
)
def test_endpoint_corruption_is_rejected(cases, corruption):
    record = copy.deepcopy(cases["valid"][0])
    if corruption == "count":
        record["endpoints"].pop()
    elif corruption == "score":
        _endpoint(record)["scores"]["gold"]["sum_logprob"] += 0.01
    elif corruption == "kv":
        _endpoint(record)["snapshot"]["cache"]["layers"][18]["keys_sha256"] = "f" * 64
    elif corruption == "both":
        _endpoint(record, "pulse_restore_both_cache")["cache_swap"][
            "both_cache_exact_clean_o8"
        ] = False
    elif corruption == "slot":
        _endpoint(record, "pulse_restore_prompt_cache")["cache_swap"][
            "absolute_positions"
        ] = [0]
    else:
        _endpoint(record)["state"] = "not_run"
    with pytest.raises(ValueError):
        validate_record(record, PLAN)


@pytest.mark.parametrize(
    "corruption",
    [
        "extra",
        "missing",
        "dtype",
        "shape",
        "nan",
        "logits",
        "raw_parity",
        "patch_sign",
        "post18",
        "both",
    ],
)
def test_saved_array_corruption_is_rejected(cases, corruption):
    record, originals, *_ = cases["valid"]
    arrays = {k: a.copy() for k, a in originals.items()}
    if corruption == "extra":
        arrays["unplanned"] = np.zeros(1, dtype=np.float32)
    elif corruption == "missing":
        arrays.pop("clean_o0_raw_z")
    elif corruption == "dtype":
        arrays["clean_o0_raw_z"] = arrays["clean_o0_raw_z"].astype(np.float64)
    elif corruption == "shape":
        arrays["clean_o0_raw_z"] = arrays["clean_o0_raw_z"].reshape(-1)
    elif corruption == "nan":
        arrays["clean_o0_raw_z"][0, 0, 0] = np.nan
    elif corruption == "logits":
        arrays["endpoint_clean_logits"][0] += 0.1
    elif corruption == "raw_parity":
        arrays["clean_full_o8_raw_z"][0, 0, 0] += 0.1
    elif corruption == "patch_sign":
        arrays["endpoint_pulse_remove_shared_requested_delta"] *= -1
    elif corruption == "post18":
        arrays["endpoint_clean_add_shared_residual_18"][0, 0, 0] += 0.1
    else:
        arrays["endpoint_pulse_restore_both_cache_raw_z"][0, 0, 0] += 0.1
    with pytest.raises(ValueError):
        validate_arrays(record, arrays)


def test_calibration_delta_is_signed_native_subtraction(cases):
    record, original, *_ = cases["calibration"]
    arrays = dict(original)
    arrays["calibration_delta32"] = -arrays["calibration_delta32"]
    with pytest.raises(ValueError, match="pulse-minus-clean"):
        validate_arrays(record, arrays)


def test_score_and_kl_tolerance_permits_roundoff_not_material_change(cases):
    original, arrays, *_ = cases["valid"]
    record = copy.deepcopy(original)
    for row in record["endpoints"]:
        row["scores"]["gold"]["sum_logprob"] += 1e-12
        row["readout"]["references"]["clean"]["reference_to_arm_KL"] += 1e-12
    assert validate_arrays(record, arrays)
    _endpoint(record)["readout"]["references"]["clean"]["reference_to_arm_KL"] += 1e-5
    with pytest.raises(ValueError, match="KL"):
        validate_arrays(record, arrays)


def test_uniform_ties_allow_topk_to_omit_lowest_argmax_index(cases):
    record, arrays, *_ = cases["valid"]
    assert record["generations"][0]["selected_token_ids"] == [0]
    assert validate_arrays(record, arrays)
