"""A calibration-only, shared temporal direction; no model or test-data fitting.

The sign of the late projection is preserved. The eight random directions are
fixed by one local PCG64 generator and are never selected by observed outcomes.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def candidate_hash(candidate):
    return hashlib.sha256(_canonical({k: v for k, v in candidate.items()
                                     if k != "candidate_sha256"})).hexdigest()


def _array_hash(value):
    return hashlib.sha256(np.asarray(value, dtype="<f4").tobytes(order="C")).hexdigest()


def _finite_vector(value, label):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a nonempty finite vector")
    return result


def _norm(value):
    # Avoid platform/thread-dependent BLAS reductions in a frozen resume hash.
    return math.sqrt(math.fsum(float(x) * float(x) for x in value))


def fit_candidate(calibration_records, *, calibration_ids, source_binding, seed=20261118):
    """Fit once from exactly eight explicitly designated calibration records.

    Each record binds its prior checkpoint and supplies B18 pulse-minus-clean
    vectors at offsets8/32 plus maximum clean/pulse cached/full L2 discrepancies.
    Numerical gates are informativeness checks, not significance tests.
    """
    ids = list(calibration_ids)
    rows = list(calibration_records)
    if len(ids) != 8 or len(set(ids)) != 8 or len(rows) != 8:
        raise ValueError("Exactly eight distinct calibration IDs/records are required")
    if any(row.get("split") != "calibration" for row in rows):
        raise ValueError("Only designated calibration records may enter fitting")
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != 8 or set(by_id) != set(ids):
        raise ValueError("Calibration records do not match the frozen calibration IDs")
    if not isinstance(source_binding, dict) or not source_binding:
        raise ValueError("A nonempty source binding is required")
    rows = [by_id[name] for name in ids]
    early = [_finite_vector(row["delta8"], "delta8") for row in rows]
    late = [_finite_vector(row["delta32"], "delta32") for row in rows]
    if len({a.shape for a in early + late}) != 1:
        raise ValueError("Calibration residual dimensions differ")
    floors = {}
    for offset in (8, 32):
        errors = [float(row[f"cache_error{offset}"]) for row in rows]
        if any(not np.isfinite(x) or x < 0 for x in errors):
            raise ValueError("Cache parity errors must be finite and nonnegative")
        floors[offset] = max(1e-6, 8.0 * max(errors))
    for row in rows:
        value = row.get("checkpoint_sha256", "")
        if (not isinstance(value, str) or len(value) != 64 or
                any(ch not in "0123456789abcdef" for ch in value)):
            raise ValueError("Each calibration row requires an exact checkpoint SHA256")
    mean8 = np.asarray([math.fsum(float(row[j]) for row in early) / 8
                        for j in range(early[0].size)], dtype=np.float64)
    norm8 = _norm(mean8)
    direction = (mean8 / norm8 if norm8 > 0 else np.zeros_like(mean8)).astype(np.float32)
    amplitude = math.fsum(math.fsum(float(a) * float(b) for a, b in zip(row, direction))
                          for row in late) / 8
    vector = (amplitude * direction.astype(np.float64)).astype(np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("Candidate cannot be represented in native FP32")
    norm = _norm(vector)
    rng = np.random.Generator(np.random.PCG64(seed))
    noise = rng.standard_normal((8, direction.size))
    noise /= np.asarray([_norm(row) for row in noise])[:, None]
    noise = (noise * norm).astype(np.float32)
    reasons = []
    if not norm8 > floors[8]:
        reasons.append("mean_offset8_signal_below_prespecified_numeric_floor")
    if not abs(amplitude) > floors[32]:
        reasons.append("signed_offset32_projection_below_prespecified_numeric_floor")
    candidate = {
        "schema": "temporal_shared_candidate_v1",
        "calibration_ids": ids,
        "source_binding": json.loads(_canonical(source_binding)),
        "calibration_checkpoints": {row["id"]: row["checkpoint_sha256"] for row in rows},
        "direction": direction.tolist(), "amplitude": amplitude,
        "vector": vector.tolist(), "noise_vectors": noise.tolist(),
        "valid": not reasons, "invalid_reasons": reasons,
        "diagnostics": {
            "mean_delta8_norm": norm8, "floor8": floors[8], "floor32": floors[32],
            "candidate_native_norm": norm,
            "direction_sha256": _array_hash(direction),
            "vector_sha256": _array_hash(vector),
            "noise_vectors_sha256": _array_hash(noise),
            "noise_seed": int(seed), "noise_generator": "NumPy PCG64",
            "fit_arithmetic": "FP64 means/projections; direction and applied vectors stored FP32",
            "reduction_policy": "fixed calibration order; math.fsum; no BLAS reductions in fit",
            "sign_policy": "signed mean late projection; never absolute-valued or flipped",
            "gate_scope": "prespecified numerical informativeness, not statistical significance",
            "invalid_policy": "do not apply shared/noise patches; continue baseline/cache controls",
        },
    }
    candidate["candidate_sha256"] = candidate_hash(candidate)
    return candidate


def verify_candidate(candidate, expected_calibration_ids=None, expected_source_binding=None):
    """Validate frozen bytes and internal geometry without fitting on new data."""
    if candidate.get("schema") != "temporal_shared_candidate_v1":
        raise ValueError("Unknown candidate schema")
    if candidate.get("candidate_sha256") != candidate_hash(candidate):
        raise ValueError("Frozen candidate hash mismatch")
    if expected_calibration_ids is not None and candidate["calibration_ids"] != list(expected_calibration_ids):
        raise ValueError("Candidate calibration IDs differ")
    if expected_source_binding is not None and candidate["source_binding"] != expected_source_binding:
        raise ValueError("Candidate source binding differs")
    direction = _finite_vector(candidate["direction"], "direction")
    vector = _finite_vector(candidate["vector"], "vector")
    amplitude = float(candidate["amplitude"])
    noise = np.asarray(candidate["noise_vectors"], dtype=np.float64)
    if (vector.shape != direction.shape or noise.shape != (8, vector.size)
            or not np.isfinite(amplitude) or not np.isfinite(noise).all()):
        raise ValueError("Invalid candidate geometry")
    expected_vector = (amplitude * direction).astype(np.float32)
    if not np.array_equal(expected_vector, vector.astype(np.float32)):
        raise ValueError("Candidate vector does not match signed amplitude/direction")
    if candidate["valid"] and not np.isclose(np.linalg.norm(direction), 1, rtol=1e-6, atol=1e-7):
        raise ValueError("Candidate direction is not unit length")
    norm = np.linalg.norm(vector)
    if not np.allclose(np.linalg.norm(noise, axis=1), norm, rtol=1e-6, atol=1e-12):
        raise ValueError("Random controls are not norm matched")
    for name, array in (("direction", direction), ("vector", vector), ("noise_vectors", noise)):
        if _array_hash(array) != candidate["diagnostics"][f"{name}_sha256"]:
            raise ValueError("Native candidate array hash mismatch")
    if bool(candidate["invalid_reasons"]) == bool(candidate["valid"]):
        raise ValueError("Candidate gate status is inconsistent")
    return True


def projection_readout(delta, candidate):
    """Evaluation only: never use held-out deltas to change the frozen patch."""
    value = _finite_vector(delta, "delta")
    direction = _finite_vector(candidate["direction"], "direction")
    if value.shape != direction.shape:
        raise ValueError("Readout dimension differs from frozen candidate")
    return {"signed_projection": float(value @ direction),
            "delta_norm": float(np.linalg.norm(value)),
            "candidate_sha256": candidate["candidate_sha256"]}
