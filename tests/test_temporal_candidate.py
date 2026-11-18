import copy

import numpy as np
import pytest

from attention_probe.temporal_candidate import fit_candidate, verify_candidate


def inputs():
    rows = [{"id": f"cal-{i}", "split": "calibration", "delta8": [1., 0., 0.],
             "delta32": [-2., .2, .1], "cache_error8": 1e-8, "cache_error32": 1e-8,
             "checkpoint_sha256": f"{i:064x}"} for i in range(8)]
    return rows, {"calibration_ids": [r["id"] for r in rows],
                  "source_binding": {"plan": "a" * 64, "items": "b" * 64}}


def test_negative_amplitude_shared_and_noise_norms_are_preserved():
    rows, kwargs = inputs()
    before = copy.deepcopy(rows)
    result = fit_candidate(rows, **kwargs)
    assert result["valid"] and result["amplitude"] == -2
    assert result["vector"] == [-2., 0., 0.]
    assert verify_candidate(result, kwargs["calibration_ids"], kwargs["source_binding"])
    assert rows == before
    assert fit_candidate(list(reversed(rows)), **kwargs) == result
    assert fit_candidate(rows, **kwargs)["noise_vectors"] == result["noise_vectors"]
    assert np.allclose(np.linalg.norm(result["noise_vectors"], axis=1), 2)


def test_rejects_heldout_records_and_missing_calibration():
    rows, kwargs = inputs()
    rows[0]["split"] = "test"
    with pytest.raises(ValueError, match="Only designated"):
        fit_candidate(rows, **kwargs)
    rows[0]["split"] = "calibration"
    rows[0]["id"] = "heldout"
    with pytest.raises(ValueError, match="do not match"):
        fit_candidate(rows, **kwargs)


@pytest.mark.parametrize("kind", ["cancellation", "late_zero", "early_noise", "late_noise"])
def test_prespecified_gate_has_no_fallback(kind):
    rows, kwargs = inputs()
    for i, row in enumerate(rows):
        if kind == "cancellation":
            row["delta8"][0] = (-1.) ** i
        elif kind == "late_zero":
            row["delta32"] = [0., 1., 0.]
        elif kind == "early_noise":
            row["cache_error8"] = .2
        else:
            row["cache_error32"] = .3
    result = fit_candidate(rows, **kwargs)
    assert not result["valid"] and result["invalid_reasons"]
    assert verify_candidate(result)


def test_candidate_hash_binds_sources_and_detects_changed_vector():
    rows, kwargs = inputs()
    result = fit_candidate(rows, **kwargs)
    changed = copy.deepcopy(result)
    changed["vector"][0] += .01
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_candidate(changed)
    rows[0]["checkpoint_sha256"] = "f" * 64
    assert fit_candidate(rows, **kwargs)["candidate_sha256"] != result["candidate_sha256"]
