"""Synthetic fixed REF-W bindings; no private values, model or NumPy imports."""
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from attention_probe import fixed_refw as F
from attention_probe.reference import MODEL_REPO, MODEL_REVISION, ModelPins, Reference


def fixture():
    coordinates = tuple((i % 28, i // 28) for i in range(286))
    reference = Reference(coordinates, "a" * 64, "b" * 64,
                          ModelPins(MODEL_REPO, MODEL_REVISION, (("synthetic", "d" * 64),)))
    document = {"schema": F.SCHEMA, "source_hn_meta_sha256": "c" * 64, "hn_sha256": "a" * 64,
                    "coordinates": [list(c) for c in coordinates],
                    "refw": {"z_mu": [float(i) for i in range(286)], "z_sd": [0.5] * 286,
                              "c_mu": [0.125] * 286, "c_sd": [0.25] * 286}}
    return reference, document


def encoded(document):
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()


def verify(document, reference=None, raw=None, **overrides):
    if reference is None:
        reference = fixture()[0]
    raw = encoded(document) if raw is None else raw
    anchors = {"expected_sha256": hashlib.sha256(raw).hexdigest(),
                   "expected_source_hn_meta_sha256": "c" * 64, "expected_hn_sha256": "a" * 64}
    anchors.update(overrides)
    return F.verify_fixed_refw(raw, reference, **anchors)


def test_exact_values_and_order_are_preserved_without_normalization():
    reference, document = fixture()
    original = copy.deepcopy(document)
    result = verify(document, reference)
    assert result == original
    assert result["refw"]["z_mu"][-1] == 285.0
    result["coordinates"][0][0] = 99
    result["refw"]["z_sd"][0] = 123
    assert document == original and reference.coordinates[0] == (0, 0)
    assert verify(document, reference) == original
    json.dumps(original, allow_nan=False)


def test_saved_zero_sd_is_preserved_and_floors_are_explicit():
    reference, document = fixture()
    document["refw"]["z_sd"][0] = 0.0
    document["refw"]["c_sd"][0] = 0.0
    result = verify(document, reference)
    assert result["refw"]["z_sd"][0] == result["refw"]["c_sd"][0] == 0.0
    assert F.CETT_EPS == 1e-9 and F.STANDARDIZATION_EPS == 1e-6


def test_checksum_covers_original_bytes_before_parsing():
    reference, document = fixture()
    raw = encoded(document)
    with pytest.raises(F.FixedRefWError, match="SHA-256 mismatch"):
        verify(document, reference, raw=raw + b" ", expected_sha256=hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize("field", ["schema", "source_hn_meta_sha256", "hn_sha256"])
def test_wrong_binding_or_schema_is_rejected(field):
    reference, document = fixture()
    document[field] = "wrong"
    with pytest.raises(F.FixedRefWError):
        verify(document, reference)


def test_reference_selection_hash_must_match_the_anchor():
    reference, document = fixture()
    with pytest.raises(F.FixedRefWError, match="different H selection"):
        verify(document, reference, expected_hn_sha256="e" * 64)


@pytest.mark.parametrize("change", ["reverse", "duplicate", "short", "bool", "float", "bounds"])
def test_coordinate_identity_order_and_types_are_strict(change):
    reference, document = fixture()
    if change == "reverse":
        document["coordinates"].reverse()
    elif change == "duplicate":
        document["coordinates"][1] = document["coordinates"][0]
    elif change == "short":
        document["coordinates"].pop()
    else:
        document["coordinates"][0][0] = {"bool": False, "float": 0.0, "bounds": 28}[change]
    with pytest.raises(F.FixedRefWError):
        verify(document, reference)


@pytest.mark.parametrize("value", [True, "1", None, 0.1, 1e100, float("nan"), float("inf")])
def test_non_numeric_non_finite_or_non_fp32_values_are_rejected(value):
    reference, document = fixture()
    document["refw"]["z_mu"][0] = value
    with pytest.raises(F.FixedRefWError):
        verify(document, reference)


@pytest.mark.parametrize("name", ["z_sd", "c_sd"])
def test_negative_sd_is_rejected(name):
    reference, document = fixture()
    document["refw"][name][0] = -0.25
    with pytest.raises(F.FixedRefWError, match="negative standard"):
        verify(document, reference)


@pytest.mark.parametrize("change", ["missing_vector", "extra_vector", "short_vector", "extra_field"])
def test_document_and_vector_shapes_are_exact(change):
    reference, document = fixture()
    if change == "missing_vector":
        document["refw"].pop("c_mu")
    elif change == "extra_vector":
        document["refw"]["new_stats"] = []
    elif change == "short_vector":
        document["refw"]["c_mu"].pop()
    else:
        document["new_stats"] = 1
    with pytest.raises(F.FixedRefWError):
        verify(document, reference)


def test_duplicate_keys_invalid_json_and_nonfinite_exponents_are_rejected():
    reference, document = fixture()
    duplicate = encoded(document).replace(b'"schema":', b'"schema":"duplicate","schema":', 1)
    for raw in [duplicate, b"{broken", b"\xff", encoded(document).replace(b"0.125", b"1e999", 1)]:
        with pytest.raises(F.FixedRefWError):
            verify(document, reference, raw=raw)


@pytest.mark.parametrize("anchor", ["expected_sha256", "expected_source_hn_meta_sha256", "expected_hn_sha256"])
def test_test_anchors_still_require_sha256_syntax(anchor):
    reference, document = fixture()
    with pytest.raises(F.FixedRefWError, match="lowercase SHA-256"):
        verify(document, reference, **{anchor: "not-a-hash"})


def test_production_loader_cannot_use_synthetic_anchors(tmp_path):
    reference, document = fixture()
    path = tmp_path / "synthetic.json"
    path.write_bytes(encoded(document))
    with pytest.raises(F.FixedRefWError):
        F.load_fixed_refw(path, reference)
    with pytest.raises(TypeError):
        F.load_fixed_refw(path, reference, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_import_is_model_free_and_stdlib_only():
    root = Path(__file__).resolve().parents[1]
    code = "import sys; import attention_probe.fixed_refw; assert not {'torch','transformers','numpy'} & set(sys.modules)"
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)
