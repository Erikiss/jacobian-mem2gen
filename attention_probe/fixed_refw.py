"""Validate the private, fixed H286 REF-W extract without model imports.

The compact file contains existing population means/SDs, already cast to FP32
as in DSSE's SetReader. This module neither fits nor recomputes normalization.
CETT uses max(||down_proj(z)||, CETT_EPS); zCETT uses the saved c_mu and
max(c_sd, STANDARDIZATION_EPS). Zero saved SD is valid and stays zero here:
the consuming measurement applies the documented denominator floor.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from pathlib import Path
from typing import Any

from .reference import (
    HN_SET_SHA256,
    MLP_WIDTH,
    N_COORDINATES,
    N_LAYERS,
    POLICY_SHA256,
    Reference,
    ReferenceError,
)

FIXED_REFW_SHA256 = "68f96129d8ab00f3c7e6a68a9b4a0bdb58798dfac4d1d405f6cce5ccca0ae5cc"
SOURCE_HN_META_SHA256 = "9c61bc34e804719e5bf5e72b644e8a38a94ad0f6c1708efaef75386acdae49ff"
CETT_EPS = 1e-9
STANDARDIZATION_EPS = 1e-6
SCHEMA = "dsse_h286_refw_v1"
_VECTOR_NAMES = {"z_mu", "z_sd", "c_mu", "c_sd"}
_FIELDS = {"schema", "source_hn_meta_sha256", "hn_sha256", "coordinates", "refw"}


class FixedRefWError(ReferenceError):
    """The supplied extract does not establish the fixed normalization."""


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise FixedRefWError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise FixedRefWError("Duplicate JSON object key.")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise FixedRefWError(f"Non-finite JSON constant: {value}")


def verify_fixed_refw(
    raw: bytes,
    reference: Reference,
    *,
    expected_sha256: str,
    expected_source_hn_meta_sha256: str,
    expected_hn_sha256: str,
) -> dict[str, Any]:
    """Low-level validation with explicit trust anchors for synthetic tests.

    Production callers must use load_fixed_refw(), which fixes all three
    anchors. The source hash binds the previously verified full hn_meta file;
    this compact loader does not claim to inspect that large file again.
    """
    expected_sha256 = _sha(expected_sha256, "Expected compact hash")
    expected_source_hn_meta_sha256 = _sha(expected_source_hn_meta_sha256, "Expected source hash")
    expected_hn_sha256 = _sha(expected_hn_sha256, "Expected H selection hash")
    if not isinstance(raw, bytes):
        raise FixedRefWError("The compact document must be immutable bytes.")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise FixedRefWError("Fixed REF-W SHA-256 mismatch.")
    if not isinstance(reference, Reference) or reference.hn_sha256 != expected_hn_sha256:
        raise FixedRefWError("Reference has a different H selection hash.")
    try:
        document = json.loads(raw, object_pairs_hook=_object, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FixedRefWError("Fixed REF-W is not valid JSON.") from exc
    if not isinstance(document, dict) or set(document) != _FIELDS:
        raise FixedRefWError("Unexpected fixed REF-W document fields.")
    if document["schema"] != SCHEMA:
        raise FixedRefWError("Unsupported fixed REF-W schema.")
    if document["source_hn_meta_sha256"] != expected_source_hn_meta_sha256:
        raise FixedRefWError("Full hn_meta source hash mismatch.")
    if document["hn_sha256"] != expected_hn_sha256:
        raise FixedRefWError("Compact H selection hash mismatch.")

    coordinates = document["coordinates"]
    if not isinstance(coordinates, list) or len(coordinates) != N_COORDINATES:
        raise FixedRefWError("Expected exactly 286 ordered coordinates.")
    checked = []
    for coordinate in coordinates:
        if (not isinstance(coordinate, list) or len(coordinate) != 2
                or any(type(value) is not int for value in coordinate)):
            raise FixedRefWError("Coordinates must be integer pairs.")
        layer, neuron = coordinate
        if not (0 <= layer < N_LAYERS and 0 <= neuron < MLP_WIDTH):
            raise FixedRefWError("Coordinate outside the Qwen3-1.7B MLP architecture.")
        checked.append((layer, neuron))
    if len(set(checked)) != N_COORDINATES or tuple(checked) != reference.coordinates:
        raise FixedRefWError("Coordinates differ from the ordered H286 reference.")

    vectors = document["refw"]
    if not isinstance(vectors, dict) or set(vectors) != _VECTOR_NAMES:
        raise FixedRefWError("Expected exactly z_mu, z_sd, c_mu and c_sd.")
    for name, vector in vectors.items():
        if not isinstance(vector, list) or len(vector) != N_COORDINATES:
            raise FixedRefWError(f"{name} must have exactly 286 values.")
        for value in vector:
            if type(value) not in (int, float):
                raise FixedRefWError(f"{name} must contain finite FP32 numbers.")
            try:
                fp32 = struct.unpack("<f", struct.pack("<f", value))[0]
            except (OverflowError, struct.error) as exc:
                raise FixedRefWError(f"{name} contains a value outside FP32.") from exc
            if not math.isfinite(fp32) or fp32 != value:
                raise FixedRefWError(f"{name} contains a non-finite or non-FP32 value.")
            if name.endswith("_sd") and value < 0:
                raise FixedRefWError(f"{name} cannot contain negative standard deviations.")
    return document


def load_fixed_refw(path: str | Path, reference: Reference) -> dict[str, Any]:
    """Load only the exact production extract bound to the original reference."""
    if not isinstance(reference, Reference) or reference.policy_sha256 != POLICY_SHA256:
        raise FixedRefWError("Expected the original production reference policy.")
    return verify_fixed_refw(
        Path(path).read_bytes(), reference,
        expected_sha256=FIXED_REFW_SHA256,
        expected_source_hn_meta_sha256=SOURCE_HN_META_SHA256,
        expected_hn_sha256=HN_SET_SHA256,
    )
