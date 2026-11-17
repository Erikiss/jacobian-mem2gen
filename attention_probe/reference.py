"""Bind a new attention pilot to the private, exact DSSE H_tq selection.

This module validates identities and provenance only. It imports no model runtime,
contains no private neuron coordinates, and does not map them to another model.

    python -m attention_probe.reference --hn-set /private/hn_set.json \
        --policy /private/htq_policy.json --out /private/reference.json
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


HN_SET_SHA256 = "11390233b62c63b9d34805482ecbd3675eadd80570b9c01ef9e268bf565c98fd"
POLICY_SHA256 = "160c420603b719afba4b909d12d31316535290e13b00aea2a6dea88110684f7c"
MODEL_ID = "qwen3-1.7b"
MODEL_REPO = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
N_COORDINATES, N_LAYERS, MLP_WIDTH = 286, 28, 6144
H_POLICY = "htq_v2"


class ReferenceError(ValueError):
    """The supplied files do not establish the required DSSE reference."""


@dataclass(frozen=True, slots=True)
class ModelPins:
    repo: str
    revision: str
    sha256: tuple[tuple[str, str], ...]
    model_id: str = MODEL_ID
    n_layers: int = N_LAYERS
    mlp_width: int = MLP_WIDTH

    def to_dict(self) -> dict[str, Any]:
        return dict(model_id=self.model_id, repo=self.repo, revision=self.revision,
                    sha256=dict(self.sha256), n_layers=self.n_layers, mlp_width=self.mlp_width)


@dataclass(frozen=True, slots=True)
class Reference:
    coordinates: tuple[tuple[int, int], ...]
    hn_sha256: str
    policy_sha256: str
    model: ModelPins
    h_policy: str = H_POLICY
    post_hoc: bool = True
    amendment: int = 2

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh JSON binding; its coordinates remain private run data."""
        return dict(schema="dsse_h286_reference_v1", coordinates=[list(x) for x in self.coordinates],
                    coordinate_order="hn_set.H_op (identical to H_tq)",
                    n_coordinates=len(self.coordinates),
                    source_sha256={"hn_set.json": self.hn_sha256, "htq_policy.json": self.policy_sha256},
                    model=self.model.to_dict(), h_policy=self.h_policy, post_hoc=self.post_hoc,
                    amendment=self.amendment, confirmatory_eligible=False,
                    scope="Neuron identities only; no CETT equivalence or cross-model coordinate mapping.")


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ReferenceError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out = {}
    for key, value in pairs:
        if key in out:
            raise ReferenceError("Duplicate JSON object key.")
        out[key] = value
    return out


def _reject_constant(value: str) -> Any:
    raise ReferenceError("Non-finite JSON constants are not allowed.")


def _document(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_object, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReferenceError(f"{label} is not valid JSON.") from exc
    if not isinstance(value, dict):
        raise ReferenceError(f"{label} must be a JSON object.")
    return value


def _coordinates(value: Any, label: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list) or len(value) != N_COORDINATES:
        raise ReferenceError(f"{label} must contain exactly {N_COORDINATES} coordinates.")
    result = []
    for i, coordinate in enumerate(value):
        if not isinstance(coordinate, list) or len(coordinate) != 2 or any(type(x) is not int for x in coordinate):
            raise ReferenceError(f"{label}[{i}] must be a pair of integer coordinates.")
        layer, index = coordinate
        if not (0 <= layer < N_LAYERS and 0 <= index < MLP_WIDTH):
            raise ReferenceError(f"{label}[{i}] is outside the 28 by 6144 MLP architecture.")
        result.append((layer, index))
    if len(set(result)) != N_COORDINATES:
        raise ReferenceError(f"{label} contains duplicate coordinates.")
    return tuple(result)


def _policy(document: dict[str, Any], label: str) -> None:
    if document.get("h_policy") != H_POLICY or document.get("post_hoc") is not True:
        raise ReferenceError(f"{label} must identify the post-hoc htq_v2 policy.")
    if type(document.get("amendment")) is not int or document["amendment"] != 2:
        raise ReferenceError(f"{label} must identify amendment 2.")
    if document.get("H_op_name") != "H_tq":
        raise ReferenceError(f"{label} must use H_tq as the operative set.")


def _model(document: dict[str, Any], label: str) -> ModelPins:
    provenance = document.get("provenance")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("models"), dict):
        raise ReferenceError(f"{label} has no model provenance.")
    for field, expected in (("h_policy", H_POLICY), ("post_hoc", True), ("amendment", 2), ("mock", False)):
        if field in provenance and (type(provenance[field]) is not type(expected) or provenance[field] != expected):
            raise ReferenceError(f"{label} has conflicting {field} provenance.")
    model = provenance["models"].get(MODEL_ID)
    if not isinstance(model, dict) or model.get("repo") != MODEL_REPO or model.get("revision") != MODEL_REVISION:
        raise ReferenceError(f"{label} does not identify the pinned Qwen3-1.7B model.")
    hashes = model.get("sha256")
    if not isinstance(hashes, dict) or not hashes or any(not isinstance(k, str) or not k for k in hashes):
        raise ReferenceError(f"{label} has no model-file SHA-256 pins.")
    pins = tuple(sorted((name, _sha(digest, f"{label} model file")) for name, digest in hashes.items()))
    return ModelPins(repo=model["repo"], revision=model["revision"], sha256=pins)


def verify_documents(hn_bytes: bytes, policy_bytes: bytes, *, expected_hn_sha256: str,
                     expected_policy_sha256: str) -> Reference:
    """Low-level verification with explicit trust anchors, including synthetic tests.

    Normal pilot code must call load_reference(), whose anchors are the production
    hashes. Checksums cover original bytes, so JSON reformatting is not equivalent.
    """
    expected_hn_sha256 = _sha(expected_hn_sha256, "Expected hn_set hash")
    expected_policy_sha256 = _sha(expected_policy_sha256, "Expected policy hash")
    if not isinstance(hn_bytes, bytes) or not isinstance(policy_bytes, bytes):
        raise ReferenceError("Reference documents must be immutable bytes.")
    hn_digest, policy_digest = _digest(hn_bytes), _digest(policy_bytes)
    if hn_digest != expected_hn_sha256:
        raise ReferenceError("hn_set.json SHA-256 mismatch.")
    if policy_digest != expected_policy_sha256:
        raise ReferenceError("htq_policy.json SHA-256 mismatch.")
    hn, policy = _document(hn_bytes, "hn_set.json"), _document(policy_bytes, "htq_policy.json")
    _policy(hn, "hn_set.json"); _policy(policy, "htq_policy.json")
    coordinates = _coordinates(hn.get("H_op"), "H_op")
    if coordinates != _coordinates(hn.get("H_tq"), "H_tq"):
        raise ReferenceError("H_op and H_tq must be identical, including coordinate order.")
    if type(hn.get("n_H")) is not int or hn["n_H"] != N_COORDINATES:
        raise ReferenceError("hn_set n_H must equal 286.")
    if type(policy.get("n_H_op")) is not int or policy["n_H_op"] != N_COORDINATES:
        raise ReferenceError("Policy n_H_op must equal 286.")
    hn_model, policy_model = _model(hn, "hn_set.json"), _model(policy, "htq_policy.json")
    if hn_model != policy_model:
        raise ReferenceError("The two files have different model-file pins.")
    return Reference(coordinates=coordinates, hn_sha256=hn_digest,
                     policy_sha256=policy_digest, model=hn_model)


def load_reference(hn_set_path: str | Path, policy_path: str | Path) -> Reference:
    """Load only the exact DSSE production files; no caller-supplied hash bypass."""
    return verify_documents(Path(hn_set_path).read_bytes(), Path(policy_path).read_bytes(),
                            expected_hn_sha256=HN_SET_SHA256, expected_policy_sha256=POLICY_SHA256)


def write_reference(reference: Reference, out: str | Path) -> None:
    """Atomically write a binding to the caller's explicit private output path."""
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(reference.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hn-set", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.resolve() in (args.hn_set.resolve(), args.policy.resolve()):
        parser.error("--out must not overwrite a source document.")
    try:
        reference = load_reference(args.hn_set, args.policy)
        write_reference(reference, args.out)
    except (OSError, ReferenceError) as exc:
        parser.error(str(exc))
    print(f"Validated {len(reference.coordinates)} H_tq coordinates; private binding written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
