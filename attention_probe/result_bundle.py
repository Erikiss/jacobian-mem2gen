"""Lossless state summary of a completed/diagnostic ZIP, never a resume archive.

Original JSON/log bytes stay unchanged. Their NPZ/logit hashes still name the
full archive; transformed state NPZs receive different filenames and their own
hashes. Full-vocabulary KL remains a recorded metric once logits are omitted.
Only the existing local ZIP is read; this module uses no model or network.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

ARCHIVE_KIND = "analysis_summary_not_resume"
PACKAGE_FILE = "COMPACT_PACKAGE.json"
MANIFEST_FILE = "COMPACT_MANIFEST.json"
FULL_MANIFEST_FILE = "full_archive_manifest.json"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()


def _vocab_sizes(document):
    sizes = set()
    if isinstance(document, dict):
        for key, value in document.items():
            if (
                key == "vocab_size"
                and isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
            ):
                sizes.add(value)
            sizes.update(_vocab_sizes(value))
    elif isinstance(document, list):
        for value in document:
            sizes.update(_vocab_sizes(value))
    return sizes


def _array_metadata(array: np.ndarray) -> dict:
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "nbytes": int(array.nbytes),
        "sha256": _sha(array.tobytes(order="C")),
        "hash_order": "C contiguous raw bytes",
    }


def _filtered_npz(raw: bytes, vocab_sizes: set[int]):
    included, omitted, states = {}, {}, {}
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("Duplicate NPZ array keys")
        for key in archive.files:
            array = archive[key]  # Object arrays are rejected by allow_pickle=False.
            metadata = _array_metadata(array)
            name = key.lower()
            named_logits = re.search(r"(?:^|[/._])(?:full_)?logits$", name) is not None
            vocabulary_vector = (
                "logits" in name and array.ndim == 1 and array.shape[0] in vocab_sizes
            )
            if named_logits or vocabulary_vector:
                omitted[key] = metadata
            else:
                included[key] = metadata
                states[key] = array
    output = io.BytesIO()
    np.savez_compressed(output, **states)
    return output.getvalue(), included, omitted


def build_compact_result(full_zip: Path, compact_zip: Path) -> dict:
    """Atomically create a separately hashed analysis ZIP, preserving the full ZIP.

    A malformed input raises and leaves an existing compact destination intact.
    The caller can then offer the full archive instead. All arrays retained in
    state NPZs keep their exact dtype/shape/values; there is no FP16 conversion.
    Excluded logits have raw-C-byte SHA256, dtype, shape and nbytes receipts.
    """
    full, compact = Path(full_zip), Path(compact_zip)
    if full.resolve() == compact.resolve() or (
        compact.exists() and os.path.samefile(full, compact)
    ):
        raise ValueError("Compact output must not overwrite the full result archive")
    full_sha, full_bytes = _file_sha(full), full.stat().st_size
    payload, transforms, vocab_sizes = {}, [], set()
    plan_sha = plan_file_sha = None
    with zipfile.ZipFile(full) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP members")
        if any(
            name in names for name in (PACKAGE_FILE, MANIFEST_FILE, FULL_MANIFEST_FILE)
        ):
            raise ValueError(
                "Input is already a compact analysis summary or uses reserved names"
            )
        for member in members:
            path = PurePosixPath(member.filename)
            if (
                path.is_absolute()
                or member.filename.rstrip("/") != str(path)
                or ".." in path.parts
                or "\\" in member.filename
                or member.flag_bits & 1
                or stat.S_ISLNK(member.external_attr >> 16)
            ):
                raise ValueError("Unsafe, encrypted or noncanonical ZIP member")
        if archive.testzip() is not None:
            raise ValueError("Full ZIP CRC verification failed")
        # Inspect metadata, preserving the exact original bytes when copied later.
        for member in members:
            if member.is_dir() or not member.filename.endswith(".json"):
                continue
            raw = archive.read(member)
            document = json.loads(raw)
            vocab_sizes.update(_vocab_sizes(document))
            if member.filename in ("run_plan.json", "plan.json") and plan_sha is None:
                canonical = json.dumps(
                    document,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode()
                plan_sha, plan_file_sha = _sha(canonical), _sha(raw)
        for member in members:
            if member.is_dir():
                continue
            name, raw = member.filename, archive.read(member)
            transformed = name
            if name.endswith(".npz"):
                original_sha = _sha(raw)
                original_bytes = len(raw)
                raw, included, excluded = _filtered_npz(raw, vocab_sizes)
                path = PurePosixPath(name)
                basename = (
                    "state_arrays.npz"
                    if path.name == "arrays.npz"
                    else path.stem + "_state_arrays.npz"
                )
                transformed = str(path.with_name(basename))
                transforms.append(
                    {
                        "original_path": name,
                        "original_sha256": original_sha,
                        "original_bytes": original_bytes,
                        "transformed_path": transformed,
                        "transformed_sha256": _sha(raw),
                        "transformed_bytes": len(raw),
                        "included_keys": list(included),
                        "excluded_keys": list(excluded),
                        "included_arrays": included,
                        "excluded_arrays": excluded,
                    }
                )
            elif name == "WRAPPER_MANIFEST.json":
                transformed = FULL_MANIFEST_FILE
            if transformed in payload:
                raise ValueError("Transformed ZIP member collision")
            payload[transformed] = raw
    included_keys = [key for row in transforms for key in row["included_keys"]]
    available = {
        "state_geometry": bool(included_keys),
        "jvp_error": any(
            any(
                word in key.lower()
                for word in ("jvp", "fd", "transport", "jacobian", "jac_")
            )
            for key in included_keys
        ),
        "full_vocab_KL": False,
    }
    metadata = {
        "schema": "compact_result_v1",
        "archive_kind": ARCHIVE_KIND,
        "resume_eligible": False,
        "full_archive": {
            "filename": full.name,
            "sha256": full_sha,
            "bytes": full_bytes,
        },
        "full_archive_payload_included": False,
        "npz_transforms": transforms,
        "plan_sha256": plan_sha,
        "plan_file_sha256": plan_file_sha,
        "plan_canonicalization": "sorted compact UTF8 JSON, ensure_ascii=False",
        "metrics_source": "runtime_recorded_fp64",
        "independent_recompute_available": available,
        "availability_scope": "Only corresponding preserved measurements are available; no new measurement is implied.",
        "original_record_hash_scope": "Unchanged JSON NPZ/logit hashes and full_archive_manifest.json refer to the full archive, not transformed state files.",
        "not_resume_reason": "Full arrays.npz members are replaced by separately named state-only NPZs; omitted full-vocabulary values are not embedded.",
    }
    payload[PACKAGE_FILE] = _json(metadata)
    manifest = {
        name: {"sha256": _sha(raw), "bytes": len(raw)}
        for name, raw in sorted(payload.items())
    }
    payload[MANIFEST_FILE] = _json(manifest)
    compact.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=compact.parent, prefix="." + compact.name + ".", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(
            temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for name, raw in sorted(payload.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o600 << 16
                archive.writestr(
                    info, raw, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6
                )
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("Compact ZIP CRC verification failed")
        os.replace(temporary, compact)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "archive_kind": ARCHIVE_KIND,
        "resume_eligible": False,
        "compact_path": str(compact),
        "compact_sha256": _file_sha(compact),
        "compact_bytes": compact.stat().st_size,
        "full_archive_filename": full.name,
        "full_archive_sha256": full_sha,
        "full_archive_bytes": full_bytes,
        "npz_count": len(transforms),
        "included_array_count": len(included_keys),
        "excluded_array_count": sum(len(row["excluded_keys"]) for row in transforms),
        "plan_sha256": plan_sha,
        "metrics_source": "runtime_recorded_fp64",
        "independent_recompute_available": available,
    }
