"""Compact result integrity; synthetic numeric arrays only, no model/network."""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile

import numpy as np
import pytest

from attention_probe.result_bundle import ARCHIVE_KIND, build_compact_result


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def npz_bytes(arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def fixture(full, *, npz_raw=None, extra=None):
    states = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    states[0, 0, 0] = -0.0
    arrays = {
        "raw_z": states,
        "jac_target_actual": np.array([1e-40, -0.0, 2.0, 3.0], np.float32),
        "jac_target_predicted": np.arange(4, dtype=np.float32),
        "central_fd": np.ones(4, np.float32),
        "transport_delta_source": np.ones(3, np.float32),
        "prefill_sham_logits": np.arange(32, dtype=np.float32),
        "nested/full_logits": np.ones((1, 32), np.float32),
        "unusual_logits_vector": np.ones(32, np.float32),
        "top10_logits_values": np.arange(10, dtype=np.float32),
        "logits_sensitivity": np.arange(4, dtype=np.float32),
    }
    raw = npz_bytes(arrays) if npz_raw is None else npz_raw
    plan = b'{"vocab_size":32,"schema":"fixture"}\n'
    result = (
        b'{"state":"complete","array_contract":{"vocab_size":32},"npz_sha256":"'
        + sha(raw).encode()
        + b'","full_vocab_KL":0.123456789012345}\n'
    )
    entries = {
        "run_plan.json": plan,
        "requests/r000/result.json": result,
        "requests/r000/arrays.npz": raw,
        "runtime.log": b"Stage: complete\noriginal log\n",
    }
    entries["WRAPPER_MANIFEST.json"] = json.dumps(
        {name: sha(value) for name, value in entries.items()}
    ).encode()
    entries.update(extra or {})
    with zipfile.ZipFile(full, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
    return arrays, entries


def test_roundtrip_states_are_bitwise_exact_logits_excluded_and_full_unchanged(
    tmp_path,
):
    full, compact = tmp_path / "full.zip", tmp_path / "compact.zip"
    arrays, entries = fixture(full)
    original = full.read_bytes()
    summary = build_compact_result(full, compact)
    assert full.read_bytes() == original
    assert summary["full_archive_sha256"] == sha(original) and summary[
        "compact_sha256"
    ] == sha(compact.read_bytes())
    assert (
        summary["archive_kind"] == ARCHIVE_KIND and summary["resume_eligible"] is False
    )
    assert summary["excluded_array_count"] == 3
    assert summary["independent_recompute_available"] == {
        "state_geometry": True,
        "jvp_error": True,
        "full_vocab_KL": False,
    }
    with zipfile.ZipFile(compact) as archive:
        assert archive.testzip() is None
        assert "requests/r000/arrays.npz" not in archive.namelist()
        assert "requests/r000/state_arrays.npz" in archive.namelist()
        assert "WRAPPER_MANIFEST.json" not in archive.namelist()
        assert (
            archive.read("full_archive_manifest.json")
            == entries["WRAPPER_MANIFEST.json"]
        )
        assert (
            archive.read("requests/r000/result.json")
            == entries["requests/r000/result.json"]
        )
        assert archive.read("runtime.log") == entries["runtime.log"]
        assert archive.read("run_plan.json") == entries["run_plan.json"]
        metadata = json.loads(archive.read("COMPACT_PACKAGE.json"))
        assert metadata["full_archive"] == {
            "filename": "full.zip",
            "sha256": sha(original),
            "bytes": len(original),
        }
        assert metadata["full_archive_payload_included"] is False
        assert metadata["metrics_source"] == "runtime_recorded_fp64"
        transform = metadata["npz_transforms"][0]
        assert transform["original_sha256"] == sha(entries["requests/r000/arrays.npz"])
        assert transform["transformed_sha256"] == sha(
            archive.read("requests/r000/state_arrays.npz")
        )
        excluded = {
            "prefill_sham_logits",
            "nested/full_logits",
            "unusual_logits_vector",
        }
        assert set(transform["excluded_keys"]) == excluded
        for key in excluded:
            receipt = transform["excluded_arrays"][key]
            assert receipt["sha256"] == sha(arrays[key].tobytes(order="C"))
            assert receipt["dtype"] == "float32" and receipt["shape"] == list(
                arrays[key].shape
            )
            assert receipt["nbytes"] == arrays[key].nbytes
        with np.load(
            io.BytesIO(archive.read("requests/r000/state_arrays.npz")),
            allow_pickle=False,
        ) as states:
            assert set(states.files) == set(arrays) - excluded
            for key in states.files:
                assert (
                    states[key].dtype == arrays[key].dtype
                    and states[key].shape == arrays[key].shape
                )
                assert states[key].tobytes(order="C") == arrays[key].tobytes(order="C")
        manifest = json.loads(archive.read("COMPACT_MANIFEST.json"))
        assert set(archive.namelist()) == set(manifest) | {"COMPACT_MANIFEST.json"}
        for name, receipt in manifest.items():
            value = archive.read(name)
            assert receipt == {"sha256": sha(value), "bytes": len(value)}
        document = json.loads(entries["run_plan.json"])
        canonical = json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        assert metadata["plan_sha256"] == summary["plan_sha256"] == sha(canonical)


def test_compact_is_explicitly_not_a_resume_archive_and_cannot_be_compacted_again(
    tmp_path,
):
    full, compact = tmp_path / "full.zip", tmp_path / "compact.zip"
    fixture(full)
    build_compact_result(full, compact)
    with zipfile.ZipFile(compact) as archive:
        meta = json.loads(archive.read("COMPACT_PACKAGE.json"))
        assert (
            meta["archive_kind"] == "analysis_summary_not_resume"
            and meta["resume_eligible"] is False
        )
        assert json.loads(archive.read("requests/r000/result.json"))[
            "npz_sha256"
        ] != sha(archive.read("requests/r000/state_arrays.npz"))
        assert all(not name.endswith("/arrays.npz") for name in archive.namelist())
    with pytest.raises(ValueError, match="already a compact"):
        build_compact_result(compact, tmp_path / "wrong.zip")


def test_empty_diagnostic_preserves_failure_and_has_no_invented_measurements(tmp_path):
    full = tmp_path / "diagnostic.zip"
    compact = tmp_path / "diagnostic_compact.zip"
    status = b'{"state":"failed","stage":"preflight"}\n'
    with zipfile.ZipFile(full, "w") as archive:
        archive.writestr("status.json", status)
        archive.writestr("traceback.txt", "no GPU")
    summary = build_compact_result(full, compact)
    assert (
        summary["npz_count"] == 0
        and summary["included_array_count"] == summary["excluded_array_count"] == 0
    )
    assert not summary["independent_recompute_available"]["state_geometry"]
    assert not summary["independent_recompute_available"]["jvp_error"]
    with zipfile.ZipFile(compact) as archive:
        assert archive.read("status.json") == status
        assert json.loads(archive.read("COMPACT_PACKAGE.json"))["plan_sha256"] is None


@pytest.mark.parametrize("alias", ["direct", "hardlink", "symlink"])
def test_full_archive_can_never_be_overwritten_through_alias(tmp_path, alias):
    full = tmp_path / "full.zip"
    fixture(full)
    original = full.read_bytes()
    out = full
    if alias == "hardlink":
        out = tmp_path / "hard.zip"
        os.link(full, out)
    if alias == "symlink":
        out = tmp_path / "sym.zip"
        out.symlink_to(full)
    with pytest.raises(ValueError, match="must not overwrite"):
        build_compact_result(full, out)
    assert full.read_bytes() == original


@pytest.mark.parametrize("bad", ["bad_npz", "object_array", "bad_json"])
def test_malformed_payload_leaves_existing_output_and_full_archive_unchanged(
    tmp_path, bad
):
    full, compact = tmp_path / "full.zip", tmp_path / "compact.zip"
    if bad == "bad_npz":
        fixture(full, npz_raw=b"not numpy")
    elif bad == "object_array":
        fixture(full, npz_raw=npz_bytes({"raw_z": np.array([object()], dtype=object)}))
    else:
        fixture(full, extra={"bad.json": b"not JSON"})
    original = full.read_bytes()
    compact.write_bytes(b"previous output")
    with pytest.raises(ValueError):
        build_compact_result(full, compact)
    assert full.read_bytes() == original and compact.read_bytes() == b"previous output"
    assert not list(tmp_path.glob(".compact.zip.*.tmp"))


def test_atomic_publish_failure_cleans_temporary_and_retains_previous_compact(
    tmp_path, monkeypatch
):
    full, compact = tmp_path / "full.zip", tmp_path / "compact.zip"
    fixture(full)
    original = full.read_bytes()
    compact.write_bytes(b"previous")

    def fail(*_args):
        raise OSError("simulated publish failure")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="publish failure"):
        build_compact_result(full, compact)
    assert full.read_bytes() == original and compact.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".compact.zip.*.tmp"))


def test_full_crc_failure_stops_before_producing_output(tmp_path):
    full, compact = tmp_path / "bad_crc.zip", tmp_path / "compact.zip"
    with zipfile.ZipFile(full, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("runtime.log", b"unique CRC payload")
    raw = bytearray(full.read_bytes())
    raw[raw.index(b"unique CRC payload")] ^= 1
    full.write_bytes(raw)
    with pytest.raises(ValueError, match="CRC"):
        build_compact_result(full, compact)
    assert not compact.exists()


@pytest.mark.parametrize(
    "name", ["../bad.txt", "/absolute.txt", "folder\\bad.txt", "folder//bad.txt"]
)
def test_noncanonical_members_are_rejected_without_extraction(tmp_path, name):
    full = tmp_path / "full.zip"
    with zipfile.ZipFile(full, "w") as archive:
        archive.writestr(name, "invalid")
    with pytest.raises(ValueError, match="noncanonical"):
        build_compact_result(full, tmp_path / "out.zip")


def test_duplicated_members_fail_instead_of_ambiguous_manifest(tmp_path):
    full = tmp_path / "full.zip"
    with zipfile.ZipFile(full, "w") as archive:
        archive.writestr("runtime.log", "first")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("runtime.log", "second")
    with pytest.raises(ValueError, match="Duplicate ZIP"):
        build_compact_result(full, tmp_path / "out.zip")
