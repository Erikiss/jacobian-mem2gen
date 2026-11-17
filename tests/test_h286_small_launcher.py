"""Offline launcher tests; never execute the actual model notebook."""

# ruff: noqa: S102 -- Execute authored synthetic notebook code under local upload/download stubs.
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import types
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "small_builder", ROOT / "notebooks/build_h286_small_launcher.py"
)
assert SPEC and SPEC.loader
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


def synthetic_notebook(final=None):
    sources = [
        "from pathlib import Path\nimport json\nassert __name__ == '__main__'\ntrace=['first']\nshared=17\n",
        "trace.append(shared+1)\n",
        final or "Path('executed.json').write_text(json.dumps(trace))\n",
    ]
    cells = [
        {
            "cell_type": "markdown",
            "source": ["raise AssertionError('markdown executed')"],
        }
    ]
    cells += [
        {"cell_type": "code", "source": source.splitlines(keepends=True)}
        for source in sources
    ]
    return json.dumps({"nbformat": 4, "cells": cells}).encode()


def execute(
    tmp_path,
    monkeypatch,
    packed,
    notebook,
    *,
    expected_package=None,
    existing=False,
    uploads=None,
    upload_error=None,
):
    document = B.launcher_document(
        expected_package or hashlib.sha256(packed).hexdigest(),
        hashlib.sha256(notebook).hexdigest(),
        len(notebook),
    )
    source = "".join(document["cells"][1]["source"])
    delivered = []
    upload_calls = []

    def upload():
        upload_calls.append(True)
        if upload_error:
            raise upload_error
        return (
            uploads if uploads is not None else {"h286_pilot_paket (1)_1_.zip": packed}
        )

    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(
        upload=upload, download=lambda path: delivered.append(Path(path))
    )
    google = types.ModuleType("google")
    google.colab = colab
    if existing:
        (tmp_path / B.PACKAGE).write_bytes(packed)
    error = None
    namespace = {"__name__": "__main__"}
    with monkeypatch.context() as p:
        p.chdir(tmp_path)
        p.setitem(sys.modules, "google", google)
        p.setitem(sys.modules, "google.colab", colab)
        try:
            exec(compile(source, "small-launcher", "exec"), namespace)
        except RuntimeError as caught:
            error = caught
    return error, namespace, delivered, upload_calls


def diagnostic(path):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        return json.loads(z.read("bootstrap_status.json"))


def test_upload_android_renames_shared_scope_and_markdown_is_never_executed(
    tmp_path, monkeypatch
):
    nb = synthetic_notebook()
    error, ns, downloads, calls = execute(
        tmp_path, monkeypatch, B.package_bytes(nb), nb
    )
    assert error is None and calls == [True] and downloads == []
    assert json.loads((tmp_path / "executed.json").read_text()) == ["first", 18]
    assert ns["started"] == 3


def test_generic_jupyter_reads_existing_zip_without_upload(tmp_path, monkeypatch):
    nb = synthetic_notebook()
    error, _, _, calls = execute(
        tmp_path, monkeypatch, B.package_bytes(nb), nb, existing=True
    )
    assert error is None and calls == [] and (tmp_path / "executed.json").exists()


def test_wrong_outer_hash_stops_before_any_code_and_downloads_diagnostic(
    tmp_path, monkeypatch
):
    nb = synthetic_notebook()
    packed = B.package_bytes(nb)
    error, _, downloads, _ = execute(
        tmp_path,
        monkeypatch,
        packed + b"changed",
        nb,
        expected_package=hashlib.sha256(packed).hexdigest(),
    )
    assert error and len(downloads) == 1 and not (tmp_path / "executed.json").exists()
    assert diagnostic(tmp_path / downloads[0])["code_cells_started"] == 0


@pytest.mark.parametrize("kind", ["extra_member", "traversal", "oversize"])
def test_member_and_size_bounds_are_checked_before_crc_or_read(
    tmp_path, monkeypatch, kind
):
    nb = synthetic_notebook()
    stream = io.BytesIO()
    content = b"0" * (2 * 1024**2 + 1) if kind == "oversize" else nb
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("../" + B.MEMBER if kind == "traversal" else B.MEMBER, content)
        if kind == "extra_member":
            z.writestr("extra.txt", "forbidden")
    monkeypatch.setattr(
        zipfile.ZipFile,
        "testzip",
        lambda self: (_ for _ in ()).throw(AssertionError("too early")),
    )
    error, ns, downloads, _ = execute(tmp_path, monkeypatch, stream.getvalue(), content)
    assert error and ns["started"] == 0 and len(downloads) == 1
    # Read status directly; the patched testzip itself must never be invoked.
    with zipfile.ZipFile(tmp_path / downloads[0]) as z:
        status = json.loads(z.read("bootstrap_status.json"))
    assert status["error_type"] == "ValueError"


def test_corrupt_crc_is_diagnosed_before_execution(tmp_path, monkeypatch):
    nb = synthetic_notebook()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr(B.MEMBER, nb)
    packed = bytearray(stream.getvalue())
    start = packed.index(nb)
    packed[start + 15] ^= 1
    error, ns, downloads, _ = execute(tmp_path, monkeypatch, bytes(packed), nb)
    assert error and ns["started"] == 0 and len(downloads) == 1


def test_exact_inner_notebook_hash_is_required(tmp_path, monkeypatch):
    nb = synthetic_notebook()
    error, ns, downloads, _ = execute(
        tmp_path, monkeypatch, B.package_bytes(nb + b" "), nb
    )
    assert error and ns["started"] == 0 and len(downloads) == 1


@pytest.mark.parametrize("mode", ["multiple", "interrupt"])
def test_bootstrap_baseexceptions_and_multiple_uploads_are_diagnostic(
    tmp_path, monkeypatch, mode
):
    nb = synthetic_notebook()
    packed = B.package_bytes(nb)
    error, _, downloads, _ = execute(
        tmp_path,
        monkeypatch,
        packed,
        nb,
        uploads={"one.zip": packed, "two.zip": packed} if mode == "multiple" else None,
        upload_error=KeyboardInterrupt() if mode == "interrupt" else None,
    )
    assert error and len(downloads) == 1
    assert diagnostic(tmp_path / downloads[0])["code_cells_started"] == 0


def test_original_failure_keeps_its_real_diagnostic_instead_of_a_second_download(
    tmp_path, monkeypatch
):
    final = """import zipfile
from google.colab import files
failure=ValueError('real original diagnostic')
target=Path('original_diagnostic.zip')
with zipfile.ZipFile(target,'w') as z:z.writestr('original.txt',str(failure))
files.download(str(target))
raise RuntimeError('Versuch angehalten; das Diagnose-ZIP enthält den tatsächlichen Fehler.')
"""
    nb = synthetic_notebook(final)
    error, _, downloads, _ = execute(tmp_path, monkeypatch, B.package_bytes(nb), nb)
    assert error and downloads == [Path("original_diagnostic.zip")]
    assert not (tmp_path / "h286_START_diagnose.zip").exists()


def test_production_packager_refuses_a_new_or_modified_pipeline(tmp_path):
    source = tmp_path / "changed.ipynb"
    source.write_bytes(synthetic_notebook())
    with pytest.raises(ValueError, match="exact already-tested"):
        B.build(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()
