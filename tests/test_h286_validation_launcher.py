"""Reuse the exercised upload/CRC/hash guards with the new pinned A100 package."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


B = load("validation_launcher_builder", "notebooks/build_h286_validation_launcher.py")
PRIOR = load("prior_launcher_fixtures", "tests/test_h286_small_launcher.py")


@pytest.fixture(autouse=True)
def bind_new_builder(monkeypatch):
    monkeypatch.setattr(PRIOR, "B", B)


@pytest.mark.parametrize("test_name", [
    "test_upload_android_renames_shared_scope_and_markdown_is_never_executed",
    "test_generic_jupyter_reads_existing_zip_without_upload",
    "test_wrong_outer_hash_stops_before_any_code_and_downloads_diagnostic",
    "test_corrupt_crc_is_diagnosed_before_execution",
    "test_exact_inner_notebook_hash_is_required",
    "test_original_failure_keeps_its_real_diagnostic_instead_of_a_second_download",
])
def test_inherited_security_and_execution_guards(tmp_path, monkeypatch, test_name):
    getattr(PRIOR, test_name)(tmp_path, monkeypatch)


@pytest.mark.parametrize("kind", ["extra_member", "traversal", "oversize"])
def test_member_bounds(tmp_path, monkeypatch, kind):
    PRIOR.test_member_and_size_bounds_are_checked_before_crc_or_read(tmp_path, monkeypatch, kind)


def test_production_binding_clean_notebook_and_public_private_split(tmp_path):
    notebook = tmp_path / "input.ipynb"
    raw = PRIOR.synthetic_notebook()
    notebook.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="reviewed SHA256"):
        B.build(notebook, tmp_path / "bad", expected_notebook_sha256="0" * 64)
    assert not (tmp_path / "bad").exists()
    starter, package = B.build(notebook, tmp_path / "out", expected_notebook_sha256=digest)
    doc = json.loads(starter.read_text())
    assert starter.stat().st_size < 8192
    assert starter.name == "h286_A100_START.ipynb"
    assert package.name == "h286_a100_paket.zip"
    assert "shared=17" not in starter.read_text()
    assert "h286_l4_pilot" not in starter.read_text()
    assert "h286_pilot_paket" not in starter.read_text()
    assert doc["metadata"]["colab"]["name"] == starter.name
    code = "".join(doc["cells"][1]["source"])
    assert digest in code and hashlib.sha256(package.read_bytes()).hexdigest() in code
    second = B.build(notebook, tmp_path / "second", expected_notebook_sha256=digest)
    assert second[1].read_bytes() == package.read_bytes()


def test_wrong_code_count_is_not_packaged(tmp_path):
    raw = json.loads(PRIOR.synthetic_notebook())
    raw["cells"].pop()
    notebook = tmp_path / "input.ipynb"
    notebook.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="code-cell count"):
        B.build(notebook, tmp_path / "out", expected_notebook_sha256=hashlib.sha256(notebook.read_bytes()).hexdigest())
