"""Exercise the self-contained notebook using synthetic references and GPU/child stubs."""
# ruff: noqa: S102 -- Authored notebook cells are intentionally executed with runtime stubs.

from __future__ import annotations

import ast
import base64
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "h286_notebook_builder", ROOT / "notebooks/build_h286_l4_pilot.py"
)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def fixture_document(private=True):
    refs = {
        name: json.dumps({"synthetic_reference": name}).encode()
        for name in BUILDER.REFERENCE_SHA256
    }
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in refs.items()}
    modules = {
        "attention_probe/__init__.py": '"""Synthetic package."""\n',
        "attention_probe/pilot_items.json": '[{"id":"synthetic-only"}]\n',
        "qwen3_h286_runtime.py": 'raise AssertionError("The real runtime must never be executed by this test")\n',
    }
    return BUILDER.notebook_document(
        modules, refs if private else {}, expected_references=hashes
    ), refs


def execute_stubbed(
    tmp_path,
    monkeypatch,
    *,
    private=True,
    cuda=True,
    free_gib=20,
    pip_code=0,
    helper_code=0,
    helper_state="complete",
    fallback=False,
    changed_torch=False,
    tampered_reference=False,
    default_directory=False,
    flood=False,
):
    document, refs = fixture_document(private)
    namespace = {"__name__": "offline_notebook_test"}
    cells = [
        "".join(cell["source"])
        for cell in document["cells"]
        if cell["cell_type"] == "code"
    ]
    exec(compile(cells[0], "config-cell", "exec"), namespace)
    exec(compile(cells[1], "embedded-cell", "exec"), namespace)
    namespace["RUN_DIRECTORY"] = "" if default_directory else str(tmp_path / "work")
    if tampered_reference:
        key = next(iter(refs))
        namespace["REFERENCE_FILES_B64"][key] = base64.b64encode(
            refs[key] + b" "
        ).decode()
    commands, delivered, links = [], [], []
    versions = {"torch": "2.14.1+cu130", "transformers": "old", "accelerate": "old"}
    torch = types.ModuleType("torch")
    torch.__version__ = versions["torch"]
    torch.version = types.SimpleNamespace(cuda="13.0")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda,
        get_device_capability=lambda index: (8, 9),
        mem_get_info=lambda index: (int(free_gib * 1024**3), 24 * 1024**3),
        get_device_name=lambda index: "synthetic CUDA GPU",
    )

    class Child:
        def __init__(self, code):
            self.code = code
            self.stdout = io.StringIO(
                ("noisy library progress\n" * 5000 if flood else "")
                + "Request 1/16: fixture\nGespeichert 1/16; Restschätzung 1 min\n"
            )

        def wait(self, timeout=None):
            return self.code

        def poll(self):
            return self.code

        def terminate(self):
            raise AssertionError("Completed stub should not be terminated")

        def kill(self):
            raise AssertionError("Completed stub should not be killed")

    def popen(command, **kwargs):
        commands.append(command)
        assert kwargs["env"]["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
        assert kwargs["env"]["TQDM_DISABLE"] == "1"
        if command[1:4] == ["-m", "pip", "install"]:
            assert "--constraint" in command and "--upgrade-strategy" in command
            constraint = Path(command[command.index("--constraint") + 1])
            assert constraint.read_text() == "torch==2.14.1+cu130\n"
            assert "transformers==5.22.2" in command and "accelerate==1.16.0" in command
            assert "torch" not in command
            versions.update(
                transformers=BUILDER.TRANSFORMERS_VERSION,
                accelerate=BUILDER.ACCELERATE_VERSION,
            )
            if changed_torch:
                versions["torch"] = "changed"
            return Child(pip_code)
        assert command[1].endswith("qwen3_h286_runtime.py")
        assert "--items" in command
        for flag, name in [
            ("--hn-set", "hn_set.json"),
            ("--policy", "htq_policy.json"),
            ("--refw", "refw_h286.json"),
        ]:
            assert Path(command[command.index(flag) + 1]).read_bytes() == refs[name]
        assert (
            Path(command[command.index("--items") + 1]).read_text()
            == namespace["MODULE_SOURCES"]["attention_probe/pilot_items.json"]
        )
        assert kwargs["env"]["PYTHONPATH"] == str(namespace["code"])
        output = Path(command[command.index("--out-dir") + 1])
        (output / "status.json").write_text(
            json.dumps(
                {
                    "state": helper_state,
                    "stage": "stub_runtime",
                    "pretrained_weights_loaded": True,
                    "success_scope": "candidate_checkpoint_integration_only",
                }
            )
        )
        (output / "metrics.json").write_text('{"synthetic":true}')
        (output / "requests").mkdir(exist_ok=True)
        (output / "requests/synthetic.json").write_text('{"not_a_model_result":true}')
        (output / "requests/.unpublished").mkdir(exist_ok=True)
        (output / "requests/.unpublished/arrays.npz").write_bytes(b"partial fixture")
        (output / "unpublished.tmp").write_bytes(b"partial fixture")
        return Child(helper_code)

    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")

    def download(path):
        if fallback:
            raise ImportError("generic Jupyter test")
        delivered.append(Path(path))

    colab.files = types.SimpleNamespace(download=download)
    google.colab = colab
    ipython = types.ModuleType("IPython")
    display = types.ModuleType("IPython.display")
    display.FileLink = lambda path: ("filelink", path)
    display.display = lambda value: links.append(value)
    error = None
    original_is_dir = Path.is_dir
    with monkeypatch.context() as patcher:
        patcher.chdir(tmp_path)
        patcher.setattr(
            Path,
            "is_dir",
            lambda path: False if str(path) == "/content" else original_is_dir(path),
        )
        patcher.setitem(sys.modules, "torch", torch)
        patcher.setitem(sys.modules, "google", google)
        patcher.setitem(sys.modules, "google.colab", colab)
        patcher.setitem(sys.modules, "IPython", ipython)
        patcher.setitem(sys.modules, "IPython.display", display)
        patcher.setattr(importlib.metadata, "version", lambda name: versions[name])
        patcher.setattr(subprocess, "Popen", popen)
        try:
            console = io.StringIO()
            with contextlib.redirect_stdout(console):
                exec(compile(cells[2], "run-cell", "exec"), namespace)
        except RuntimeError as caught:
            error = caught
    archive = tmp_path / (
        "h286_l4_pilot_diagnostic.zip" if error else "h286_l4_pilot_results.zip"
    )
    assert archive.exists()
    assert delivered == ([] if fallback else [archive])
    if fallback:
        assert links == [("filelink", archive.name)]
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        entries = {name: zipped.read(name) for name in zipped.namelist()}
    manifest = json.loads(entries["WRAPPER_MANIFEST.json"])
    assert set(manifest) == set(entries) - {"WRAPPER_MANIFEST.json"}
    for name, digest in manifest.items():
        assert hashlib.sha256(entries[name]).hexdigest() == digest
    assert not any(
        "model_cache" in name or name.endswith(".safetensors") for name in entries
    )
    assert not any(".unpublished" in name or name.endswith(".tmp") for name in entries)
    assert len(console.getvalue()) < 15000
    if flood:
        assert "noisy library progress" not in console.getvalue()
        assert "Request 1/16" in console.getvalue()
        assert entries["runtime.log"].count(b"noisy library progress") == 5000
    return error, json.loads(entries["status.json"]), entries, commands


@pytest.mark.parametrize("state", ["complete", "paused"])
def test_private_run_all_is_fresh_subprocess_preserves_torch_and_bundles_requests(
    tmp_path, monkeypatch, state
):
    error, status, entries, commands = execute_stubbed(
        tmp_path, monkeypatch, helper_state=state
    )
    assert (
        error is None
        and status["state"] == state
        and status["wrapper_verified"] is True
    )
    assert len(commands) == 2 and commands[1][1].endswith("qwen3_h286_runtime.py")
    assert "requests/synthetic.json" in entries and "helper_status.json" in entries
    assert json.loads(entries["environment_installed.json"])["torch"] == "2.14.1+cu130"


@pytest.mark.parametrize("cuda,free", [(False, 20), (True, 11)])
def test_gpu_guard_aborts_before_pip_or_any_runtime_download(
    tmp_path, monkeypatch, cuda, free
):
    error, status, entries, commands = execute_stubbed(
        tmp_path, monkeypatch, cuda=cuda, free_gib=free
    )
    assert error and status["state"] == "failed" and commands == []
    assert status["pretrained_weights_loaded"] is False
    assert "wrapper_diagnostic.txt" in entries


def test_public_template_stops_without_private_references_before_installation(
    tmp_path, monkeypatch
):
    error, status, _, commands = execute_stubbed(tmp_path, monkeypatch, private=False)
    assert error and status["stage"] == "reference_validation" and commands == []


def test_tampered_reference_aborts_before_pip(tmp_path, monkeypatch):
    error, status, _, commands = execute_stubbed(
        tmp_path, monkeypatch, tampered_reference=True
    )
    assert error and status["state"] == "failed" and commands == []


@pytest.mark.parametrize("pip_code,changed_torch", [(1, False), (0, True)])
def test_dependency_failure_cannot_start_the_helper(
    tmp_path, monkeypatch, pip_code, changed_torch
):
    error, status, _, commands = execute_stubbed(
        tmp_path, monkeypatch, pip_code=pip_code, changed_torch=changed_torch
    )
    assert error and status["state"] == "failed" and len(commands) == 1
    assert status["pretrained_weights_loaded"] is False


@pytest.mark.parametrize("state,code", [("failed", 1), ("failed", 0), ("complete", 1)])
def test_exit_code_or_failed_child_status_never_becomes_a_pass(
    tmp_path, monkeypatch, state, code
):
    error, status, entries, _ = execute_stubbed(
        tmp_path, monkeypatch, helper_state=state, helper_code=code
    )
    assert error and status["state"] == "failed" and status["wrapper_verified"] is False
    assert json.loads(entries["helper_status.json"])["state"] == state


def test_generic_jupyter_offers_a_relative_file_link(tmp_path, monkeypatch):
    error, status, _, _ = execute_stubbed(tmp_path, monkeypatch, fallback=True)
    assert error is None and status["state"] == "complete"


def test_library_flood_is_saved_on_disk_without_filling_notebook_output(
    tmp_path, monkeypatch
):
    error, status, _, _ = execute_stubbed(tmp_path, monkeypatch, flood=True)
    assert error is None and status["state"] == "complete"


def test_default_work_directory_is_stable_and_fingerprinted(tmp_path, monkeypatch):
    execute_stubbed(tmp_path, monkeypatch, default_directory=True)
    folders = list(tmp_path.glob("h286_attention_pilot_*"))
    assert len(folders) == 1 and len(folders[0].name.rsplit("_", 1)[1]) == 12
    execute_stubbed(tmp_path, monkeypatch, default_directory=True)
    assert list(tmp_path.glob("h286_attention_pilot_*")) == folders


def test_renderer_ids_ast_and_public_reference_privacy():
    doc, _ = fixture_document(private=False)
    assert len(doc["cells"]) == 5
    assert len({cell["id"] for cell in doc["cells"]}) == 5
    for cell in doc["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None
    namespace = {}
    exec("".join(doc["cells"][2]["source"]), namespace)
    assert namespace["REFERENCE_FILES_B64"] == {}


def test_actual_public_notebook_matches_builder_and_original_sources(tmp_path):
    generated = BUILDER.build(tmp_path / "h286_l4_pilot.ipynb")
    assert json.loads(generated.read_text()) == json.loads(
        (ROOT / "notebooks/h286_l4_pilot.ipynb").read_text()
    )
    doc = json.loads(generated.read_text())
    namespace = {}
    exec("".join(doc["cells"][2]["source"]), namespace)
    assert namespace["REFERENCE_FILES_B64"] == {}
    assert namespace["MODULE_SOURCES"] == BUILDER.source_bundle()


def test_private_build_cannot_overwrite_the_public_repository(tmp_path):
    inputs = {}
    for name in BUILDER.REFERENCE_SHA256:
        path = tmp_path / name
        path.write_bytes(b"untrusted fixture")
        inputs[name] = path
    with pytest.raises(ValueError, match="outside"):
        BUILDER.build(
            ROOT / "notebooks/do_not_write_private.ipynb", reference_paths=inputs
        )
    assert not (ROOT / "notebooks/do_not_write_private.ipynb").exists()
