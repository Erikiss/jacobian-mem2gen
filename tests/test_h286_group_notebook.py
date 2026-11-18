"""Offline group wrapper/source/package checks; never install or run a model."""

# ruff: noqa: S102 -- Execute authored notebook cells with explicitly stubbed subprocesses.
from __future__ import annotations

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


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = load("group_notebook_builder", "notebooks/build_h286_group_restore.py")
OLD_LAUNCHER_TESTS = load(
    "group_inherited_launcher_tests", "tests/test_h286_small_launcher.py"
)


def fixture_document(private=True):
    references = {
        name: json.dumps({"synthetic": name}).encode() for name in B.REFERENCE_SHA256
    }
    anchors = {
        name: hashlib.sha256(raw).hexdigest() for name, raw in references.items()
    }
    names = tuple(B.PRIOR.source_bundle()) + tuple(B.NEW_SOURCES)
    modules = {
        name: "[]\n"
        if name.endswith(".json")
        else '"""Synthetic test source, not executed."""\n'
        for name in names
    }
    return B.notebook_document(
        modules, references if private else {}, expected_references=anchors
    ), references


def execute(
    tmp_path,
    monkeypatch,
    *,
    private=True,
    cuda=True,
    free_gib=20,
    helper_code=0,
    helper_state="complete",
    loaded=True,
    metrics_mode="fresh",
    pip_code=0,
    flood=False,
    interrupt=False,
    default_directory=False,
    mutation=None,
    torch_drift=False,
    colab_available=True,
):
    document, references = fixture_document(private)
    namespace = {"__name__": "offline_group_test"}
    codes = [
        "".join(cell["source"])
        for cell in document["cells"]
        if cell["cell_type"] == "code"
    ]
    exec(compile(codes[0], "config", "exec"), namespace)
    exec(compile(codes[1], "payload", "exec"), namespace)
    namespace["RUN_DIRECTORY"] = "" if default_directory else str(tmp_path / "work")
    if mutation:
        mutation(namespace)
    versions = {"torch": "2.14.1+cu130", "transformers": "old", "accelerate": "old"}
    commands, downloads, children = [], [], []
    fake_torch = types.ModuleType("torch")
    fake_torch.version = types.SimpleNamespace(cuda="13.0")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda,
        get_device_capability=lambda _i: (8, 0),
        mem_get_info=lambda _i: (int(free_gib * 1024**3), 80 * 1024**3),
        get_device_name=lambda _i: "synthetic A100",
    )

    class Child:
        def __init__(self, code, *, interrupted=False):
            self.code, self.interrupted, self.terminated = code, interrupted, False
            self.stdout = (
                self.lines()
                if interrupted
                else io.StringIO(
                    ("library noise\n" * 5000 if flood else "")
                    + "Status: group\nStage groups\nRequest fixture\nGespeichert fixture\n"
                )
            )

        def lines(self):
            yield "Stage group prefill\n"
            raise KeyboardInterrupt("synthetic interrupt")

        def wait(self, timeout=None):
            return self.code

        def poll(self):
            return None if self.interrupted and not self.terminated else self.code

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.terminated = True

    def popen(command, **kwargs):
        commands.append(command)
        assert (
            kwargs["env"]["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
            and kwargs["env"]["TQDM_DISABLE"] == "1"
        )
        if command[1:4] == ["-m", "pip", "install"]:
            assert "--constraint" in command and "torch" not in command
            constraint = Path(command[command.index("--constraint") + 1])
            assert constraint.read_text() == "torch==2.14.1+cu130\n"
            assert "transformers==5.22.2" in command and "accelerate==1.16.0" in command
            versions.update(
                transformers=B.TRANSFORMERS_VERSION, accelerate=B.ACCELERATE_VERSION
            )
            if torch_drift:
                versions["torch"] = "unexpected replacement"
            child = Child(pip_code)
        else:
            assert command[1].endswith("/qwen3_h286_group_runtime.py")
            assert command[command.index("--items") + 1].endswith(
                "/attention_probe/group_probe_items.json"
            )
            assert kwargs["env"]["PYTHONPATH"] == str(namespace["code"])
            for flag, name in (
                ("--hn-set", "hn_set.json"),
                ("--policy", "htq_policy.json"),
                ("--refw", "refw_h286.json"),
                ("--continuity", "group_continuity.json"),
            ):
                assert (
                    Path(command[command.index(flag) + 1]).read_bytes()
                    == references[name]
                )
            assert (
                Path(command[command.index("--cache-dir") + 1])
                == tmp_path / "h286_model_cache"
            )
            output = Path(command[command.index("--out-dir") + 1])
            (output / "status.json").write_text(json.dumps({"state": helper_state}))
            if metrics_mode != "keep":
                (output / "metrics.json").write_text(
                    json.dumps(
                        {
                            "weights_loaded": loaded,
                            "wrapper_attempt_id": kwargs["env"]["H286_WRAPPER_ATTEMPT"]
                            if metrics_mode == "fresh"
                            else "old-attempt",
                        }
                    )
                )
            (output / "group_results.json").write_text('{"synthetic":true}')
            child = Child(helper_code, interrupted=interrupt)
        children.append(child)
        return child

    google, colab = types.ModuleType("google"), types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(
        download=lambda path: downloads.append(Path(path))
    )
    google.colab = colab
    old_is_dir = Path.is_dir
    error, console = None, io.StringIO()
    with monkeypatch.context() as patch:
        patch.chdir(tmp_path)
        patch.setattr(
            Path,
            "is_dir",
            lambda path: False if str(path) == "/content" else old_is_dir(path),
        )
        patch.setitem(sys.modules, "torch", fake_torch)
        if colab_available:
            patch.setitem(sys.modules, "google", google)
            patch.setitem(sys.modules, "google.colab", colab)
        else:
            patch.setitem(sys.modules, "google.colab", None)
            display = types.ModuleType("IPython.display")
            display.FileLink = lambda path: (Path.cwd() / path).resolve()
            display.display = lambda link: downloads.append(link)
            patch.setitem(sys.modules, "IPython.display", display)
        patch.setattr(importlib.metadata, "version", lambda name: versions[name])
        patch.setattr(subprocess, "Popen", popen)
        try:
            with contextlib.redirect_stdout(console):
                exec(compile(codes[2], "group-run", "exec"), namespace)
        except RuntimeError as caught:
            error = caught
    target = tmp_path / (
        "h286_group_restore_diagnostic.zip"
        if error
        else "h286_group_restore_results.zip"
    )
    assert downloads == [target]
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries["WRAPPER_MANIFEST.json"])
    assert set(manifest) == set(entries) - {"WRAPPER_MANIFEST.json"}
    assert all(
        hashlib.sha256(entries[name]).hexdigest() == digest
        for name, digest in manifest.items()
    )
    return (
        error,
        json.loads(entries["status.json"]),
        entries,
        commands,
        children,
        console.getvalue(),
    )


@pytest.mark.parametrize("loaded", [True, False])
def test_success_fresh_metrics_required_continuity_and_isolated_child(
    tmp_path, monkeypatch, loaded
):
    error, status, entries, commands, _, _ = execute(
        tmp_path, monkeypatch, loaded=loaded
    )
    assert error is None and status["state"] == "complete"
    assert status["purpose"] == "original_qwen3_h286_group_restoration"
    assert status["pretrained_weights_loaded"] is loaded
    assert status["weights_loaded_evidence"] == "fresh_child_metrics"
    assert len(commands) == 2 and "group_results.json" in entries


def test_generic_jupyter_offers_the_same_verified_zip_as_a_local_link(
    tmp_path, monkeypatch
):
    error, status, entries, commands, _, _ = execute(
        tmp_path, monkeypatch, colab_available=False
    )
    assert error is None and status["state"] == "complete"
    assert len(commands) == 2 and "WRAPPER_MANIFEST.json" in entries


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cuda": False},
        {"free_gib": 11},
        {"private": False},
        {"pip_code": 1},
        {"torch_drift": True},
    ],
)
def test_early_failures_never_start_group_runtime(tmp_path, monkeypatch, kwargs):
    error, status, _, commands, _, _ = execute(tmp_path, monkeypatch, **kwargs)
    assert (
        error
        and status["state"] == "failed"
        and status["pretrained_weights_loaded"] is False
    )
    assert len(commands) == (
        1 if kwargs.get("pip_code") or kwargs.get("torch_drift") else 0
    )


@pytest.mark.parametrize("kind", ["reference", "source"])
def test_tampering_stops_before_install_or_model_download(tmp_path, monkeypatch, kind):
    def mutation(ns):
        if kind == "reference":
            ns["REFERENCE_FILES_B64"]["group_continuity.json"] = "aW52YWxpZA=="
        else:
            ns["MODULE_SOURCES"]["attention_probe/group_restoration.py"] = (
                'raise RuntimeError("tampered")'
            )

    error, status, _, commands, _, _ = execute(tmp_path, monkeypatch, mutation=mutation)
    assert error and status["state"] == "failed" and commands == []


def test_child_error_keeps_partial_outputs_and_does_not_accept_complete_claim(
    tmp_path, monkeypatch
):
    error, status, entries, _, _, _ = execute(
        tmp_path, monkeypatch, helper_code=1, loaded=True
    )
    assert error and status["state"] == "failed" and status["wrapper_verified"] is False
    assert (
        status["pretrained_weights_loaded"] is True and "group_results.json" in entries
    )
    assert json.loads(entries["helper_status.json"])["state"] == "complete"


def test_unsupported_paused_status_is_not_a_complete_group_experiment(
    tmp_path, monkeypatch
):
    error, status, _, _, _, _ = execute(tmp_path, monkeypatch, helper_state="paused")
    assert error and status["state"] == "failed"


def test_stale_metrics_are_not_fresh_loaded_evidence(tmp_path, monkeypatch):
    execute(tmp_path, monkeypatch, loaded=True)
    error, status, _, _, _, _ = execute(
        tmp_path, monkeypatch, helper_code=1, metrics_mode="keep"
    )
    assert error and status["pretrained_weights_loaded"] is None
    assert (
        status["weights_loaded_evidence"] == "missing_unreadable_or_stale_child_metrics"
    )


def test_interrupt_terminates_active_child_and_retains_real_diagnostic(
    tmp_path, monkeypatch
):
    error, status, entries, _, children, _ = execute(
        tmp_path, monkeypatch, interrupt=True
    )
    assert error and status["state"] == "failed" and children[-1].terminated
    assert (
        status["error_type"] == "KeyboardInterrupt"
        and b"synthetic interrupt" in entries["wrapper_diagnostic.txt"]
    )


def test_console_bounded_full_logs_retained_and_default_path_stable(
    tmp_path, monkeypatch
):
    original = tmp_path / "old_result"
    original.write_text("unchanged")
    error, _, entries, _, _, console = execute(
        tmp_path, monkeypatch, flood=True, default_directory=True
    )
    assert error is None and len(console) < 15000 and "library noise" not in console
    assert entries["runtime.log"].count(b"library noise") == 5000
    folders = [path for path in tmp_path.glob("h286_group_restore_*") if path.is_dir()]
    assert len(folders) == 1
    execute(tmp_path, monkeypatch, default_directory=True)
    assert [
        path for path in tmp_path.glob("h286_group_restore_*") if path.is_dir()
    ] == folders
    assert original.read_text() == "unchanged"


def test_public_renderer_has_no_private_bytes_and_preserves_old_wrapper():
    old_run = B.VALIDATION.validation_run_source()
    document, _ = fixture_document(private=False)
    namespace = {}
    exec("".join(document["cells"][2]["source"]), namespace)
    assert (
        namespace["REFERENCE_FILES_B64"] == {}
        and len(namespace["REFERENCE_SHA256"]) == 4
    )
    assert len([c for c in document["cells"] if c["cell_type"] == "code"]) == 3
    assert len({c["id"] for c in document["cells"]}) == len(document["cells"])
    assert B.VALIDATION.validation_run_source() == old_run
    assert document["metadata"]["colab"]["name"] == B.MEMBER


def test_production_source_bindings_and_saved_public_template(tmp_path):
    old = B.PRIOR.source_bundle()
    sources = B.source_bundle()
    assert len(old) == 10 and len(sources) == 13
    assert all(sources[name] == value for name, value in old.items())
    assert all(
        sources[name] == path.read_text() for name, path in B.NEW_SOURCES.items()
    )
    path = B.build(tmp_path / "public.ipynb")
    doc = json.loads(path.read_text())
    assert doc == json.loads((ROOT / "notebooks/h286_group_restore.ipynb").read_text())
    namespace = {}
    exec("".join(doc["cells"][2]["source"]), namespace)
    assert (
        namespace["MODULE_SOURCES"] == sources
        and namespace["REFERENCE_FILES_B64"] == {}
    )
    for name, source in sources.items():
        assert (
            namespace["MODULE_SHA256"][name]
            == hashlib.sha256(source.encode()).hexdigest()
        )


def test_private_reference_binding_and_repository_write_guards(tmp_path):
    doc, refs = fixture_document()
    anchors = {name: hashlib.sha256(raw).hexdigest() for name, raw in refs.items()}
    with pytest.raises(ValueError, match="production anchors"):
        B.notebook_document(
            {},
            dict(refs, **{"group_continuity.json": b"changed"}),
            expected_references=anchors,
        )
    paths = {name: tmp_path / name for name in refs}
    for name, path in paths.items():
        path.write_bytes(refs[name])
    with pytest.raises(ValueError, match="outside"):
        B.build(ROOT / "notebooks/forbidden_private_group.ipynb", reference_paths=paths)
    assert not (ROOT / "notebooks/forbidden_private_group.ipynb").exists()
    raw = json.dumps(doc).encode()
    notebook = tmp_path / "input.ipynb"
    notebook.write_bytes(raw)
    with pytest.raises(ValueError, match="outside"):
        B.package(
            notebook,
            ROOT / "notebooks/forbidden_package",
            expected_notebook_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_packaging_exact_reviewed_notebook_size_hash_and_determinism(tmp_path):
    doc, _ = fixture_document()
    raw = json.dumps(doc).encode()
    notebook = tmp_path / "private.ipynb"
    notebook.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="reviewed SHA256"):
        B.package(notebook, tmp_path / "bad", expected_notebook_sha256="0" * 64)
    starter, packed = B.package(
        notebook, tmp_path / "out", expected_notebook_sha256=digest
    )
    assert starter.stat().st_size < 8192 and len(raw) < 2 * 1024**2
    assert starter.name == B.STARTER and packed.name == B.PACKAGE
    code = starter.read_text()
    assert digest in code and hashlib.sha256(packed.read_bytes()).hexdigest() in code
    assert "Synthetic test source" not in code and "REFERENCE_FILES_B64" not in code
    with zipfile.ZipFile(packed) as archive:
        assert (
            archive.namelist() == [B.MEMBER]
            and archive.read(B.MEMBER) == raw
            and archive.testzip() is None
        )
    again = B.package(notebook, tmp_path / "again", expected_notebook_sha256=digest)
    assert (
        again[0].read_bytes() == starter.read_bytes()
        and again[1].read_bytes() == packed.read_bytes()
    )


@pytest.mark.parametrize(
    "test_name",
    [
        "test_upload_android_renames_shared_scope_and_markdown_is_never_executed",
        "test_generic_jupyter_reads_existing_zip_without_upload",
        "test_wrong_outer_hash_stops_before_any_code_and_downloads_diagnostic",
        "test_corrupt_crc_is_diagnosed_before_execution",
        "test_exact_inner_notebook_hash_is_required",
        "test_original_failure_keeps_its_real_diagnostic_instead_of_a_second_download",
    ],
)
def test_inherited_mobile_loader_guards(tmp_path, monkeypatch, test_name):
    monkeypatch.setattr(OLD_LAUNCHER_TESTS, "B", B)
    getattr(OLD_LAUNCHER_TESTS, test_name)(tmp_path, monkeypatch)


@pytest.mark.parametrize("kind", ["extra_member", "traversal", "oversize"])
def test_loader_member_bounds_before_crc_read(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(OLD_LAUNCHER_TESTS, "B", B)
    OLD_LAUNCHER_TESTS.test_member_and_size_bounds_are_checked_before_crc_or_read(
        tmp_path, monkeypatch, kind
    )


@pytest.mark.parametrize("mode", ["multiple", "interrupt"])
def test_loader_cancel_and_multiple_uploads_have_a_diagnostic(
    tmp_path, monkeypatch, mode
):
    monkeypatch.setattr(OLD_LAUNCHER_TESTS, "B", B)
    OLD_LAUNCHER_TESTS.test_bootstrap_baseexceptions_and_multiple_uploads_are_diagnostic(
        tmp_path, monkeypatch, mode
    )


@pytest.mark.parametrize("mutation", ["wrong_cell_count", "executed_cell"])
def test_package_requires_three_clean_unexecuted_cells(tmp_path, mutation):
    document, _ = fixture_document()
    if mutation == "wrong_cell_count":
        document["cells"].pop(3)
    else:
        document["cells"][1]["execution_count"] = 1
        document["cells"][1]["outputs"] = [
            {"output_type": "stream", "text": "private output"}
        ]
    raw = json.dumps(document).encode()
    path = tmp_path / "input.ipynb"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="code-cell count|clean unexecuted"):
        B.package(
            path,
            tmp_path / "out",
            expected_notebook_sha256=hashlib.sha256(raw).hexdigest(),
        )
    assert not (tmp_path / "out").exists()
