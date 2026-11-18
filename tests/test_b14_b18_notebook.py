"""Exercise the actual next-stage wrapper with stubbed external processes.

The existing offline harness checks package installation, result archives and
fresh-child evidence. Only its literal runtime/output names change here.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import types
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = load("b14_b18_builder_test", "notebooks/build_b14_b18.py")
OLD = load("b14_b18_wrapper_harness", "tests/test_h286_a100_validation_notebook.py")


def fixture_document(private=True):
    refs = {
        name: json.dumps({"synthetic": name}).encode() for name in B.REFERENCE_SHA256
    }
    anchors = {name: hashlib.sha256(raw).hexdigest() for name, raw in refs.items()}
    modules = {
        name: '"""Synthetic module; not executed."""\n' for name in B.NEW_SOURCES
    }
    return B.notebook_document(
        modules, refs if private else {}, expected_references=anchors
    ), refs


def rename(code):
    replacements = (
        ("qwen3_h286_validation_runtime.py", "qwen3_b14_b18_runtime.py"),
        (
            "original_qwen3_h286_answer_validation",
            "original_qwen3_b14_b18_local_transport",
        ),
        ("h286_a100_validation_diagnostic.zip", "b14_b18_diagnostic.zip"),
        ("h286_a100_validation_results.zip", "b14_b18_results.zip"),
        ("h286_a100_validation_", "b14_b18_local_transport_"),
    )
    constants = []
    for value in code.co_consts:
        if isinstance(value, types.CodeType):
            value = rename(value)
        elif isinstance(value, str):
            for before, after in replacements:
                value = value.replace(before, after)
        constants.append(value)
    return code.replace(co_consts=tuple(constants))


BINDINGS = dict(OLD.__dict__, B=B, fixture_document=fixture_document)


def inherit(function):
    result = types.FunctionType(
        rename(function.__code__),
        BINDINGS,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    result.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return result


execute = inherit(OLD.execute)
BINDINGS["execute"] = execute


@pytest.mark.parametrize("state", ["complete", "paused"])
def test_bound_inputs_runtime_and_result_archive(tmp_path, monkeypatch, state):
    error, status, entries, commands, _ = execute(
        tmp_path, monkeypatch, helper_state=state
    )
    assert error is None and status["state"] == state
    assert status["purpose"] == "original_qwen3_b14_b18_local_transport"
    assert status["pretrained_weights_loaded"] is True
    assert status["weights_loaded_evidence"] == "fresh_child_metrics"
    command = commands[-1]
    _, refs = fixture_document()
    for flag, name in (
        ("--items", "b14_b18_items.json"),
        ("--plan", "b14_b18_plan.json"),
    ):
        assert Path(command[command.index(flag) + 1]).read_bytes() == refs[name]
    assert "WRAPPER_MANIFEST.json" in entries


test_early_failure_never_starts_model_runtime = pytest.mark.parametrize(
    "kwargs", [{"cuda": False}, {"free_gib": 11}, {"private": False}, {"pip_code": 1}]
)(inherit(OLD.test_early_failure_never_starts_model_runtime))
test_stale_metrics_from_previous_attempt_cannot_claim_loaded_weights = inherit(
    OLD.test_stale_metrics_from_previous_attempt_cannot_claim_loaded_weights
)
test_paused_is_explicit_and_console_stays_small = inherit(
    OLD.test_paused_is_explicit_and_console_stays_small
)
test_new_default_folder_is_stable_and_isolates_old_pilot = inherit(
    OLD.test_new_default_folder_is_stable_and_isolates_old_pilot
)


@pytest.mark.parametrize("name", list(B.REFERENCE_SHA256))
def test_tampered_private_input_fails_before_dependency_install(
    tmp_path, monkeypatch, name
):
    def corrupt(private=True):
        document, refs = fixture_document(private)
        source = "".join(document["cells"][2]["source"])
        source += f"\nREFERENCE_FILES_B64[{name!r}] = 'dGFtcGVyZWQ='\n"
        document["cells"][2]["source"] = [source]
        return document, refs

    monkeypatch.setitem(BINDINGS, "fixture_document", corrupt)
    error, status, _, commands, _ = execute(tmp_path, monkeypatch)
    assert error and status["state"] == "failed" and commands == []


def test_private_package_exact_bytes_and_hash_only_small_starter(tmp_path):
    document, refs = fixture_document()
    raw = (json.dumps(document) + "\n").encode()
    notebook = tmp_path / B.MEMBER
    notebook.write_bytes(raw)
    starter, package = B.package(
        notebook, tmp_path, expected_notebook_sha256=hashlib.sha256(raw).hexdigest()
    )
    with zipfile.ZipFile(package) as archive:
        assert archive.namelist() == [B.MEMBER]
        assert archive.read(B.MEMBER) == raw and archive.testzip() is None
    launch = json.loads(starter.read_bytes())
    code = "".join(launch["cells"][1]["source"])
    ast.parse(code)
    assert hashlib.sha256(package.read_bytes()).hexdigest() in code
    assert hashlib.sha256(raw).hexdigest() in code
    assert B.PACKAGE in code and B.MEMBER in code
    assert starter.stat().st_size < 8192
    assert all(payload.decode() not in starter.read_text() for payload in refs.values())


def test_unsafe_private_destination_and_changed_notebook_rejected(tmp_path):
    assert (
        inspect.signature(B.build).parameters["output"].default
        == ROOT / "notebooks" / B.MEMBER
    )
    with pytest.raises(ValueError, match="outside"):
        B.build(ROOT / "notebooks" / B.MEMBER, reference_paths={"test": Path(__file__)})
    notebook = tmp_path / B.MEMBER
    notebook.write_text("{}")
    with pytest.raises(ValueError, match="reviewed"):
        B.package(notebook, tmp_path, expected_notebook_sha256="0" * 64)
