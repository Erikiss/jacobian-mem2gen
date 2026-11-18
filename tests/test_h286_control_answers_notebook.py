"""Exercise the inherited offline harness against the separate nine-arm wrapper.

Only authored synthetic notebook cells run. Existing process/upload stubs and
security checks are reused with private globals and renamed output constants;
no scientific runner, model, install command or network request is executed.
"""

# ruff: noqa: S102 -- Only authored payload/config cells with stubbed runtimes.
from __future__ import annotations

import hashlib
import importlib.util
import json
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


B = load("control_answers_builder", "notebooks/build_h286_control_answers.py")
GROUP = load("inherited_group_notebook_checks", "tests/test_h286_group_notebook.py")


def renamed_code(code):
    constants = []
    for value in code.co_consts:
        if isinstance(value, types.CodeType):
            value = renamed_code(value)
        elif isinstance(value, str):
            value = value.replace(
                "qwen3_h286_group_runtime.py", "qwen3_h286_control_answers_runtime.py"
            )
            value = value.replace("h286_group_restore", "h286_control_answers")
            value = value.replace(
                "original_qwen3_h286_group_restoration",
                "original_qwen3_h286_control_answers",
            )
        constants.append(value)
    return code.replace(co_consts=tuple(constants))


BINDINGS = dict(GROUP.__dict__, B=B)


def inherit(function, *, rename=False):
    result = types.FunctionType(
        renamed_code(function.__code__) if rename else function.__code__,
        BINDINGS,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    result.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return result


fixture_document = inherit(GROUP.fixture_document)
BINDINGS["fixture_document"] = fixture_document
execute = inherit(GROUP.execute, rename=True)
BINDINGS["execute"] = execute


@pytest.mark.parametrize("loaded", [True, False])
def test_nine_arm_wrapper_uses_new_runtime_and_all_five_bound_inputs(
    tmp_path, monkeypatch, loaded
):
    error, status, entries, commands, _, _ = execute(
        tmp_path, monkeypatch, loaded=loaded
    )
    assert error is None and status["state"] == "complete"
    assert status["purpose"] == "original_qwen3_h286_control_answers"
    assert (
        status["pretrained_weights_loaded"] is loaded
        and status["weights_loaded_evidence"] == "fresh_child_metrics"
    )
    assert len(commands) == 2 and commands[1][1].endswith(
        "/qwen3_h286_control_answers_runtime.py"
    )
    _, references = fixture_document()
    flags = {
        "--hn-set": "hn_set.json",
        "--policy": "htq_policy.json",
        "--refw": "refw_h286.json",
        "--continuity": "group_continuity.json",
        "--control-baseline": "control_answers_baseline.json",
    }
    for flag, name in flags.items():
        assert (
            Path(commands[1][commands[1].index(flag) + 1]).read_bytes()
            == references[name]
        )
    assert "WRAPPER_MANIFEST.json" in entries


@pytest.mark.parametrize("name", tuple(B.REFERENCE_SHA256))
def test_each_private_input_hash_is_required_before_any_install_or_model(
    tmp_path, monkeypatch, name
):
    def mutate(namespace):
        namespace["REFERENCE_FILES_B64"][name] = "dGFtcGVyZWQ="

    error, status, _, commands, _, _ = execute(tmp_path, monkeypatch, mutation=mutate)
    assert (
        error
        and status["state"] == "failed"
        and status["pretrained_weights_loaded"] is False
    )
    assert commands == []


def test_changed_new_runner_source_stops_before_install_or_model(tmp_path, monkeypatch):
    def mutate(namespace):
        namespace["MODULE_SOURCES"]["qwen3_h286_control_answers_runtime.py"] = (
            'raise RuntimeError("changed runner")\n'
        )

    error, status, _, commands, _, _ = execute(tmp_path, monkeypatch, mutation=mutate)
    assert error and status["state"] == "failed" and commands == []
    assert status["pretrained_weights_loaded"] is False


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
def test_inherited_preflight_guard_stops_without_runtime(tmp_path, monkeypatch, kwargs):
    inherit(GROUP.test_early_failures_never_start_group_runtime, rename=True)(
        tmp_path, monkeypatch, kwargs
    )


@pytest.mark.parametrize(
    "name",
    [
        "test_generic_jupyter_offers_the_same_verified_zip_as_a_local_link",
        "test_child_error_keeps_partial_outputs_and_does_not_accept_complete_claim",
        "test_unsupported_paused_status_is_not_a_complete_group_experiment",
        "test_stale_metrics_are_not_fresh_loaded_evidence",
        "test_interrupt_terminates_active_child_and_retains_real_diagnostic",
        "test_console_bounded_full_logs_retained_and_default_path_stable",
    ],
)
def test_existing_failure_cleanup_resume_and_output_checks(tmp_path, monkeypatch, name):
    inherit(getattr(GROUP, name), rename=True)(tmp_path, monkeypatch)


def test_renderer_keeps_old_group_document_and_globals_unchanged():
    old_source = B.PRIOR.group_run_source()
    old_member = B.PRIOR.MEMBER
    document, _ = fixture_document(private=False)
    namespace = {}
    exec("".join(document["cells"][2]["source"]), namespace)
    assert (
        namespace["REFERENCE_FILES_B64"] == {}
        and len(namespace["REFERENCE_SHA256"]) == 5
    )
    code = "".join(document["cells"][3]["source"])
    assert (
        "'--control-baseline'" in code
        and "qwen3_h286_control_answers_runtime.py" in code
    )
    assert (
        B.PRIOR.group_run_source() == old_source
        and B.PRIOR.MEMBER == old_member == "h286_group_restore.ipynb"
    )
    for name in ("package", "package_bytes"):
        fn = getattr(B, name)
        old = getattr(B.PRIOR, name)
        assert fn.__code__ is old.__code__ and fn.__globals__ is not old.__globals__
        assert fn.__globals__["MEMBER"] == "h286_control_answers.ipynb"
    assert len([c for c in document["cells"] if c["cell_type"] == "code"]) == 3
    assert len({c["id"] for c in document["cells"]}) == len(document["cells"])
    assert document["metadata"]["colab"]["name"] == B.MEMBER
    assert len(namespace["MODULE_SOURCES"]) == 14


def test_source_bundle_keeps_original_thirteen_and_saved_public_template(tmp_path):
    old = B.PRIOR.source_bundle()
    sources = B.source_bundle()
    assert (
        len(old) == 13
        and len(sources) == 14
        and all(sources[k] == v for k, v in old.items())
    )
    assert (
        sources["qwen3_h286_control_answers_runtime.py"]
        == (ROOT / "notebooks/qwen3_h286_control_answers_runtime.py").read_text()
    )
    path = B.build(tmp_path / "public.ipynb")
    document = json.loads(path.read_bytes())
    assert document == json.loads(
        (ROOT / "notebooks/h286_control_answers.ipynb").read_bytes()
    )
    namespace = {}
    exec("".join(document["cells"][2]["source"]), namespace)
    assert (
        namespace["REFERENCE_FILES_B64"] == {}
        and namespace["MODULE_SOURCES"] == sources
    )
    for name, source in sources.items():
        assert (
            namespace["MODULE_SHA256"][name]
            == hashlib.sha256(source.encode()).hexdigest()
        )


def test_production_fifth_reference_anchor_is_explicit_and_old_anchors_preserved():
    assert (
        B.CONTROL_BASELINE_SHA256
        == "b14f0fe12ff6c1d059a6e5954944feb67c87ea7cff61b0f7ad4533d586425143"
    )
    assert set(B.REFERENCE_SHA256) == set(B.PRIOR.REFERENCE_SHA256) | {
        "control_answers_baseline.json"
    }
    assert all(B.REFERENCE_SHA256[k] == v for k, v in B.PRIOR.REFERENCE_SHA256.items())
    with pytest.raises(ValueError, match="production anchors"):
        B.notebook_document({}, {name: b"changed" for name in B.REFERENCE_SHA256})


def test_private_build_and_package_repository_guards(tmp_path):
    doc, refs = fixture_document()
    paths = {name: tmp_path / name for name in refs}
    for name, path in paths.items():
        path.write_bytes(refs[name])
    with pytest.raises(ValueError, match="outside"):
        B.build(
            ROOT / "notebooks/forbidden_control_private.ipynb", reference_paths=paths
        )
    raw = json.dumps(doc).encode()
    nb = tmp_path / "private.ipynb"
    nb.write_bytes(raw)
    with pytest.raises(ValueError, match="outside"):
        B.package(
            nb,
            ROOT / "notebooks/forbidden_control_package",
            expected_notebook_sha256=hashlib.sha256(raw).hexdigest(),
        )
    assert not (ROOT / "notebooks/forbidden_control_private.ipynb").exists()


def test_small_package_pins_exact_new_notebook_without_rendering_payload(tmp_path):
    document, _ = fixture_document()
    raw = json.dumps(document).encode()
    nb = tmp_path / "private.ipynb"
    nb.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="reviewed SHA256"):
        B.package(nb, tmp_path / "bad", expected_notebook_sha256="0" * 64)
    start, package = B.package(nb, tmp_path / "out", expected_notebook_sha256=digest)
    assert (
        start.name == B.STARTER
        and package.name == B.PACKAGE
        and start.stat().st_size < 8192
    )
    loader = start.read_text()
    assert (
        digest in loader and hashlib.sha256(package.read_bytes()).hexdigest() in loader
    )
    assert "REFERENCE_FILES_B64" not in loader and "Synthetic test source" not in loader
    with zipfile.ZipFile(package) as z:
        assert (
            z.namelist() == [B.MEMBER]
            and z.read(B.MEMBER) == raw
            and z.testzip() is None
        )
    again = B.package(nb, tmp_path / "again", expected_notebook_sha256=digest)
    assert (
        again[0].read_bytes() == start.read_bytes()
        and again[1].read_bytes() == package.read_bytes()
    )


@pytest.mark.parametrize(
    "name",
    [
        "test_upload_android_renames_shared_scope_and_markdown_is_never_executed",
        "test_generic_jupyter_reads_existing_zip_without_upload",
        "test_wrong_outer_hash_stops_before_any_code_and_downloads_diagnostic",
        "test_corrupt_crc_is_diagnosed_before_execution",
        "test_exact_inner_notebook_hash_is_required",
        "test_original_failure_keeps_its_real_diagnostic_instead_of_a_second_download",
    ],
)
def test_inherited_mobile_upload_hash_crc_and_error_guards(tmp_path, monkeypatch, name):
    monkeypatch.setattr(GROUP.OLD_LAUNCHER_TESTS, "B", B)
    getattr(GROUP.OLD_LAUNCHER_TESTS, name)(tmp_path, monkeypatch)


@pytest.mark.parametrize("kind", ["extra_member", "traversal", "oversize"])
def test_loader_rejects_members_before_crc_and_read(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(GROUP.OLD_LAUNCHER_TESTS, "B", B)
    GROUP.OLD_LAUNCHER_TESTS.test_member_and_size_bounds_are_checked_before_crc_or_read(
        tmp_path, monkeypatch, kind
    )
