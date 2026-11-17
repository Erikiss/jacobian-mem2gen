"""Package the follow-up A100 answer validation without changing the prior pilot.

The renderer reuses the tested three-cell wrapper via exact, counted substitutions
on a fresh document. Private references may only be written outside the repository.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
_BASE_SPEC = importlib.util.spec_from_file_location(
    "h286_validation_base_builder", HERE / "build_h286_l4_pilot.py"
)
assert _BASE_SPEC is not None and _BASE_SPEC.loader is not None
BASE = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(BASE)
RUNTIME = HERE / "qwen3_h286_validation_runtime.py"
TRANSFORMERS_VERSION = BASE.TRANSFORMERS_VERSION
ACCELERATE_VERSION = BASE.ACCELERATE_VERSION
REFERENCE_SHA256 = dict(BASE.REFERENCE_SHA256)

INTRO = """# Original Qwen3-1.7B: A100-Antwortvalidierung mit fester H286-Auswahl

**A100 auswählen → Alle ausführen.** Das private Notebook enthält die unveränderten
Originalreferenzen bereits. Kein Drive-Zugriff, keine zusätzlichen Datenuploads.
Die kleine Startdatei lädt das vollständige Notebook aus dem bereitgestellten ZIP.
Das öffentliche Template enthält keine privaten Neuronen oder Referenzdaten und
stoppt ohne sie vor dem Modell-Download.

Die festgelegten Qwen3-1.7B-Gewichte, 286 Koordinaten und originalen REF-W-Werte
bleiben erhalten: FP32, TF32 aus, festes Modell und feste Paketversionen. Der
vorhandene GPU-Torch bleibt installiert. Bereits vorhandene Gewichte im
`/content/h286_model_cache` werden wiederverwendet.

Die neue, begrenzte Antwortvalidierung prüft Antwortformen und technische Kontrollen
mit dem neuen Runtime-Modul. Sie ist vom früheren Präfix-Pilot getrennt. Ergebnisse
bleiben explorativ und bestätigen keinen allgemeinen Wahrheits-, Gedächtnis- oder
Selbstüberwachungsmechanismus; Energieeinsparung wird nicht gemessen.

Kurzer Fortschritt und vollständiges Log werden getrennt gespeichert. Am Ende wird
das Ergebnis- oder Diagnose-ZIP angeboten. Ein Status `paused` ist ein fortsetzbarer
Checkpoint, kein abgeschlossener Versuch. GPU-Sitzung danach bei Bedarf beenden.
"""


def _replace_once(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise ValueError(f"Base-wrapper anchor changed or is ambiguous: {before!r}")
    return source.replace(before, after, 1)


def validation_run_source() -> str:
    source = BASE.RUN
    replacements = (
        ("import zipfile\n\nbase", "import zipfile\nimport uuid\n\nwrapper_attempt_id = uuid.uuid4().hex\nbase"),
        ("'h286_attention_pilot_'", "'h286_a100_validation_'"),
        ("'original_qwen3_h286_attention_pilot'", "'original_qwen3_h286_answer_validation'"),
        ("status['pretrained_weights_loaded'] = None", "status['pretrained_weights_loaded'] = False"),
        ("env = dict(child_env, PYTHONPATH=str(code))",
         "env = dict(child_env, PYTHONPATH=str(code), H286_WRAPPER_ATTEMPT=wrapper_attempt_id)"),
        ("str(code / 'qwen3_h286_runtime.py')", "str(code / 'qwen3_h286_validation_runtime.py')"),
        ("Original-FP32-Modell, feste H286/REF-W, 16 kurze Aufgaben.",
         "Original-FP32-Modell, feste H286/REF-W, getrennte Antwortvalidierung."),
        ("'h286_l4_pilot_diagnostic.zip'", "'h286_a100_validation_diagnostic.zip'"),
        ("'h286_l4_pilot_results.zip'", "'h286_a100_validation_results.zip'"),
        ("Begrenzter explorativer H286-Versuch abgeschlossen. Keine allgemeine Wahrheitsvalidierung.",
         "Begrenzte explorative H286-Antwortvalidierung abgeschlossen. Keine allgemeine Wahrheitsvalidierung."),
    )
    for before, after in replacements:
        source = _replace_once(source, before, after)
    evidence = """    # Only this attempt's metrics can establish whether its model weights loaded.
    # Stable work folders may contain metrics from an earlier successful attempt.
    if helper_started:
        status['pretrained_weights_loaded'] = None
        status['weights_loaded_evidence'] = 'missing_unreadable_or_stale_child_metrics'
        try:
            child_metrics = json.loads((output / 'metrics.json').read_text())
            loaded = child_metrics.get('weights_loaded')
            if (child_metrics.get('wrapper_attempt_id') == wrapper_attempt_id
                    and type(loaded) is bool):
                status['pretrained_weights_loaded'] = loaded
                status['weights_loaded_evidence'] = 'fresh_child_metrics'
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    status['wrapper_attempt_id'] = wrapper_attempt_id
"""
    source = _replace_once(source, "    status['written_utc'] =", evidence + "    status['written_utc'] =")
    ast.parse(source)
    return source


def source_bundle() -> dict[str, str]:
    modules = BASE.source_bundle()
    modules["qwen3_h286_validation_runtime.py"] = RUNTIME.read_text()
    modules["attention_probe/answer_validation.py"] = (ROOT / "attention_probe/answer_validation.py").read_text()
    return modules


def notebook_document(modules: dict[str, str], reference_files: dict[str, bytes] | None = None,
                      *, expected_references: dict[str, str] | None = None) -> dict:
    document = BASE.notebook_document(modules, reference_files, expected_references=expected_references)
    document["cells"][0]["source"] = INTRO.splitlines(keepends=True)
    document["cells"][3]["source"] = validation_run_source().splitlines(keepends=True)
    document["cells"][4]["source"] = [
        "Das ZIP enthält Status, Antwortvalidierung, vollständige Logs und fortsetzbare Ergebnisse.\n"
    ]
    for index, cell in enumerate(document["cells"]):
        cell["id"] = hashlib.sha256(f"h286-a100-validation:{index}".encode()).hexdigest()[:12]
    document["metadata"]["colab"]["name"] = "h286_a100_validation.ipynb"
    return document


def build(output: Path = HERE / "h286_a100_validation.ipynb", *,
          reference_paths: dict[str, Path] | None = None) -> Path:
    references = {name: path.read_bytes() for name, path in (reference_paths or {}).items()}
    if references and output.resolve().is_relative_to(ROOT):
        raise ValueError("Private notebook must be written outside the public repository.")
    document = notebook_document(source_bundle(), references)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n")
    if references:
        output.chmod(0o600)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "h286_a100_validation.ipynb")
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--refw", type=Path)
    args = parser.parse_args()
    paths = None
    if args.reference_dir or args.refw:
        if not args.reference_dir or not args.refw:
            parser.error("--reference-dir and --refw must be provided together for the private notebook.")
        paths = {name: args.reference_dir / name for name in ("hn_set.json", "htq_policy.json")}
        paths["refw_h286.json"] = args.refw
    print(build(args.out, reference_paths=paths))


if __name__ == "__main__":
    main()
