"""One unchanged Pip prompt, nine prespecified arms, separate private package.

Reuse the exercised group wrapper/packager with isolated function globals. Never
mutate old builders or any of the thirteen original scientific source files.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MEMBER = "h286_control_answers.ipynb"
PACKAGE = "h286_control_answers_paket.zip"
STARTER = "h286_control_answers_START.ipynb"
CONTROL_BASELINE_SHA256 = (
    "b14f0fe12ff6c1d059a6e5954944feb67c87ea7cff61b0f7ad4533d586425143"
)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRIOR = load("control_answers_prior_builder", HERE / "build_h286_group_restore.py")
REFERENCE_SHA256 = dict(
    PRIOR.REFERENCE_SHA256, **{"control_answers_baseline.json": CONTROL_BASELINE_SHA256}
)
TRANSFORMERS_VERSION = PRIOR.TRANSFORMERS_VERSION
ACCELERATE_VERSION = PRIOR.ACCELERATE_VERSION
NEW_SOURCES = {
    "qwen3_h286_control_answers_runtime.py": HERE
    / "qwen3_h286_control_answers_runtime.py"
}
INTRO = """# Pip: neun festgelegte Kontrollantworten

**A100 auswählen → Alle ausführen.** Im kleinen Starter das bereitgestellte
private Paket auswählen. Danach läuft nur der unveränderte Original-Prompt r013:
„Every wug is green. Pip is a wug. Draft: brown. Give Pip's color only.“

Sham, B7-Puls und G7-Transfer werden zusammen mit allen sechs bereits festgelegten
G7-Transferkontrollen geprüft. Es gibt keine Auswahl anhand der neuen Antworten,
keinen weiteren Prompt und keine neue Neuronenauswahl. Jeder Arm beginnt frisch
beim gleichen Prompt und endet bei EOS oder höchstens 128 ausgewählten Tokens.

Geplant sind neun Mess-Prefills, drei technische Vergleichs-Prefills und neun
Generierungs-Prefills, insgesamt 21 Prefill-Forwards. Die Decode-Schritte kommen
hinzu. Original Qwen3-1.7B, FP32/TF32 aus, H286 und REF-W bleiben unverändert.
Vorhandene Gewichte werden aus dem Modellcache wiederverwendet; der installierte
GPU-Torch bleibt erhalten. Fortschritt und tatsächliche Zeiten erscheinen im Log.

Kein Drive, kein Warten auf eine GPU und keine manuellen Zwischenstufen.
Am Ende wird das Ergebnis- oder Diagnose-ZIP angeboten. Ein Abbruch ist kein
vollständiger Versuch; bereits gespeicherte Daten bleiben erhalten.
Die Auswertung bleibt explorativ. Die neun Antworten sind kein allgemeiner
Leistungs-, Gedächtnis- oder Selbstmodellnachweis. Das private Paket nicht öffentlich
hochladen; die öffentliche Vorlage enthält keine privaten Referenzbytes.
"""


def source_bundle():
    sources = PRIOR.source_bundle()
    sources.update({name: path.read_text() for name, path in NEW_SOURCES.items()})
    return sources


def control_run_source():
    source = PRIOR.group_run_source()
    for before, after in (
        ("'h286_group_restore_'", "'h286_control_answers_'"),
        (
            "'original_qwen3_h286_group_restoration'",
            "'original_qwen3_h286_control_answers'",
        ),
        (
            "str(code / 'qwen3_h286_group_runtime.py')",
            "str(code / 'qwen3_h286_control_answers_runtime.py')",
        ),
        (
            "    if RESUME_ZIP:\n",
            "    command += ['--control-baseline', str(inputs / 'control_answers_baseline.json')]\n    if RESUME_ZIP:\n",
        ),
        (
            "'h286_group_restore_diagnostic.zip'",
            "'h286_control_answers_diagnostic.zip'",
        ),
        ("'h286_group_restore_results.zip'", "'h286_control_answers_results.zip'"),
        (
            "zehn Eingaben, begrenzte Gruppenrücksetzung und Antwortläufe.",
            "ein unveränderter Pip-Prompt, neun festgelegte Kontrollantworten.",
        ),
        (
            "Begrenzter explorativer H286-Gruppenversuch abgeschlossen. Inhaltliche Auswertung bleibt getrennt.",
            "Neun explorative H286-Kontrollantworten gespeichert. Inhaltliche Auswertung bleibt getrennt.",
        ),
    ):
        source = PRIOR.VALIDATION._replace_once(source, before, after)
    ast.parse(source)
    return source


def notebook_document(modules, reference_files=None, *, expected_references=None):
    if expected_references is None:
        if (
            not isinstance(CONTROL_BASELINE_SHA256, str)
            or len(CONTROL_BASELINE_SHA256) != 64
        ):
            raise ValueError(
                "The production control-baseline SHA must be bound before building"
            )
        expected_references = dict(REFERENCE_SHA256)
    document = PRIOR.notebook_document(
        modules, reference_files, expected_references=expected_references
    )
    document["cells"][0]["source"] = INTRO.splitlines(keepends=True)
    document["cells"][3]["source"] = control_run_source().splitlines(keepends=True)
    document["cells"][4]["source"] = [
        "Ergebnis-/Diagnose-ZIP herunterladen. EOS, Inhalt und Antwortformat werden getrennt ausgewertet.\n"
    ]
    document["metadata"]["colab"]["name"] = MEMBER
    for index, cell in enumerate(document["cells"]):
        cell["id"] = hashlib.sha256(
            f"h286-control-answers:{index}".encode()
        ).hexdigest()[:12]
    return document


def launcher_document(package_sha, notebook_sha, notebook_size):
    document = PRIOR.launcher_document(package_sha, notebook_sha, notebook_size)
    source = "".join(document["cells"][1]["source"])
    for before, after in (
        (PRIOR.MEMBER, MEMBER),
        (PRIOR.PACKAGE, PACKAGE),
        (
            "h286_group_restore_START_diagnose.zip",
            "h286_control_answers_START_diagnose.zip",
        ),
        ("h286_group_restore_cell_", "h286_control_answers_cell_"),
    ):
        if before not in source:
            raise ValueError("Prior launcher contract changed: " + before)
        source = source.replace(before, after)
    ast.parse(source)
    document["metadata"]["colab"]["name"] = STARTER
    document["cells"][0]["source"] = [
        "# Pip: kleiner Neun-Arm-Starter\n\n**A100 → Alle ausführen.** In der Dateiauswahl **"
        + PACKAGE
        + "** auswählen.\n"
        "Danach läuft nur der unveränderte Pip-Prompt mit den neun festgelegten Armen. Kein Drive und kein Warten auf eine GPU.\n"
        "Ergebnis-/Diagnose-ZIP am Ende herunterladen. Das private Paket nicht öffentlich hochladen.\n"
        "In normalem Jupyter die Paket-ZIP neben diesen Starter legen.\n"
    ]
    document["cells"][1]["source"] = source.splitlines(keepends=True)
    document["cells"][0]["id"] = "h286-control-start-intro"
    document["cells"][1]["id"] = "h286-control-start-code"
    return document


def inherit(function):
    """Reuse exact tested code objects with a private copy of builder globals."""
    bindings = dict(PRIOR.__dict__)
    bindings.update(
        HERE=HERE,
        ROOT=ROOT,
        MEMBER=MEMBER,
        PACKAGE=PACKAGE,
        STARTER=STARTER,
        notebook_document=notebook_document,
        source_bundle=source_bundle,
        launcher_document=launcher_document,
    )
    if "package_bytes" in globals():
        bindings["package_bytes"] = package_bytes
    inherited = types.FunctionType(
        function.__code__,
        bindings,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    inherited.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return inherited


package_bytes = inherit(PRIOR.package_bytes)
package = inherit(PRIOR.package)
_build = inherit(PRIOR.build)


def build(output=HERE / MEMBER, *, reference_paths=None):
    return _build(output, reference_paths=reference_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / MEMBER)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--refw", type=Path)
    parser.add_argument("--continuity", type=Path)
    parser.add_argument("--control-baseline", type=Path)
    parser.add_argument("--package-dir", type=Path)
    args = parser.parse_args()
    paths = None
    if any((args.reference_dir, args.refw, args.continuity, args.control_baseline)):
        if not all(
            (args.reference_dir, args.refw, args.continuity, args.control_baseline)
        ):
            parser.error("All four private reference options are required together")
        paths = {
            name: args.reference_dir / name
            for name in ("hn_set.json", "htq_policy.json")
        }
        paths.update(
            {
                "refw_h286.json": args.refw,
                "group_continuity.json": args.continuity,
                "control_answers_baseline.json": args.control_baseline,
            }
        )
    notebook = build(args.out, reference_paths=paths)
    print(notebook)
    if args.package_dir:
        reviewed_sha = hashlib.sha256(notebook.read_bytes()).hexdigest()
        for artifact in package(
            notebook, args.package_dir, expected_notebook_sha256=reviewed_sha
        ):
            print(artifact)
