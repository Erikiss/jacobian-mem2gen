"""Separate fixed-H286 group-restoration notebook and mobile package builder.

The original ten embedded sources are copied unchanged. Private reference bytes
can only be written outside this public repository; the small public starter
contains exact package/notebook hashes, never reference payloads.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import re
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MEMBER = "h286_group_restore.ipynb"
PACKAGE = "h286_group_restore_paket.zip"
STARTER = "h286_group_restore_START.ipynb"
CONTINUITY_SHA256 = "111d7213f76374a459c38ca17f4701eb9228693708ee1657a7e25f987c65ddc1"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRIOR = load("group_prior_completion_builder", HERE / "build_h286_completion_128.py")
VALIDATION = PRIOR.PRIOR
LAUNCHER = PRIOR.LAUNCHER
REFERENCE_SHA256 = dict(
    VALIDATION.REFERENCE_SHA256, **{"group_continuity.json": CONTINUITY_SHA256}
)
TRANSFORMERS_VERSION = VALIDATION.TRANSFORMERS_VERSION
ACCELERATE_VERSION = VALIDATION.ACCELERATE_VERSION
NEW_SOURCES = {
    "attention_probe/group_restoration.py": ROOT
    / "attention_probe/group_restoration.py",
    "attention_probe/group_probe_items.json": ROOT
    / "attention_probe/group_probe_items.json",
    "qwen3_h286_group_runtime.py": HERE / "qwen3_h286_group_runtime.py",
}
INTRO = """# H286: begrenzter Gruppenversuch

**A100 auswählen → Alle ausführen.** Der kleine Starter lädt das bereitgestellte
private Paket, ohne das große Notebook im Browser zu öffnen. Kein Drive-Zugriff,
keine GPU-Warteschleife und keine weiteren manuellen Zwischenstufen.

Das originale Qwen3-1.7B, FP32 mit ausgeschaltetem TF32, dieselben 286 Koordinaten
und die feste REF-W-Normierung bleiben erhalten. Gruppenmessungen, begrenzte
Antwortläufe und Abschlussprüfungen laufen automatisch nacheinander.
Geplant sind zehn Eingaben, 350 kurze Prefills und 46 freie
Antwortläufe mit höchstens 128 ausgewählten Tokens. Fortschritt und tatsächliche
Zeiten werden protokolliert; Laden und technische Prüfungen kommen hinzu.

Die H286-Gruppen stammen aus der bestehenden Auswahl; dieser Versuch ist explorativ.
Er prüft keine allgemeine Leistungsverbesserung, kein Lernen und kein Gedächtnis.
Inhalt und verlangtes Antwortformat müssen später getrennt beurteilt werden.

Am Ende wird das Ergebnis- oder Diagnose-ZIP angeboten. Ein Abbruch wird als Fehler
gekennzeichnet; bereits gespeicherte Einzelresultate bleiben erhalten. Alte Ergebnisse werden nicht verändert;
vorhandene Modellgewichte im Cache werden wiederverwendet. Das private Paket nicht
öffentlich hochladen. Das öffentliche Template enthält keine privaten Referenzbytes.
"""


def source_bundle():
    sources = PRIOR.source_bundle()
    sources.update({name: path.read_text() for name, path in NEW_SOURCES.items()})
    return sources


def group_run_source():
    source = VALIDATION.validation_run_source()
    for before, after in (
        ("'h286_a100_validation_'", "'h286_group_restore_'"),
        (
            "'original_qwen3_h286_answer_validation'",
            "'original_qwen3_h286_group_restoration'",
        ),
        (
            "str(code / 'qwen3_h286_validation_runtime.py')",
            "str(code / 'qwen3_h286_group_runtime.py')",
        ),
        (
            "'--items', str(code / 'attention_probe/pilot_items.json'),",
            "'--items', str(code / 'attention_probe/group_probe_items.json'),",
        ),
        (
            "    if RESUME_ZIP:\n",
            "    command += ['--continuity', str(inputs / 'group_continuity.json')]\n    if RESUME_ZIP:\n",
        ),
        (
            "if helper_status.get('state') not in ('complete', 'paused'):",
            "if helper_status.get('state') != 'complete':",
        ),
        (
            "Runtime hat keinen vollständigen oder bewusst pausierten Abschluss gemeldet.",
            "Runtime hat keinen vollständigen Abschluss gemeldet.",
        ),
        (
            "'h286_a100_validation_diagnostic.zip'",
            "'h286_group_restore_diagnostic.zip'",
        ),
        ("'h286_a100_validation_results.zip'", "'h286_group_restore_results.zip'"),
        (
            "getrennte Antwortvalidierung.",
            "zehn Eingaben, begrenzte Gruppenrücksetzung und Antwortläufe.",
        ),
        (
            "Begrenzte explorative H286-Antwortvalidierung abgeschlossen. Keine allgemeine Wahrheitsvalidierung.",
            "Begrenzter explorativer H286-Gruppenversuch abgeschlossen. Inhaltliche Auswertung bleibt getrennt.",
        ),
    ):
        source = VALIDATION._replace_once(source, before, after)
    source = VALIDATION._replace_once(
        source,
        "if status['state'] == 'paused':\n    print('Bewusst pausierter Checkpoint; kein abgeschlossener Versuch und kein PASS.')\nelse:\n    print('Begrenzter explorativer H286-Gruppenversuch abgeschlossen. Inhaltliche Auswertung bleibt getrennt.')",
        "print('Begrenzter explorativer H286-Gruppenversuch abgeschlossen. Inhaltliche Auswertung bleibt getrennt.')",
    )
    ast.parse(source)
    return source


def notebook_document(modules, reference_files=None, *, expected_references=None):
    references = reference_files or {}
    if expected_references is None:
        expected_references = dict(REFERENCE_SHA256)
    document = VALIDATION.notebook_document(
        modules, references, expected_references=expected_references
    )
    document["cells"][0]["source"] = INTRO.splitlines(keepends=True)
    document["cells"][3]["source"] = group_run_source().splitlines(keepends=True)
    document["cells"][4]["source"] = [
        "Ergebnis-/Diagnose-ZIP herunterladen; vollständige Texte, Phasenstatus und Messwerte bleiben darin erhalten.\n"
    ]
    document["metadata"]["colab"]["name"] = MEMBER
    for index, cell in enumerate(document["cells"]):
        cell["id"] = hashlib.sha256(f"h286-group-restore:{index}".encode()).hexdigest()[
            :12
        ]
    return document


def build(output=HERE / MEMBER, *, reference_paths=None):
    output = Path(output)
    references = {
        name: Path(path).read_bytes() for name, path in (reference_paths or {}).items()
    }
    if references and output.resolve().is_relative_to(ROOT):
        raise ValueError(
            "Private group artifacts must be written outside the public repository"
        )
    document = notebook_document(source_bundle(), references)
    raw = (json.dumps(document, ensure_ascii=False, indent=1) + "\n").encode()
    if len(raw) > 2 * 1024**2:
        raise ValueError("Notebook exceeds the 2 MiB mobile-loader bound")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    if references:
        output.chmod(0o600)
    return output


def package_bytes(notebook):
    stream = io.BytesIO()
    info = zipfile.ZipInfo(MEMBER, date_time=(1980, 1, 1, 0, 0, 0))
    info.external_attr = 0o600 << 16
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            info, notebook, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9
        )
    return stream.getvalue()


def launcher_document(package_sha, notebook_sha, notebook_size):
    document = LAUNCHER.launcher_document(package_sha, notebook_sha, notebook_size)
    source = "".join(document["cells"][1]["source"])
    for before, after in (
        ("h286_a100_validation.ipynb", MEMBER),
        ("h286_a100_paket.zip", PACKAGE),
        ("h286_A100_START_diagnose.zip", "h286_group_restore_START_diagnose.zip"),
        ("h286_validation_cell_", "h286_group_restore_cell_"),
    ):
        if before not in source:
            raise ValueError("Prior launcher contract changed: " + before)
        source = source.replace(before, after)
    ast.parse(source)
    document["metadata"]["colab"]["name"] = STARTER
    document["cells"][0]["source"] = [
        "# H286-Gruppenversuch: kleiner Starter\n\n**A100 → Alle ausführen.** In der Dateiauswahl das bereitgestellte **"
        + PACKAGE
        + "** auswählen.\n"
        "Danach laufen alle Gruppen- und Antwortstufen automatisch. Kein Drive und kein Warten auf eine GPU.\n"
        "Am Ende Ergebnis-/Diagnose-ZIP herunterladen. Das private Paket nicht öffentlich hochladen.\n"
        "In normalem Jupyter die Paket-ZIP neben diesen Starter legen.\n"
    ]
    document["cells"][1]["source"] = source.splitlines(keepends=True)
    document["cells"][0]["id"] = "h286-group-start-intro"
    document["cells"][1]["id"] = "h286-group-start-code"
    return document


def package(notebook, directory, *, expected_notebook_sha256):
    if not re.fullmatch("[0-9a-f]{64}", expected_notebook_sha256):
        raise ValueError("An explicit reviewed notebook SHA256 is required")
    raw = Path(notebook).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_notebook_sha256:
        raise ValueError("Notebook does not match its reviewed SHA256")
    document = json.loads(raw)
    code = [
        cell for cell in document.get("cells", []) if cell.get("cell_type") == "code"
    ]
    if document.get("nbformat") != 4 or len(code) != 3 or len(raw) > 2 * 1024**2:
        raise ValueError("Unexpected notebook format, code-cell count or size")
    for cell in code:
        ast.parse(
            "".join(cell["source"])
            if isinstance(cell["source"], list)
            else cell["source"]
        )
        if cell.get("outputs") or cell.get("execution_count") is not None:
            raise ValueError("Distribute a clean unexecuted notebook")
    payload = package_bytes(raw)
    starter = launcher_document(
        hashlib.sha256(payload).hexdigest(), expected_notebook_sha256, len(raw)
    )
    serialized = json.dumps(starter, ensure_ascii=False, separators=(",", ":")) + "\n"
    if len(serialized.encode()) >= 8192 or len(payload) > 2 * 1024**2:
        raise ValueError("Mobile starter/package exceeds its size limit")
    directory = Path(directory)
    # Packages may carry private references; publish the hash-only starter separately.
    if directory.resolve().is_relative_to(ROOT):
        raise ValueError(
            "Notebook packages must be written outside the public repository"
        )
    directory.mkdir(parents=True, exist_ok=True)
    target, start = directory / PACKAGE, directory / STARTER
    target.write_bytes(payload)
    target.chmod(0o600)
    start.write_text(serialized)
    return start, target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / MEMBER)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--refw", type=Path)
    parser.add_argument("--continuity", type=Path)
    parser.add_argument("--package-dir", type=Path)
    args = parser.parse_args()
    paths = None
    if args.reference_dir or args.refw or args.continuity:
        if not args.reference_dir or not args.refw or not args.continuity:
            parser.error(
                "--reference-dir, --refw and --continuity are all required for private references"
            )
        paths = {
            name: args.reference_dir / name
            for name in ("hn_set.json", "htq_policy.json")
        }
        paths["refw_h286.json"] = args.refw
        paths["group_continuity.json"] = args.continuity
    notebook = build(args.out, reference_paths=paths)
    print(notebook)
    if args.package_dir:
        reviewed_sha = hashlib.sha256(notebook.read_bytes()).hexdigest()
        for artifact in package(
            notebook, args.package_dir, expected_notebook_sha256=reviewed_sha
        ):
            print(artifact)
