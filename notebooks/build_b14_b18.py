"""Build the native B14→B18 local-transport pilot and its small mobile starter.

The exercised wrapper and mobile loader are reused without modifying any prior
experiment. Public templates contain source and hashes only; the actual inputs
and H286 reference bytes are confined to a private package outside this repo.
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
MEMBER = "b14_b18_local_transport.ipynb"
PACKAGE = "b14_b18_paket.zip"
STARTER = "b14_b18_START.ipynb"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRIOR = load("b14_b18_group_packager", HERE / "build_h286_group_restore.py")
VALIDATION = PRIOR.VALIDATION
TRANSFORMERS_VERSION = VALIDATION.TRANSFORMERS_VERSION
ACCELERATE_VERSION = VALIDATION.ACCELERATE_VERSION
# Bound once the fixed private input selection and protocol have been reviewed.
ITEMS_SHA256 = "cc482e87fc49dcea8fa77b0ccb4bf63ed8df41e5f75c5c47c434decd4e3d1a5e"
PLAN_SHA256 = "76dcc81a06142b2fb662383ec82a6dc17834e9612543163036d2a7c278643bbf"
REFERENCE_SHA256 = dict(
    VALIDATION.REFERENCE_SHA256,
    **{
        "b14_b18_items.json": ITEMS_SHA256,
        "b14_b18_plan.json": PLAN_SHA256,
    },
)
NEW_SOURCES = {
    "attention_probe/native_jacobian.py": ROOT / "attention_probe/native_jacobian.py",
    "attention_probe/local_transport.py": ROOT / "attention_probe/local_transport.py",
    "attention_probe/result_bundle.py": ROOT / "attention_probe/result_bundle.py",
    "qwen3_b14_b18_runtime.py": HERE / "qwen3_b14_b18_runtime.py",
}
INTRO = """# Block 14 → Block 18: lokale Ausbreitung und Rücktransport

**A100 → Alle ausführen.** Im kleinen Starter das bereitgestellte
**b14_b18_paket.zip** auswählen. Danach laufen alle Stufen automatisch.

16 bestehende Wissens- und Rechenaufgaben, originales Qwen3-1.7B, dieselben
286 Koordinaten und dieselbe REF-W-Normierung; FP32 und TF32 aus.
Attention-Eingriffe an Block 13/14/15 werden an Block 17/18/19 gemessen.
Identische vorgegebene Fortsetzungen trennen Zustandsänderungen von abweichenden
Antworten. Freie Antworten werden zusätzlich und getrennt gespeichert.

Der lokale Jacobian der Strecke 14→18 wird mit automatischer Differentiation
gemessen und durch Finite Differenzen geprüft. Für vier festgelegte Aufgaben
wird eine Veränderung an Block 18 mit einer regularisierten lokalen Abbildung
nach Block 14 zurücktransportiert, einschließlich Kopier- und Zufallskontrollen.
Das ist eine kontextabhängige Erweiterung der Transportidee aus Mem2Gen;
es ist keine bereits gefittete globale Jacobian-Lens und kein Gedächtnisnachweis.

Es gibt keinen Drive-Zugriff und kein Warten auf eine GPU. Vorhandene Gewichte
werden wiederverwendet; der GPU-Torch bleibt installiert. Der Rechenteil hat
ein Zeitbudget von 45 Minuten mit Checkpoints zwischen Aufgaben. Modellladen
und Installation kommen hinzu. Eine Pause ist ausdrücklich kein vollständiger Lauf.

Am Ende wird **b14_b18_results.zip** als kompakte Auswertung angeboten: diese
Datei hier hochladen. Die zusätzliche **b14_b18_results_RAW.zip** bitte ebenfalls
sichern; sie enthält alle Logit-Rohdaten und dient auch zur Wiederaufnahme.
Die kompakte Datei ist kein Resume-Paket. Bei Fehlern wird ein Diagnose-ZIP
angeboten. Anschließend die GPU-Sitzung beenden, wenn sie nicht mehr gebraucht
wird. Das private Paket nicht öffentlich hochladen.
"""


def source_bundle():
    sources = VALIDATION.BASE.source_bundle()
    sources["attention_probe/answer_validation.py"] = (
        ROOT / "attention_probe/answer_validation.py"
    ).read_text()
    sources.update({name: path.read_text() for name, path in NEW_SOURCES.items()})
    return sources


def run_source():
    source = VALIDATION.validation_run_source()
    for before, after in (
        ("'h286_a100_validation_'", "'b14_b18_local_transport_'"),
        (
            "'original_qwen3_h286_answer_validation'",
            "'original_qwen3_b14_b18_local_transport'",
        ),
        (
            "str(code / 'qwen3_h286_validation_runtime.py')",
            "str(code / 'qwen3_b14_b18_runtime.py')",
        ),
        (
            "'--items', str(code / 'attention_probe/pilot_items.json'),",
            "'--items', str(inputs / 'b14_b18_items.json'),\n               '--plan', str(inputs / 'b14_b18_plan.json'),",
        ),
        (
            "getrennte Antwortvalidierung.",
            "B14→B18: 16 Aufgaben, lokale Jacobians und vier Rücktransporte.",
        ),
        ("'h286_a100_validation_diagnostic.zip'", "'b14_b18_diagnostic.zip'"),
        ("'h286_a100_validation_results.zip'", "'b14_b18_results.zip'"),
        (
            "Begrenzte explorative H286-Antwortvalidierung abgeschlossen. Keine allgemeine Wahrheitsvalidierung.",
            "B14→B18-Versuch abgeschlossen. Ergebnisse für die getrennte wissenschaftliche Auswertung gespeichert.",
        ),
    ):
        source = VALIDATION._replace_once(source, before, after)
    compact = """    os.replace(temporary, target)
    raw_target = None
    if helper_started:
        # Compact analysis retains every state/derivative array. Full logit
        # vectors and exact resume checkpoints stay in a separate RAW archive.
        try:
            import importlib.util
            bundle_path = code / 'attention_probe/result_bundle.py'
            expected_bundle = MODULE_SHA256['attention_probe/result_bundle.py']
            if hashlib.sha256(bundle_path.read_bytes()).hexdigest() != expected_bundle:
                raise ValueError('Result-export source SHA mismatch')
            bundle_spec = importlib.util.spec_from_file_location('b14_b18_result_export', bundle_path)
            bundle_module = importlib.util.module_from_spec(bundle_spec)
            bundle_spec.loader.exec_module(bundle_module)
            raw_target = target.with_name(target.stem + '_RAW.zip')
            os.replace(target, raw_target)
            compact_info = bundle_module.build_compact_result(raw_target, target)
            print('Kompakte Auswertung:', target, '| Rohdaten ebenfalls sichern:', raw_target, flush=True)
        except Exception as compact_error:
            print('Kompaktexport nicht verfügbar; vollständiges Original-ZIP verwenden:', str(compact_error)[:300], flush=True)
            if raw_target is not None and raw_target.is_file():
                os.replace(raw_target, target)
            raw_target = None
"""
    source = VALIDATION._replace_once(
        source, "    os.replace(temporary, target)\n", compact
    )
    source = VALIDATION._replace_once(
        source,
        "        files.download(str(target))\n",
        "        files.download(str(target))\n        if raw_target is not None:\n            files.download(str(raw_target))\n",
    )
    source = VALIDATION._replace_once(
        source,
        "            display(FileLink(os.path.relpath(target, Path.cwd())))\n",
        "            display(FileLink(os.path.relpath(target, Path.cwd())))\n            if raw_target is not None:\n                display(FileLink(os.path.relpath(raw_target, Path.cwd())))\n",
    )
    ast.parse(source)
    return source


def notebook_document(modules, reference_files=None, *, expected_references=None):
    expected = REFERENCE_SHA256 if expected_references is None else expected_references
    if any(len(value) != 64 for value in expected.values()):
        raise ValueError("Bind and review every input SHA256 before building")
    document = VALIDATION.notebook_document(
        modules, reference_files, expected_references=expected
    )
    document["cells"][0]["source"] = INTRO.splitlines(keepends=True)
    document["cells"][3]["source"] = run_source().splitlines(keepends=True)
    document["cells"][4]["source"] = [
        (
            "Ergebnis-/Diagnose-ZIP sichern. Status, fehlende Fortsetzungspositionen, "
            "EOS/Tokenlimit und Solver-Konvergenz getrennt auswerten.\n"
        )
    ]
    document["metadata"]["colab"]["name"] = MEMBER
    for index, cell in enumerate(document["cells"]):
        cell["id"] = hashlib.sha256(f"b14-b18:{index}".encode()).hexdigest()[:12]
    return document


def launcher_document(package_sha, notebook_sha, notebook_size):
    document = PRIOR.launcher_document(package_sha, notebook_sha, notebook_size)
    source = "".join(document["cells"][1]["source"])
    for before, after in (
        (PRIOR.MEMBER, MEMBER),
        (PRIOR.PACKAGE, PACKAGE),
        ("h286_group_restore_START_diagnose.zip", "b14_b18_START_diagnose.zip"),
        ("h286_group_restore_cell_", "b14_b18_cell_"),
    ):
        if before not in source:
            raise ValueError("Prior launcher contract changed: " + before)
        source = source.replace(before, after)
    ast.parse(source)
    document["metadata"]["colab"]["name"] = STARTER
    document["cells"][0]["source"] = [
        (
            "# Block 14 → Block 18\n\n**A100 → Alle ausführen.** Bei der Dateiauswahl "
            "**b14_b18_paket.zip** auswählen. Danach laufen die 16 Aufgaben und "
            "die Jacobian-/Transportprüfungen automatisch. Kein Drive nötig.\n\n"
            "Am Ende **b14_b18_results.zip** herunterladen und hier zur Auswertung "
            "bereitstellen. Die zusätzliche **b14_b18_results_RAW.zip** ebenfalls sichern; "
            "sie enthält die vollständigen Rohdaten. Bei Fehlern das Diagnose-ZIP verwenden. Das private Paket "
            "nicht öffentlich hochladen. In Jupyter die Paket-ZIP neben den Starter legen.\n"
        )
    ]
    document["cells"][1]["source"] = source.splitlines(keepends=True)
    document["cells"][0]["id"] = "b14-b18-start-intro"
    document["cells"][1]["id"] = "b14-b18-start-code"
    return document


def inherit(function):
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
    result = types.FunctionType(
        function.__code__,
        bindings,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    result.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return result


package_bytes = inherit(PRIOR.package_bytes)
package = inherit(PRIOR.package)
_build = inherit(PRIOR.build)


def build(output=HERE / MEMBER, *, reference_paths=None):
    return _build(output, reference_paths=reference_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / MEMBER)
    parser.add_argument("--inputs-dir", type=Path)
    parser.add_argument("--package-dir", type=Path)
    args = parser.parse_args()
    paths = (
        {name: args.inputs_dir / name for name in REFERENCE_SHA256}
        if args.inputs_dir
        else None
    )
    notebook = build(args.out, reference_paths=paths)
    print(notebook)
    if args.package_dir:
        for path in package(
            notebook,
            args.package_dir,
            expected_notebook_sha256=hashlib.sha256(notebook.read_bytes()).hexdigest(),
        ):
            print(path)
