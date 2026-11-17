"""Bind a tiny public A100 starter to an explicitly reviewed private notebook."""
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
SPEC = importlib.util.spec_from_file_location("h286_prior_launcher", HERE / "build_h286_small_launcher.py")
assert SPEC and SPEC.loader
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

MEMBER = "h286_a100_validation.ipynb"
PACKAGE = "h286_a100_paket.zip"
STARTER = "h286_A100_START.ipynb"
INTRO = """# A100: Antworten und H286 prüfen

**Nur diesen kleinen Starter in Colab öffnen.** Laufzeittyp **A100**, dann
**Alle ausführen**. In der Dateiauswahl **h286_a100_paket.zip** auswählen.

Danach laufen die 16 Aufgaben mit je fünf Varianten automatisch: bisherige
Zustandsmessung, frei erzeugte Antworten und wahrscheinlichste Antworttoken.
Die 286 Koordinaten und das originale Qwen3-1.7B bleiben gleich.
Der Eingriff erfolgt einmal am Ende des Prompts. Kein Training.

Kein Drive-Zugriff nötig. Fortschritt und Restzeit erscheinen während des Laufs;
am Ende wird die Ergebnis- oder Diagnose-ZIP angeboten. Danach die GPU-Sitzung
beenden, wenn sie nicht mehr gebraucht wird. Das private Paket nicht öffentlich
hochladen. In normalem Jupyter die ZIP neben den Starter legen.
"""


def package_bytes(notebook: bytes) -> bytes:
    stream = io.BytesIO()
    info = zipfile.ZipInfo(MEMBER, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr(info, notebook, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return stream.getvalue()


def launcher_document(package_sha: str, notebook_sha: str, notebook_size: int) -> dict:
    document = BASE.launcher_document(package_sha, notebook_sha, notebook_size)
    source = "".join(document["cells"][1]["source"])
    replacements = {
        "h286_l4_pilot.ipynb": MEMBER,
        "h286_pilot_paket.zip": PACKAGE,
        "h286_START_diagnose.zip": "h286_A100_START_diagnose.zip",
        "h286_original_cell_": "h286_validation_cell_",
    }
    for old, new in replacements.items():
        if old not in source:
            raise ValueError("Prior launcher contract changed: " + old)
        source = source.replace(old, new)
    ast.parse(source)
    document["metadata"]["colab"]["name"] = STARTER
    document["cells"][0]["source"] = INTRO.splitlines(keepends=True)
    document["cells"][0]["id"] = "h286-a100-intro"
    document["cells"][1]["source"] = source.splitlines(keepends=True)
    document["cells"][1]["id"] = "h286-a100-loader"
    return document


def build(notebook: Path, directory: Path, *, expected_notebook_sha256: str) -> tuple[Path, Path]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_notebook_sha256):
        raise ValueError("An explicit reviewed notebook SHA256 is required")
    raw = notebook.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_notebook_sha256:
        raise ValueError("Notebook does not match its reviewed SHA256")
    document = json.loads(raw)
    cells = [cell for cell in document.get("cells", []) if cell.get("cell_type") == "code"]
    if document.get("nbformat") != 4 or len(cells) != 3 or len(raw) > 2 * 1024**2:
        raise ValueError("Unexpected notebook format, code-cell count, or size")
    for cell in cells:
        source = cell["source"]
        ast.parse("".join(source) if isinstance(source, list) else source)
        if cell.get("outputs") or cell.get("execution_count") is not None:
            raise ValueError("Distribute a clean unexecuted notebook")
    packed = package_bytes(raw)
    launcher = launcher_document(hashlib.sha256(packed).hexdigest(), expected_notebook_sha256, len(raw))
    serialized = json.dumps(launcher, ensure_ascii=False, separators=(",", ":")) + "\n"
    if len(serialized.encode()) >= 8 * 1024 or len(packed) > 2 * 1024**2:
        raise ValueError("Mobile starter/package exceeds its size limit")
    directory.mkdir(parents=True, exist_ok=True)
    package, starter = directory / PACKAGE, directory / STARTER
    package.write_bytes(packed)
    package.chmod(0o600)
    starter.write_text(serialized)
    return starter, package


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--expected-notebook-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("/workspace/artifacts"))
    args = parser.parse_args()
    for path in build(args.notebook, args.out_dir, expected_notebook_sha256=args.expected_notebook_sha256):
        print(path)
