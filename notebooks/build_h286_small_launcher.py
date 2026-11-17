"""Package the exact tested private notebook behind a tiny public Colab loader."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import zipfile
from pathlib import Path

NOTEBOOK_SHA256 = "348fc41423a924447a3743cfc01537529667ac2b53555680b2e7d2c1820d33d2"
MEMBER = "h286_l4_pilot.ipynb"
PACKAGE = "h286_pilot_paket.zip"

INTRO = """# Kleiner H286-Starter

**Nur diese kleine Datei in Colab öffnen.** A100 bevorzugt; L4 ist möglich.

1. Dieses Notebook öffnen und **Alle ausführen**.
2. Erst in der erscheinenden Dateiauswahl **h286_pilot_paket.zip** auswählen.
3. Der unveränderte Versuch startet automatisch; das Ergebnis-ZIP wird angeboten.

Das große Notebook wird nur im Python-Speicher gelesen und nicht im Browser
gerendert. Kein Drive nötig. Das Paket enthält private Referenzen; nicht öffentlich
hochladen. In normalem Jupyter die ZIP neben diesen Starter legen.
"""

CODE = r"""import hashlib, io, json, traceback, zipfile
from pathlib import Path
PACKAGE_SHA = __PACKAGE_SHA__
NOTEBOOK_SHA = __NOTEBOOK_SHA__
NOTEBOOK_BYTES = __NOTEBOOK_BYTES__
MEMBER = 'h286_l4_pilot.ipynb'
PAKET = Path('h286_pilot_paket.zip')
stage, started = 'Paket laden', 0

def anbieten(path):
    try:
        from google.colab import files
        files.download(str(path))
    except Exception:
        try:
            from IPython.display import FileLink, display
            display(FileLink(str(path)))
        except Exception:
            print('Datei:', path)

try:
    if PAKET.is_file():
        if PAKET.stat().st_size > 2*1024**2:
            raise ValueError('Paket zu groß.')
        raw = PAKET.read_bytes()
    else:
        try:
            from google.colab import files
        except ImportError:
            raise FileNotFoundError('h286_pilot_paket.zip neben den Starter legen.') from None
        print('Jetzt genau das heruntergeladene h286_pilot_paket.zip auswählen.')
        uploaded = files.upload()
        if len(uploaded) != 1:
            raise ValueError('Bitte genau eine Paket-ZIP auswählen.')
        raw = bytes(next(iter(uploaded.values())))
    stage = 'ZIP prüfen'
    if len(raw) > 2*1024**2 or hashlib.sha256(raw).hexdigest() != PACKAGE_SHA:
        raise ValueError('Paket-Prüfsumme stimmt nicht; Originalpaket verwenden.')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if (len(members) != 1 or members[0].filename != MEMBER or members[0].is_dir()
            or members[0].flag_bits & 1 or members[0].file_size != NOTEBOOK_BYTES
            or members[0].file_size > 2*1024**2):
            raise ValueError('Unerwarteter oder zu großer ZIP-Inhalt.')
        if archive.testzip() is not None:
            raise ValueError('ZIP-CRC-Prüfung fehlgeschlagen.')
        notebook = archive.read(members[0])
    stage = 'Notebook prüfen'
    if hashlib.sha256(notebook).hexdigest() != NOTEBOOK_SHA:
        raise ValueError('Notebook-Prüfsumme stimmt nicht.')
    document = json.loads(notebook)
    if document.get('nbformat') != 4 or not isinstance(document.get('cells'), list):
        raise ValueError('Ungültiges Notebookformat.')
    cells = [cell for cell in document['cells'] if cell.get('cell_type') == 'code']
    if len(cells) != 3:
        raise ValueError('Unerwartete Anzahl Original-Codezellen.')
    scope = {'__name__': '__main__'}
    for number, cell in enumerate(cells, 1):
        stage = 'Original-Codezelle ' + str(number)
        source = cell['source']
        if isinstance(source, list):
            source = ''.join(source)
        if not isinstance(source, str):
            raise ValueError('Ungültige Codezelle.')
        print('Original-Codezelle', number, '/', len(cells), 'startet.', flush=True)
        started += 1
        exec(compile(source, 'h286_original_cell_' + str(number), 'exec'), scope)
except BaseException as error:
    if (started == 3 and isinstance(error, RuntimeError)
        and str(error).startswith('Versuch angehalten;') and 'scope' in globals()
        and scope.get('failure') is not None and isinstance(scope.get('target'), Path)
        and scope['target'].is_file()):
        print('Originalversuch angehalten. Sein vollständiges Diagnose-ZIP wurde bereits angeboten:', scope['target'])
        raise RuntimeError('Bitte das bereits angebotene Original-Diagnose-ZIP verwenden.') from None
    status = {'state': 'failed', 'stage': stage, 'code_cells_started': started,
              'error_type': type(error).__name__, 'error': str(error)[:2000],
              'expected_package_sha256': PACKAGE_SHA, 'expected_notebook_sha256': NOTEBOOK_SHA}
    target = Path('h286_START_diagnose.zip')
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as diagnostic:
        diagnostic.writestr('bootstrap_status.json', json.dumps(status, ensure_ascii=False, indent=2))
        diagnostic.writestr('traceback.txt', traceback.format_exc()[-12000:])
    print('Starter angehalten bei:', stage, '| Diagnose:', target)
    anbieten(target)
    raise RuntimeError('Starter angehalten; das Diagnose-ZIP enthält den Fehler.') from None
"""


def package_bytes(notebook: bytes) -> bytes:
    stream = io.BytesIO()
    info = zipfile.ZipInfo(MEMBER, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    with zipfile.ZipFile(
        stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        archive.writestr(
            info, notebook, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9
        )
    return stream.getvalue()


def launcher_document(package_sha: str, notebook_sha: str, notebook_size: int) -> dict:
    source = CODE.replace("__PACKAGE_SHA__", repr(package_sha)).replace(
        "__NOTEBOOK_SHA__", repr(notebook_sha)
    )
    source = source.replace("__NOTEBOOK_BYTES__", str(notebook_size))
    ast.parse(source)
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "h286_START_KLEIN.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
        },
        "cells": [
            {
                "id": "h286-small-intro",
                "cell_type": "markdown",
                "metadata": {},
                "source": INTRO.splitlines(keepends=True),
            },
            {
                "id": "h286-small-code",
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": source.splitlines(keepends=True),
            },
        ],
    }


def build(notebook: Path, directory: Path) -> tuple[Path, Path]:
    raw = notebook.read_bytes()
    if hashlib.sha256(raw).hexdigest() != NOTEBOOK_SHA256:
        raise ValueError(
            "Input is not the exact already-tested private notebook; refuse silent pipeline changes."
        )
    packed = package_bytes(raw)
    document = launcher_document(
        hashlib.sha256(packed).hexdigest(), NOTEBOOK_SHA256, len(raw)
    )
    directory.mkdir(parents=True, exist_ok=True)
    package, launcher = directory / PACKAGE, directory / "h286_START_KLEIN.ipynb"
    package.write_bytes(packed)
    package.chmod(0o600)
    launcher.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    return launcher, package


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("/workspace/artifacts/h286_l4_pilot.ipynb"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("/workspace/artifacts"))
    args = parser.parse_args()
    for path in build(args.notebook, args.out_dir):
        print(path)
