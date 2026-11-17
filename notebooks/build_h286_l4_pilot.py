"""Build public/private self-contained notebooks for the fixed H286 GPU pilot."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNTIME = HERE / "qwen3_h286_runtime.py"
TRANSFORMERS_VERSION = "5.22.2"
ACCELERATE_VERSION = "1.16.0"
REFERENCE_SHA256 = {
    "hn_set.json": "11390233b62c63b9d34805482ecbd3675eadd80570b9c01ef9e268bf565c98fd",
    "htq_policy.json": "160c420603b719afba4b909d12d31316535290e13b00aea2a6dea88110684f7c",
    "refw_h286.json": "68f96129d8ab00f3c7e6a68a9b4a0bdb58798dfac4d1d405f6cce5ccca0ae5cc",
}

INTRO = """# Original Qwen3-1.7B: unveränderte H286-Auswahl

**A100 bevorzugt, L4 möglich → Alle ausführen.** B200 ist bei passendem
Torch/CUDA und bestandenem Preflight ebenfalls möglich.
Dieselbe Datei funktioniert in Colab und in Jupyter. Eine CUDA-GPU mit mindestens
12 GiB freiem Speicher ist erforderlich; es gibt keine GPU-Warteschleife.

Die private Nutzdatei enthält die vorhandenen Referenzen bereits. Sie benötigt
keinen Drive-Zugriff und keine weiteren Uploads oder Pfadangaben. Das öffentliche
Template enthält keine privaten Referenzdaten und stoppt ohne sie vor Downloads.

Es werden die echten, festgelegten Qwen3-1.7B-Gewichte geladen (etwa 3,5 GB Download,
etwa 7 GB Modellgewichte in FP32 plus Laufzeitbedarf). Die 286 Koordinaten und ihre
originalen REF-W-Statistiken bleiben unverändert. Nach Warm-up und zwei technischen
Checks läuft der begrenzte Versuch mit 16 kurzen Fragen und fünf Bedingungen
automatisch. Fortschritt und Zeitabschätzung erscheinen während des Laufs.

Ergebnisse sind explorativ. Gemessene Rohaktivierung und geerbte Normierung werden
im Paket bezeichnet; dieser Versuch bestätigt keinen allgemeinen Wahrheits- oder
Selbstüberwachungsmechanismus. Atlas/Qwen3.6 bleibt separat vorbereitet.

Am Ende wird das Ergebnis- oder Diagnose-ZIP direkt angeboten. Ein bewusstes
Zeitlimit erzeugt einen fortsetzbaren Checkpoint mit Status `paused`, keinen PASS.
Nach dem Lauf die GPU-Sitzung beenden, wenn sie nicht mehr gebraucht wird.
"""

CONFIG = """# Private Nutzdatei: unverändert lassen und Alle ausführen.
# Öffentliches Template: optional vorhandene Originaldateien in diesem Verzeichnis.
REFERENCE_DIRECTORY = ""
RESUME_ZIP = ""  # Nur für einen vorhandenen Checkpoint von einer anderen Sitzung.
RUN_DIRECTORY = ""  # Leer: automatisch, wiederholbar in derselben Sitzung.
MIN_FREE_VRAM_GIB = 12.0
"""

RUN = r"""import base64
from collections import deque
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import zipfile

base = Path('/content') if Path('/content').is_dir() else Path.cwd()
binding_fingerprint = hashlib.sha256(json.dumps({'modules': MODULE_SHA256, 'references': REFERENCE_SHA256},
                                                sort_keys=True).encode()).hexdigest()[:12]
work = (Path(RUN_DIRECTORY).expanduser() if RUN_DIRECTORY
        else base / ('h286_attention_pilot_' + binding_fingerprint))
output = work / 'output'
output.mkdir(parents=True, exist_ok=True)
failure = None
helper_started = False
helper_status = None
return_code = None
status = {'state': 'running', 'stage': 'gpu_preflight', 'purpose': 'original_qwen3_h286_attention_pilot',
          'pretrained_weights_loaded': False, 'inference': 'exploratory (post hoc)'}

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + chr(10))

def run_command(command, logname, env=None):
    process = None
    tail = deque(maxlen=10)
    shown = 0
    try:
        with (output / logname).open('a', encoding='utf-8') as log:
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                log.write(line); log.flush()
                short = line.strip().replace(chr(13), ' ')[:500]
                tail.append(short)
                if logname == 'runtime.log' and shown < 120 and short.startswith(
                        ('Request ', 'Gespeichert ', 'Stage ', 'Status:', 'Status ', 'CHECK ', 'HARD FAILURE:')):
                    print(short, flush=True); shown += 1
            code = process.wait()
        if code:
            print('Kindprozess fehlgeschlagen; vollständiges Log im ZIP. Letzte Zeilen:', flush=True)
            for short in tail:
                print(short, flush=True)
            raise subprocess.CalledProcessError(code, command)
        return code
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        raise

try:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Keine CUDA-GPU. A100, L4 oder B200 auswählen; kein Warten.')
    capability = tuple(torch.cuda.get_device_capability(0))
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    if capability[0] < 8 or free_bytes < MIN_FREE_VRAM_GIB * 1024**3:
        raise RuntimeError('Ampere-oder-neuere CUDA-GPU mit mindestens 12 GiB freiem Speicher erforderlich.')
    torch_before = importlib.metadata.version('torch')
    environment = {'python': sys.version, 'torch': torch_before, 'torch_cuda': torch.version.cuda,
                   'gpu': torch.cuda.get_device_name(0), 'capability': list(capability),
                   'free_vram_gib': free_bytes / 1024**3, 'total_vram_gib': total_bytes / 1024**3,
                   'transformers_required': TRANSFORMERS_VERSION, 'accelerate_required': ACCELERATE_VERSION,
                   'execution_dtype': 'float32', 'tf32': False}
    write_json(output / 'environment_before.json', environment)
    print('GPU:', environment['gpu'], '| frei:', round(environment['free_vram_gib'], 2), 'GiB')
    print('Torch/CUDA bleibt erhalten:', torch_before, torch.version.cuda)
    status['stage'] = 'reference_validation'
    inputs = work / 'inputs'; inputs.mkdir(exist_ok=True)
    for name, expected in REFERENCE_SHA256.items():
        if name in REFERENCE_FILES_B64:
            raw = base64.b64decode(REFERENCE_FILES_B64[name], validate=True)
        elif REFERENCE_DIRECTORY:
            raw = (Path(REFERENCE_DIRECTORY).expanduser() / name).read_bytes()
        else:
            raise RuntimeError('Öffentliches Template ohne private Referenz. Die bereitgestellte private Nutzdatei verwenden.')
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Originalreferenz-Prüfsumme weicht ab: ' + name)
        target = inputs / name
        if target.exists() and target.read_bytes() != raw:
            raise ValueError('Vorhandene Referenz gehört zu einer anderen Auswahl; Ergebnisse nicht mischen.')
        target.write_bytes(raw)
    binding = {'sha256': MODULE_SHA256, 'reference_sha256': REFERENCE_SHA256,
               'model': 'Qwen/Qwen3-1.7B', 'revision': '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'}
    binding_path = output / 'embedded_source.json'
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError('Vorhandener Lauf gehört zu anderem Quellcode oder Referenzen; Ergebnisse nicht mischen.')
    code = work / 'code'; code.mkdir(exist_ok=True)
    for name, source in MODULE_SOURCES.items():
        if hashlib.sha256(source.encode()).hexdigest() != MODULE_SHA256[name]:
            raise ValueError('Eingebetteter Quellcode weicht ab: ' + name)
        path = code / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(source)
    write_json(binding_path, binding)
    status['stage'] = 'dependencies'
    constraint = work / 'preserve_torch.txt'
    constraint.write_text('torch==' + torch_before + chr(10))
    print('Exakte Transformers-/Accelerate-Version installieren; GPU-Torch bleibt festgelegt.', flush=True)
    child_env = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS='1', TQDM_DISABLE='1',
                     TOKENIZERS_PARALLELISM='false')
    run_command([sys.executable, '-m', 'pip', 'install', '--quiet', '--upgrade', '--upgrade-strategy', 'only-if-needed',
                 '--constraint', str(constraint), 'transformers==' + TRANSFORMERS_VERSION,
                 'accelerate==' + ACCELERATE_VERSION], 'install.log', child_env)
    installed = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'accelerate')}
    if installed != {'torch': torch_before, 'transformers': TRANSFORMERS_VERSION, 'accelerate': ACCELERATE_VERSION}:
        raise RuntimeError('Paketversionen weichen ab; keine Modellgewichte angefordert.')
    write_json(output / 'environment_installed.json', installed)
    status['stage'] = 'runtime'
    helper_started = True
    status['pretrained_weights_loaded'] = None
    env = dict(child_env, PYTHONPATH=str(code))
    command = [sys.executable, str(code / 'qwen3_h286_runtime.py'), '--hn-set', str(inputs / 'hn_set.json'),
               '--policy', str(inputs / 'htq_policy.json'), '--refw', str(inputs / 'refw_h286.json'),
               '--items', str(code / 'attention_probe/pilot_items.json'),
               '--out-dir', str(output), '--cache-dir', str(base / 'h286_model_cache')]
    if RESUME_ZIP:
        command += ['--resume-zip', str(Path(RESUME_ZIP).expanduser())]
    print('Frischer Python-Prozess: Original-FP32-Modell, feste H286/REF-W, 16 kurze Aufgaben.', flush=True)
    return_code = run_command(command, 'runtime.log', env)
    helper_status = json.loads((output / 'status.json').read_text())
    if helper_status.get('state') not in ('complete', 'paused'):
        raise RuntimeError('Runtime hat keinen vollständigen oder bewusst pausierten Abschluss gemeldet.')
    status.update(helper_status, helper_return_code=return_code, wrapper_verified=True)
except BaseException as error:
    failure = error
    status.update(state='failed', error_type=type(error).__name__, error=str(error), wrapper_verified=False)
    (output / 'wrapper_diagnostic.txt').write_text(traceback.format_exc())
finally:
    # Keep the child's original status for diagnosis, then publish one authoritative final status.
    child_path = output / 'status.json'
    if helper_started and child_path.exists():
        try:
            helper_status = json.loads(child_path.read_text())
            write_json(output / 'helper_status.json', helper_status)
            if failure is not None:
                status['helper_state'] = helper_status.get('state')
                status['pretrained_weights_loaded'] = helper_status.get('pretrained_weights_loaded')
        except Exception:
            status['helper_state'] = 'unreadable'
    status['written_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(output / 'status.json', status)
    members = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(output.rglob('*')) if path.is_file()
               and path.name not in ('WRAPPER_MANIFEST.json',) and path.suffix not in ('.zip', '.tmp')
               and not any(part.startswith('.') for part in path.relative_to(output).parts)}
    write_json(output / 'WRAPPER_MANIFEST.json', members)
    target = base / ('h286_l4_pilot_diagnostic.zip' if failure is not None else 'h286_l4_pilot_results.zip')
    temporary = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if (path.is_file() and path.suffix not in ('.zip', '.tmp')
                    and not any(part.startswith('.') for part in path.relative_to(output).parts)):
                archive.write(path, str(path.relative_to(output)))
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Ergebnis-ZIP ist beschädigt.')
    os.replace(temporary, target)
    print('Status:', status['state'], '| ZIP:', target, flush=True)
    try:
        from google.colab import files
        files.download(str(target))
    except Exception:
        try:
            from IPython.display import FileLink, display
            display(FileLink(os.path.relpath(target, Path.cwd())))
        except Exception:
            print('ZIP aus diesem Verzeichnis herunterladen:', target)
if failure is not None:
    raise RuntimeError('Versuch angehalten; das Diagnose-ZIP enthält den tatsächlichen Fehler.') from None
if status['state'] == 'paused':
    print('Bewusst pausierter Checkpoint; kein abgeschlossener Versuch und kein PASS.')
else:
    print('Begrenzter explorativer H286-Versuch abgeschlossen. Keine allgemeine Wahrheitsvalidierung.')
"""


def source_bundle() -> dict[str, str]:
    return {
        "attention_probe/__init__.py": (
            ROOT / "attention_probe/__init__.py"
        ).read_text(),
        "attention_probe/reference.py": (
            ROOT / "attention_probe/reference.py"
        ).read_text(),
        "attention_probe/fixed_refw.py": (
            ROOT / "attention_probe/fixed_refw.py"
        ).read_text(),
        "attention_probe/interventions.py": (
            ROOT / "attention_probe/interventions.py"
        ).read_text(),
        "attention_probe/pilot_items.json": (
            ROOT / "attention_probe/pilot_items.json"
        ).read_text(),
        "qwen3_h286_runtime.py": RUNTIME.read_text(),
    }


def notebook_document(
    modules: dict[str, str],
    reference_files: dict[str, bytes] | None = None,
    *,
    expected_references: dict[str, str] | None = None,
) -> dict:
    """Low-level renderer supports synthetic offline fixtures with explicit anchors."""
    references = reference_files or {}
    expected = REFERENCE_SHA256 if expected_references is None else expected_references
    if references and (
        set(references) != set(expected)
        or any(
            hashlib.sha256(raw).hexdigest() != expected[name]
            for name, raw in references.items()
        )
    ):
        raise ValueError(
            "Private reference bytes do not match their exact production anchors."
        )
    for name, source in modules.items():
        if name.endswith(".py"):
            ast.parse(source)
    payload = (
        "# Eingebettete Quellen und optionale private Originalbytes; nicht manuell ändern.\n"
        + "TRANSFORMERS_VERSION = "
        + repr(TRANSFORMERS_VERSION)
        + "\n"
        + "ACCELERATE_VERSION = "
        + repr(ACCELERATE_VERSION)
        + "\n"
        + "REFERENCE_SHA256 = "
        + repr(expected)
        + "\n"
        + "REFERENCE_FILES_B64 = "
        + repr(
            {name: base64.b64encode(raw).decode() for name, raw in references.items()}
        )
        + "\n"
        + "MODULE_SOURCES = "
        + repr(modules)
        + "\n"
        + "MODULE_SHA256 = "
        + repr(
            {
                name: hashlib.sha256(source.encode()).hexdigest()
                for name, source in modules.items()
            }
        )
        + "\n"
    )
    items = [
        ("markdown", INTRO),
        ("code", CONFIG),
        ("code", payload),
        ("code", RUN),
        (
            "markdown",
            "Das ZIP enthält Status, Messwerte, Zeitangaben und fortsetzbare Einzelaufgaben. GPU danach bei Bedarf beenden.\n",
        ),
    ]
    cells = []
    for i, (kind, source) in enumerate(items):
        value = {
            "id": hashlib.sha256(f"h286-pilot:{i}".encode()).hexdigest()[:12],
            "cell_type": kind,
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if kind == "code":
            ast.parse(source)
            value.update(execution_count=None, outputs=[])
        cells.append(value)
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "h286_l4_pilot.ipynb", "provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
        },
        "cells": cells,
    }


def build(
    output: Path = HERE / "h286_l4_pilot.ipynb",
    *,
    reference_paths: dict[str, Path] | None = None,
) -> Path:
    reference_files = {
        name: path.read_bytes() for name, path in (reference_paths or {}).items()
    }
    if reference_files and output.resolve().is_relative_to(ROOT):
        raise ValueError(
            "Private notebook must be written outside the public repository."
        )
    document = notebook_document(source_bundle(), reference_files)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n")
    if reference_files:
        output.chmod(0o600)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "h286_l4_pilot.ipynb")
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--refw", type=Path)
    args = parser.parse_args()
    paths = None
    if args.reference_dir or args.refw:
        if not args.reference_dir or not args.refw:
            parser.error(
                "--reference-dir and --refw must be provided together for the private notebook."
            )
        paths = {
            name: args.reference_dir / name
            for name in ("hn_set.json", "htq_policy.json")
        }
        paths["refw_h286.json"] = args.refw
        if args.out.resolve().is_relative_to(ROOT):
            parser.error(
                "Private notebook must be written outside the public repository."
            )
    print(build(args.out, reference_paths=paths))


if __name__ == "__main__":
    main()
