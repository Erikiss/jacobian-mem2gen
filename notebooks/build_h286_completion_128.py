"""Separate four-input completion notebook/package; never alter the old 48 run."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MEMBER = 'h286_completion_128.ipynb'
PACKAGE = 'h286_completion_128_paket.zip'
STARTER = 'h286_completion_128_START.ipynb'
BASELINE_SHA256 = '930d772f999b91931023cc1dbbc4bbd98ad68eb27717d6b6fdc6356d1f2e1bd0'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


PRIOR = load('completion_prior_builder', HERE/'build_h286_a100_validation.py')
LAUNCHER = load('completion_prior_launcher', HERE/'build_h286_validation_launcher.py')
REFERENCE_SHA256 = dict(PRIOR.REFERENCE_SHA256, **{'completion_48_baseline.json': BASELINE_SHA256})
INTRO = '''# Vier offene Antworten: Gesamtkappe128

A100 auswählen und **Alle ausführen**. Dieses getrennte Paket wiederholt nur
p07/p08 mit beiden Entwürfen (Original-IDs r012–r015), jeweils alle fünf bisherigen
Arme. Original Qwen3-1.7B, FP32, H286, REF-W und Kontrollen bleiben gleich.

Jeder Arm startet frisch vom unveränderten Prompt mit einmaligem Prefill-Puls;
maximal 128 ausgewählte Tokens insgesamt, bei EOS endet er früher. Alte Tokens
werden niemals als erzwungenes Fortsetzungspräfix eingespeist. Die ersten 48 Tokens
beziehungsweise ein früheres EOS müssen zum bisherigen Lauf passen. Numerische
Logit-/Prefill-Differenzen werden zusätzlich protokolliert. Abweichende Tokenpräfixe
führen zum Diagnose-Status; bereits erzeugte Antworten bleiben im ZIP erhalten.

Abschluss, striktes Antwortformat und später zu prüfende semantische Antwort bleiben
getrennt. Eine abgeschnittene Antwort wird nicht als richtig oder falsch gerettet.
Der Nachlauf wurde nach Sichtung der alten Ergebnisse gewählt und ist diagnostisch.
Kein Drive-Zugriff. Am Ende Ergebnis-/Diagnose-ZIP herunterladen; alte 48-Ergebnisse
werden nicht verändert. Gewichte im vorhandenen Modellcache werden wiederverwendet.
'''


def source_bundle():
    result = PRIOR.source_bundle()
    result['attention_probe/completion_128.py'] = (ROOT/'attention_probe/completion_128.py').read_text()
    result['qwen3_h286_completion_128_runtime.py'] = (HERE/'qwen3_h286_completion_128_runtime.py').read_text()
    return result


def notebook_document(modules, references=None, *, expected_references=None):
    document = PRIOR.notebook_document(modules, references,
        expected_references=REFERENCE_SHA256 if expected_references is None else expected_references)
    code = ''.join(document['cells'][3]['source'])
    for before, after in (
        ("'h286_a100_validation_'", "'h286_completion_128_'"),
        ("'original_qwen3_h286_answer_validation'", "'original_qwen3_h286_completion_128'"),
        ("str(code / 'qwen3_h286_validation_runtime.py')", "str(code / 'qwen3_h286_completion_128_runtime.py')"),
        ("'--items', str(code / 'attention_probe/pilot_items.json'),", "'--items', str(code / 'attention_probe/pilot_items.json'),\n               '--baseline', str(inputs / 'completion_48_baseline.json'),"),
        ("'h286_a100_validation_diagnostic.zip'", "'h286_completion_128_diagnostic.zip'"),
        ("'h286_a100_validation_results.zip'", "'h286_completion_128_results.zip'"),
        ('getrennte Antwortvalidierung.', 'vier Originalinputs, Gesamtkappe128.'),
        ('Begrenzte explorative H286-Antwortvalidierung abgeschlossen. Keine allgemeine Wahrheitsvalidierung.',
         'Vier diagnostische EOS-Fortsetzungen abgeschlossen; semantische Prüfung bleibt getrennt.'),
    ):
        code = PRIOR._replace_once(code, before, after)
    ast.parse(code)
    document['cells'][0]['source'] = INTRO.splitlines(keepends=True)
    document['cells'][3]['source'] = code.splitlines(keepends=True)
    document['cells'][4]['source'] = ['Ergebnis-ZIP enthält die Antworten und den gebundenen 48-Präfixvergleich.\n']
    document['metadata']['colab']['name'] = MEMBER
    for i, cell in enumerate(document['cells']):
        cell['id'] = hashlib.sha256(f'h286-completion-128:{i}'.encode()).hexdigest()[:12]
    return document


def build(output=HERE/MEMBER, *, reference_paths=None):
    output = Path(output)
    references = {name: Path(path).read_bytes() for name, path in (reference_paths or {}).items()}
    if references and output.resolve().is_relative_to(ROOT):
        raise ValueError('Private completion artifacts must be written outside the public repository')
    document = notebook_document(source_bundle(), references)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, ensure_ascii=False, indent=1)+'\n')
    if references: output.chmod(0o600)
    return output


def launcher_document(package_sha, notebook_sha, notebook_size):
    document = LAUNCHER.launcher_document(package_sha, notebook_sha, notebook_size)
    code = ''.join(document['cells'][1]['source'])
    for before, after in (('h286_a100_validation.ipynb', MEMBER), ('h286_a100_paket.zip', PACKAGE),
                          ('h286_A100_START_diagnose.zip', 'h286_completion_128_START_diagnose.zip'),
                          ('h286_validation_cell_', 'h286_completion_128_cell_')):
        if before not in code: raise ValueError('Prior launcher contract changed: '+before)
        code = code.replace(before, after)
    ast.parse(code)
    document['cells'][0]['source'] = ['# Vier Antworten bis EOS, maximal 128 Tokens\n\n**A100 → Alle ausführen.** Genau **'+PACKAGE+'** auswählen.\n'
        'Danach laufen nur die vier Originalinputs mit fünf Armen. Das frühere48-Präfix wird automatisch geprüft.\n'
        'Kein Drive. Ergebnis-/Diagnose-ZIP am Ende herunterladen; privates Paket nicht öffentlich hochladen.\n']
    document['cells'][1]['source'] = code.splitlines(keepends=True)
    document['metadata']['colab']['name'] = STARTER
    return document


def package(notebook, directory):
    raw = Path(notebook).read_bytes()
    document = json.loads(raw)
    if len(raw)>2*1024**2 or len([cell for cell in document['cells'] if cell['cell_type']=='code'])!=3:
        raise ValueError('Private notebook exceeds the tested small-loader bounds')
    stream = io.BytesIO()
    info = zipfile.ZipInfo(MEMBER, date_time=(1980,1,1,0,0,0)); info.external_attr = 0o600<<16
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as packed:
        packed.writestr(info,raw,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    payload = stream.getvalue()
    starter = launcher_document(hashlib.sha256(payload).hexdigest(), hashlib.sha256(raw).hexdigest(), len(raw))
    text = json.dumps(starter,ensure_ascii=False,separators=(',',':'))+'\n'
    if len(text.encode())>=8192: raise ValueError('Starter exceeds 8 KiB')
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    target, start = directory/PACKAGE, directory/STARTER
    target.write_bytes(payload);target.chmod(0o600);start.write_text(text)
    return start,target


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE/MEMBER)
    parser.add_argument('--reference-dir',type=Path);parser.add_argument('--refw',type=Path);parser.add_argument('--baseline',type=Path)
    parser.add_argument('--package-dir',type=Path)
    args=parser.parse_args(); paths=None
    if any((args.reference_dir,args.refw,args.baseline)):
        if not all((args.reference_dir,args.refw,args.baseline)): parser.error('All three private reference options are required')
        paths={name:args.reference_dir/name for name in ('hn_set.json','htq_policy.json')}
        paths.update({'refw_h286.json':args.refw,'completion_48_baseline.json':args.baseline})
    notebook=build(args.out,reference_paths=paths);print(notebook)
    if args.package_dir:
        for artifact in package(notebook,args.package_dir):print(artifact)
