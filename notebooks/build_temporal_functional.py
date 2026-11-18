"""Package the separate temporal/functional pilot and a small mobile starter.

Only new files are written. The exercised wrapper/loader and compact exporter
are reused, while original model/reference modules remain byte-identical.
Private reference/input bytes are restricted to artifacts outside this repo.
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
MEMBER = 'temporal_functional.ipynb'
PACKAGE = 'temporal_functional_paket.zip'
STARTER = 'temporal_functional_START.ipynb'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    assert spec and spec.loader
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


PRIOR=load('temporal_functional_prior_builder',HERE/'build_b14_b18_refine.py')
VALIDATION=PRIOR.VALIDATION
BASE=VALIDATION.BASE
TRANSFORMERS_VERSION=PRIOR.TRANSFORMERS_VERSION
ACCELERATE_VERSION=PRIOR.ACCELERATE_VERSION
ITEMS_SHA256='997f59b8ab3fae7190a609958def98227bcad2546666e1cc66bcdf89778e225e'
PLAN_SHA256='af42159b61e639da19cdb99c140e048cd5e88ef2c82bd467b5217f88dfb40b62'
REFERENCE_SHA256=dict(BASE.REFERENCE_SHA256,**{
    'temporal_items.json':ITEMS_SHA256,'temporal_plan.json':PLAN_SHA256,
})
NEW_SOURCES={
    'attention_probe/result_bundle.py':ROOT/'attention_probe/result_bundle.py',
    'attention_probe/answer_validation.py':ROOT/'attention_probe/answer_validation.py',
    'attention_probe/temporal_cache.py':ROOT/'attention_probe/temporal_cache.py',
    'attention_probe/temporal_validation.py':ROOT/'attention_probe/temporal_validation.py',
    'qwen3_temporal_functional_runtime.py':HERE/'qwen3_temporal_functional_runtime.py',
}
CALIBRATION_SOURCE='attention_probe/temporal_candidate.py'
INTRO="""# Zeitliche Ausbreitung und funktionale Kontrollen

**A100 → Alle ausführen.** Im kleinen Starter das bereitgestellte
**temporal_functional_paket.zip** auswählen. Kein Drive und kein weiterer Upload.

16 festgelegte Eingaben werden in acht Kalibrierungs- und acht Testfälle getrennt.
Eine gemeinsame Richtung wird auf der Kalibrierung festgelegt und vor den
Testmessungen eingefroren. Identische Fortsetzungen und native Cache-Kontrollen
trennen zeitliche Zustandsänderungen von abweichenden Eingabetokens.
Funktionale Endpunkte werden getrennt von der reinen Geometrie gespeichert.

Originales Qwen3-1.7B, FP32 mit ausgeschaltetem TF32, dieselben 286 Koordinaten
und dieselbe REF-W-Normierung bleiben erhalten. Modellgewichte werden nicht
trainiert. Der Versuch ist explorativ und belegt für sich kein Gedächtnis und
keinen allgemeinen Selbstüberwachungsmechanismus.

GPU-Torch bleibt installiert; vorhandene Modellgewichte im Cache werden
wiederverwendet. Das Rechenbudget beträgt 20 Minuten mit Checkpoints;
Installation und Modellladen kommen hinzu. Das ist keine Laufzeitgarantie.
Ein Status `paused` bedeutet einen fortsetzbaren Teilstand.

Am Ende **temporal_functional_results.zip** für die Auswertung herunterladen.
**temporal_functional_results_RAW.zip** ebenfalls sichern: nur die Rohdatei ist
zur Wiederaufnahme geeignet. Bei Fehlern wird ein Diagnose-ZIP angeboten.
Das private Paket nicht öffentlich hochladen. Die GPU-Sitzung danach beenden,
wenn sie nicht mehr gebraucht wird.
"""


def source_bundle():
    if not CALIBRATION_SOURCE:
        raise ValueError('Bind the reviewed shared-calibration module before building')
    sources=BASE.source_bundle()
    # This stage uses its own private input file, not the old color-demo items.
    del sources['attention_probe/pilot_items.json']
    sources.update({name:path.read_text() for name,path in NEW_SOURCES.items()})
    sources[CALIBRATION_SOURCE]=(ROOT/CALIBRATION_SOURCE).read_text()
    return sources


def run_source():
    source=PRIOR.run_source()
    for before,after in (
        ("'b14_b18_refinement_'","'temporal_functional_'"),
        ("'original_qwen3_b14_b18_refinement'","'original_qwen3_temporal_functional'"),
        ("str(code / 'qwen3_b14_b18_refine_runtime.py')","str(code / 'qwen3_temporal_functional_runtime.py')"),
        ("'--items', str(inputs / 'b14_b18_items.json'),\n               '--plan', str(inputs / 'b14_b18_plan.json'),\n               '--refine-plan', str(inputs / 'b14_b18_refine_plan.json'),\n               '--anchors', str(inputs / 'b14_b18_refine_anchors.json'),",
         "'--items', str(inputs / 'temporal_items.json'),\n               '--plan', str(inputs / 'temporal_plan.json'),"),
        ('B14→B18: vier feste Fälle, Konvergenz und Dämpfung.',
         'Zeitliche und funktionale Stufe: feste Kalibrierung, Testeingaben und Cache-Kontrollen.'),
        ("'b14_b18_refine_diagnostic.zip'","'temporal_functional_diagnostic.zip'"),
        ("'b14_b18_refine_results.zip'","'temporal_functional_results.zip'"),
        ('B14→B18-Folgelauf abgeschlossen. Konvergenz und Rekonstruktion getrennt auswerten.',
         'Zeitliche und funktionale Stufe abgeschlossen. Messungen und Verhalten getrennt auswerten.'),
    ):
        source=VALIDATION._replace_once(source,before,after)
    ast.parse(source)
    return source


def notebook_document(modules,reference_files=None,*,expected_references=None):
    expected=REFERENCE_SHA256 if expected_references is None else expected_references
    if any(not isinstance(value,str) or len(value)!=64 for value in expected.values()):
        raise ValueError('Bind and review all five reference/input SHA256 values before building')
    document=VALIDATION.notebook_document(modules,reference_files,expected_references=expected)
    document['cells'][0]['source']=INTRO.splitlines(keepends=True)
    document['cells'][3]['source']=run_source().splitlines(keepends=True)
    document['cells'][4]['source']=[
        'Kompakte Auswertung und vollständige Rohdaten sichern. Kalibrierung, Testfälle, Cache-Kontrollen und funktionale Endpunkte getrennt beurteilen.\n']
    document['metadata']['colab']['name']=MEMBER
    for index,cell in enumerate(document['cells']):
        cell['id']=hashlib.sha256(f'temporal-functional:{index}'.encode()).hexdigest()[:12]
    return document


def launcher_document(package_sha,notebook_sha,notebook_size):
    document=PRIOR.launcher_document(package_sha,notebook_sha,notebook_size)
    source=''.join(document['cells'][1]['source'])
    for before,after in ((PRIOR.MEMBER,MEMBER),(PRIOR.PACKAGE,PACKAGE),
                         ('b14_b18_refine_START_diagnose.zip','temporal_functional_START_diagnose.zip'),
                         ('b14_b18_refine_cell_','temporal_functional_cell_')):
        if before not in source:
            raise ValueError('Prior mobile-loader anchor changed: '+before)
        source=source.replace(before,after)
    ast.parse(source)
    document['cells'][0]['source']=[(
        '# Zeitliche und funktionale Stufe\n\n**A100 → Alle ausführen.** Bei der Dateiauswahl '
        '**temporal_functional_paket.zip** auswählen. Danach läuft die festgelegte Stufe automatisch. '
        'Kein Drive nötig.\n\nAm Ende **temporal_functional_results.zip** zur Auswertung herunterladen. '
        '**temporal_functional_results_RAW.zip** ebenfalls sichern; nur RAW ist fortsetzbar. '
        'Bei Fehlern das Diagnose-ZIP verwenden. Das private Paket nicht öffentlich hochladen. '
        'In Jupyter die Paket-ZIP neben diesen Starter legen.\n')]
    document['cells'][1]['source']=source.splitlines(keepends=True)
    document['cells'][0]['id']='temporal-functional-start-intro'
    document['cells'][1]['id']='temporal-functional-start-code'
    document['metadata']['colab']['name']=STARTER
    return document


def inherit(function):
    bindings=dict(function.__globals__)
    bindings.update(HERE=HERE,ROOT=ROOT,MEMBER=MEMBER,PACKAGE=PACKAGE,STARTER=STARTER,
                    notebook_document=notebook_document,source_bundle=source_bundle,
                    launcher_document=launcher_document)
    if 'package_bytes' in globals():
        bindings['package_bytes']=package_bytes
    clone=types.FunctionType(function.__code__,bindings,function.__name__,function.__defaults__,function.__closure__)
    clone.__kwdefaults__=dict(function.__kwdefaults__ or {})
    return clone


package_bytes=inherit(PRIOR.package_bytes)
package=inherit(PRIOR.package)
_build=inherit(PRIOR._build)


def build(output=HERE/MEMBER,*,reference_paths=None):
    return _build(output,reference_paths=reference_paths)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE/MEMBER)
    parser.add_argument('--inputs-dir',type=Path)
    parser.add_argument('--package-dir',type=Path)
    args=parser.parse_args()
    references={name:args.inputs_dir/name for name in REFERENCE_SHA256} if args.inputs_dir else None
    notebook=build(args.out,reference_paths=references)
    print(notebook)
    if args.package_dir:
        for path in package(notebook,args.package_dir,expected_notebook_sha256=hashlib.sha256(notebook.read_bytes()).hexdigest()):
            print(path)
