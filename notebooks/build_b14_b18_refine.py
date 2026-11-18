"""Package the four-case B14→B18 convergence/damping follow-up separately.

The exercised prior wrapper, lossless compact exporter and small mobile loader
are reused. Prior scientific sources and private references remain unchanged.
Private notebooks/packages must be written outside the public repository.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.util
import types
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MEMBER = 'b14_b18_refinement.ipynb'
PACKAGE = 'b14_b18_refine_paket.zip'
STARTER = 'b14_b18_refine_START.ipynb'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRIOR = load('b14_b18_refine_prior_builder',HERE/'build_b14_b18.py')
VALIDATION = PRIOR.VALIDATION
TRANSFORMERS_VERSION = PRIOR.TRANSFORMERS_VERSION
ACCELERATE_VERSION = PRIOR.ACCELERATE_VERSION
# Bound after review of the new fixed private protocol and prior-result anchors.
REFINE_PLAN_SHA256 = 'ff4cd84d3ff1e01e87074d5c2938d79224c83634fa441a095a70c7d1d9b78afb'
ANCHORS_SHA256 = 'd56873915a388bcf9c42e606a64f512d02098bf1747e45306196938e9284b2c7'
REFERENCE_SHA256 = dict(PRIOR.REFERENCE_SHA256,**{
    'b14_b18_refine_plan.json':REFINE_PLAN_SHA256,
    'b14_b18_refine_anchors.json':ANCHORS_SHA256,
})
NEW_SOURCES = {
    'qwen3_b14_b18_refine_runtime.py':HERE/'qwen3_b14_b18_refine_runtime.py',
    'attention_probe/transport_refinement.py':ROOT/'attention_probe/transport_refinement.py',
}
INTRO = """# Block 14 → Block 18: Konvergenz und Dämpfung

**A100 → Alle ausführen.** Im kleinen Starter das bereitgestellte
**b14_b18_refine_paket.zip** auswählen. Kein Drive und kein weiterer Upload.

Vier festgelegte Fälle des vorigen Laufs werden mit genauer begrenzter
Konvergenz- und Dämpfungsprüfung untersucht. Originales Qwen3-1.7B, FP32,
ausgeschaltetes TF32, dieselben 286 Koordinaten, REF-W-Werte und Quellprompts
bleiben erhalten. Die alten Resultate werden geprüft und unverändert bewahrt.
Es gibt keine neuen Fragen, freien Antworten oder Anpassung der Gewichte.

Der Versuch misst lokalen, promptgebundenen Transport. Ein ausgeführter Solver
bedeutet keine bestätigte Konvergenz oder gelungene Rekonstruktion; diese Angaben
werden getrennt gespeichert. Es ist keine globale Jacobian-Lens und kein
Gedächtnisnachweis.

GPU-Torch bleibt installiert; Modellgewichte im vorhandenen Cache werden
wiederverwendet. Das Rechenbudget beträgt 20 Minuten mit sicheren Checkpoints;
Installation und Modellladen kommen hinzu. Das ist keine Laufzeitgarantie.
Ein Status `paused` bedeutet einen fortsetzbaren Teilstand.

Am Ende **b14_b18_refine_results.zip** als kompakte Auswertung herunterladen
und hier bereitstellen. **b14_b18_refine_results_RAW.zip** zusätzlich sichern;
nur die Rohdatei enthält vollständige Logits und dient zur Wiederaufnahme.
Bei Fehlern wird ein Diagnose-ZIP angeboten. Das private Paket nicht öffentlich
hochladen. Die GPU-Sitzung danach beenden, wenn sie nicht mehr gebraucht wird.
"""


def source_bundle():
    sources = PRIOR.source_bundle()
    sources.update({name:path.read_text() for name,path in NEW_SOURCES.items()})
    return sources


def run_source():
    source = PRIOR.run_source()
    for before,after in (
        ("'b14_b18_local_transport_'","'b14_b18_refinement_'"),
        ("'original_qwen3_b14_b18_local_transport'","'original_qwen3_b14_b18_refinement'"),
        ("str(code / 'qwen3_b14_b18_runtime.py')","str(code / 'qwen3_b14_b18_refine_runtime.py')"),
        ("'--plan', str(inputs / 'b14_b18_plan.json'),",
         "'--plan', str(inputs / 'b14_b18_plan.json'),\n               '--refine-plan', str(inputs / 'b14_b18_refine_plan.json'),\n               '--anchors', str(inputs / 'b14_b18_refine_anchors.json'),"),
        ('B14→B18: 16 Aufgaben, lokale Jacobians und vier Rücktransporte.',
         'B14→B18: vier feste Fälle, Konvergenz und Dämpfung.'),
        ("'b14_b18_diagnostic.zip'","'b14_b18_refine_diagnostic.zip'"),
        ("'b14_b18_results.zip'","'b14_b18_refine_results.zip'"),
        ('B14→B18-Versuch abgeschlossen. Ergebnisse für die getrennte wissenschaftliche Auswertung gespeichert.',
         'B14→B18-Folgelauf abgeschlossen. Konvergenz und Rekonstruktion getrennt auswerten.'),
    ):
        source = VALIDATION._replace_once(source,before,after)
    ast.parse(source)
    return source


def notebook_document(modules,reference_files=None,*,expected_references=None):
    expected = REFERENCE_SHA256 if expected_references is None else expected_references
    if any(not isinstance(value,str) or len(value)!=64 for value in expected.values()):
        raise ValueError('Bind and review all seven reference SHA256 values before building')
    document = PRIOR.notebook_document(modules,reference_files,expected_references=expected)
    # The byte-bound prior-result anchor is about 1 MiB. Compress its transport
    # representation only; restore the identical bytes before the exercised
    # wrapper validates hashes or writes inputs. The small loader's 2 MiB bound
    # and every scientific/reference file remain unchanged.
    anchor_name = 'b14_b18_refine_anchors.json'
    if reference_files and anchor_name in reference_files:
        source = ''.join(document['cells'][2]['source'])
        raw = reference_files[anchor_name]
        literal = repr(base64.b64encode(raw).decode())
        compressed = repr(base64.b64encode(zlib.compress(raw,9)).decode())
        source = VALIDATION._replace_once(source,literal,compressed)
        source += (
            '\n# Restore original anchor bytes before the normal SHA checks.\n'
            'import base64 as _anchor_b64\nimport zlib as _anchor_zlib\n'
            f'REFERENCE_FILES_B64[{anchor_name!r}] = _anchor_b64.b64encode(_anchor_zlib.decompress('
            f'_anchor_b64.b64decode(REFERENCE_FILES_B64[{anchor_name!r}], validate=True))).decode()\n')
        ast.parse(source)
        document['cells'][2]['source'] = source.splitlines(keepends=True)
    document['cells'][0]['source'] = INTRO.splitlines(keepends=True)
    document['cells'][3]['source'] = run_source().splitlines(keepends=True)
    document['cells'][4]['source'] = [
        'Kompakte Auswertung und vollständige Rohdaten sichern; Solver-Konvergenz, Dämpfung und Rekonstruktionsfehler getrennt beurteilen.\n']
    document['metadata']['colab']['name'] = MEMBER
    for index,cell in enumerate(document['cells']):
        cell['id'] = hashlib.sha256(f'b14-b18-refine:{index}'.encode()).hexdigest()[:12]
    return document


def launcher_document(package_sha,notebook_sha,notebook_size):
    document = PRIOR.launcher_document(package_sha,notebook_sha,notebook_size)
    source = ''.join(document['cells'][1]['source'])
    for before,after in ((PRIOR.MEMBER,MEMBER),(PRIOR.PACKAGE,PACKAGE),
                         ('b14_b18_START_diagnose.zip','b14_b18_refine_START_diagnose.zip'),
                         ('b14_b18_cell_','b14_b18_refine_cell_')):
        if before not in source:
            raise ValueError('Prior mobile-loader anchor changed: '+before)
        source = source.replace(before,after)
    ast.parse(source)
    document['cells'][0]['source'] = [
        ('# Block 14 → Block 18: vier Fälle\n\n**A100 → Alle ausführen.** Bei der Dateiauswahl '
        '**b14_b18_refine_paket.zip** auswählen. Danach laufen Konvergenz- und Dämpfungsprüfungen '
        'automatisch. Kein Drive nötig.\n\nAm Ende **b14_b18_refine_results.zip** für die Auswertung '
        'herunterladen. **b14_b18_refine_results_RAW.zip** ebenfalls sichern; nur RAW ist fortsetzbar. '
        'Bei Fehlern das Diagnose-ZIP verwenden. Das private Paket nicht öffentlich hochladen. '
        'In Jupyter die Paket-ZIP neben diesen Starter legen.\n')]
    document['cells'][1]['source'] = source.splitlines(keepends=True)
    document['cells'][0]['id'] = 'b14-b18-refine-start-intro'
    document['cells'][1]['id'] = 'b14-b18-refine-start-code'
    document['metadata']['colab']['name'] = STARTER
    return document


def inherit(function):
    # Fresh globals preserve every dependency of the exercised function while
    # changing only this package's explicit destinations/renderers. No global
    # monkeypatch can leak into a prior builder or modify an old experiment.
    bindings = dict(function.__globals__)
    bindings.update(HERE=HERE,ROOT=ROOT,MEMBER=MEMBER,PACKAGE=PACKAGE,STARTER=STARTER,
                    notebook_document=notebook_document,source_bundle=source_bundle,
                    launcher_document=launcher_document)
    if 'package_bytes' in globals():
        bindings['package_bytes'] = package_bytes
    result = types.FunctionType(function.__code__,bindings,function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return result


package_bytes = inherit(PRIOR.package_bytes)
package = inherit(PRIOR.package)
_build = inherit(PRIOR._build)


def build(output=HERE/MEMBER,*,reference_paths=None):
    return _build(output,reference_paths=reference_paths)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE/MEMBER)
    parser.add_argument('--inputs-dir',type=Path)
    parser.add_argument('--package-dir',type=Path)
    args = parser.parse_args()
    references = {name:args.inputs_dir/name for name in REFERENCE_SHA256} if args.inputs_dir else None
    notebook = build(args.out,reference_paths=references)
    print(notebook)
    if args.package_dir:
        for path in package(notebook,args.package_dir,expected_notebook_sha256=hashlib.sha256(notebook.read_bytes()).hexdigest()):
            print(path)
