"""Exercise the new four-case wrapper/loader with model and pip stubbed."""
from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import inspect
import json
import types
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name,path):
    spec = importlib.util.spec_from_file_location(name,ROOT/path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


B = load('refine_notebook_builder','notebooks/build_b14_b18_refine.py')
OLD = load('refine_old_wrapper_harness','tests/test_b14_b18_notebook.py')
COMPACT = load('refine_old_compact_harness','tests/test_b14_b18_compact_wrapper.py')


def fixture_document(private=True):
    refs={name:json.dumps({'synthetic':name}).encode() for name in B.REFERENCE_SHA256}
    anchors={name:hashlib.sha256(raw).hexdigest() for name,raw in refs.items()}
    modules={name:'"""Synthetic module; never executed."""\n' for name in B.NEW_SOURCES}
    return B.notebook_document(modules,refs if private else {},expected_references=anchors),refs


def rename(code):
    replacements=(('qwen3_b14_b18_runtime.py','qwen3_b14_b18_refine_runtime.py'),
                  ('original_qwen3_b14_b18_local_transport','original_qwen3_b14_b18_refinement'),
                  ('b14_b18_diagnostic.zip','b14_b18_refine_diagnostic.zip'),
                  ('b14_b18_results.zip','b14_b18_refine_results.zip'),
                  ('b14_b18_results_RAW.zip','b14_b18_refine_results_RAW.zip'),
                  ('b14_b18_local_transport_','b14_b18_refinement_'))
    constants=[]
    for value in code.co_consts:
        if isinstance(value,types.CodeType):
            value=rename(value)
        elif isinstance(value,str):
            for before,after in replacements:
                value=value.replace(before,after)
        constants.append(value)
    return code.replace(co_consts=tuple(constants))


def inherit(function,**overrides):
    bindings=dict(function.__globals__,B=B,fixture_document=fixture_document,**overrides)
    clone=types.FunctionType(rename(function.__code__),bindings,function.__name__,function.__defaults__,function.__closure__)
    clone.__kwdefaults__=dict(function.__kwdefaults__ or {})
    return clone


execute=inherit(OLD.execute)


@pytest.mark.parametrize('state',['complete','paused'])
def test_actual_wrapper_seven_references_fresh_runtime_and_result_archive(tmp_path,monkeypatch,state):
    error,status,entries,commands,_=execute(tmp_path,monkeypatch,helper_state=state)
    assert error is None and status['state']==state
    assert status['purpose']=='original_qwen3_b14_b18_refinement'
    assert status['pretrained_weights_loaded'] is True and status['weights_loaded_evidence']=='fresh_child_metrics'
    command=commands[-1]; _,refs=fixture_document()
    for flag,name in (('--hn-set','hn_set.json'),('--policy','htq_policy.json'),('--refw','refw_h286.json'),
                      ('--items','b14_b18_items.json'),('--plan','b14_b18_plan.json'),
                      ('--refine-plan','b14_b18_refine_plan.json'),('--anchors','b14_b18_refine_anchors.json')):
        assert Path(command[command.index(flag)+1]).read_bytes()==refs[name]
    assert 'WRAPPER_MANIFEST.json' in entries


@pytest.mark.parametrize('kwargs',[{'cuda':False},{'free_gib':11},{'private':False},{'pip_code':1}])
def test_early_failure_materializes_diagnostic_without_starting_science(tmp_path,monkeypatch,kwargs):
    error,status,entries,commands,_=execute(tmp_path,monkeypatch,**kwargs)
    assert error and status['state']=='failed' and status['pretrained_weights_loaded'] is False
    assert len(commands)==(1 if kwargs.get('pip_code') else 0)
    assert 'WRAPPER_MANIFEST.json' in entries


@pytest.mark.parametrize('name',list(B.REFERENCE_SHA256))
def test_every_tampered_private_input_fails_before_pip_or_model(tmp_path,monkeypatch,name):
    def corrupt(private=True):
        document,refs=fixture_document(private)
        source=''.join(document['cells'][2]['source'])
        document['cells'][2]['source']=[source+f'\nREFERENCE_FILES_B64[{name!r}] = "dGFtcGVyZWQ="\n']
        return document,refs
    monkeypatch.setitem(execute.__globals__,'fixture_document',corrupt)
    error,status,_,commands,_=execute(tmp_path,monkeypatch)
    assert error and status['state']=='failed' and not commands


def test_failed_child_cannot_claim_complete_or_reuse_stale_weight_evidence(tmp_path,monkeypatch):
    execute(tmp_path,monkeypatch)
    error,status,entries,_,_=execute(tmp_path,monkeypatch,helper_code=1,metrics_mode='keep')
    assert error and status['state']=='failed' and status['wrapper_verified'] is False
    assert status['pretrained_weights_loaded'] is None
    assert json.loads(entries['helper_status.json'])['state']=='complete'


def test_paused_output_is_bounded_and_explicitly_incomplete(tmp_path,monkeypatch):
    error,status,entries,_,console=execute(tmp_path,monkeypatch,helper_state='paused',flood=True)
    assert error is None and status['state']=='paused' and 'kein PASS' in console
    assert len(console)<15000 and 'library noise' not in console
    assert entries['runtime.log'].count(b'library noise')==5000


def test_new_folder_and_renderers_do_not_mutate_old_experiment(tmp_path,monkeypatch):
    prior=(B.PRIOR.run_source(),B.PRIOR.INTRO,B.PRIOR.MEMBER,B.PRIOR.PACKAGE)
    old=tmp_path/'b14_b18_local_transport_keep'; old.mkdir(); (old/'preserve').write_text('old')
    execute(tmp_path,monkeypatch,default_directory=True)
    folders=[path for path in tmp_path.glob('b14_b18_refinement_*') if path.is_dir()]
    assert len(folders)==1
    execute(tmp_path,monkeypatch,default_directory=True)
    assert [path for path in tmp_path.glob('b14_b18_refinement_*') if path.is_dir()]==folders
    assert (old/'preserve').read_text()=='old'
    document,_=fixture_document(private=False)
    assert len([cell for cell in document['cells'] if cell['cell_type']=='code'])==3
    assert len({cell['id'] for cell in document['cells']})==len(document['cells'])
    assert (B.PRIOR.run_source(),B.PRIOR.INTRO,B.PRIOR.MEMBER,B.PRIOR.PACKAGE)==prior


def test_private_package_exact_bytes_and_hash_only_mobile_starter(tmp_path):
    document,refs=fixture_document(); raw=(json.dumps(document)+'\n').encode()
    notebook=tmp_path/B.MEMBER; notebook.write_bytes(raw)
    starter,package=B.package(notebook,tmp_path,expected_notebook_sha256=hashlib.sha256(raw).hexdigest())
    with zipfile.ZipFile(package) as archive:
        assert archive.namelist()==[B.MEMBER] and archive.testzip() is None
        assert archive.read(B.MEMBER)==raw
    launch=json.loads(starter.read_bytes()); source=''.join(launch['cells'][1]['source']); ast.parse(source)
    assert hashlib.sha256(package.read_bytes()).hexdigest() in source
    assert hashlib.sha256(raw).hexdigest() in source
    assert B.PACKAGE in source and B.MEMBER in source
    assert starter.stat().st_size<8192
    assert all(payload.decode() not in starter.read_text() for payload in refs.values())


def test_large_anchor_transport_is_compressed_but_materializes_identical_original_bytes():
    refs={name:json.dumps({'synthetic':name}).encode() for name in B.REFERENCE_SHA256}
    refs['b14_b18_refine_anchors.json']=json.dumps([
        {'sha256':hashlib.sha256(str(i).encode()).hexdigest(),'shape':[1,1,286],'source':'frozen prior result'}
        for i in range(9000)]).encode()
    anchors={name:hashlib.sha256(raw).hexdigest() for name,raw in refs.items()}
    document=B.notebook_document({},refs,expected_references=anchors)
    source=''.join(document['cells'][2]['source'])
    assert base64.b64encode(refs['b14_b18_refine_anchors.json']).decode() not in source
    namespace={}
    exec(source,namespace)  # noqa: S102 — authored transport decoder under test
    assert all(base64.b64decode(namespace['REFERENCE_FILES_B64'][name])==raw for name,raw in refs.items())
    assert len(json.dumps(document).encode())<2*1024**2


def test_private_destination_and_changed_reviewed_notebook_rejected(tmp_path):
    assert inspect.signature(B.build).parameters['output'].default==ROOT/'notebooks'/B.MEMBER
    with pytest.raises(ValueError,match='outside'):
        B.build(ROOT/'notebooks'/B.MEMBER,reference_paths={'test':Path(__file__)})
    notebook=tmp_path/B.MEMBER; notebook.write_text('{}')
    with pytest.raises(ValueError,match='reviewed'):
        B.package(notebook,tmp_path,expected_notebook_sha256='0'*64)


def test_real_compact_exporter_with_new_wrapper_names(tmp_path,monkeypatch):
    run=inherit(COMPACT.test_actual_wrapper_downloads_lossless_compact_and_original_raw_archives,_builder=lambda:B)
    run(tmp_path,monkeypatch)


def test_public_sources_preserve_all_prior_bytes_and_template_is_reproducible(tmp_path):
    sources=B.source_bundle(); old=B.PRIOR.source_bundle()
    assert len(sources)==len(old)+2 and all(sources[name]==value for name,value in old.items())
    for name,path in B.NEW_SOURCES.items():
        assert sources[name]==path.read_text()
    path=B.build(tmp_path/'public.ipynb')
    document=json.loads(path.read_bytes())
    assert document==json.loads((ROOT/'notebooks'/B.MEMBER).read_bytes())
    namespace={}
    exec(''.join(document['cells'][2]['source']),namespace)  # noqa: S102 — authored payload under test
    assert namespace['REFERENCE_FILES_B64']=={} and namespace['MODULE_SOURCES']==sources
