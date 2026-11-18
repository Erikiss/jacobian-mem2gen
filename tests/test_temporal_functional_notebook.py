"""Exercise temporal-stage materialization, failures and compact/RAW downloads."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import types
import zipfile
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    assert spec and spec.loader
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


B=load('temporal_functional_test_builder','notebooks/build_temporal_functional.py')
OLD=load('temporal_functional_old_harness','tests/test_b14_b18_notebook.py')
COMPACT=load('temporal_functional_old_compact','tests/test_b14_b18_compact_wrapper.py')


def fixture_document(private=True):
    refs={name:json.dumps({'synthetic':name}).encode() for name in B.REFERENCE_SHA256}
    anchors={name:hashlib.sha256(raw).hexdigest() for name,raw in refs.items()}
    modules={name:'"""Synthetic module; never executed."""\n' for name in B.NEW_SOURCES}
    return B.notebook_document(modules,refs if private else {},expected_references=anchors),refs


def rename(code):
    replacements=(('qwen3_b14_b18_runtime.py','qwen3_temporal_functional_runtime.py'),
        ('original_qwen3_b14_b18_local_transport','original_qwen3_temporal_functional'),
        ('b14_b18_diagnostic.zip','temporal_functional_diagnostic.zip'),
        ('b14_b18_results.zip','temporal_functional_results.zip'),
        ('b14_b18_results_RAW.zip','temporal_functional_results_RAW.zip'),
        ('b14_b18_local_transport_','temporal_functional_'),
        ('b14_b18_items.json','temporal_items.json'),('b14_b18_plan.json','temporal_plan.json'))
    def replace(value):
        if isinstance(value,types.CodeType):
            return rename(value)
        if isinstance(value,tuple):
            return tuple(replace(item) for item in value)
        if isinstance(value,str):
            for before,after in replacements:
                value=value.replace(before,after)
        return value
    return code.replace(co_consts=tuple(replace(value) for value in code.co_consts))


def inherit(function,**overrides):
    bindings=dict(function.__globals__,B=B,fixture_document=fixture_document,**overrides)
    clone=types.FunctionType(rename(function.__code__),bindings,function.__name__,function.__defaults__,function.__closure__)
    clone.__kwdefaults__=dict(function.__kwdefaults__ or {})
    return clone


execute=inherit(OLD.execute)


@pytest.mark.parametrize('state',['complete','paused'])
def test_actual_wrapper_five_inputs_new_runtime_and_fresh_weight_evidence(tmp_path,monkeypatch,state):
    error,status,entries,commands,_=execute(tmp_path,monkeypatch,helper_state=state)
    assert error is None and status['state']==state
    assert status['purpose']=='original_qwen3_temporal_functional'
    assert status['pretrained_weights_loaded'] is True and status['weights_loaded_evidence']=='fresh_child_metrics'
    command=commands[-1];_,refs=fixture_document()
    for flag,name in (('--hn-set','hn_set.json'),('--policy','htq_policy.json'),('--refw','refw_h286.json'),
                      ('--items','temporal_items.json'),('--plan','temporal_plan.json')):
        assert Path(command[command.index(flag)+1]).read_bytes()==refs[name]
    assert '--refine-plan' not in command and '--anchors' not in command
    assert 'WRAPPER_MANIFEST.json' in entries


@pytest.mark.parametrize('kwargs',[{'cuda':False},{'free_gib':11},{'private':False},{'pip_code':1}])
def test_early_failure_downloads_diagnostic_without_starting_scientific_runtime(tmp_path,monkeypatch,kwargs):
    error,status,entries,commands,_=execute(tmp_path,monkeypatch,**kwargs)
    assert error and status['state']=='failed' and status['pretrained_weights_loaded'] is False
    assert len(commands)==(1 if kwargs.get('pip_code') else 0)
    assert 'WRAPPER_MANIFEST.json' in entries


@pytest.mark.parametrize('name',list(B.REFERENCE_SHA256))
def test_all_five_tampered_inputs_stop_before_dependency_install_or_model(tmp_path,monkeypatch,name):
    def corrupt(private=True):
        document,refs=fixture_document(private)
        payload=''.join(document['cells'][2]['source'])
        document['cells'][2]['source']=[payload+f'\nREFERENCE_FILES_B64[{name!r}]= "dGFtcGVyZWQ="\n']
        return document,refs
    monkeypatch.setitem(execute.__globals__,'fixture_document',corrupt)
    error,status,_,commands,_=execute(tmp_path,monkeypatch)
    assert error and status['state']=='failed' and not commands


def test_failed_child_and_stale_metrics_cannot_claim_success_or_fresh_weights(tmp_path,monkeypatch):
    execute(tmp_path,monkeypatch)
    error,status,entries,_,_=execute(tmp_path,monkeypatch,helper_code=1,metrics_mode='keep')
    assert error and status['state']=='failed' and status['wrapper_verified'] is False
    assert status['pretrained_weights_loaded'] is None
    assert json.loads(entries['helper_status.json'])['state']=='complete'


@pytest.mark.parametrize('state',['running','failed'])
def test_zero_exit_with_nonterminal_or_failed_child_status_is_still_diagnostic(tmp_path,monkeypatch,state):
    error,status,entries,_,_=execute(tmp_path,monkeypatch,helper_state=state)
    assert error and status['state']=='failed' and status['wrapper_verified'] is False
    assert json.loads(entries['helper_status.json'])['state']==state


def test_paused_status_is_explicit_and_large_child_output_stays_in_log(tmp_path,monkeypatch):
    error,status,entries,_,console=execute(tmp_path,monkeypatch,helper_state='paused',flood=True)
    assert error is None and status['state']=='paused' and 'kein PASS' in console
    assert len(console)<15000 and 'library noise' not in console
    assert entries['runtime.log'].count(b'library noise')==5000


def test_new_directory_is_repeatable_and_preserves_previous_experiment(tmp_path,monkeypatch):
    original=(B.PRIOR.run_source(),B.PRIOR.INTRO,B.PRIOR.MEMBER,B.PRIOR.PACKAGE)
    prior=tmp_path/'b14_b18_refinement_keep';prior.mkdir();(prior/'original').write_text('old')
    execute(tmp_path,monkeypatch,default_directory=True)
    folders=[path for path in tmp_path.glob('temporal_functional_*') if path.is_dir()]
    assert len(folders)==1
    execute(tmp_path,monkeypatch,default_directory=True)
    assert [path for path in tmp_path.glob('temporal_functional_*') if path.is_dir()]==folders
    assert (prior/'original').read_text()=='old'
    document,_=fixture_document(private=False)
    assert len([cell for cell in document['cells'] if cell['cell_type']=='code'])==3
    assert len({cell['id'] for cell in document['cells']})==len(document['cells'])
    assert (B.PRIOR.run_source(),B.PRIOR.INTRO,B.PRIOR.MEMBER,B.PRIOR.PACKAGE)==original


def test_exact_private_notebook_package_and_small_hash_only_starter(tmp_path):
    document,refs=fixture_document();raw=(json.dumps(document)+'\n').encode()
    notebook=tmp_path/B.MEMBER;notebook.write_bytes(raw)
    starter,package=B.package(notebook,tmp_path,expected_notebook_sha256=hashlib.sha256(raw).hexdigest())
    with zipfile.ZipFile(package) as archive:
        assert archive.namelist()==[B.MEMBER] and archive.testzip() is None and archive.read(B.MEMBER)==raw
    launch=json.loads(starter.read_bytes());source=''.join(launch['cells'][1]['source']);ast.parse(source)
    assert hashlib.sha256(package.read_bytes()).hexdigest() in source and hashlib.sha256(raw).hexdigest() in source
    assert B.PACKAGE in source and B.MEMBER in source and starter.stat().st_size<8192
    assert all(value.decode() not in starter.read_text() for value in refs.values())


def test_private_repo_destinations_and_changed_notebook_are_rejected(tmp_path):
    assert inspect.signature(B.build).parameters['output'].default==ROOT/'notebooks'/B.MEMBER
    with pytest.raises(ValueError,match='outside'):
        B.build(ROOT/'notebooks'/B.MEMBER,reference_paths={'test':Path(__file__)})
    notebook=tmp_path/B.MEMBER;notebook.write_text('{}')
    with pytest.raises(ValueError,match='reviewed'):
        B.package(notebook,tmp_path,expected_notebook_sha256='0'*64)


def test_actual_compact_export_and_preserved_raw_checkpoint(tmp_path,monkeypatch):
    run=inherit(COMPACT.test_actual_wrapper_downloads_lossless_compact_and_original_raw_archives,_builder=lambda:B)
    run(tmp_path,monkeypatch)


def test_public_template_preserves_original_sources_and_contains_no_private_bytes(tmp_path):
    sources=B.source_bundle();original=B.BASE.source_bundle()
    assert all(sources[name]==value for name,value in original.items() if name!='attention_probe/pilot_items.json')
    assert 'attention_probe/pilot_items.json' not in sources
    assert len(sources)==11
    assert 'qwen3_b14_b18_refine_runtime.py' not in sources
    for name,path in B.NEW_SOURCES.items():
        assert sources[name]==path.read_text()
    path=B.build(tmp_path/'public.ipynb');document=json.loads(path.read_bytes())
    assert document==json.loads((ROOT/'notebooks'/B.MEMBER).read_bytes())
    namespace={}
    exec(''.join(document['cells'][2]['source']),namespace)  # noqa: S102 — authored payload under test
    assert namespace['REFERENCE_FILES_B64']=={} and namespace['MODULE_SOURCES']==sources
