"""Four-input cap extension, private prior-prefix binding and small package tests."""
# ruff: noqa: S102 -- Execute only authored notebook payloads, never uploaded code.
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'notebooks'))
R = importlib.import_module('qwen3_h286_completion_128_runtime')
S = importlib.util.spec_from_file_location('completion_builder', ROOT/'notebooks/build_h286_completion_128.py')
B = importlib.util.module_from_spec(S); S.loader.exec_module(B)


def baseline_generation(*, eos=False):
    tokens = [7]*9+[0] if eos else [7]*48
    steps = [{'step': i, 'selected_token_id': token, 'selected_token_logprob': -1.,
              'top_tokens': [{'token_id': token, 'logprob': -1.}]} for i, token in enumerate(tokens)]
    return {'arm': {'name': 'synthetic'}, 'seed': 20261129, 'selected_token_ids': tokens,
            'stop_reason': 'eos' if eos else 'max_new_tokens', 'steps': steps,
            'first_step_logits_sha256': 'a'*64, 'eos_token_ids': [0], 'eos_source': 'generation_config',
            'requested_delta_norms': [[[1.]]], 'actual_delta_norms': [[[1.]]]}


def prior_record(eos=False):
    generation = baseline_generation(eos=eos)
    arrays = {'gen_raw_z': np.ones((1, 4, 2), np.float32)}
    return {'generation': generation, 'arm': generation['arm'],
            'gen_array_sha256': {k: hashlib.sha256(v.tobytes(order='C')).hexdigest() for k,v in arrays.items()},
            'gen_array_shapes': {k: list(v.shape) for k,v in arrays.items()}}, arrays


@pytest.mark.parametrize('old_eos', [False, True])
def test_exact_selected_prefix_includes_old_early_eos(old_eos):
    old, arrays = prior_record(old_eos)
    new = copy.deepcopy(old['generation'])
    if not old_eos:
        new['selected_token_ids'] += [8]*79+[0]
        new['stop_reason'] = 'eos'
        new['steps'] += [{'step': i, 'selected_token_id': token, 'selected_token_logprob': -1.,
                          'top_tokens': [{'token_id': token, 'logprob': -1.}]}
                         for i,token in enumerate(new['selected_token_ids'][48:],48)]
    assert R.compare_prefix(new, old, arrays)['compatible'] is True
    new['selected_token_ids'][2] = 9
    assert R.compare_prefix(new, old, arrays)['compatible'] is False


def test_old_eos_cannot_be_forced_into_longer_text():
    old, arrays = prior_record(True)
    new = copy.deepcopy(old['generation'])
    new['selected_token_ids'].append(8)
    assert R.compare_prefix(new, old, arrays)['compatible'] is False


def test_eos_configuration_is_protocol_identity_even_for_same_token_prefix():
    old, arrays = prior_record()
    new = copy.deepcopy(old['generation'])
    new['eos_token_ids'] = [0, 7]
    result = R.compare_prefix(new, old, arrays)
    assert result['tokens_equal'] is True and result['eos_protocol_equal'] is False
    assert result['compatible'] is False


def test_numerical_differences_are_visible_and_not_silent_token_failures():
    old, arrays = prior_record()
    new = copy.deepcopy(old['generation'])
    new['first_step_logits_sha256'] = 'b'*64
    new['steps'][0]['selected_token_logprob'] = -1.001
    changed = {'gen_raw_z': arrays['gen_raw_z']*2}
    result = R.compare_prefix(new, old, changed)
    assert result['compatible'] is True
    assert result['first_logits_exact'] is result['prefix_steps_logprobs_topk_exact'] is result['prefill_arrays_exact'] is False
    assert result['max_abs_selected_logprob_difference'] == pytest.approx(.001)


def test_fork_globals_are_consistent_and_originals_stay_48():
    assert R.previous.MAX_NEW_TOKENS == 48
    scope = R.FUNCTIONS
    for name in ('verify_proof','technical_checks','validate_generation','validate_payload','compute_request','verified_request','restore_resume'):
        assert scope[name].__globals__ is scope
        assert scope[name].__code__ is getattr(R.previous,name).__code__
    assert scope['MAX_NEW_TOKENS'] == 128 and scope['SCHEMA'] == R.SCHEMA
    assert scope['generation_module']() is R.generation_module()
    assert R.INDICES == (12,13,14,15) and R.REQUEST_IDS == ('r012','r013','r014','r015')


def test_request_identity_guard_retains_original_ids_seeds_and_prompts():
    request = {'request_id':'r012','item_id':'x','pair_id':'p07','prompt_ids':[1,2],
               'formatted_prompt_sha256':'a','answer_ids':[3],'pulse_positions':[1]}
    R.check_original_request(request, {'request':copy.deepcopy(request)})
    wrong = dict(request,request_id='r000')
    with pytest.raises(ValueError,match='request_id'):
        R.check_original_request(wrong,{'request':request})


def test_missing_private_source_produces_diagnostic_without_gpu(tmp_path):
    result=R.run('missing','missing','missing','missing','missing',tmp_path/'out')
    assert result['state']=='failed' and result['stage']=='reference'
    with zipfile.ZipFile(tmp_path/'out/h286_completion_128_results.zip') as packed:
        assert {'status.json','metrics.json','traceback.txt'} <= set(packed.namelist())
        assert json.loads(packed.read('metrics.json'))['weights_loaded'] is False


def test_public_template_has_no_private_baseline_and_preserves_old_sources(tmp_path):
    old=B.PRIOR.source_bundle(); old_run=B.PRIOR.validation_run_source()
    modules=B.source_bundle()
    assert len(modules)==10 and all(modules[k]==v for k,v in old.items())
    doc=B.notebook_document(modules)
    ns={};exec(''.join(doc['cells'][2]['source']),ns)
    assert ns['REFERENCE_FILES_B64']=={}
    code=''.join(doc['cells'][3]['source'])
    assert 'qwen3_h286_completion_128_runtime.py' in code and "'--baseline'" in code
    assert "base / 'h286_model_cache'" in code
    assert "'h286_completion_128_'" in code
    assert B.PRIOR.validation_run_source()==old_run
    assert len([c for c in doc['cells'] if c['cell_type']=='code'])==3
    path=B.build(tmp_path/'public.ipynb')
    assert json.loads(path.read_text())==doc


def test_private_build_guard_and_pinned_small_loader(tmp_path):
    refs={name:tmp_path/name for name in B.REFERENCE_SHA256}
    for path in refs.values():path.write_bytes(b'synthetic')
    with pytest.raises(ValueError,match='outside'):
        B.build(ROOT/'notebooks/forbidden_completion_private.ipynb',reference_paths=refs)
    raw=json.dumps(B.notebook_document(B.source_bundle())).encode()
    nb=tmp_path/'public.ipynb';nb.write_bytes(raw)
    starter,package=B.package(nb,tmp_path/'package')
    assert starter.stat().st_size<8192
    doc=json.loads(starter.read_text());code=''.join(doc['cells'][1]['source'])
    assert hashlib.sha256(package.read_bytes()).hexdigest() in code
    assert hashlib.sha256(raw).hexdigest() in code
    assert B.PACKAGE in code and B.MEMBER in code
    with zipfile.ZipFile(package) as packed:
        assert packed.namelist()==[B.MEMBER] and packed.read(B.MEMBER)==raw


@pytest.fixture(scope='module')
def native():
    torch=pytest.importorskip('torch')
    transformers=pytest.importorskip('transformers')
    if transformers.__version__!='5.22.2':pytest.skip('Original pinned Transformers5.22.2 optional runtime')
    from transformers import Qwen3Config, Qwen3ForCausalLM
    previous_threads=torch.get_num_threads();torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(1)
        c=Qwen3Config(vocab_size=24,hidden_size=16,intermediate_size=16,num_hidden_layers=2,
                     num_attention_heads=2,num_key_value_heads=1,head_dim=8,eos_token_id=None)
        c._attn_implementation='sdpa';model=Qwen3ForCausalLM(c).float().eval().requires_grad_(False)
    class Tokenizer:
        def decode(self,ids,**_):return ' '.join(str(x) for x in ids)
        def __call__(self,_text,**_):return {'input_ids':[3]}
    yield torch,model,Tokenizer()
    torch.set_num_threads(previous_threads)


def test_native_greedy128_has_same_first48_and_original_decoder_still_rejects128(native):
    torch,model,tok=native
    original=R.previous.generation_module();extended=R.generation_module()
    coords=((0,1),(1,2));stats={'refw':{'c_mu':[0.,0.],'c_sd':[1.,1.]}}
    req={'request_id':'r012','prompt_ids':[1,2,3,4],'answer':'gold'}
    arm={'name':'random','block':1,'beta':1/1.08,'mode':'random'}
    rng=torch.random.get_rng_state().clone()
    old,old_arrays=original.run_generation_arm(model,tok,req,coords,stats,arm,efficient_gpu=False,max_new_tokens=48)
    new,new_arrays=extended.run_generation_arm(model,tok,req,coords,stats,arm,efficient_gpu=False,max_new_tokens=128)
    assert len(new['selected_token_ids'])==128 and old['selected_token_ids']==new['selected_token_ids'][:48]
    assert new['steps'][:48]==old['steps'] and new['seed']==20261129
    assert all(np.array_equal(old_arrays[k],new_arrays[k]) for k in old_arrays)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert original.MAX_NEW_TOKENS==48
    with pytest.raises(ValueError,match='through 48'):
        original.run_generation_arm(model,tok,req,coords,stats,arm,efficient_gpu=False,max_new_tokens=128)
    assert sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules())==0
