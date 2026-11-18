"""Native tiny-model integration and strict stage-one checkpoint contracts."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'notebooks'))
R = importlib.import_module('qwen3_b14_b18_runtime')


@pytest.fixture(scope='module')
def native():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    old_threads = torch.get_num_threads(); torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(4002)
        config = Qwen3Config(vocab_size=41,hidden_size=16,intermediate_size=64,
            num_hidden_layers=28,num_attention_heads=2,num_key_value_heads=1,
            head_dim=8,eos_token_id=0,bos_token_id=1,pad_token_id=0)
        config._attn_implementation = 'sdpa'
        model = Qwen3ForCausalLM(config).float().eval().requires_grad_(False)
    model.generation_config.eos_token_id = 0
    prompt = [1,5,7,9,2,3,4,8]
    with torch.no_grad(),R.attention_context(torch,'MATH'):
        last = model.model(input_ids=torch.tensor([prompt]),use_cache=False).last_hidden_state[0,-1]
        model.lm_head.weight.zero_(); model.lm_head.weight[0].copy_(last)
    class Tokenizer:
        def __call__(self,_text,**_):
            return {'input_ids':[3,4,5,6,7,8,9,10]}
        def decode(self,values,**_):
            return ' '.join(str(int(value)) for value in values)
    coordinates = tuple((layer,channel) for layer in R.RESIDUAL_BLOCKS for channel in (0,1))
    reference = SimpleNamespace(coordinates=coordinates)
    stats = {'refw':{'c_mu':[.1]*len(coordinates),'c_sd':[.2]*len(coordinates)}}
    item = {'id':'synthetic','request_id':'s3:synthetic','family':'s3','prompt_ids':prompt,
            'fixed_continuation_text':'synthetic fixed gold','prompt':'synthetic'}
    engine = {'model':model,'tokenizer':Tokenizer()}
    request = R.prepare_request(engine['tokenizer'],0,item,41)
    result,arrays = R.compute_request(engine,reference,stats,request,backend='MATH')
    yield torch,engine,reference,stats,request,result,arrays
    torch.set_num_threads(old_threads)


def test_native_full_pipeline_same_prefixes_causality_jvp_fd_transport_and_free_answers(native):
    torch,engine,_,_,request,result,arrays = native
    assert request['available_offsets']==[0,1,4,8] and request['pulse_positions']==[7]
    assert len(result['measurements'])==28
    assert all(row['capture_positions']==[7+row['offset']] and row['pulse_positions']==[7]
               for row in result['measurements'])
    assert result['checks']==dict.fromkeys(R.PROOF_KEYS,True)
    assert result['jacobian']['b15_source14_unchanged'] is True
    assert result['jacobian']['nonlinear_actual_pulse_parity']['error_norm']<1e-5
    assert result['calibration']['fd_gate_state']=='passed'
    assert result['transport']['state']=='complete'
    assert len(result['transport']['power_iterations'])==4
    assert result['transport']['solver']['max_iter']==16
    assert [row['name'] for row in result['transport']['arms']]==[
        'ridge_transport','direct_delta18_at14','source_norm_matched_noise']
    assert all(row['record']['selected_token_ids']==[0] and row['record']['max_new_tokens']==64
               for row in result['generations'])
    assert R.hook_count(engine['model'])==0
    assert np.array_equal(arrays['math_clean_residual_14'][0,0],arrays['jac_baseline_source'])
    assert np.array_equal(arrays['math_clean_residual_18'][0,0],arrays['jac_baseline_target'])
    for row in result['b18_controls']+result['transport']['arms']:
        assert set(row['logit_references'])=={'math_clean','math_pulse'}
    assert torch.equal(torch.from_numpy(arrays['jac_target_actual']),torch.from_numpy(arrays['jac_target_nonlinear']))


def test_original_decoder_cap_and_global_rng_not_modified(native):
    torch,engine,reference,stats,request,_,_ = native
    from attention_probe import answer_validation
    assert answer_validation.MAX_NEW_TOKENS==48
    with pytest.raises(ValueError,match='48'):
        answer_validation.run_generation_arm(engine['model'],engine['tokenizer'],
            {'request_id':'r000','prompt_ids':request['prompt_ids'],'answer':'synthetic'},
            reference.coordinates,stats,R.ARMS[0],efficient_gpu=False,max_new_tokens=64)
    state = torch.random.get_rng_state().clone()
    a = R.fixed_noise_like(torch.ones(16),123)
    b = R.fixed_noise_like(torch.ones(16),123)
    assert torch.equal(a,b) and torch.equal(state,torch.random.get_rng_state())
    assert float(a.double().norm())==pytest.approx(4.,rel=1e-7)


def test_no_gold_padding_or_moving_residual_intervention(native):
    _,engine,reference,stats,request,_,_ = native
    class Short:
        def __call__(self,_text,**_):
            return {'input_ids':[3]}
    short = R.prepare_request(Short(),2,request['item'],41)
    assert short['available_offsets']==[0,1] and short['missing_offsets']==[4,8]
    import torch
    with pytest.raises(ValueError,match='offset zero'):
        R.forward_probe(engine['model'],request,1,reference.coordinates,stats,R.ARMS[0],backend='MATH',
                        patch={'block':18,'delta':torch.zeros(16)})


def test_full_vocab_kl_arrays_are_independently_recomputable(native):
    _,_,_,_,_,result,arrays = native
    for row in result['measurements']:
        current = arrays[f'o{row["offset"]}_'+row['arm']+'_logits'].astype(np.float64)
        baseline = arrays[f'o{row["offset"]}_clean_logits'].astype(np.float64)
        def logsoftmax(x):
            x=x-x.max(); return x-np.log(np.exp(x).sum())
        current,baseline=logsoftmax(current),logsoftmax(baseline)
        assert row['clean_to_arm_KL']==pytest.approx(np.sum(np.exp(baseline)*(baseline-current)),abs=1e-13)
        assert row['arm_to_clean_KL']==pytest.approx(np.sum(np.exp(current)*(current-baseline)),abs=1e-13)


def test_atomic_exact_checkpoint_resume_and_json_stable_plan(tmp_path,native):
    _,_,_,_,request,result,arrays = native
    plan = {'schema':R.SCHEMA,'controls':{'14':'source','18':'target'}}
    digest=hashlib.sha256(R.base.canonical(plan)).hexdigest()
    assert hashlib.sha256(R.base.canonical(json.loads(R.base.canonical(plan)))).hexdigest()==digest
    R.base.atomic_json(tmp_path/'run_plan.json',plan)
    saved=R.write_checkpoint(tmp_path,result,arrays,digest,{'device':'cpu/native synthetic'})
    assert R.verified(tmp_path,0,digest,request['item'])==saved
    with pytest.raises(ValueError,match='overwrite'):
        R.write_checkpoint(tmp_path,result,arrays,digest,{})
    R.archive(tmp_path)
    destination=tmp_path/'resumed'; destination.mkdir()
    R.restore_resume(tmp_path/'b14_b18_results.zip',destination,digest,[request['item']])
    assert R.verified(destination,0,digest,request['item'])==saved
    with pytest.raises(ValueError,match='another'):
        R.restore_resume(tmp_path/'b14_b18_results.zip',destination,'wrong',[request['item']])
    (destination/'requests/b14b18_000/arrays.npz').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='SHA'):
        R.verified(destination,0,digest,request['item'])


@pytest.mark.parametrize('mutation',['proof','source_ids','pulse_position','missing_arm','missing_array','shape','dtype','extra',
                                     'generation','calibration','jacobian','control_name','control_receiver','logits_hash'])
def test_saved_checkpoint_rejects_incomplete_or_changed_protocol(native,mutation):
    _,_,_,_,_,result,arrays=native
    record=copy.deepcopy(result)
    record.update(schema=R.SCHEMA,state='complete',plan_sha256='plan',
        technical_checks={'state':'passed','plan_sha256':'plan','checks':copy.deepcopy(result['checks'])})
    arrays={k:v.copy() for k,v in arrays.items()}
    if mutation=='proof':record['technical_checks']['checks'].pop('sham_exact')
    elif mutation=='source_ids':record['request']['prompt_ids'][0]=12
    elif mutation=='pulse_position':record['request']['pulse_positions']=[8]
    elif mutation=='missing_arm':record['measurements'].pop()
    elif mutation=='missing_array':arrays.pop('o0_clean_logits')
    elif mutation=='shape':arrays['o0_clean_raw_z']=np.zeros((2,1,12),np.float32)
    elif mutation=='dtype':arrays['o0_clean_logits']=arrays['o0_clean_logits'].astype(np.float64)
    elif mutation=='extra':arrays['extra']=np.zeros(1,np.float32)
    elif mutation=='generation':record['generations'][0]['record']['max_new_tokens']=48
    elif mutation=='calibration':record['calibration']['fd_gate_state']='failed'
    elif mutation=='jacobian':record['jacobian']['target']=19
    elif mutation=='control_name':record['b18_controls'][0]['name']='other'
    elif mutation=='control_receiver':record['b18_controls'][0]['receiver']='b14_soft_108'
    else:arrays['o8_clean_logits'][0]+=1
    with pytest.raises(ValueError):
        R.validate_record(record,'plan');R.validate_arrays(record,arrays)


def test_missing_references_diagnostic_without_model_or_waiting(tmp_path):
    status=R.run('missing','missing','missing','missing','missing',tmp_path)
    assert status['state']=='failed' and status['stage']=='reference'
    assert (tmp_path/'b14_b18_results.zip').is_file()
    metrics=json.loads((tmp_path/'metrics.json').read_text())
    assert metrics['weights_loaded'] is False and metrics['pretrained_model_run'] is False


def test_budget_uses_next_atomic_input_not_the_whole_remaining_queue():
    assert R.should_pause(300,[300],15) is False  # 75-minute queue ETA alone cannot stop this run.
    assert R.should_pause(2600,[120],5) is True
    assert R.should_pause(2700,[1],1) is True
    assert R.should_pause(3000,[120],0) is False
