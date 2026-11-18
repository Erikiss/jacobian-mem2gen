"""Exactly nine prespecified control answers for unchanged original r013.

Original thirteen scientific sources are imported, never patched. Donors are
fresh same-prompt states; all seven patches are calibrated before their outcomes.
One atomic checkpoint contains all nine prefills and all nine greedy answers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

import qwen3_h286_group_runtime as group

base=group.base
SCHEMA='qwen3_h286_control_answers_v1'
CONTROL_BASELINE_SHA256='b14f0fe12ff6c1d059a6e5954944feb67c87ea7cff61b0f7ad4533d586425143'
SOURCE_PLAN_SHA256='c9bf357e934fe308363e6131415b316158e2ba542ab0f127adb701220a3ed10d'
REQUEST_ID='r013'
ARM_NAMES=('sham','other_block_108','b7_transfer','b7_outside_transfer_0','b7_outside_transfer_1',
           'b7_outside_transfer_2','b7_within_transfer_0','b7_within_transfer_1','b7_within_transfer_2')
BASELINES=ARM_NAMES[:2]
KNOWN_GENERATIONS=ARM_NAMES[:3]
PROOF_CHECKS=('native_clean_logits','identity_sham','identity_pulse','all_norm_controls_valid',
              'fresh_same_prompt_donors','generation_prefill_logits','all_nine_prior_first_tokens',
              'three_prior_full_generations','frozen_parameters','global_rng','hook_cleanup')


def specs():
    lookup={row['name']:row for row in group.arm_specs()}
    return [lookup[name] for name in ARM_NAMES]


def load_control_baseline(path,reference):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=CONTROL_BASELINE_SHA256:
        raise ValueError('Exact private control-answer baseline SHA mismatch')
    value=json.loads(raw)
    if (value.get('schema')!='h286_control_answers_baseline_v1'
        or value.get('source_plan_sha256')!=SOURCE_PLAN_SHA256
        or hashlib.sha256(base.canonical(value['source_plan'])).hexdigest()!=SOURCE_PLAN_SHA256
        or value.get('model')!=reference.model.to_dict() or value.get('arm_names')!=list(ARM_NAMES)
        or value.get('request',{}).get('request_id')!=REQUEST_ID):
        raise ValueError('Original r013/model/nine-arm source identity changed')
    if len(value['source_module_sha256'])!=13:raise ValueError('Need all thirteen unchanged scientific sources')
    for name,digest in value['source_module_sha256'].items():
        if base.sha_file(group.source_path(name))!=digest:raise ValueError('Original source changed: '+name)
    return value


def plan_document(reference,stats,items,continuity,controls,prior):
    parent=group.plan_document(reference,stats,items,continuity,controls)
    if hashlib.sha256(base.canonical(parent)).hexdigest()!=SOURCE_PLAN_SHA256:
        raise ValueError('Embedded original group plan does not reproduce its source SHA')
    item=next(value for value in items if value['request_id']==REQUEST_ID)
    return {'schema':SCHEMA,'purpose':'nine fixed negative/target control answers for original r013 only',
        'helper_sha256':base.sha_file(__file__),'unchanged_source_sha256':prior['source_module_sha256'],
        'source_archive_sha256':prior['source_archive_sha256'],'source_plan_sha256':SOURCE_PLAN_SHA256,
        'control_baseline_sha256':CONTROL_BASELINE_SHA256,'reference_sha256':prior['reference_sha256'],
        'model':reference.model.to_dict(),'coordinates':[list(pair) for pair in reference.coordinates],
        'item':item,'request_id':REQUEST_ID,'seed':base.SEED+13,'controls':{'7':controls[7]},
        'prefill_arms':specs(),'arm_names':list(ARM_NAMES),'array_contract':group.ARRAY_CONTRACT,
        'prefill_slots':9,'identity_prefills':2,'native_clean_prefills':1,'generation_replay_prefills':9,
        'planned_total_prefill_forwards':21,'generation_count':9,'max_new_tokens':128,
        'dtype':'float32','tf32':False,'attention':'sdpa/EFFICIENT_ATTENTION',
        'transformers':base.TRANSFORMERS,'normalization':parent['normalization'],'numeric_guards':parent['numeric_guards'],
        'donors':'fresh same-prompt clean and block7-pulse states; never private baseline values',
        'hard_checks':'identity/native exact; each generation=current prefill hash; all9 prior first tokens; old3 full tokens/EOS/protocol',
        'numeric_prior_comparisons':'separate diagnostics, not extra hard numeric conditions',
        'queue_budget':{'wallclock_limit_seconds':None,'completion_guaranteed':False},
        'claim':'Post-result exploratory control follow-up; no training or confirmatory/semantic-accuracy claim'}


def collect_prefills(engine,reference,stats,request,controls,*,efficient_gpu=True):
    import numpy as np
    import torch
    model=engine['model'];tokenizer=engine['tokenizer'];norms=engine.get('weight_norms')
    hooks=group.previous.hook_count(model)
    if hooks!=0:raise ValueError('Native clean check requires genuinely uninstrumented model')
    ids=torch.tensor([request['prompt_ids']],device=next(model.parameters()).device,dtype=torch.long)
    with torch.inference_mode(),group.previous.attention_context(torch,efficient_gpu):
        output=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=True)
        native=group.tensor_sha(output.logits[0,-1]);del output
    donors={};rows={};arrays={};patches={};calibrations={};snapshots={}
    for name in BASELINES:
        fresh=group.run_prefill(model,request,reference.coordinates,stats,group.receiver_arm(name),
                               efficient_gpu=efficient_gpu,weight_norms=norms)
        donors[name]=fresh
        snapshots[f'baseline_{name}_logits']=group.tensor_sha(fresh['logits'])
        for block in group.BLOCKS:snapshots[f'baseline_{name}_z_block_{block}']=group.tensor_sha(fresh['z'][block])
        arrays[f'donor_{name}_z_block_7']=fresh['z'][7].numpy()
    if native!=group.tensor_sha(donors['sham']['logits']):raise ValueError('Native clean logits differ from passive sham')
    # Every fixed patch is calibrated now, before identities/control outcomes.
    for spec in specs()[2:]:
        args,target,cal=group.make_patch(spec,donors,controls)
        if args is None:raise ValueError('Invalid norm control; none omitted/resampled: '+spec['name']+': '+cal['invalid_reason'])
        patches[spec['name']]=dict(args,block=7);calibrations[spec['name']]=(target,cal)
    identity_checks={}
    identity_specs={s['name']:s for s in group.arm_specs()}
    for receiver,suffix in (('sham','sham'),('other_block_108','pulse')):
        spec=identity_specs['b7_identity_'+suffix]
        args,_,_=group.make_patch(spec,donors,controls)
        observed=group.run_prefill(model,request,reference.coordinates,stats,group.receiver_arm(receiver),
            patch_args=dict(args,block=7),efficient_gpu=efficient_gpu,weight_norms=norms)
        same=(group.tensor_sha(observed['logits'])==group.tensor_sha(donors[receiver]['logits'])
              and all(np.array_equal(observed['arrays'][key],value) for key,value in donors[receiver]['arrays'].items()))
        if not same:raise ValueError('Identity group patch changed native logits/readouts')
        identity_checks['identity_'+suffix]=True
    for spec in specs():
        name=spec['name'];patch=None
        if name in BASELINES:observed=donors[name]
        else:
            observed=group.run_prefill(model,request,reference.coordinates,stats,group.receiver_arm(spec['receiver']),
                patch_args=patches[name],efficient_gpu=efficient_gpu,weight_norms=norms)
            target,cal=calibrations[name]
            patch=group.patch_summary(observed['patch'],model,spec,target,controls,cal)
            if not patch['raw_norm_match_passed']:raise ValueError('Applied norm control invalid; no omission/replacement: '+name)
        rows[name]={'name':name,'spec':spec,'state':'complete','prefill_executed':True,'patch':patch,
                    **group.logit_metrics(observed['logits'],donors,tokenizer,request)}
        arrays[f'prefill_{name}_logits']=observed['logits'].numpy()
        for key,value in observed['arrays'].items():arrays[f'prefill_{name}_{key}']=value
    if group.previous.hook_count(model)!=hooks:raise ValueError('Prefill/identity hooks remain installed')
    return {'donors':donors,'patches':patches,'rows':rows,'arrays':arrays,
            'checks':dict(native_clean_logits=True,hook_cleanup=True,**identity_checks),
            'baseline_numeric_snapshots':snapshots}


def continuity_checks(request,rows,arrays,prior):
    import numpy as np
    group.completion.check_original_request(request,prior)
    result={}
    for name in ARM_NAMES:
        row=rows[name];generation=row['generation']['record'];old=prior['arms'][name]
        if row['spec']!=old['spec']:raise ValueError('Prespecified control spec changed: '+name)
        first=generation['selected_token_ids'][0]
        if first!=old['first_token_id']:raise ValueError('New generated first token differs from old observed top1: '+name)
        old_prefill=old['recorded_prefill']
        array_hashes={key:hashlib.sha256(np.asarray(arrays[f'prefill_{name}_{key}']).tobytes(order='C')).hexdigest()
                      for key in ('raw_z','cett','zcett','down_output_norm_raw')}
        checks={'prior_first_token_equal':True,'old_first_token_id':old['first_token_id'],
                'prefill_logits_exact':row['logits_sha256']==old['logits_sha256'],
                'prefill_arrays_exact':array_hashes==old['prefill_array_sha256'],
                'prefill_shapes_exact':{key:list(arrays[f'prefill_{name}_{key}'].shape) for key in array_hashes}==old['prefill_array_shapes'],
                'old_top10_exact':row['top10']==old_prefill['top10']}
        if name in KNOWN_GENERATIONS:
            oldgen=old['generation']['record']
            keys=('selected_token_ids','generated_token_ids','stop_reason','seed','prompt_length',
                  'eos_token_ids','eos_source','max_new_tokens')
            if any(generation[key]!=oldgen[key] for key in keys):raise ValueError('Known full token/EOS/seed protocol changed: '+name)
            hashes={key:hashlib.sha256(np.asarray(value).tobytes(order='C')).hexdigest()
                    for key,value in ((k[len('generation_'+name+'_'):],v) for k,v in arrays.items() if k.startswith('generation_'+name+'_'))}
            checks.update(prior_full_tokens_protocol_equal=True,
                prior_generation_first_logits_exact=generation['first_step_logits_sha256']==oldgen['first_step_logits_sha256'],
                prior_steps_exact=hashlib.sha256(base.canonical(generation['steps'])).hexdigest()==old['generation']['steps_sha256'],
                prior_generation_arrays_exact=hashes==old['generation']['gen_array_sha256'])
        else:checks['prior_generation']='not_run; new prespecified negative-control answer'
        result[name]=checks
    return result


def compute_request(engine,reference,stats,request,controls,control_baseline,*,efficient_gpu=True):
    import numpy as np
    import torch
    model=engine['model'];frozen=base.parameter_evidence(model)
    cpu_rng=torch.random.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone() if efficient_gpu else None
    measured=collect_prefills(engine,reference,stats,request,controls,efficient_gpu=efficient_gpu)
    rows,arrays=measured['rows'],measured['arrays']
    print('Stage r013: neun Prefills und Identitäten geprüft; alle neun Antworten folgen.',flush=True)
    for spec in specs():
        name=spec['name']
        gen,values=group.run_generation(model,engine['tokenizer'],request,reference.coordinates,stats,spec,
            measured['patches'].get(name),efficient_gpu=efficient_gpu,weight_norms=engine.get('weight_norms'))
        if gen['first_step_logits_sha256']!=rows[name]['logits_sha256']:
            raise ValueError('Generation differs from current corresponding prefill logits: '+name)
        rows[name]['generation']={'state':'complete','record':gen}
        for key,value in values.items():arrays[f'generation_{name}_{key}']=value
    continuity=continuity_checks(request,rows,arrays,control_baseline)
    snapshots=measured['baseline_numeric_snapshots']
    numeric_baselines={key:value==control_baseline['baseline_array_sha256'][key] for key,value in snapshots.items()}
    if (base.parameter_evidence(model)!=frozen or not torch.equal(cpu_rng,torch.random.get_rng_state())
        or (cuda_rng is not None and not torch.equal(cuda_rng,torch.cuda.get_rng_state()))
        or group.previous.hook_count(model)!=0):raise ValueError('Frozen parameter/RNG/hook evidence changed')
    if any(np.asarray(v).dtype!=np.float32 or not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite/non-FP32 readout')
    checks=dict(measured['checks'],all_norm_controls_valid=True,fresh_same_prompt_donors=True,
        generation_prefill_logits=True,all_nine_prior_first_tokens=True,three_prior_full_generations=True,
        frozen_parameters=True,global_rng=True)
    return {'request':request,'arms':[rows[name] for name in ARM_NAMES],'array_contract':{
        'vocab_size':int(model.config.vocab_size),'hidden_size':int(model.config.hidden_size),
        'mlp_width':int(model.config.intermediate_size),'n_coordinates':len(reference.coordinates)},
        'prefill_slots':9,'generation_count':9,'total_prefill_forwards':21,'continuity':continuity,
        'baseline_numeric_diagnostics':numeric_baselines,'technical_checks':{'state':'passed','checks':checks}},arrays


def validate_record(record,digest):
    if (record.get('schema')!=SCHEMA or record.get('state')!='complete' or record.get('request_id')!=REQUEST_ID
        or record.get('plan_sha256')!=digest or record.get('array_contract')!=group.ARRAY_CONTRACT
        or record.get('prefill_slots')!=9 or record.get('generation_count')!=9 or record.get('total_prefill_forwards')!=21):
        raise ValueError('Incomplete/differently bound nine-arm checkpoint')
    req=record['request'];item=group.frozen_items()[REQUEST_ID]
    if (req.get('request_id')!=REQUEST_ID or req.get('group_item')!=item or req.get('question')!=item['prompt']
        or req.get('answer')!=item['target'] or req.get('prefill_capture_positions')!=[len(req['prompt_ids'])-1]
        or req.get('pulse_positions')!=[len(req['prompt_ids'])-1]
        or req.get('generation_capture_positions')!=list(range(max(0,len(req['prompt_ids'])-4),len(req['prompt_ids'])))
        or hashlib.sha256(req['formatted_prompt'].encode()).hexdigest()!=req['formatted_prompt_sha256']):
        raise ValueError('Original r013 prompt/item/position changed')
    proof=record.get('technical_checks',{})
    if proof.get('state')!='passed' or proof.get('plan_sha256')!=digest or any(proof.get('checks',{}).get(k) is not True for k in PROOF_CHECKS):
        raise ValueError('Missing plan-bound complete technical checks')
    if [row.get('spec') for row in record['arms']]!=specs():raise ValueError('Exact nine-arm subset/order changed')
    for name,row in zip(ARM_NAMES,record['arms'],strict=True):
        if row.get('name')!=name or row.get('state')!='complete' or row.get('generation',{}).get('state')!='complete':
            raise ValueError('Every prescribed negative/target control must be complete')
        if row['spec']['kind']!='baseline' and row.get('patch',{}).get('raw_norm_match_passed') is not True:
            raise ValueError('Invalid norm control cannot be omitted or accepted')
        gen=row['generation']['record']
        group.completion.FUNCTIONS['validate_generation'](gen,req,group.receiver_arm(row['spec']['receiver']))
        if gen.get('seed')!=base.SEED+13 or gen.get('first_step_logits_sha256')!=row.get('logits_sha256'):
            raise ValueError('Original seed/current prefill generation hash changed')
        if set(row.get('full_vocab_kl',{}))!=set(BASELINES):raise ValueError('Need both full-vocabulary baseline KL endpoints')
        if any(set(end)!= {'baseline_to_arm','arm_to_baseline'} or not all(npfinite(v) for v in end.values())
               for end in row['full_vocab_kl'].values()):raise ValueError('Missing/wrong/nonfinite KL endpoints')
        continuity=record['continuity'][name]
        if continuity.get('prior_first_token_equal') is not True or (name in KNOWN_GENERATIONS and continuity.get('prior_full_tokens_protocol_equal') is not True):
            raise ValueError('Missing hard first-token/full-known-generation continuity')


def npfinite(value):
    import math
    return isinstance(value,(int,float)) and math.isfinite(value)


def validate_arrays(record,arrays):
    import numpy as np
    c=group.ARRAY_CONTRACT;expected={};length=min(4,len(record['request']['prompt_ids']))
    for name in BASELINES:expected[f'donor_{name}_z_block_7']=(1,1,c['mlp_width'])
    for name in ARM_NAMES:
        expected[f'prefill_{name}_logits']=(c['vocab_size'],)
        for key in ('raw_z','cett','zcett','down_output_norm_raw'):
            expected[f'prefill_{name}_{key}']=(1,1,c['n_coordinates'])
            expected[f'generation_{name}_{key}']=(1,length,c['n_coordinates'])
        for block in (3,7,14,27):expected[f'generation_{name}_residual_layer_{block}']=(1,length,c['hidden_size'])
    keys=set(arrays.files) if hasattr(arrays,'files') else set(arrays)
    if keys!=set(expected):raise ValueError('Missing/unexpected nine-arm arrays or full-vocabulary vectors')
    for key,shape in expected.items():
        value=np.asarray(arrays[key])
        if value.shape!=shape or value.dtype!=np.float32 or not np.isfinite(value).all():raise ValueError('Bad array shape/dtype/finite: '+key)
    for row in record['arms']:
        if int(np.asarray(arrays[f'prefill_{row["name"]}_logits']).argmax())!=row['generation']['record']['selected_token_ids'][0]:
            raise ValueError('Saved full logits disagree with generated first token')
        if hashlib.sha256(np.asarray(arrays[f'prefill_{row["name"]}_logits']).tobytes()).hexdigest()!=row['logits_sha256']:
            raise ValueError('Full saved arm logits hash changed')


def verified(output,digest,prior):
    import numpy as np
    directory=Path(output)/'requests'/REQUEST_ID
    if not directory.exists():return None
    record=json.loads((directory/'result.json').read_text());validate_record(record,digest)
    if base.sha_file(directory/'arrays.npz')!=record['npz_sha256']:raise ValueError('Nine-arm NPZ SHA mismatch')
    with np.load(directory/'arrays.npz',allow_pickle=False) as arrays:
        validate_arrays(record,arrays)
        checks=continuity_checks(record['request'],{row['name']:row for row in record['arms']},arrays,prior)
        if checks!=record['continuity']:raise ValueError('Stored continuity evidence changed')
    return record


def write_checkpoint(output,result,arrays,digest,hardware,prior):
    import numpy as np
    parent=Path(output)/'requests';parent.mkdir(parents=True,exist_ok=True);dest=parent/REQUEST_ID
    if dest.exists():raise ValueError('Refusing to overwrite complete r013 checkpoint')
    tmp=Path(tempfile.mkdtemp(prefix='.r013_',dir=parent))
    try:
        proof=dict(result['technical_checks'],plan_sha256=digest)
        record=dict(result,schema=SCHEMA,state='complete',request_id=REQUEST_ID,plan_sha256=digest,
                    technical_checks=proof,hardware=hardware)
        validate_record(record,digest);validate_arrays(record,arrays)
        np.savez_compressed(tmp/'arrays.npz',**arrays);record['npz_sha256']=base.sha_file(tmp/'arrays.npz')
        base.atomic_json(tmp/'result.json',record);os.rename(tmp,dest)
    finally:
        if tmp.exists():shutil.rmtree(tmp)
    return verified(output,digest,prior)


def archive(output):
    target=Path(output)/'h286_control_answers_results.zip';tmp=target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(Path(output).rglob('*')):
            if path.is_file() and path.suffix not in ('.zip','.tmp') and not any(p.startswith('.') for p in path.relative_to(output).parts):z.write(path,path.relative_to(output))
    os.replace(tmp,target)


def restore_resume(path,output,digest,prior):
    if not path:return
    with zipfile.ZipFile(path) as z:
        plan=json.loads(z.read('run_plan.json'))
        if plan.get('schema')!=SCHEMA or hashlib.sha256(base.canonical(plan)).hexdigest()!=digest:raise ValueError('Resume has another nine-arm plan')
        names=[f'requests/{REQUEST_ID}/result.json',f'requests/{REQUEST_ID}/arrays.npz']
        if not all(name in z.namelist() for name in names):raise ValueError('Only complete all-nine r013 input may be resumed')
        if (Path(output)/'requests'/REQUEST_ID).exists():
            verified(output,digest,prior)
            if any((Path(output)/name).read_bytes()!=z.read(name) for name in names):raise ValueError('Resume conflict; no overwrite')
            return
        temporary=Path(tempfile.mkdtemp(prefix='.resume_',dir=output))
        try:
            dest=temporary/'requests'/REQUEST_ID;dest.mkdir(parents=True)
            for name in names:
                if z.getinfo(name).file_size>32*(1<<20):raise ValueError('Oversized single-input checkpoint')
                (temporary/name).write_bytes(z.read(name))
            verified(temporary,digest,prior)
            (Path(output)/'requests').mkdir(exist_ok=True);os.rename(dest,Path(output)/'requests'/REQUEST_ID)
        finally:shutil.rmtree(temporary)


def run(hn,policy,refw,items_path,continuity_path,control_baseline,out_dir,cache_dir=None,resume_zip=None):
    output=Path(out_dir);output.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    status={'state':'running','stage':'reference','scope':SCHEMA,'completed_inputs':0,'confirmatory_eligible':False}
    metrics={'weights_loaded':False,'pretrained_model_run':False,'wrapper_attempt_id':os.environ.get('H286_WRAPPER_ATTEMPT'),
             'no_training':True,'no_confirmatory_claim':True}
    def save():base.atomic_json(output/'status.json',status);base.atomic_json(output/'metrics.json',metrics)
    save()
    try:
        reference,stats,items,continuity=group.load_inputs(hn,policy,refw,items_path,continuity_path)
        prior=load_control_baseline(control_baseline,reference);controls=group.make_controls(reference.coordinates,reference.model.mlp_width)
        plan=plan_document(reference,stats,items,continuity,controls,prior);digest=hashlib.sha256(base.canonical(plan)).hexdigest()
        path=output/'run_plan.json'
        if path.exists() and path.read_bytes()!=base.canonical(plan)+b'\n':raise ValueError('Different output plan; prior sources/results must not be mixed')
        if not path.exists():base.atomic_json(path,plan)
        restore_resume(resume_zip,output,digest,prior);record=verified(output,digest,prior)
        if record is None:
            status['stage']='hardware_and_original_model_load';save()
            def loaded(hardware):metrics.update(weights_loaded=True,hardware=hardware);save()
            engine=group.previous.initialize_engine(reference,Path(cache_dir or Path.home()/'.cache/h286_pilot'),on_loaded=loaded)
            model=engine['model'];contract={k:int(getattr(model.config,attr)) for k,attr in
                [('vocab_size','vocab_size'),('hidden_size','hidden_size'),('mlp_width','intermediate_size')]}
            contract['n_coordinates']=len(reference.coordinates)
            if contract!=group.ARRAY_CONTRACT:raise ValueError('Original model/readout architecture changed')
            item=next(row for row in items if row['request_id']==REQUEST_ID)
            request=group.prepare_request(engine['tokenizer'],13,item)
            group.completion.check_original_request(request,prior)
            status['stage']='nine_prespecified_controls';save()
            print('Request r013: neun feste Arme, 21 Prefills insgesamt, alle neun Antworten bis EOS/128.',flush=True)
            result,arrays=compute_request(engine,reference,stats,request,controls,prior)
            metrics['pretrained_model_run']=True
            record=write_checkpoint(output,result,arrays,digest,engine['hardware'],prior)
        metrics.update(generation_count=9,prefill_slots=9,total_prefill_forwards=21,retained_technical_checks=record['technical_checks'])
        status.update(state='complete',stage='complete',completed_inputs=1,generation_count=9,prefill_slots=9)
        print('Gespeichert r013: alle neun Antworten und festen Kontinuitätsprüfungen vollständig.',flush=True)
    except BaseException as error:  # noqa: BLE001 -- preserve diagnostics on interruption.
        status.update(state='failed',error_type=type(error).__name__,error=str(error));(output/'traceback.txt').write_text(traceback.format_exc())
        print('HARD FAILURE: '+str(error),flush=True)
    finally:
        metrics['elapsed_seconds']=time.monotonic()-started;save();archive(output)
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('hn-set','policy','refw','items','continuity','control-baseline','out-dir'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--cache-dir');parser.add_argument('--resume-zip');args=parser.parse_args()
    result=run(args.hn_set,args.policy,args.refw,args.items,args.continuity,args.control_baseline,args.out_dir,args.cache_dir,args.resume_zip)
    print(json.dumps(result),flush=True);raise SystemExit(0 if result['state']=='complete' else 1)
