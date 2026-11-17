"""Frozen group protocol, norm controls and atomic source-bound checkpoints."""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'notebooks'))
R=importlib.import_module('qwen3_h286_group_runtime')


def coordinates():
    return tuple([(7,i) for i in range(9)]+[(14,i) for i in range(25)])


def test_fixed_cardinality_masks_seeds_and_generation_budget():
    controls=R.make_controls(coordinates(),64)
    assert controls==R.make_controls(coordinates(),64)
    assert len(R.arm_specs())==35
    assert sum(len(R.generated_names(rid)) for rid in R.REQUEST_IDS)==46
    for block,value in controls.items():
        masks=value['outside_masks']
        assert len({tuple(mask) for mask in masks})==3
        assert all(len(mask)==R.COUNTS[block] and not set(mask)&set(value['group']) for mask in masks)
        assert value['outside_overlaps']==[{'a':a,'b':b,'count':len(set(masks[a])&set(masks[b]))}
                                           for a in range(3) for b in range(a+1,3)]
        assert len(value['outside_mask_seeds'])==len(value['within_direction_seeds'])==3


def test_duplicate_fixed_masks_fail_without_resampling(monkeypatch):
    class RNG:
        def __init__(self,*_):pass
        def sample(self,population,count):return list(population)[:count]
        def gauss(self,*_):return 1.
    monkeypatch.setattr(R.random,'Random',RNG)
    with pytest.raises(ValueError,match='Duplicate'):
        R.make_controls(coordinates(),64)


def test_plan_json_roundtrip_preserves_hash_and_no_false_time_budget():
    from attention_probe.reference import ModelPins, Reference
    ref=Reference(coordinates(),'a'*64,'b'*64,ModelPins(R.base.MODEL,R.base.REVISION,(('test','c'*64),)))
    continuity={'source_module_sha256':{},'source_archive_sha256':'d'*64,'source_plan_sha256':'e'*64}
    plan=R.plan_document(ref,{'source_hn_meta_sha256':'f'*64},[],continuity,R.make_controls(coordinates(),64))
    assert R.base.canonical(plan)==R.base.canonical(json.loads(R.base.canonical(plan)))
    assert set(plan['controls'])=={'7','14'}
    assert plan['generation']['planned_count']==46 and plan['prefill_slots']==350
    assert plan['planned_total_prefill_forwards']==406
    assert plan['queue_budget']['wallclock_limit_seconds'] is None


def donors(torch, zero=False):
    clean=torch.arange(64,dtype=torch.float32).reshape(1,1,64)
    delta=torch.arange(1,65,dtype=torch.float32).reshape(1,1,64)/100
    if zero:delta.zero_()
    return {'sham':{'z':{7:clean.clone(),14:clean.clone()}},
            'soft_108':{'z':{14:clean+delta,7:clean+delta}},
            'other_block_108':{'z':{7:clean+delta,14:clean+delta}}}


def test_outside_uses_actual_donor_direction_and_only_raw_norm_scaling():
    torch=pytest.importorskip('torch');controls=R.make_controls(coordinates(),64);d=donors(torch)
    spec=next(s for s in R.arm_specs() if s['name']=='b14_outside_restore_0')
    args,target,cal=R.make_patch(spec,d,controls)
    ix=args['channels'];actual=(d['soft_108']['z'][14][:,:,ix]-d['sham']['z'][14][:,:,ix]).double()
    expected=(-actual/actual.norm()*target.double().norm()).float()
    torch.testing.assert_close(args['delta'],expected,rtol=1e-6,atol=0)
    assert cal['direction_source']=='same-prompt actual pulseZ_C-cleanZ_C'
    assert torch.allclose(args['delta'].double().norm(),target.double().norm(),rtol=1e-6)


def test_zero_target_is_explicit_noop_but_zero_nonzero_target_direction_is_invalid():
    torch=pytest.importorskip('torch');controls=R.make_controls(coordinates(),64)
    spec=next(s for s in R.arm_specs() if s['name']=='b7_outside_transfer_0')
    args,_,cal=R.make_patch(spec,donors(torch,zero=True),controls)
    assert cal['zero_target_noop'] is True and torch.equal(args['delta'],torch.zeros_like(args['delta']))
    d=donors(torch);mask=controls[7]['outside_masks'][0]
    d['other_block_108']['z'][7][:,:,mask]=d['sham']['z'][7][:,:,mask]
    args,_,cal=R.make_patch(spec,d,controls)
    assert args is None and 'degenerate' in cal['invalid_reason']


def synthetic_request(rid='r100'):
    item=R.frozen_items()[rid]
    text='synthetic stable prompt'
    return {'request_id':rid,'item_id':item['id'],'pair_id':item['pair_id'],'question':item['prompt'],
            'answer':item['target'],'answer_ids':[5],'prompt_ids':[1,2,3,4],
            'formatted_prompt':text,'formatted_prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'group_item':item,'validation_schema':R.SCHEMA,'pulse_positions':[3],
            'prefill_capture_positions':[3],'generation_capture_positions':[0,1,2,3]}


def synthetic_input(rid='r100',plan_hash='p'):
    req=synthetic_request(rid);arms=[];arrays={};c=R.ARRAY_CONTRACT
    logits=np.zeros(c['vocab_size'],np.float32);digest=hashlib.sha256(logits.tobytes()).hexdigest()
    for name in R.BASELINE_NAMES:
        arrays[f'baseline_{name}_logits']=logits.copy()
        for block in R.BLOCKS:arrays[f'baseline_{name}_z_block_{block}']=np.zeros((1,1,c['mlp_width']),np.float32)
    for spec in R.arm_specs():
        name=spec['name'];row={'name':name,'spec':spec,'state':'complete','prefill_executed':True,
            'logits_sha256':digest,'full_vocab_kl':{n:{'baseline_to_arm':0.,'arm_to_baseline':0.} for n in R.BASELINE_NAMES},
            'patch':None if spec['kind']=='baseline' else {'raw_norm_match_passed':True}}
        for key in ('raw_z','cett','zcett','down_output_norm_raw'):
            arrays[f'prefill_{name}_{key}']=np.zeros((1,1,c['n_coordinates']),np.float32)
        if name not in R.generated_names(rid):row['generation']={'state':'not_run','reason':'fixed'}
        else:
            steps=[{'selected_token_id':7,'selected_token_logprob':-1.,'is_eos':False,'top_tokens':[{'logprob':-1.}]},
                   {'selected_token_id':0,'selected_token_logprob':-1.,'is_eos':True,'top_tokens':[{'logprob':-1.}]}]
            gen={'schema':'h286_prompt_only_generation_v1','request_id':rid,'arm':R.receiver_arm(spec['receiver']),
                 'max_new_tokens':128,'top_k':10,'prompt_length':4,'prefill_forward_count':1,'pulse_fired':1,'capture_fired':1,
                 'prefill_capture_positions':[0,1,2,3],'pulse_positions':[3],'selected_token_ids':[7,0],
                 'generated_token_ids':[7],'steps':steps,'decode_forward_count':1,'final_cache_length':5,
                 'stop_reason':'eos','truncated':False,'classification':'noncanonical','seed':R.base.SEED+int(rid[1:]),
                 'first_step_logits_sha256':digest}
            row['generation']={'state':'complete','record':gen}
            for key in ('raw_z','cett','zcett','down_output_norm_raw'):
                arrays[f'generation_{name}_{key}']=np.zeros((1,4,c['n_coordinates']),np.float32)
            for block in (3,7,14,27):arrays[f'generation_{name}_residual_layer_{block}']=np.zeros((1,4,c['hidden_size']),np.float32)
        arms.append(row)
    proof={'state':'passed','plan_sha256':plan_hash,'checks':{name:True for name in R.PROOF_CHECKS}}
    rec={'request':req,'arms':arms,'schema':R.SCHEMA,'state':'complete','request_id':rid,'plan_sha256':plan_hash,
         'planned_prefill_slots':35,'generation_count':len(R.generated_names(rid)),'array_contract':c,
         'executed_prefills':35,'technical_checks':proof,'continuity':{}}
    return rec,arrays


@pytest.mark.parametrize('mutation',['proof','missing_check','request','item','positions','not_run','norm','missing_array','shape','dtype','extra'])
def test_strict_common_validator_rejects_incomplete_or_misbound_checkpoints(mutation):
    rec,arrays=synthetic_input()
    if mutation=='proof':rec['technical_checks']['plan_sha256']='wrong'
    elif mutation=='missing_check':del rec['technical_checks']['checks']['native_clean_logits']
    elif mutation=='request':rec['request']['request_id']='r000'
    elif mutation=='item':rec['request']['group_item']=dict(rec['request']['group_item'],prompt='changed')
    elif mutation=='positions':rec['request']['prefill_capture_positions']=[0,1,2,3]
    elif mutation=='not_run':rec['arms'][3]['generation']={'state':'complete','record':{}}
    elif mutation=='norm':rec['arms'][7]['patch']['raw_norm_match_passed']=False
    elif mutation=='missing_array':del arrays['prefill_sham_raw_z']
    elif mutation=='shape':arrays['baseline_sham_logits']=np.zeros(1,np.float32)
    elif mutation=='dtype':arrays['baseline_sham_logits']=arrays['baseline_sham_logits'].astype(np.float64)
    else:arrays['unexpected_full_logits']=np.zeros(2,np.float32)
    with pytest.raises(ValueError):
        R.validate_record(rec,'p','r100');R.validate_arrays(rec,arrays)


def test_atomic_complete_input_resume_and_tamper(tmp_path):
    rec,arrays=synthetic_input();plan={'schema':R.SCHEMA,'controls':{'7':{},'14':{}}}
    digest=hashlib.sha256(R.base.canonical(plan)).hexdigest()
    rec['technical_checks']['plan_sha256']=digest
    R.base.atomic_json(tmp_path/'run_plan.json',plan)
    saved=R.write_input(tmp_path,'r100',rec,arrays,digest,{'gpu':'synthetic'},rec['technical_checks'])
    assert R.verified_input(tmp_path,'r100',digest)==saved
    with pytest.raises(ValueError,match='overwrite'):
        R.write_input(tmp_path,'r100',rec,arrays,digest,{},rec['technical_checks'])
    R.archive(tmp_path);other=tmp_path/'other';other.mkdir()
    R.restore_resume(tmp_path/'h286_group_restore_results.zip',other,digest)
    assert R.verified_input(other,'r100',digest)==saved
    (other/'requests/r100/arrays.npz').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='SHA'):
        R.verified_input(other,'r100',digest)


def test_reference_failure_keeps_diagnostic_without_model(tmp_path):
    result=R.run('missing','missing','missing','missing','missing',tmp_path/'out')
    assert result['state']=='failed' and result['stage']=='reference'
    assert (tmp_path/'out/h286_group_restore_results.zip').exists()
    assert json.loads((tmp_path/'out/metrics.json').read_text())['weights_loaded'] is False


# Native integration fragments independently authored against the frozen hook API.
@pytest.fixture(scope="module")
def group_native():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "5.22.2":
        pytest.skip("Requires the original pinned optional transformers==5.22.2 runtime")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(1793)
        config = Qwen3Config(vocab_size=41, hidden_size=16, intermediate_size=64,
                            num_hidden_layers=28, num_attention_heads=2,
                            num_key_value_heads=1, head_dim=8, eos_token_id=None,
                            bos_token_id=1, pad_token_id=0)
        config._attn_implementation = "sdpa"
        model = Qwen3ForCausalLM(config).float().eval().requires_grad_(False)

    class Tokenizer:
        def decode(self, values, **_):
            return " ".join(str(int(x)) for x in values)

        def __call__(self, text, **_):
            return {"input_ids": [3 if text == "gold" else 4]}

    # Public synthetic local indices: nine channels at 7, twenty-five at 14.
    coordinates = tuple((7, x) for x in range(9)) + tuple((14, x) for x in range(25))
    stats = {"refw": {"c_mu": [0.] * len(coordinates), "c_sd": [1.] * len(coordinates)}}
    request = {"request_id": "r012", "prompt_ids": [1, 7, 3, 2], "answer": "gold",
               "answer_ids": [3], "pulse_positions": [3],
               "group_item": {"semantic_alternative": "foil"}}
    yield torch, model, Tokenizer(), coordinates, stats, request
    torch.set_num_threads(old_threads)


def _group_hook_count(model):
    return sum(len(module._forward_pre_hooks) + len(module._forward_hooks)
               for module in model.modules())


def test_group_native_prefill_baseline_identity_and_frozen_rng(group_native):
    torch, model, tokenizer, coordinates, stats, request = group_native
    ids = torch.tensor([request["prompt_ids"]])
    with torch.inference_mode(), R.previous.attention_context(torch, False):
        direct = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True).logits[0, -1]
    params = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    sham = R.run_prefill(model, request, coordinates, stats, R.receiver_arm("sham"), efficient_gpu=False)
    assert torch.equal(sham["logits"], direct)
    assert sham["pulse_fired"] == sham["capture_fired"] == 1
    assert sham["arrays"]["raw_z"].shape == (1, 1, 34)
    assert sham["z"][7].shape == sham["z"][14].shape == (1, 1, 64)
    for block, count in ((7, 9), (14, 25)):
        channels = [channel for layer, channel in coordinates if layer == block]
        identity = R.run_prefill(model, request, coordinates, stats, R.receiver_arm("sham"),
            patch_args={"block": block, "channels": channels,
                        "donor_values": sham["z"][block][:, :, channels]}, efficient_gpu=False)
        assert len(channels) == count
        assert torch.equal(identity["logits"], sham["logits"])
        assert all(np.array_equal(identity["arrays"][name], value)
                   for name, value in sham["arrays"].items())
        assert identity["patch"].fired_count == identity["patch"].completed_count == 1
        assert identity["patch"].removed_before_decode
        assert not identity["patch"].record["raw_delta_norm"].any()
        assert _group_hook_count(model) == 0
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(params[name], value) for name, value in model.state_dict().items())
    metrics = R.logit_metrics(sham["logits"], {"sham": sham}, tokenizer, request)
    assert metrics["full_vocab_kl"]["sham"] == {"baseline_to_arm": 0., "arm_to_baseline": 0.}


@pytest.mark.parametrize("block,count,offset", [(7, 9, 0), (14, 25, 9)])
def test_group_native_donor_replacement_is_postpatch_readout(group_native, block, count, offset):
    torch, model, _tokenizer, coordinates, stats, request = group_native
    baseline = R.run_prefill(model, request, coordinates, stats, R.receiver_arm("sham"), efficient_gpu=False)
    channels = [channel for layer, channel in coordinates if layer == block]
    donor = torch.linspace(-.4, .6, count, dtype=torch.float32)[None, None, :]
    donor_saved = donor.clone()
    result = R.run_prefill(model, request, coordinates, stats, R.receiver_arm("sham"),
        patch_args={"block": block, "channels": channels, "donor_values": donor}, efficient_gpu=False)
    assert torch.equal(donor, donor_saved)
    assert torch.equal(result["z"][block][:, :, channels], donor)
    assert np.array_equal(result["arrays"]["raw_z"][:, :, offset:offset+count], donor.numpy())
    outside = sorted(set(range(64)) - set(channels))
    assert torch.equal(result["z"][block][:, :, outside], baseline["z"][block][:, :, outside])
    if block == 14:
        assert torch.equal(result["z"][7], baseline["z"][7])
    assert not torch.equal(result["logits"], baseline["logits"])
    assert result["patch"].record["raw_delta_norm"].item() > 0
    assert result["patch"].record["projected_delta_norm"].item() > 0
    assert _group_hook_count(model) == 0


def test_group_native_generation_outer_hooks_removed_before_cached_decode(group_native, monkeypatch):
    torch, model, tokenizer, coordinates, stats, request = group_native
    import attention_probe.completion_128 as decoder
    actual_decoder = decoder.run_generation_arm
    seen_caps = []

    def cap_three(*args, **kwargs):
        seen_caps.append(kwargs["max_new_tokens"])
        assert kwargs["max_new_tokens"] == 128  # Production caller stays frozen.
        kwargs["max_new_tokens"] = 3             # Test-only short real decoder.
        return actual_decoder(*args, **kwargs)

    monkeypatch.setattr(decoder, "run_generation_arm", cap_three)
    baseline = R.run_prefill(model, request, coordinates, stats, R.receiver_arm("sham"), efficient_gpu=False)
    channels = list(range(9))
    donor = baseline["z"][7][:, :, channels].clone()
    rng = torch.random.get_rng_state().clone()
    cache_events = []

    def check_decode_hooks(module, args, kwargs):
        cache = kwargs.get("past_key_values")
        if cache is not None and int(cache.get_seq_length()) > 0:
            cache_events.append(int(cache.get_seq_length()))
            # Only this independent test observer may remain. This catches even
            # inert GroupZPatch/CoordinateCapture/CETT hooks left during decode.
            assert _group_hook_count(model) == 1

    observer = model.register_forward_pre_hook(check_decode_hooks, with_kwargs=True)
    try:
        # CPU SDPA can otherwise choose its fused kernel, while run_prefill pins
        # MATH. Compare identity with one common backend, as the GPU runtime does.
        with R.previous.attention_context(torch, False):
            clean, arrays0 = R.run_generation(model, tokenizer, request, coordinates, stats,
                {"receiver": "sham"}, None, efficient_gpu=False)
            restored, arrays1 = R.run_generation(model, tokenizer, request, coordinates, stats,
                {"receiver": "sham"}, {"block": 7, "channels": channels, "donor_values": donor}, efficient_gpu=False)
        assert _group_hook_count(model) == 1
    finally:
        observer.remove()
    assert seen_caps == [128, 128]
    assert clean["max_new_tokens"] == restored["max_new_tokens"] == 3
    assert clean["decode_forward_count"] == restored["decode_forward_count"] == 2
    assert cache_events == [4, 5, 4, 5]
    assert clean["selected_token_ids"] == restored["selected_token_ids"]
    assert clean["steps"] == restored["steps"]
    assert clean["first_step_logits_sha256"] == restored["first_step_logits_sha256"]
    assert {"residual_layer_3", "residual_layer_7", "residual_layer_14", "residual_layer_27"} <= set(arrays0)
    assert all(np.array_equal(arrays0[name], value) for name, value in arrays1.items())
    assert torch.equal(rng, torch.random.get_rng_state())
    assert _group_hook_count(model) == 0


def test_group_full_vocab_kl_known_distribution_both_directions_and_offset(group_native):
    torch, _model, tokenizer, _coordinates, _stats, request = group_native
    p = torch.tensor([.5, .25, .25], dtype=torch.float64)
    q = torch.tensor([.25, .5, .25], dtype=torch.float64)
    local_request = dict(request, answer_ids=[0])

    class KLTokenizer(type(tokenizer)):
        def __call__(self, text, **_):
            return {"input_ids": [1]}

    tok = KLTokenizer()
    baselines = {"sham": {"logits": p.log()}, "arm_copy": {"logits": q.log()}}
    result = R.logit_metrics(q.log(), baselines, tok, local_request)
    expected = .25 * math.log(2.)
    assert result["full_vocab_kl"]["sham"]["baseline_to_arm"] == pytest.approx(expected, abs=1e-15)
    assert result["full_vocab_kl"]["sham"]["arm_to_baseline"] == pytest.approx(expected, abs=1e-15)
    assert result["full_vocab_kl"]["arm_copy"] == {"baseline_to_arm": 0., "arm_to_baseline": 0.}
    assert result["first_token_correct_minus_foil_logprob"] == pytest.approx(-math.log(2.), abs=1e-15)
    assert result["first_token_contrast_informative"] is True
    shifted = R.logit_metrics(q.log() + 256., baselines, tok, local_request)
    for direction in ("baseline_to_arm", "arm_to_baseline"):
        assert shifted["full_vocab_kl"]["sham"][direction] == pytest.approx(expected, abs=2e-14)
    # Unequal KL directions catch swapped labels; use all three vocabulary entries.
    r = torch.tensor([.8, .1, .1], dtype=torch.float64)
    result = R.logit_metrics(r.log(), baselines, tok, local_request)
    forward = sum(float(p[i]) * math.log(float(p[i] / r[i])) for i in range(3))
    reverse = sum(float(r[i]) * math.log(float(r[i] / p[i])) for i in range(3))
    assert forward != pytest.approx(reverse, abs=1e-5)
    assert result["full_vocab_kl"]["sham"]["baseline_to_arm"] == pytest.approx(forward, abs=1e-15)
    assert result["full_vocab_kl"]["sham"]["arm_to_baseline"] == pytest.approx(reverse, abs=1e-15)
