"""New nine-arm subset/count/continuity/checkpoint checks; no model downloads."""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'notebooks'))
C=importlib.import_module('qwen3_h286_control_answers_runtime')
def _control_runtime():
    return C


_CONTROL_NAMES=C.ARM_NAMES


S=importlib.util.spec_from_file_location('control_previous_fixtures',ROOT/'tests/test_h286_group_runtime.py')
F=importlib.util.module_from_spec(S);S.loader.exec_module(F)


def fixture():
    old,old_arrays=F.synthetic_input('r013','plan')
    rows={row['name']:copy.deepcopy(row) for row in old['arms']}
    request=copy.deepcopy(old['request']);arrays={};prior={'request':copy.deepcopy(request),'arms':{},'baseline_array_sha256':{}}
    for name in C.ARM_NAMES:
        row=rows[name];row['full_vocab_kl']={n:row['full_vocab_kl'][n] for n in C.BASELINES}
        if row['generation']['state']=='not_run':row['generation']=copy.deepcopy(rows['b7_transfer']['generation'])
        row['generation']['record']['arm']=C.group.receiver_arm(row['spec']['receiver'])
        arrays[f'prefill_{name}_logits']=old_arrays['baseline_sham_logits'].copy()
        arrays[f'prefill_{name}_logits'][7]=1.
        row['logits_sha256']=hashlib.sha256(arrays[f'prefill_{name}_logits'].tobytes()).hexdigest()
        row['generation']['record']['first_step_logits_sha256']=row['logits_sha256']
        row['top10']=[{'token_id':7 if i==0 else i-1,'text':'synthetic','logprob':-float(i+1)} for i in range(10)]
        for key in ('raw_z','cett','zcett','down_output_norm_raw'):
            arrays[f'prefill_{name}_{key}']=old_arrays[f'prefill_{name}_{key}'].copy()
        for key in ('raw_z','cett','zcett','down_output_norm_raw','residual_layer_3','residual_layer_7','residual_layer_14','residual_layer_27'):
            source=name if f'generation_{name}_{key}' in old_arrays else 'b7_transfer'
            arrays[f'generation_{name}_{key}']=old_arrays[f'generation_{source}_{key}'].copy()
        gen=row['generation']['record']
        gen.update(eos_token_ids=[0],eos_source='generation_config')
        entry={'spec':copy.deepcopy(row['spec']),'first_token_id':gen['selected_token_ids'][0],
            'logits_sha256':row['logits_sha256'],'recorded_prefill':copy.deepcopy(row),
            'prefill_array_sha256':{key:hashlib.sha256(arrays[f'prefill_{name}_{key}'].tobytes()).hexdigest()
                                  for key in ('raw_z','cett','zcett','down_output_norm_raw')},
            'prefill_array_shapes':{key:list(arrays[f'prefill_{name}_{key}'].shape)
                                   for key in ('raw_z','cett','zcett','down_output_norm_raw')},
            'generation':{'state':'not_run'}}
        if name in C.KNOWN_GENERATIONS:
            prefix=f'generation_{name}_'
            values={key[len(prefix):]:value for key,value in arrays.items() if key.startswith(prefix)}
            entry['generation']={'state':'complete','record':{k:v for k,v in gen.items() if k!='steps'},
                'steps_sha256':hashlib.sha256(C.base.canonical(gen['steps'])).hexdigest(),
                'gen_array_sha256':{key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in values.items()}}
        prior['arms'][name]=entry
    for name in C.BASELINES:
        arrays[f'donor_{name}_z_block_7']=old_arrays[f'baseline_{name}_z_block_7'].copy()
    continuity=C.continuity_checks(request,rows,arrays,prior)
    result={'request':request,'arms':[rows[n] for n in C.ARM_NAMES],'array_contract':C.group.ARRAY_CONTRACT,
            'prefill_slots':9,'generation_count':9,'total_prefill_forwards':21,'continuity':continuity,
            'technical_checks':{'state':'passed','plan_sha256':'plan','checks':{k:True for k in C.PROOF_CHECKS}},
            'schema':C.SCHEMA,'state':'complete','request_id':'r013','plan_sha256':'plan'}
    return result,arrays,prior


def test_exact_nine_subset_and_old_sources_are_not_modified():
    assert C.ARM_NAMES==('sham','other_block_108','b7_transfer','b7_outside_transfer_0','b7_outside_transfer_1',
                        'b7_outside_transfer_2','b7_within_transfer_0','b7_within_transfer_1','b7_within_transfer_2')
    assert len(C.specs())==9 and len(C.group.arm_specs())==35
    assert all(row['direction']=='transfer' and row['block']==7 for row in C.specs()[2:])
    assert C.base.SEED+13==20261130


def test_original_numeric_arrays_exact_and_drift_is_diagnostic_only():
    result,arrays,prior=fixture();rows={r['name']:r for r in result['arms']}
    checks=C.continuity_checks(result['request'],rows,arrays,prior)
    assert all(checks[n]['prior_generation_arrays_exact'] and checks[n]['prior_steps_exact'] for n in C.KNOWN_GENERATIONS)
    changed={k:v.copy() for k,v in arrays.items()};changed['generation_sham_raw_z'][0,0,0]=1.
    checks=C.continuity_checks(result['request'],rows,changed,prior)
    assert checks['sham']['prior_generation_arrays_exact'] is False
    assert checks['sham']['prior_full_tokens_protocol_equal'] is True


@pytest.mark.parametrize('mutation',['first_token','known_tokens','seed','prompt'])
def test_hard_continuity_does_not_accept_changed_tokens_protocol_or_tokenization(mutation):
    result,arrays,prior=fixture();rows={r['name']:r for r in result['arms']};request=result['request']
    if mutation=='first_token':prior['arms']['b7_within_transfer_2']['first_token_id']=8
    elif mutation=='known_tokens':prior['arms']['sham']['generation']['record']['selected_token_ids']=[8,0]
    elif mutation=='seed':prior['arms']['other_block_108']['generation']['record']['seed']=1
    else:request['prompt_ids']=[9,2,3,4]
    with pytest.raises(ValueError):C.continuity_checks(request,rows,arrays,prior)


@pytest.mark.parametrize('mutation',['empty_kl','wrong_kl_keys','missing_arm','not_run','proof','seed','missing_logits','shape','dtype','extra'])
def test_new_checkpoint_requires_complete_nine_answers_and_full_vocab_contract(mutation):
    record,arrays,_=fixture()
    if mutation=='empty_kl':record['arms'][0]['full_vocab_kl']['sham']={}
    elif mutation=='wrong_kl_keys':record['arms'][0]['full_vocab_kl']['sham']={'wrong':0.}
    elif mutation=='missing_arm':record['arms'].pop()
    elif mutation=='not_run':record['arms'][-1]['generation']={'state':'not_run'}
    elif mutation=='proof':record['technical_checks']['plan_sha256']='wrong'
    elif mutation=='seed':record['arms'][0]['generation']['record']['seed']=1
    elif mutation=='missing_logits':del arrays['prefill_b7_within_transfer_2_logits']
    elif mutation=='shape':arrays['prefill_sham_logits']=np.zeros(2,np.float32)
    elif mutation=='dtype':arrays['prefill_sham_logits']=arrays['prefill_sham_logits'].astype(np.float64)
    else:arrays['unexpected']=np.ones(1,np.float32)
    with pytest.raises(ValueError):C.validate_record(record,'plan');C.validate_arrays(record,arrays)


def test_one_atomic_all_nine_checkpoint_resume_and_no_old_group_mix(tmp_path):
    result,arrays,prior=fixture();plan={'schema':C.SCHEMA,'arms':list(C.ARM_NAMES)}
    digest=hashlib.sha256(C.base.canonical(plan)).hexdigest();C.base.atomic_json(tmp_path/'run_plan.json',plan)
    saved=C.write_checkpoint(tmp_path,result,arrays,digest,{'gpu':'synthetic'},prior)
    assert C.verified(tmp_path,digest,prior)==saved
    with pytest.raises(ValueError,match='overwrite'):C.write_checkpoint(tmp_path,result,arrays,digest,{},prior)
    C.archive(tmp_path);target=tmp_path/'resume';target.mkdir()
    C.restore_resume(tmp_path/'h286_control_answers_results.zip',target,digest,prior)
    assert C.verified(target,digest,prior)==saved
    with pytest.raises(ValueError,match='another'):C.restore_resume(tmp_path/'h286_control_answers_results.zip',tmp_path,'wrong',prior)
    (target/'requests/r013/arrays.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA'):C.verified(target,digest,prior)


def test_missing_five_private_refs_aborts_with_diagnostic_and_no_model(tmp_path):
    status=C.run('missing','missing','missing','missing','missing','missing',tmp_path/'out')
    assert status['state']=='failed' and status['stage']=='reference'
    assert (tmp_path/'out/h286_control_answers_results.zip').exists()
    metrics=json.loads((tmp_path/'out/metrics.json').read_text());assert metrics['weights_loaded'] is False


# Native end-to-end fragments independently authored against the unchanged hook API.
@pytest.fixture(scope="module")
def control_native():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "5.22.2":
        pytest.skip("Requires the original pinned optional transformers==5.22.2 runtime")
    import qwen3_h286_group_runtime as group
    from transformers import Qwen3Config, Qwen3ForCausalLM
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(1793)
        config = Qwen3Config(vocab_size=41, hidden_size=16, intermediate_size=64,
                            num_hidden_layers=28, num_attention_heads=2,
                            num_key_value_heads=1, head_dim=8, eos_token_id=0,
                            bos_token_id=1, pad_token_id=0)
        config._attn_implementation = "sdpa"
        model = Qwen3ForCausalLM(config).float().eval().requires_grad_(False)
    model.generation_config.eos_token_id = 0

    class Tokenizer:
        def decode(self, values, **_):
            return " ".join(str(int(value)) for value in values)

        def __call__(self, text, **_):
            return {"input_ids": [3 if text == "gold" else 4]}

    coordinates = tuple((7, x) for x in range(9)) + tuple((14, x) for x in range(25))
    reference = SimpleNamespace(coordinates=coordinates)
    stats = {"refw": {"c_mu": [0.] * len(coordinates), "c_sd": [1.] * len(coordinates)}}
    request = {"request_id": "r013", "item_id": "synthetic_r013", "pair_id": "synthetic_pair",
               "formatted_prompt": "synthetic native-test prompt",
               "formatted_prompt_sha256": hashlib.sha256(b"synthetic native-test prompt").hexdigest(),
               "prompt_ids": [1, 7, 3, 2], "answer": "gold",
               "answer_ids": [3], "pulse_positions": [3],
               "prefill_capture_positions": [3], "generation_capture_positions": [0, 1, 2, 3],
               "group_item": {"semantic_alternative": "foil"}}
    ids = torch.tensor([request["prompt_ids"]])
    # Test-only model construction: one positive EOS readout, every competing
    # row exactly zero. All nine real cap128 generations then end after one EOS
    # instead of bypassing/stubbing the production decoder or resource bounds.
    with torch.no_grad(), group.previous.attention_context(torch, False):
        last = model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state[0, -1]
        model.lm_head.weight.zero_()
        model.lm_head.weight[0].copy_(last)
    engine = {"model": model, "tokenizer": Tokenizer()}
    controls = group.make_controls(coordinates, 64)
    yield _control_runtime(), torch, engine, reference, stats, request, controls
    torch.set_num_threads(old_threads)


def _control_hook_count(model):
    return sum(len(module._forward_pre_hooks) + len(module._forward_hooks)
               for module in model.modules())


_CONTROL_NAMES = (
    "sham", "other_block_108", "b7_transfer",
    "b7_outside_transfer_0", "b7_outside_transfer_1", "b7_outside_transfer_2",
    "b7_within_transfer_0", "b7_within_transfer_1", "b7_within_transfer_2",
)


def _control_array_hash(value):
    import hashlib
    return hashlib.sha256(np.asarray(value).tobytes(order="C")).hexdigest()


def _control_synthetic_prior(runtime, state, engine, reference, stats, request):
    """Derive only synthetic continuity evidence from actual tiny model calls."""
    import hashlib
    group = runtime.group
    arrays = state["arrays"]
    prior = {"request": dict(request), "arm_names": list(_CONTROL_NAMES), "arms": {},
             "baseline_array_sha256": {}, "baseline_array_shapes": {}}
    for name in _CONTROL_NAMES:
        row = state["rows"][name]
        prefix = f"prefill_{name}_"
        selected = {key[len(prefix):]: value for key, value in arrays.items()
                    if key.startswith(prefix) and key[len(prefix):] != "logits"}
        assert set(selected) == {"raw_z", "cett", "zcett", "down_output_norm_raw"}
        entry = {"spec": row["spec"], "first_token_id": row["top10"][0]["token_id"],
                 "logits_sha256": row["logits_sha256"], "recorded_prefill": dict(row),
                 "prefill_array_sha256": {key: _control_array_hash(value) for key, value in selected.items()},
                 "prefill_array_shapes": {key: list(value.shape) for key, value in selected.items()},
                 "prefill_array_dtypes": {key: str(value.dtype) for key, value in selected.items()},
                 "generation": {"state": "not_run"}}
        if name in _CONTROL_NAMES[:3]:
            # This is the exact previous decoder at the actual cap 128. EOS0
            # makes it cheap without monkeypatching any generation function.
            with group.previous.attention_context(__import__("torch"), False):
                generation, gen_arrays = group.run_generation(
                    engine["model"], engine["tokenizer"], request, reference.coordinates, stats,
                    row["spec"], state["patches"].get(name), efficient_gpu=False)
            assert generation["selected_token_ids"] == [0]
            assert generation["first_step_logits_sha256"] == row["logits_sha256"]
            entry["generation"] = {
                "state": "complete", "record": {key: value for key, value in generation.items() if key != "steps"},
                "steps_sha256": hashlib.sha256(group.base.canonical(generation["steps"])).hexdigest(),
                "gen_array_sha256": {key: _control_array_hash(value) for key, value in gen_arrays.items()},
                "gen_array_shapes": {key: list(value.shape) for key, value in gen_arrays.items()},
                "gen_array_dtypes": {key: str(value.dtype) for key, value in gen_arrays.items()},
            }
        prior["arms"][name] = entry
    for name, donor in state["donors"].items():
        snapshot = {"logits": donor["logits"].numpy()}
        snapshot.update({f"z_block_{block}": value.numpy() for block, value in donor["z"].items()})
        for field, value in snapshot.items():
            key = f"baseline_{name}_{field}"
            prior["baseline_array_sha256"][key] = _control_array_hash(value)
            prior["baseline_array_shapes"][key] = list(value.shape)
    return prior


def test_control_native_collect_nine_prefills_and_exact_donor_readout(control_native):
    runtime, torch, engine, reference, stats, request, controls = control_native
    model = engine["model"]
    params = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    state = runtime.collect_prefills(engine, reference, stats, request, controls, efficient_gpu=False)
    assert set(state["donors"]) == {"sham", "other_block_108"}
    assert set(state["rows"]) == set(_CONTROL_NAMES)
    assert all(state["checks"][key] is True for key in
               ("native_clean_logits", "identity_sham", "identity_pulse", "hook_cleanup"))
    for name in _CONTROL_NAMES:
        row = state["rows"][name]
        assert row["state"] == "complete"
        assert row["top10"][0]["token_id"] == 0
        assert row["top10"][0]["logprob"] > row["top10"][1]["logprob"]
        assert state["arrays"][f"prefill_{name}_logits"].shape == (41,)
        assert state["arrays"][f"prefill_{name}_raw_z"].shape == (1, 1, 34)
        assert row["logits_sha256"] == _control_array_hash(state["arrays"][f"prefill_{name}_logits"])
    # Fixed G7 transfer assigns pulse donor values directly to the nine H rows.
    donor = state["donors"]["other_block_108"]["z"][7][:, :, controls[7]["group"]]
    assert np.array_equal(state["arrays"]["prefill_b7_transfer_raw_z"][:, :, :9], donor.numpy())
    assert not np.array_equal(state["arrays"]["prefill_b7_transfer_raw_z"][:, :, :9],
                              state["arrays"]["prefill_sham_raw_z"][:, :, :9])
    assert _control_hook_count(model) == 0
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(params[name], value) for name, value in model.state_dict().items())


def test_control_native_complete_nine_real_eos_generations_with_prior_binding(control_native, monkeypatch):
    runtime, torch, engine, reference, stats, request, controls = control_native
    model = engine["model"]
    state = runtime.collect_prefills(engine, reference, stats, request, controls, efficient_gpu=False)
    prior = _control_synthetic_prior(runtime, state, engine, reference, stats, request)
    rng = torch.random.get_rng_state().clone()
    calls = []
    original_forward = model.forward

    def counted_forward(*args, **kwargs):
        calls.append((tuple(kwargs["input_ids"].shape), kwargs.get("past_key_values")))
        return original_forward(*args, **kwargs)

    # A forward wrapper counts actual calls without installing an observer hook;
    # the production native-clean check correctly requires hook_count == 0.
    monkeypatch.setattr(model, "forward", counted_forward)
    # CPU SDPA's default fused backend can round differently from the MATH
    # backend explicitly used by collect_prefills. Hold the same backend across
    # this identity test; production pins EFFICIENT_ATTENTION on the GPU.
    with runtime.group.previous.attention_context(torch, False):
        result, arrays = runtime.compute_request(engine, reference, stats, request, controls, prior, efficient_gpu=False)
    assert len(calls) == 21
    assert all(shape == (1, 4) and cache is None for shape, cache in calls)
    rows = {row["name"]: row for row in result["arms"]}
    assert set(rows) == set(_CONTROL_NAMES)
    assert result["generation_count"] == 9
    assert result["total_prefill_forwards"] == 21
    assert all(result["technical_checks"]["checks"][key] is True for key in runtime.PROOF_CHECKS)
    assert all(result["baseline_numeric_diagnostics"].values())
    for name in _CONTROL_NAMES:
        row = rows[name]
        assert row["generation"]["state"] == "complete"
        generation = row["generation"]["record"]
        assert generation["max_new_tokens"] == 128
        assert generation["seed"] == runtime.group.base.SEED + 13
        assert generation["selected_token_ids"] == [0]
        assert generation["generated_token_ids"] == []
        assert generation["stop_reason"] == "eos" and not generation["truncated"]
        assert generation["prefill_forward_count"] == 1 and generation["decode_forward_count"] == 0
        assert generation["first_step_logits_sha256"] == row["logits_sha256"] == prior["arms"][name]["logits_sha256"]
        assert generation["pulse_fired"] == generation["capture_fired"] == 1
        comparison = result["continuity"][name]
        assert comparison["prior_first_token_equal"]
        assert comparison["prefill_logits_exact"] and comparison["prefill_arrays_exact"] and comparison["prefill_shapes_exact"]
        if name in _CONTROL_NAMES[:3]:
            assert comparison["prior_full_tokens_protocol_equal"] and comparison["prior_generation_first_logits_exact"]
            assert comparison["prior_steps_exact"] and comparison["prior_generation_arrays_exact"]
        assert arrays[f"generation_{name}_raw_z"].shape == (1, 4, 34)
        assert arrays[f"generation_{name}_residual_layer_27"].shape == (1, 4, 16)
        for field in ("raw_z", "cett", "zcett", "down_output_norm_raw"):
            assert np.array_equal(arrays[f"generation_{name}_{field}"][:, -1:, :],
                                  arrays[f"prefill_{name}_{field}"])
    assert all(np.asarray(value).dtype == np.float32 and np.isfinite(value).all() for value in arrays.values())
    assert _control_hook_count(model) == 0
    assert torch.equal(rng, torch.random.get_rng_state())
