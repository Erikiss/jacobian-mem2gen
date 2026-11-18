"""Native tiny random Qwen3 tests; no pretrained weights or GPU required."""
import pytest
import torch

from attention_probe.temporal_cache import (
    AbsoluteCapture,
    cache_length,
    cache_manifest,
    clone_cache,
    forward_capture,
    swap_cache_positions,
)


@pytest.fixture(scope='module')
def tiny():
    tf=pytest.importorskip('transformers')
    if tf.__version__!='5.22.2':
        pytest.skip('Pinned native cache API is transformers==5.22.2')
    from transformers import Qwen3Config, Qwen3ForCausalLM
    threads=torch.get_num_threads();torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(2031)
        config=Qwen3Config(vocab_size=64,hidden_size=24,intermediate_size=24,
                          num_hidden_layers=20,num_attention_heads=3,num_key_value_heads=1,head_dim=8)
        config._attn_implementation='sdpa'
        model=Qwen3ForCausalLM(config).float().eval().requires_grad_(False)
    coordinates=tuple((block,channel) for block in range(20) for channel in (1,3))
    stats={'refw':{'c_mu':[.01]*len(coordinates),'c_sd':[.2]*len(coordinates)}}
    yield model,coordinates,stats
    torch.set_num_threads(threads)


def call(tiny,ids,**kwargs):
    return forward_capture(*[tiny[0],ids,tiny[1],tiny[2]],**kwargs)


def no_hooks(model):
    return sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules())==0


def native(model,ids,pulse=None):
    from contextlib import nullcontext

    from attention_probe.interventions import AttentionPulse
    ids=torch.tensor([ids])
    ctx=(AttentionPulse(model,14,[4],beta=1/1.08) if pulse else nullcontext())
    with torch.no_grad(),torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),ctx:
        return model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=True)


def test_cached_capture_absolute_positions_and_native_rebuild_parity(tiny):
    model,coordinates,stats=tiny
    prompt=[1,2,3,4,5]; continuation=[6,7,8,9,10,11,12,13]*4
    rng=torch.random.get_rng_state().clone()
    for pulsed in (False,True):
        pulse={'block':14,'beta':1/1.08,'seed':17} if pulsed else None
        step=call(tiny,prompt,pulse=pulse)
        assert step.evidence['pulse_fired']==int(pulsed)
        consumed=0
        for count in (1,7,23,1):
            previous=cache_manifest(step.cache)
            old=step.cache
            step=call(tiny,continuation[consumed:consumed+count],cache=old)
            consumed+=count
            assert cache_manifest(old)==previous
            assert step.evidence['absolute_capture_positions']==[len(prompt)+consumed-1]
            assert step.evidence['cache_output_length']==len(prompt)+consumed
            assert step.evidence['pulse_fired']==0
            assert step.arrays['raw_z'].shape==(1,1,len(coordinates))
            assert torch.isfinite(step.arrays['zcett']).all()
            expected=(step.arrays['cett']-.01)/.2
            torch.testing.assert_close(step.arrays['zcett'],expected,rtol=0,atol=0)
            if consumed in (8,32):
                full=native(model,prompt+continuation[:consumed],pulse=pulsed)
                torch.testing.assert_close(step.logits,full.logits[:,-1:].cpu(),rtol=2e-5,atol=2e-6)
                full_via_api=call(tiny,prompt+continuation[:consumed],
                                  pulse={'block':14,'beta':1/1.08,'positions':[len(prompt)-1]} if pulsed else None)
                torch.testing.assert_close(full_via_api.logits,full.logits[:,-1:].cpu(),rtol=0,atol=0)
                assert full_via_api.evidence['pulse_absolute_positions']==([len(prompt)-1] if pulsed else [])
                for a,b in zip(step.cache.layers,full.past_key_values.layers):
                    torch.testing.assert_close(a.keys,b.keys,rtol=2e-5,atol=2e-6)
                    torch.testing.assert_close(a.values,b.values,rtol=2e-5,atol=2e-6)
                # Full forward capture at the same absolute token uses exactly
                # the same readout definition as the cached one-token/chunk path.
                with (
                    torch.no_grad(),
                    torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),
                    AbsoluteCapture(model,coordinates,[len(prompt)+consumed-1],stats) as observed,
                ):
                    native(model,prompt+continuation[:consumed],pulse=pulsed)
                for key,value in observed.arrays().items():
                    torch.testing.assert_close(step.arrays[key],value,rtol=3e-5,atol=3e-6)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert no_hooks(model)


def test_pulse_causal_cache_pattern_swap_and_donors_survive_future_forwards(tiny):
    prompt=[1,2,3,4,5];tail=[6,7,8,9,10,11,12,13]
    clean0=call(tiny,prompt)
    pulse0=call(tiny,prompt,pulse={'block':14,'beta':1/1.08})
    clean=call(tiny,tail,cache=clean0.cache)
    pulse=call(tiny,tail,cache=pulse0.cache)
    L=len(prompt)
    for block,(a,b) in enumerate(zip(clean.cache.layers,pulse.cache.layers)):
        for field in ('keys','values'):
            x,y=getattr(a,field),getattr(b,field)
            assert torch.equal(x[...,:L-1,:],y[...,:L-1,:])
            if block<=14:
                assert torch.equal(x,y)
            if block==15:
                assert torch.equal(x[...,L:,:],y[...,L:,:])
    assert any(not torch.equal(a.keys,b.keys) for a,b in zip(clean.cache.layers[15:],pulse.cache.layers[15:]))
    clean_hash,pulse_hash=cache_manifest(clean.cache),cache_manifest(pulse.cache)
    old_logits=pulse.logits.clone()
    for positions in ([L-1],list(range(L,L+8)),list(range(L-1,L+8))):
        swapped,evidence=swap_cache_positions(pulse.cache,clean.cache,layer_indices=range(15,20),absolute_positions=positions)
        assert evidence['donor_unchanged'] and evidence['receiver_unchanged'] and evidence['untouched_slots_exact']
        assert torch.equal(pulse.logits,old_logits)
        assert cache_length(swapped)==L+8
        step=call(tiny,[14],cache=swapped)
        assert step.evidence['absolute_capture_positions']==[L+8]
        assert step.evidence['logit_predicts_absolute_positions']==[L+9]
        assert cache_manifest(clean.cache)==clean_hash and cache_manifest(pulse.cache)==pulse_hash
        if len(positions)==9:
            # Both causal sources removed: all remaining cache slots equal clean.
            for a,b in zip(swapped.layers,clean.cache.layers):
                assert torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values)
            rebuilt=call(tiny,[14],cache=clean.cache)
            assert torch.equal(step.logits,rebuilt.logits)
    identity,_=swap_cache_positions(clean.cache,clean.cache,layer_indices=[15],absolute_positions=[L-1])
    assert cache_manifest(identity)==clean_hash
    assert no_hooks(tiny[0])


def test_clone_preserves_metadata_no_shared_storage_and_no_capture_path(tiny):
    original=call(tiny,[1,2,3])
    original.cache.test_counter={'consumed':3}
    original.cache.layers[0].test_counter=7
    cloned=clone_cache(original.cache)
    assert cloned.test_counter=={'consumed':3} and cloned.layers[0].test_counter==7
    cloned.test_counter['consumed']=99
    assert original.cache.test_counter['consumed']==3
    for a,b in zip(original.cache.layers,cloned.layers):
        assert a.keys.data_ptr()!=b.keys.data_ptr() and a.values.data_ptr()!=b.values.data_ptr()
        assert torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values)
    captured=call(tiny,[4],cache=original.cache)
    bare=call(tiny,[4],cache=original.cache,capture=False)
    assert bare.arrays=={} and bare.evidence['capture_fired']==0
    assert torch.equal(captured.logits,bare.logits)
    assert cache_manifest(captured.cache)==cache_manifest(bare.cache)


def test_fixed_post18_patch_is_once_and_capture_observes_patched_output(tiny):
    pre=call(tiny,[1,2,3])
    base=call(tiny,[4],cache=pre.cache)
    delta=torch.linspace(-.001,.001,tiny[0].config.hidden_size)
    patched=call(tiny,[4],cache=pre.cache,residual_patch={'block':18,'delta':delta})
    assert patched.evidence['residual_patch']['fired_count']==1
    assert torch.equal(base.arrays['residual_17'],patched.arrays['residual_17'])
    assert torch.equal(base.arrays['residual_18']+delta,patched.arrays['residual_18'])
    coords=torch.tensor([l for l,_ in tiny[1]])
    assert torch.equal(base.arrays['raw_z'][...,coords<=18],patched.arrays['raw_z'][...,coords<=18])
    for a,b in zip(base.cache.layers[:19],patched.cache.layers[:19]):
        assert torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values)
    assert not torch.equal(base.logits,patched.logits)
    assert no_hooks(tiny[0])


@pytest.mark.parametrize('bad', [[],[1,1],[-1],[True],[99]])
def test_bad_swap_positions_fail_explicitly(tiny,bad):
    step=call(tiny,[1,2,3])
    with pytest.raises(ValueError):
        swap_cache_positions(step.cache,step.cache,layer_indices=[15],absolute_positions=bad)


def test_invalid_cache_and_absolute_positions_cleanup(tiny):
    step=call(tiny,[1,2,3])
    bad=clone_cache(step.cache)
    bad.layers[1].keys=bad.layers[1].keys[...,:2,:]
    bad.layers[1].values=bad.layers[1].values[...,:2,:]
    with pytest.raises(ValueError,match='lengths'):
        clone_cache(bad)
    bad=clone_cache(step.cache);bad.offloading=True
    with pytest.raises(ValueError,match='non-offloaded'):
        clone_cache(bad)
    with pytest.raises(ValueError,match='current chunk'):
        call(tiny,[4],cache=step.cache,capture_positions=[0])
    with pytest.raises(ValueError,match='prefill-only'):
        call(tiny,[4],cache=step.cache,pulse={'block':14})
    with pytest.raises(ValueError,match='one original'):
        call(tiny,[1,2,3],pulse={'block':14,'positions':[0,2]})
    with pytest.raises(ValueError,match='Position IDs'), AbsoluteCapture(tiny[0],tiny[1],[3],tiny[2]):
        tiny[0](input_ids=torch.tensor([[4]]),past_key_values=clone_cache(step.cache),position_ids=torch.tensor([[0]]),use_cache=True)
    assert no_hooks(tiny[0])


def test_hook_cleanup_on_native_forward_failure(tiny,monkeypatch):
    def fail(*args,**kwargs):
        raise RuntimeError('injected failure')
    monkeypatch.setattr(tiny[0].model.layers[17].mlp,'forward',fail)
    with pytest.raises(RuntimeError,match='injected'):
        call(tiny,[1,2,3],pulse={'block':14},residual_patch={'block':18,'delta':torch.zeros(24)})
    assert no_hooks(tiny[0])
