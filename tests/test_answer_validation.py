"""Native tiny Qwen3 cache/generation checks; no pretrained downloads or GPU."""
from __future__ import annotations

import contextlib
import hashlib
import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'notebooks'))
torch = pytest.importorskip('torch')
from attention_probe.answer_validation import classify_answer, run_generation_arm


class SmallTokenizer:
    pieces: ClassVar[dict[int, str]] = {0:'<pad>', 1:'<bos>', 2:'<eos>', 3:'<other_eos>', 7:'blue', 8:'Blue',
              9:'15', 10:'.', 11:'5', 12:'Answer', 13:' green', 14:'B', 15:'Wrong'}

    def decode(self, ids, *, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        assert skip_special_tokens is False and clean_up_tokenization_spaces is False
        return ''.join(self.pieces.get(int(t), f't{t}') for t in ids)

    def __call__(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return {'input_ids': {'blue':[7], '5':[11], '15':[9], 'B':[14]}[text]}


class NoTeacherForcedFields(dict):
    forbidden: ClassVar[set[str]] = {'ids','input_ids','answer_ids','capture_positions','pulse_positions','answer_logit_positions'}

    def __getitem__(self, key):
        assert key not in self.forbidden, f'Generation accessed old teacher-forced field: {key}'
        return super().__getitem__(key)

    def get(self, key, default=None):
        assert key not in self.forbidden
        return super().get(key, default)


@pytest.fixture(scope='module')
def native():
    transformers = pytest.importorskip('transformers')
    if transformers.__version__ != '5.22.2':
        pytest.skip('Requires the pinned optional transformers==5.22.2 runtime')
    from transformers import Qwen3Config, Qwen3ForCausalLM
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(871)
        config = Qwen3Config(vocab_size=48, hidden_size=16, intermediate_size=16,
            num_hidden_layers=28, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, eos_token_id=None, bos_token_id=1, pad_token_id=0)
        config._attn_implementation = 'sdpa'
        model = Qwen3ForCausalLM(config).float().eval().requires_grad_(False)
    coordinates = tuple((l,j) for l in range(28) for j in range(16))[:286]
    stats = {'refw': {'c_mu':[0.]*286, 'c_sd':[1.]*286}}
    yield model, SmallTokenizer(), coordinates, stats
    torch.set_num_threads(previous)


def request(answer='blue'):
    return NoTeacherForcedFields(request_id='r007', prompt_ids=[1,4,5,6,8], answer=answer,
        ids='FORBIDDEN', input_ids='FORBIDDEN', answer_ids='FORBIDDEN',
        capture_positions='FORBIDDEN', pulse_positions='FORBIDDEN', answer_logit_positions='FORBIDDEN')


SHAM = {'name': 'sham', 'block': 14, 'beta': 1., 'mode': 'scale'}
SOFT = {'name': 'soft_108', 'block': 14, 'beta': 1/1.08, 'mode': 'scale'}


def hook_count(model):
    return sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules())


@contextlib.contextmanager
def eos_config(model, generation=None, config=None):
    old_generation, old_config = model.generation_config.eos_token_id, model.config.eos_token_id
    model.generation_config.eos_token_id = generation
    model.config.eos_token_id = config
    try:
        yield
    finally:
        model.generation_config.eos_token_id = old_generation
        model.config.eos_token_id = old_config


@contextlib.contextmanager
def scripted_choices(model, choices, *, fail_at=None, nan_at=None):
    """Keep real Qwen3 attention/cache/capture, override only output logits."""
    calls = []
    def hook(_model, args, kwargs, output):
        index = len(calls)
        calls.append(kwargs['input_ids'].detach().cpu().clone())
        if index == fail_at:
            raise RuntimeError('Injected forward failure')
        output.logits = torch.full_like(output.logits, -20.)
        output.logits[..., choices[min(index, len(choices)-1)]] = 20.
        if index == nan_at:
            output.logits[..., 0] = float('nan')
        return output
    handle = model.register_forward_hook(hook, with_kwargs=True)
    try:
        yield calls
    finally:
        handle.remove()


def test_native_greedy_generate_matches_manual_sham_and_no_pulse(native):
    from transformers import GenerationConfig
    model, tokenizer, coords, stats = native
    q = request()
    ids = torch.tensor([q['prompt_ids']])
    before_rng = torch.random.get_rng_state().clone()
    before_parameters = [(id(p),p.data_ptr(),p._version) for p in model.parameters()]
    with eos_config(model), torch.no_grad():
        generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=6,
            use_cache=True, eos_token_id=None, pad_token_id=0, bos_token_id=1)
        vanilla = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
            generation_config=generation)
        raw_prefill = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True).logits[0,-1].float()
        sham, arrays = run_generation_arm(model, tokenizer, q, coords, stats, SHAM,
            efficient_gpu=False, max_new_tokens=6)
        baseline, base_arrays = run_generation_arm(model, tokenizer, q, coords, stats, None,
            efficient_gpu=False, max_new_tokens=6)
    assert sham['generated_token_ids'] == vanilla[0,ids.shape[1]:].tolist() == baseline['generated_token_ids']
    digest = hashlib.sha256(raw_prefill.numpy().tobytes()).hexdigest()
    assert sham['first_step_logits_sha256'] == baseline['first_step_logits_sha256'] == digest
    assert sham['pulse_fired'] == sham['capture_fired'] == baseline['capture_fired'] == 1
    assert baseline['pulse_fired'] == 0
    assert sham['prefill_forward_count'] == 1 and sham['decode_forward_count'] == 5
    assert sham['prefill_capture_positions'] == [1,2,3,4] and sham['pulse_positions'] == [4]
    assert sham['final_cache_length'] == 10  # Last capped token was selected, not fed back.
    assert sham['truncated'] and sham['classification'] == 'truncated'
    assert arrays['raw_z'].shape == (1,4,286)
    assert set(arrays) == {'raw_z','cett','zcett','down_output_norm_raw',*[f'residual_layer_{l}' for l in (3,7,14,27)]}
    for name in arrays:
        np.testing.assert_array_equal(arrays[name], base_arrays[name])
        assert np.isfinite(arrays[name]).all()
    assert all(len(s['top_tokens']) == 10 for s in sham['steps'])
    np.testing.assert_allclose(sham['first_token_gold']['logprob'], raw_prefill.log_softmax(-1)[7].item(), atol=0, rtol=0)
    assert sham['first_token_gold']['rank'] == int((raw_prefill>raw_prefill[7]).sum())+1
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert before_parameters == [(id(p),p.data_ptr(),p._version) for p in model.parameters()]
    assert hook_count(model) == 0


def test_pulse_only_prefill_never_gold_inputs_and_eos_logging(native):
    model, tokenizer, coords, stats = native
    q = request()
    observed = []
    def spy(_module, _args, kwargs):
        observed.append((kwargs['input_ids'].tolist(), len(model.model.layers[14].self_attn.q_norm._forward_hooks),
                         len(model.model.layers[0].mlp.down_proj._forward_pre_hooks)))
    handle = model.register_forward_pre_hook(spy, with_kwargs=True)
    try:
        with eos_config(model, generation=[2,3]), scripted_choices(model, [7,10,2]):
            result, arrays = run_generation_arm(model, tokenizer, q, coords, stats, SOFT, efficient_gpu=False)
    finally:
        handle.remove()
    assert [x[0] for x in observed] == [[[1,4,5,6,8]], [[7]], [[10]]]
    assert observed[0][1:] == (1,1) and all(x[1:] == (0,0) for x in observed[1:])
    assert result['pulse_fired'] == result['capture_fired'] == 1
    assert result['generated_token_ids'] == [7,10]
    assert result['selected_token_ids'] == [7,10,2]
    assert result['generated_text'] == 'blue.'
    assert result['steps'][-1]['is_eos'] and result['stop_reason'] == 'eos' and not result['truncated']
    assert result['classification'] == 'format_equivalent'
    assert result['num_steps'] == 3 and result['decode_forward_count'] == 2 and result['final_cache_length'] == 7
    assert result['seed'] == 20261124
    assert arrays['raw_z'].shape == (1,4,286) and hook_count(model) == 0


def test_changing_known_answer_does_not_change_generation_or_capture(native):
    model, tokenizer, coords, stats = native
    outputs = []
    initial_caches = []
    def spy(_module, _args, kwargs):
        if kwargs['input_ids'].shape[1] > 1:
            initial_caches.append(kwargs.get('past_key_values'))
    handle = model.register_forward_pre_hook(spy, with_kwargs=True)
    try:
        for answer in ('blue','5'):
            with eos_config(model, generation=2), scripted_choices(model, [7,2]):
                outputs.append(run_generation_arm(model, tokenizer, request(answer), coords, stats, SOFT, efficient_gpu=False))
    finally:
        handle.remove()
    assert initial_caches == [None,None]
    assert outputs[0][0]['generated_token_ids'] == outputs[1][0]['generated_token_ids'] == [7]
    assert outputs[0][0]['classification'] == 'exact' and outputs[1][0]['classification'] == 'noncanonical'
    assert outputs[0][0]['first_token_gold']['token_id'] != outputs[1][0]['first_token_gold']['token_id']
    for key in outputs[0][1]:
        np.testing.assert_array_equal(outputs[0][1][key], outputs[1][1][key])


def test_eos_fallback_and_immediate_eos_not_emitted(native):
    model, tokenizer, coords, stats = native
    with eos_config(model, generation=None, config=[2,3]), scripted_choices(model, [3]):
        result, _ = run_generation_arm(model, tokenizer, request(), coords, stats, None, efficient_gpu=False)
    assert result['eos_source'] == 'config' and result['eos_token_ids'] == [2,3]
    assert result['generated_token_ids'] == [] and result['generated_text'] == ''
    assert result['selected_token_ids'] == [3] and result['num_steps'] == 1
    assert result['decode_forward_count'] == 0 and result['stop_reason'] == 'eos'
    assert result['classification'] == 'noncanonical' and not result['truncated']


def test_cap_even_current_exact_prefix_is_truncated(native):
    model, tokenizer, coords, stats = native
    with eos_config(model, generation=2), scripted_choices(model, [7]):
        result, _ = run_generation_arm(model, tokenizer, request(), coords, stats, SHAM,
            efficient_gpu=False, max_new_tokens=1, top_k=3)
    assert result['generated_text'] == 'blue' and result['classification'] == 'truncated'
    assert result['classification_details']['surface_classification'] == 'exact'
    assert result['num_steps'] == 1 and len(result['steps'][0]['top_tokens']) == 3
    assert result['stop_reason'] == 'max_new_tokens'


@pytest.mark.parametrize('generated,expected,kind',[
    ('5','5','exact'), (' 5. ','5','format_equivalent'), ('BLUE','blue','format_equivalent'),
    ('b!','B','format_equivalent'), ('5.0','5','noncanonical'), ('+5','5','noncanonical'),
    ('05','5','noncanonical'), ('The answer is 5','5','noncanonical'),
    ('5 or 6','5','noncanonical'), ('5\n5','5','noncanonical'), ('5..','5','noncanonical'),
])
def test_strict_classification_never_rescues_explanations(generated, expected, kind):
    assert classify_answer(generated, expected)['classification'] == kind
    assert classify_answer(generated, expected, truncated=True)['classification'] == 'truncated'


@pytest.mark.parametrize('failure', ['prefill_nan','decode_exception'])
def test_cleanup_on_prefill_or_decode_failure(native, failure):
    model, tokenizer, coords, stats = native
    kwargs = {'nan_at':0} if failure=='prefill_nan' else {'fail_at':1}
    with eos_config(model, generation=2), scripted_choices(model, [7,7], **kwargs):
        with pytest.raises((ValueError,RuntimeError)):
            run_generation_arm(model, tokenizer, request(), coords, stats, SOFT, efficient_gpu=False)
        # The external test hook remains; every production hook has been removed.
        assert hook_count(model) == 1
    assert hook_count(model) == 0


@pytest.mark.parametrize('option,value',[('max_new_tokens',0),('max_new_tokens',49),('max_new_tokens',True),('top_k',0),('top_k',11)])
def test_resource_bounds_fail_before_hooks(native, option, value):
    model, tokenizer, coords, stats = native
    with pytest.raises(ValueError):
        run_generation_arm(model, tokenizer, request(), coords, stats, SHAM, efficient_gpu=False, **{option:value})
    assert hook_count(model) == 0
