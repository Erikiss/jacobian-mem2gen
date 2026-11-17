"""CPU-only random-weight Qwen3-shaped tests; no weights or tokenizer downloaded."""
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from attention_probe.interventions import AttentionPulse, CoordinateCapture


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-10) * self.weight


class TinyAttention(nn.Module):
    def __init__(self, width=8, heads=2):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.head_dim), RMSNorm(self.head_dim)

    def forward(self, x):
        b, t, _ = x.shape
        shape = (b, t, self.heads, self.head_dim)
        # Same q_norm layout/order as HF Qwen3: normalization before transpose.
        q = self.q_proj(x).reshape(shape)
        if getattr(self, "q_norm", None) is not None:
            q = self.q_norm(q)
        q = q.transpose(1, 2)
        k = self.k_norm(self.k_proj(x).reshape(shape)).transpose(1, 2)
        v = self.v_proj(x).reshape(shape).transpose(1, 2)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        causal = torch.ones(t, t, dtype=torch.bool).tril()
        logits = logits.masked_fill(~causal, -torch.inf)
        prob = logits.softmax(-1)
        self.last_q, self.last_logits, self.last_prob = (z.detach().clone() for z in (q, logits, prob))
        y = (prob @ v).transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(y)


class TinyMLP(nn.Module):
    def __init__(self, width=8, intermediate=128):
        super().__init__()
        self.gate_proj = nn.Linear(width, intermediate, bias=False)
        self.up_proj = nn.Linear(width, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, width, bias=False)

    def forward(self, x):
        z = torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        self.last_z = z.detach().clone()
        return self.down_proj(z)


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttention()
        self.mlp = TinyMLP()

    def forward(self, x):
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        self.last_residual = x.detach().clone()
        return (x,)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3", num_attention_heads=2, head_dim=4)
        self.embed = nn.Embedding(41, 8)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyLayer() for _ in range(3)])
        self.head = nn.Linear(8, 41, bias=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, cache_position=None):
        x = self.embed(input_ids)
        for layer in self.model.layers:
            x = layer(x)[0]
        return self.head(x)


@pytest.fixture
def model():
    with torch.random.fork_rng():
        torch.manual_seed(31)
        return TinyModel().eval()


@pytest.fixture
def ids():
    return torch.tensor([[3, 8, 13, 5, 21], [4, 11, 1, 6, 24]])


def coordinates():
    # Exactly 286 distinct channels; non-sorted order tests stable feature identity.
    return tuple(reversed([(i//128, i % 128) for i in range(286)]))


def hook_count(model):
    return sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules())


def test_beta_one_sham_is_bitwise_exact_and_leaves_no_hooks(model, ids):
    baseline = model(ids).detach()
    count = hook_count(model)
    with AttentionPulse(model, 1, [3, 1], beta=1) as pulse:
        observed = model(ids).detach()
    assert pulse.fired_count == 1
    assert torch.equal(baseline, observed)
    assert hook_count(model) == count
    assert torch.equal(model(ids), baseline)


@pytest.mark.parametrize("beta", [.95, .9])
def test_query_scaling_is_after_norm_and_scales_logits(model, ids, beta):
    model(ids)
    attn = model.model.layers[1].self_attn
    baseline = attn.last_logits.clone()
    base_q = attn.last_q.clone()
    baseline_entropy = -(attn.last_prob[:, :, 3, :4].clamp_min(1e-30).log()*attn.last_prob[:, :, 3, :4]).sum(-1)
    with AttentionPulse(model, 1, [3], beta=beta):
        model(ids)
    assert torch.allclose(attn.last_q[:, :, 3], base_q[:, :, 3]*beta, atol=1e-6)
    assert torch.allclose(attn.last_logits[:, :, 3, :4], baseline[:, :, 3, :4]*beta, atol=1e-6)
    entropy = -(attn.last_prob[:, :, 3, :4].clamp_min(1e-30).log()*attn.last_prob[:, :, 3, :4]).sum(-1)
    assert torch.all(entropy >= baseline_entropy)
    assert torch.equal(attn.last_q[:, :, :3], base_q[:, :, :3])


def test_scaling_before_q_norm_would_cancel_but_our_pulse_does_not(model, ids):
    baseline = model(ids).detach()
    attn = model.model.layers[1].self_attn
    handle = attn.q_proj.register_forward_hook(lambda module, args, out: out*.9)
    try:
        before = model(ids).detach()
    finally:
        handle.remove()
    with AttentionPulse(model, 1, [3], beta=.9):
        after = model(ids).detach()
    assert (before-baseline).abs().max() < 1e-6
    assert (after-baseline).abs().max() > 1e-4


def test_zero_query_gives_uniform_causal_attention_at_selected_token(model, ids):
    with AttentionPulse(model, 1, [3], beta=0) as pulse:
        model(ids)
    attn = model.model.layers[1].self_attn
    assert torch.count_nonzero(attn.last_q[:, :, 3]) == 0
    assert torch.equal(attn.last_prob[:, :, 3, :4], torch.full((2,2,4), .25))
    assert torch.count_nonzero(attn.last_prob[:, :, 3, 4:]) == 0
    assert pulse.fired_count == 1


def test_random_delta_matches_each_head_and_preserves_global_rng(model, ids):
    model(ids)
    baseline_q = model.model.layers[1].self_attn.last_q.clone()
    state = torch.random.get_rng_state().clone()
    with AttentionPulse(model, 1, [1, 3], beta=.9, mode="random", seed=712) as pulse:
        model(ids)
    assert torch.equal(state, torch.random.get_rng_state())
    changed = model.model.layers[1].self_attn.last_q
    actual = (changed[:, :, [1,3]]-baseline_q[:, :, [1,3]]).double().norm(dim=-1)
    target = .1*baseline_q[:, :, [1,3]].double().norm(dim=-1)
    assert torch.allclose(actual, target, rtol=2e-6, atol=1e-7)
    assert torch.allclose(pulse.actual_delta_norms, pulse.requested_delta_norms, rtol=2e-6, atol=1e-7)
    assert torch.equal(changed[:, :, [0,2,4]], baseline_q[:, :, [0,2,4]])
    with AttentionPulse(model, 1, [1,3], beta=.9, mode="random", seed=712):
        model(ids)
    assert torch.equal(changed, model.model.layers[1].self_attn.last_q)


def test_zero_random_delta_sham_does_not_draw_randomness(model, ids):
    base = model(ids).detach()
    state = torch.random.get_rng_state().clone()
    with AttentionPulse(model, 1, [3], beta=1, mode="random", seed=42) as pulse:
        actual = model(ids).detach()
    assert torch.equal(actual, base)
    assert torch.equal(state, torch.random.get_rng_state())
    assert not torch.count_nonzero(pulse.actual_delta_norms)


def test_causal_earlier_layers_and_earlier_tokens_unchanged(model, ids):
    with CoordinateCapture(model, coordinates(), range(5), residual_layers=[0,1,2]) as cap0:
        baseline = model(ids).detach()
    with CoordinateCapture(model, coordinates(), range(5), residual_layers=[0,1,2]) as cap1:
        with AttentionPulse(model, 1, [3], beta=.9):
            perturbed = model(ids).detach()
    assert torch.equal(cap0.residuals[0], cap1.residuals[0])
    assert torch.equal(cap0.residuals[1][:,:3], cap1.residuals[1][:,:3])
    assert torch.equal(baseline[:,:3], perturbed[:,:3])
    assert not torch.equal(cap0.residuals[1][:,3], cap1.residuals[1][:,3])


def test_once_even_with_two_full_forwards_and_later_decode(model, ids):
    base = model(ids).detach()
    with AttentionPulse(model, 1, [3], beta=.9) as pulse:
        changed = model(ids).detach()
        untouched = model(ids).detach()
        model(ids[:, :1], past_key_values=SimpleNamespace(get_seq_length=lambda:5), cache_position=torch.tensor([5]))
    assert pulse.fired_count == 1
    assert not torch.equal(changed, base)
    assert torch.equal(untouched, base)


@pytest.mark.parametrize("positional", [False,True])
def test_cached_chunk_is_skipped_before_position_bounds_check(model, ids, positional):
    cache = SimpleNamespace(get_seq_length=lambda:4)
    with AttentionPulse(model, 1, [3], beta=.9) as pulse:
        if positional:
            model(ids[:,:1], None, cache, torch.tensor([4]))
        else:
            model(ids[:,:1], past_key_values=cache, cache_position=torch.tensor([4]))
        assert pulse.fired_count == 0
        model(ids)
    assert pulse.fired_count == 1
    assert pulse.skipped_cached_passes == 1


def test_decode_only_fails_validation_without_applying_pulse(model, ids):
    base = model(ids[:,:1]).detach()
    with pytest.raises(RuntimeError, match="fire once"):
        with AttentionPulse(model, 1, [0], beta=.9) as pulse:
            actual = model(ids[:,:1], past_key_values=SimpleNamespace(get_seq_length=lambda:5)).detach()
    assert pulse.fired_count == 0
    assert torch.equal(actual, base)
    assert hook_count(model) == 0


def test_capture_286_raw_channels_and_residuals_preserves_coordinate_order(model, ids):
    baseline = model(ids).detach()
    expected = torch.stack([model.model.layers[b].mlp.last_z[:,[3,1],j] for b,j in coordinates()], -1)
    with CoordinateCapture(model, coordinates(), [3,1], residual_layers=[2,0]) as cap:
        actual = model(ids).detach()
        values = cap.values.clone()
    assert values.shape == (2,2,286)
    assert torch.equal(values, expected)
    assert torch.equal(actual, baseline)
    assert cap.fired_count == 1
    assert tuple(cap.coordinates) == coordinates()
    for layer in [0,2]:
        assert torch.equal(cap.residuals[layer], model.model.layers[layer].last_residual[:,[3,1]])
    assert "not CETT" in cap.feature_kind
    assert hook_count(model) == 0


def test_capture_once_does_not_overwrite_on_a_second_input(model, ids):
    with CoordinateCapture(model, coordinates(), [3]) as cap:
        model(ids)
        initial = cap.values.clone()
        model(ids.flip(1))
    assert torch.equal(cap.values, initial)


@pytest.mark.parametrize("positions", [[-1],[True],[1,1],[9]])
def test_invalid_positions_fail_closed(model, ids, positions):
    with pytest.raises(ValueError):
        with AttentionPulse(model, 1, positions, beta=.9):
            model(ids)
    assert hook_count(model) == 0


def test_padding_is_rejected_and_hooks_clean_up(model, ids):
    mask = torch.ones_like(ids);mask[1,3]=0
    with pytest.raises(ValueError, match="padding"):
        with AttentionPulse(model, 1, [3], beta=.9):
            model(ids, attention_mask=mask)
    assert hook_count(model) == 0


def test_cleanup_on_forward_failure_does_not_mask_original_error(model, ids):
    def fail(module,args,output):
        raise OSError("deliberate model failure")
    h = model.model.layers[2].register_forward_hook(fail)
    try:
        with pytest.raises(OSError, match="deliberate"):
            with CoordinateCapture(model, coordinates(), [3]):
                with AttentionPulse(model, 1, [3], beta=.9):
                    model(ids)
        assert hook_count(model) == 1
    finally:
        h.remove()
    assert hook_count(model) == 0


def test_nonfinite_residual_is_rejected_before_it_is_reported(model, ids):
    def corrupt(module, args, output):
        value = output[0].clone()
        value[:,3,0] = torch.nan
        return (value,)
    handle = model.model.layers[1].register_forward_hook(corrupt)
    try:
        with pytest.raises(ValueError, match="Nonfinite selected decoder residual"):
            with CoordinateCapture(model,coordinates(),[3],residual_layers=[1]):
                model(ids)
        assert hook_count(model) == 1
    finally:
        handle.remove()
    assert hook_count(model) == 0


def test_architecture_and_missing_q_norm_do_not_silently_fall_back(model, ids):
    del model.model.layers[1].self_attn.q_norm
    with pytest.raises(ValueError, match="fallback is forbidden"):
        AttentionPulse(model, 1, [0], beta=.9)
    model.config.model_type="qwen2"
    with pytest.raises(ValueError, match="Unsupported architecture"):
        AttentionPulse(model, 1, [0], beta=.9)
    # Qwen2 is opt-in only; its q_proj hook does not rely on q_norm.
    pulse=AttentionPulse(model,1,[0],beta=.9,architecture="qwen2")
    assert pulse.hook_module is model.model.layers[1].self_attn.q_proj
    model(ids)
    baseline_q=model.model.layers[1].self_attn.last_q.clone()
    with pulse:
        model(ids)
    assert torch.allclose(model.model.layers[1].self_attn.last_q[:,:,0],baseline_q[:,:,0]*.9)
    assert torch.equal(model.model.layers[1].self_attn.last_q[:,:,1:],baseline_q[:,:,1:])


def test_coordinate_count_bounds_duplicates_and_eval_requirement(model):
    for coords in [[(0,1)], coordinates()+((0,1),), tuple([(0,128)]+list(coordinates()[1:]))]:
        with pytest.raises(ValueError):
            CoordinateCapture(model,coords,[0])
    bad=list(coordinates());bad[0]=bad[1]
    with pytest.raises(ValueError,match="Duplicate"):
        CoordinateCapture(model,bad,[0])
    model.train()
    with pytest.raises(ValueError,match="eval"):
        with AttentionPulse(model,1,[0]):
            pass
    assert hook_count(model)==0
