"""Random-weight native CPU Qwen3 tests; never download pretrained weights.

The only dense Jacobian below has width16 and is an independent tiny-model test
oracle. Production utilities remain matrix-free and fixed-prompt last->last.
"""
from contextlib import contextmanager

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from attention_probe.native_jacobian import build_residual_map


@pytest.fixture
def model():
    transformers = pytest.importorskip('transformers')
    if transformers.__version__ != '5.22.2':
        pytest.skip('Requires the pinned optional transformers==5.22.2 runtime')
    from transformers import Qwen3Config, Qwen3ForCausalLM
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(803)
        config = Qwen3Config(vocab_size=41, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=6, num_attention_heads=2, num_key_value_heads=1,
                            head_dim=8, eos_token_id=None, pad_token_id=0)
        config._attn_implementation = 'sdpa'
        result = Qwen3ForCausalLM(config).float().eval().requires_grad_(False)
    yield result
    torch.set_num_threads(old)


@pytest.fixture
def ids():
    return torch.tensor([[1, 5, 3, 7, 2]])


def hooks(model):
    return sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in model.modules())


@contextmanager
def reference_pass(model, source, target, vector=None):
    """Independent full native-model path, without the suffix-map machinery."""
    seen = {}

    def source_hook(module, args, output):
        seen['source_native'] = output.detach().clone()
        if vector is None:
            seen['source_injected'] = output.detach().clone()
            return None
        changed = output.clone()
        changed[0, -1, :] = vector
        seen['source_injected'] = changed.detach().clone()
        return changed

    def target_hook(module, args, output):
        seen['target'] = output

    handles = [model.model.layers[source].register_forward_hook(source_hook),
               model.model.layers[target].register_forward_hook(target_hook)]
    try:
        yield seen
    finally:
        for handle in reversed(handles):
            handle.remove()


def full_reference(model, ids, source, target, vector=None, mask=None):
    with reference_pass(model, source, target, vector) as seen, torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
        model(input_ids=ids, attention_mask=torch.ones_like(ids) if mask is None else mask,
              use_cache=False)
    return seen


def test_baseline_exact_full_native_forward_preserves_model_inputs_rng(model, ids):
    params = {key: value.clone() for key, value in model.state_dict().items()}
    original = ids.clone()
    rng = torch.random.get_rng_state().clone()
    reference = full_reference(model, ids, 2, 4)
    mapping = build_residual_map(model, ids, 2, 4)
    assert torch.equal(mapping.baseline_source, reference['source_native'][0, -1])
    assert torch.equal(mapping.baseline_target, reference['target'][0, -1])
    assert torch.equal(mapping(mapping.baseline_source), mapping.baseline_target)
    assert mapping.position == 4
    assert mapping.source_block == 2 and mapping.target_block == 4
    assert all(call['use_cache'] is False and call['past_key_values'] is None
               for call in mapping._calls.values())
    assert all(call['position_embeddings'] is not None and call['position_ids'] is not None
               for call in mapping._calls.values())
    copy = mapping.baseline_source
    copy.zero_()
    assert torch.equal(mapping.baseline_source, reference['source_native'][0, -1])
    assert hooks(model) == 0
    assert torch.equal(original, ids) and torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(params[key], value) for key, value in model.state_dict().items())


@pytest.mark.parametrize('left_padding', [False, True])
def test_replacement_matches_full_native_suffix_and_only_last_source_row(model, ids, left_padding):
    mask = torch.tensor([[0, 0, 1, 1, 1]]) if left_padding else torch.ones_like(ids)
    mapping = build_residual_map(model, ids, 1, 4, attention_mask=mask)
    source = mapping.baseline_source + torch.linspace(-.02, .03, 16)
    original = source.clone()
    reference = full_reference(model, ids, 1, 4, vector=source, mask=mask)
    target = mapping(source)
    assert torch.equal(target, reference['target'][0, -1])
    assert torch.equal(reference['source_injected'][:, :-1], reference['source_native'][:, :-1])
    assert torch.equal(reference['source_injected'][0, -1], source)
    assert torch.equal(source, original)
    assert hooks(model) == 0
    # A last-position causal perturbation cannot alter earlier target positions.
    clean = full_reference(model, ids, 1, 4, mask=mask)
    assert torch.equal(reference['target'][:, :-1], clean['target'][:, :-1])


def test_jvp_vjp_dense_tiny_oracle_adjoint_and_zero(model, ids):
    mapping = build_residual_map(model, ids, 1, 4)
    point = mapping.baseline_source
    v = torch.linspace(-.6, .7, 16)
    u = torch.linspace(.3, -.8, 16)
    jacobian = torch.autograd.functional.jacobian(mapping, point)
    jv, vju = mapping.jvp(v), mapping.vjp(u)
    torch.testing.assert_close(jv, jacobian @ v, rtol=3e-6, atol=1e-7)
    torch.testing.assert_close(vju, jacobian.T @ u, rtol=3e-6, atol=1e-7)
    torch.testing.assert_close(jv.double().dot(u.double()), v.double().dot(vju.double()), rtol=3e-6, atol=1e-7)
    torch.testing.assert_close(mapping.jvp(2*v), 2*jv, rtol=3e-6, atol=1e-7)
    assert torch.equal(mapping.jvp(torch.zeros_like(v)), torch.zeros_like(v))
    assert torch.equal(mapping.vjp(torch.zeros_like(u)), torch.zeros_like(u))
    assert not jv.requires_grad and not vju.requires_grad
    assert all(p.grad is None for p in model.parameters())
    assert hooks(model) == 0


def test_central_fd_validation_and_alternative_linearization_point(model, ids):
    mapping = build_residual_map(model, ids, 1, 4)
    direction = torch.linspace(-.5, .5, 16)
    direction /= direction.norm()
    diagnostic = mapping.validate_jvp(direction, .001, rtol=2e-3, atol=2e-5)
    assert diagnostic['passed']
    assert diagnostic['relative_l2_error'] < 2e-3
    assert diagnostic['source_block'] == 1 and diagnostic['target_block'] == 4
    assert diagnostic['position'] == 4
    point = mapping.baseline_source + .004*torch.arange(16)/16
    jvp = mapping.jvp(direction, at=point)
    finite = mapping.central_fd(direction, .001, at=point)
    torch.testing.assert_close(jvp, finite, rtol=2e-3, atol=2e-5)
    assert torch.equal(mapping.fd(direction, .001, at=point), finite)
    assert torch.equal(mapping.central_fd(torch.zeros_like(direction), .001), torch.zeros_like(direction))


def test_callable_keeps_autograd_under_outer_no_grad(model, ids):
    mapping = build_residual_map(model, ids, 1, 4)
    point = mapping.baseline_source.requires_grad_(True)
    with torch.no_grad():
        target = mapping(point)
    assert target.requires_grad
    gradient = torch.autograd.grad(target.sum(), point)[0]
    torch.testing.assert_close(gradient, mapping.vjp(torch.ones(16)), rtol=2e-6, atol=1e-7)


def test_repeated_operators_call_only_native_suffix_layers(model, ids, monkeypatch):
    mapping = build_residual_map(model, ids, 2, 4)
    calls = []
    for index, layer in enumerate(model.model.layers):
        original = layer.forward

        def wrapper(*args, index=index, original=original, **kwargs):
            calls.append(index)
            assert kwargs['use_cache'] is False and kwargs['past_key_values'] is None
            return original(*args, **kwargs)

        monkeypatch.setattr(layer, 'forward', wrapper)
    mapping(mapping.baseline_source)
    mapping.jvp(torch.ones(16))
    mapping.vjp(torch.ones(16))
    assert calls == [3, 4, 3, 4, 3, 4]
    assert hooks(model) == 0


def test_errors_leave_no_baseline_hooks_or_busy_state(model, ids, monkeypatch):
    original = model.model.layers[4].forward

    def fail(*args, **kwargs):
        raise RuntimeError('native target failure')

    with monkeypatch.context() as scoped:
        scoped.setattr(model.model.layers[4], 'forward', fail)
        with pytest.raises(RuntimeError, match='native target failure'):
            build_residual_map(model, ids, 1, 4)
    assert hooks(model) == 0
    mapping = build_residual_map(model, ids, 1, 4)
    with monkeypatch.context() as scoped:
        scoped.setattr(model.model.layers[4], 'forward', fail)
        with pytest.raises(RuntimeError, match='native target failure'):
            mapping(mapping.baseline_source)
    assert not mapping._busy and hooks(model) == 0
    assert torch.equal(mapping(mapping.baseline_source), mapping.baseline_target)
    assert model.model.layers[4].forward == original


@pytest.mark.parametrize('source,target', [(2, 2), (4, 2), (-1, 3), (1, 6), (True, 3)])
def test_invalid_block_geometry_rejected_before_hooks(model, ids, source, target):
    with pytest.raises(ValueError):
        build_residual_map(model, ids, source, target)
    assert hooks(model) == 0


@pytest.mark.parametrize('bad', [torch.zeros(1,16), torch.zeros(16,dtype=torch.float64),
                               torch.full((16,),float('nan'))])
def test_strict_vector_guards(model, ids, bad):
    mapping = build_residual_map(model, ids, 1, 4)
    for operation in (mapping, mapping.jvp, mapping.vjp):
        with pytest.raises(ValueError):
            operation(bad)
    assert hooks(model) == 0


@pytest.mark.parametrize('step', [0, -1, float('nan'), True])
def test_fd_step_guard(model, ids, step):
    mapping = build_residual_map(model, ids, 1, 4)
    with pytest.raises(ValueError, match='step'):
        mapping.central_fd(torch.ones(16), step)


def test_input_mask_model_and_execution_guards(model, ids):
    for bad in (ids.repeat(2,1), ids.float(), torch.tensor([[999]])):
        with pytest.raises(ValueError):
            build_residual_map(model, bad, 1, 4)
    with pytest.raises(ValueError, match='attention_mask'):
        build_residual_map(model, ids, 1, 4, attention_mask=torch.tensor([[1,1,1,1,0]]))
    with pytest.raises(ValueError, match='attention_mask'):
        build_residual_map(model, ids, 1, 4, attention_mask=torch.ones(1,1,1,5))
    with torch.inference_mode(), pytest.raises(ValueError, match='inference_mode'):
        build_residual_map(model, ids, 1, 4)
    with torch.autocast('cpu',dtype=torch.bfloat16), pytest.raises(ValueError, match='Autocast'):
        build_residual_map(model, ids, 1, 4)
    handle = model.register_forward_hook(lambda *args: None)
    try:
        with pytest.raises(ValueError, match='pre-existing'):
            build_residual_map(model, ids, 1, 4)
        assert hooks(model) == 1
    finally:
        handle.remove()
    mapping = build_residual_map(model, ids, 1, 4)
    direction = torch.ones(16)
    with torch.inference_mode():
        for operation in (mapping, mapping.jvp, mapping.vjp):
            with pytest.raises(ValueError, match='inference_mode'):
                operation(direction)
        with pytest.raises(ValueError, match='inference_mode'):
            mapping.central_fd(direction,.001)
    model.model.layers[2].train()
    with pytest.raises(ValueError, match='eval'):
        mapping(direction)
    model.eval()
    with torch.no_grad():
        next(model.parameters()).add_(.01)
    with pytest.raises(ValueError, match='parameters changed'):
        mapping(mapping.baseline_source)
    assert hooks(model) == 0


def test_real_zero_based_indices_14_to_18_without_full_model_loading():
    transformers = pytest.importorskip('transformers')
    if transformers.__version__ != '5.22.2':
        pytest.skip('Pinned optional runtime')
    from transformers import Qwen3Config, Qwen3ForCausalLM
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(119)
            config = Qwen3Config(vocab_size=13,hidden_size=8,intermediate_size=16,num_hidden_layers=20,
                                num_attention_heads=2,num_key_value_heads=1,head_dim=4)
            config._attn_implementation = 'sdpa'
            native = Qwen3ForCausalLM(config).float().eval().requires_grad_(False)
        mapping = build_residual_map(native,torch.tensor([[1,3,2]]),14,18)
        assert set(mapping._calls) == {15,16,17,18}
        assert torch.equal(mapping(mapping.baseline_source),mapping.baseline_target)
        assert mapping.validate_jvp(torch.linspace(-.2,.2,8),.001,rtol=5e-3,atol=2e-5)['passed']
        assert hooks(native) == 0
    finally:
        torch.set_num_threads(old)
