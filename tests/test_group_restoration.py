"""Native random-weight CPU Qwen3 only; no checkpoints or private coordinates."""
# ruff: noqa: SIM117 -- nested contexts make hook lifetime boundaries explicit.
import random
from contextlib import contextmanager

import pytest
import torch

from attention_probe.group_restoration import (
    GroupZPatch,
    ProbeCollector,
    channels_in_block,
    random_matched_delta,
    sample_control_masks,
)
from attention_probe.interventions import CoordinateCapture


@pytest.fixture
def model():
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "5.22.2":
        pytest.skip("Requires the pinned optional transformers==5.22.2 runtime")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(809)
        config = Qwen3Config(vocab_size=41, hidden_size=16, intermediate_size=24,
                            num_hidden_layers=3, num_attention_heads=2,
                            num_key_value_heads=1, head_dim=8, eos_token_id=None,
                            bos_token_id=1, pad_token_id=0)
        config._attn_implementation = "sdpa"
        instance = Qwen3ForCausalLM(config).float().eval().requires_grad_(False)
    yield instance
    torch.set_num_threads(threads)


@pytest.fixture
def ids():
    return torch.tensor([[2, 5, 3, 7], [11, 4, 2, 9]])


def hook_count(model):
    return sum(len(m._forward_pre_hooks) + len(m._forward_hooks) for m in model.modules())


@contextmanager
def z_tap(model, block):
    values = []
    handle = model.model.layers[block].mlp.down_proj.register_forward_pre_hook(
        lambda module, args: values.append(args[0].detach().clone()))
    try:
        yield values
    finally:
        handle.remove()


def run(model, ids, **kwargs):
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=torch.ones_like(ids), **kwargs)


def test_identity_exact_rng_parameters_and_removal_before_decode(model, ids):
    baseline = run(model, ids, use_cache=True).logits.clone()
    state = torch.random.get_rng_state().clone()
    params = {k: v.clone() for k, v in model.state_dict().items()}
    count = hook_count(model)
    with GroupZPatch(model, 1, [3], [9, 2]) as patch:
        output = run(model, ids, use_cache=True)
        assert torch.equal(output.logits, baseline)
        assert hook_count(model) == count
        assert patch.removed_before_decode
        assert patch.fired_count == patch.completed_count == 1
        next_ids = output.logits[:, -1].argmax(-1, keepdim=True)
        with torch.no_grad():
            model(input_ids=next_ids, past_key_values=output.past_key_values, use_cache=True)
        assert hook_count(model) == count
        assert patch.fired_count == 1
    assert torch.equal(state, torch.random.get_rng_state())
    assert all(torch.equal(params[k], v) for k, v in model.state_dict().items())
    assert torch.equal(patch.record["receiver_before"], patch.record["receiver_after"])
    assert not patch.record["raw_delta_norm"].any()
    assert not patch.record["projected_delta_norm"].any()
    assert hook_count(model) == count


@pytest.mark.parametrize("mode", ["replace", "add"])
def test_exact_selected_cells_unselected_input_and_donor_unchanged(model, ids, mode):
    positions, channels = [3, 1], [9, 2, 7]
    with z_tap(model, 1) as baseline_tap:
        run(model, ids, use_cache=False)
    original = baseline_tap[0]
    donor = torch.tensor([[[.25, -.5, 1.], [.1, .2, -.3]],
                          [[-.2, .7, -.8], [.6, -.1, .4]]], dtype=torch.float32)
    saved = donor.clone()
    options = {"donor_values" if mode == "replace" else "delta": donor}
    with z_tap(model, 1) as seen:
        with GroupZPatch(model, 1, positions, channels, **options) as patch:
            run(model, ids, use_cache=False)
    expected = original.clone()
    for p_index, p in enumerate(positions):
        for c_index, c in enumerate(channels):
            before = original[:, p, c]
            expected[:, p, c] = donor[:, p_index, c_index] if mode == "replace" else (
                before + donor[:, p_index, c_index])
    assert torch.equal(seen[0], expected)
    assert torch.equal(donor, saved)
    before = original[:, torch.tensor(positions)[:, None], torch.tensor(channels)[None, :]]
    after = expected[:, torch.tensor(positions)[:, None], torch.tensor(channels)[None, :]]
    assert torch.equal(patch.record["receiver_before"], before)
    assert torch.equal(patch.record["receiver_after"], after)
    difference = after - before
    expected_projection = torch.nn.functional.linear(
        difference, model.model.layers[1].mlp.down_proj.weight[:, channels])
    assert torch.equal(patch.record["projected_delta_vector"], expected_projection)
    assert torch.equal(patch.record["raw_delta_norm"], difference.double().norm(dim=-1))
    assert torch.equal(patch.record["projected_delta_norm"], expected_projection.double().norm(dim=-1))
    assert all(value.device.type == "cpu" and not value.requires_grad
               for value in patch.record.values())


def test_identity_donor_and_zero_delta_preserve_input_object(model, ids):
    with z_tap(model, 1) as seen:
        baseline = run(model, ids, use_cache=False).logits
    donor = seen[0][:, [3], :][:, :, [9, 2]]
    for options in ({"donor_values": donor}, {"delta": torch.zeros_like(donor)}):
        input_objects = []
        before = model.model.layers[1].mlp.down_proj.register_forward_pre_hook(
            lambda module, args, observed=input_objects: observed.append(args[0]), prepend=True)
        try:
            # Patch is prepended after the observer, so add another leading tap.
            with GroupZPatch(model, 1, [3], [9, 2], **options):
                first = model.model.layers[1].mlp.down_proj.register_forward_pre_hook(
                    lambda module, args, observed=input_objects: observed.append(args[0]), prepend=True)
                try:
                    result = run(model, ids, use_cache=False)
                finally:
                    first.remove()
        finally:
            before.remove()
        assert len(input_objects) == 2 and input_objects[0] is input_objects[1]
        assert torch.equal(result.logits, baseline)


def test_probe_and_coordinate_capture_see_postpatch_even_installed_first(model, ids):
    donor = torch.tensor([[[1., -2.]], [[3., -4.]]])
    with CoordinateCapture(model, [(1, 9), (1, 2)], [3], expected_count=2) as passive:
        with ProbeCollector(model, blocks=[1, 2]) as probe:
            with GroupZPatch(model, 1, [3], [9, 2], donor_values=donor) as patch:
                output = run(model, ids, use_cache=False)
                # Only the passive existing context still has its own hooks.
                assert not patch._handles and not probe._handles
    assert torch.equal(passive.values, donor)
    assert torch.equal(probe.z_by_block[1][:, :, [9, 2]], donor)
    assert set(probe.z_by_block) == {1, 2}
    assert probe.z_by_block[1].shape == (2, 1, 24)
    assert torch.equal(probe.first_logits, output.logits[:, -1])
    assert probe.first_logits.device.type == "cpu"
    assert not probe.first_logits.requires_grad
    assert probe.fired_count == probe.completed_count == 1
    assert hook_count(model) == 0


def test_probe_left_padding_last_column_and_plain_no_cache_parity(model, ids):
    mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    with torch.no_grad():
        baseline = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    state = torch.random.get_rng_state().clone()
    with ProbeCollector(model, blocks=[0, 2]) as probe, torch.no_grad():
        result = model(input_ids=ids, attention_mask=mask, use_cache=False)
    assert probe._positions == (3,)
    assert torch.equal(probe.first_logits, baseline[:, -1])
    assert torch.equal(result.logits, baseline)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert hook_count(model) == 0


def test_cached_call_before_prefill_skipped_and_only_cached_rejected(model, ids):
    cache = run(model, ids, use_cache=True).past_key_values
    with GroupZPatch(model, 1, [3], [2]) as patch:
        with torch.no_grad():
            model(input_ids=ids[:, :1], past_key_values=cache, use_cache=True)
        assert patch.fired_count == 0 and patch.skipped_cached_passes == 1
        assert patch._handles
        run(model, ids, use_cache=False)
        assert patch.fired_count == 1 and not patch._handles
    cache2 = run(model, ids, use_cache=True).past_key_values
    with pytest.raises(RuntimeError, match="exactly one"):
        with ProbeCollector(model, blocks=[1]) as probe:
            with torch.no_grad():
                model(input_ids=ids[:, :1], past_key_values=cache2, use_cache=True)
    assert probe.completed_count == 0 and hook_count(model) == 0


def test_failure_after_patch_preserves_original_error_and_cleans_hooks(model, ids):
    def fail(module, args):
        raise RuntimeError("injected after patch")
    count = hook_count(model)
    with pytest.raises(RuntimeError, match="injected after patch"):
        with GroupZPatch(model, 1, [3], [2], delta=torch.ones(2, 1, 1)) as patch:
            with ProbeCollector(model, blocks=[1, 2]) as probe:
                handle = model.model.layers[2].mlp.down_proj.register_forward_pre_hook(fail)
                try:
                    run(model, ids, use_cache=False)
                finally:
                    handle.remove()
    assert patch.fired_count == 1 and patch.completed_count == 0
    assert probe.completed_count == 0
    assert hook_count(model) == count


def test_failure_in_foreign_model_post_hook_preserves_error_and_cleans(model, ids):
    def fail(module, args, output):
        raise RuntimeError("injected foreign post-hook")
    handle = model.register_forward_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="injected foreign post-hook"):
            with GroupZPatch(model, 1, [3], [2]) as patch:
                with ProbeCollector(model, blocks=[1]) as probe:
                    run(model, ids, use_cache=False)
        assert patch.completed_count == probe.completed_count == 0
        assert hook_count(model) == 1  # Only the foreign hook remains.
    finally:
        handle.remove()
    assert hook_count(model) == 0


@pytest.mark.parametrize("options,match", [
    ({"donor_values": torch.zeros(1, 1, 1)}, "exact shape"),
    ({"donor_values": torch.zeros(2, 1, 1), "delta": torch.zeros(2, 1, 1)}, "OR"),
    ({"donor_values": torch.zeros(2, 1, 1, dtype=torch.float64)}, "float32"),
    ({"donor_values": torch.ones(2, 1, 1, requires_grad=True)}, "detached"),
    ({"donor_values": torch.full((2, 1, 1), float("nan"))}, "finite"),
])
def test_strict_source_guards_cleanup(model, ids, options, match):
    with pytest.raises(ValueError, match=match):
        with GroupZPatch(model, 1, [3], [2], **options):
            run(model, ids, use_cache=False)
    assert hook_count(model) == 0


@pytest.mark.parametrize("positions,channels", [([4], [2]), ([1], [24]), ([3, 3], [2]), ([3], [2, 2])])
def test_indices_rejected_without_leftover_hooks(model, ids, positions, channels):
    with pytest.raises(ValueError), GroupZPatch(model, 1, positions, channels):
        run(model, ids, use_cache=False)
    assert hook_count(model) == 0


def test_padding_autocast_and_model_guards(model, ids):
    with pytest.raises(ValueError, match="real, unmasked"):
        with ProbeCollector(model, blocks=[1], positions=[0]):
            with torch.no_grad():
                model(input_ids=ids, attention_mask=torch.tensor([[0,1,1,1],[1,1,1,1]]))
    assert hook_count(model) == 0
    with pytest.raises(ValueError, match="autocast"):
        with ProbeCollector(model, blocks=[1]):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                run(model, ids, use_cache=False)
    assert hook_count(model) == 0
    model.train()
    with pytest.raises(ValueError, match="eval"):
        GroupZPatch(model, 1, [3], [2])
    model.eval().requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        ProbeCollector(model, blocks=[1])
    model.requires_grad_(False).double()
    with pytest.raises(ValueError, match="FP32"):
        ProbeCollector(model, blocks=[1])
    with pytest.raises(ValueError, match="native"):
        ProbeCollector(torch.nn.Linear(2, 2), blocks=[0])


def test_empty_context_and_reuse_rejected(model, ids):
    patch = GroupZPatch(model, 1, [3], [2])
    with pytest.raises(RuntimeError, match="exactly one"), patch:
        pass
    with pytest.raises(RuntimeError, match="fresh"), patch:
        run(model, ids, use_cache=False)
    assert hook_count(model) == 0


def test_control_masks_deterministic_exclusions_no_global_rng_mutation():
    state = random.getstate()
    masks = sample_control_masks(24, [7, 2, 9], 3, seeds=[104729, 130363, 155921])
    assert masks == sample_control_masks(24, [7, 2, 9], 3, seeds=[104729, 130363, 155921])
    assert random.getstate() == state
    assert len(set(masks)) == 3
    assert all(len(mask) == 3 and tuple(sorted(mask)) == mask
               and not set(mask).intersection({7, 2, 9}) for mask in masks)
    with pytest.raises(ValueError, match="duplicate complete masks"):
        sample_control_masks(3, [2], 2, seeds=[1, 2])
    with pytest.raises(ValueError, match="Not enough"):
        sample_control_masks(3, [2], 3, seeds=[1])
    with pytest.raises(ValueError):
        sample_control_masks(3, [], 1, seeds=[True])
    assert channels_in_block([(2, 4), (1, 7), (1, 3), (2, 1)], 1) == (7, 3)
    with pytest.raises(ValueError):
        channels_in_block([(1, 3), (1, 3)], 1)


def test_matched_directions_per_row_norms_zero_and_local_rng():
    source = torch.tensor([[[3., 4., 0.], [0., 0., 0.]], [[-2., 0., 0.], [1., -2., 2.]]])
    saved = source.clone()
    state = torch.random.get_rng_state().clone()
    result = random_matched_delta(source, seed=27011)
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(source, saved)
    assert torch.equal(result, random_matched_delta(source, seed=27011))
    assert not torch.equal(result, random_matched_delta(source, seed=27043))
    assert torch.allclose(result.double().norm(dim=-1), source.double().norm(dim=-1), rtol=1e-7, atol=1e-7)
    assert not result[0, 1].any()
    assert torch.equal(random_matched_delta(torch.zeros(1, 1, 3), seed=8), torch.zeros(1, 1, 3))
    assert result.dtype == torch.float32 and result.device.type == "cpu"
    assert not result.requires_grad
