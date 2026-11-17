"""One-prefill signed gated-MLP interventions for the frozen native Qwen3.

A channel is a column of ``SiLU(gate_proj(x)) * up_proj(x)`` immediately before
``down_proj``. It is neither an attention head nor a standardized CETT score.
Positions are absolute physical columns of a full, unfilled prompt. The caller
must supply only the intended last-prompt column for the group experiment.

Every context removes all its own hooks when the first fresh prefill succeeds.
On a failed forward the model-post hooks are removed at context exit (PyTorch's
exception handler iterates that hook dictionary directly). Cached decoding
inside the still-open successful context therefore has no intervention/collector
hooks. This module performs no model loading,
coordinate selection, generation, device transfer of models, or parameter edits.
"""
from __future__ import annotations

import inspect
import random
from collections.abc import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .interventions import _cache_length, _integers, _layers


def _integer(value, name: str, *, positive: bool = False) -> int:
    result = _integers((value,), name)[0]
    if positive and result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _cpu_copy(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", copy=True)


def _source(value, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise ValueError(f"{name} must be a CPU float32 tensor with shape [B,P,C]")
    if value.device.type != "cpu" or value.dtype != torch.float32:
        raise ValueError(f"{name} must be CPU float32; implicit casts are forbidden")
    if value.requires_grad or value.grad_fn is not None:
        raise ValueError(f"{name} must be detached, without a live autograd graph")
    if any(size <= 0 for size in value.shape) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must have nonempty axes and finite values")
    return value.detach().clone()


def _native_model(model):
    cls = type(model)
    if (cls.__name__ != "Qwen3ForCausalLM"
            or cls.__module__ != "transformers.models.qwen3.modeling_qwen3"
            or getattr(getattr(model, "config", None), "model_type", None) != "qwen3"):
        raise ValueError("Group restoration requires native Hugging Face Qwen3ForCausalLM")
    if model.training:
        raise ValueError("Group restoration requires model.eval(); mode is never changed")
    params = tuple(model.parameters())
    if not params or any(p.requires_grad for p in params):
        raise ValueError("Group restoration requires frozen model parameters")
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in params):
        raise ValueError("Group restoration requires native FP32 parameters")
    devices = {p.device for p in params}
    if len(devices) != 1 or next(iter(devices)).type not in ("cpu", "cuda"):
        raise ValueError("Group restoration requires one CPU/CUDA device; no offload or meta parameters")
    return next(iter(devices))


class _AutoPrefill:
    """One fresh forward; remove own hooks at its outer model return."""

    def __init__(self, model, positions: Sequence[int] | None):
        self.model = model
        self.device = _native_model(model)
        self.positions = None if positions is None else _integers(positions, "positions")
        self.layers = _layers(model)
        self._handles = []
        self._post_handles = []
        self._normal_post_ran = False
        self._used = False
        self._claimed = False
        self._window = None
        self._positions = None
        self.full_passes = 0
        self.completed_count = 0
        self.skipped_cached_passes = 0
        self.removed_before_decode = False

    def _model_pre(self, module, args, kwargs):
        self._normal_post_ran = False
        self._window = None
        if self._claimed:
            return
        # Bind positional cache/mask arguments too; HF forwards also accept kwargs.
        try:
            bound = inspect.signature(module.forward).bind_partial(*args, **kwargs).arguments
        except (TypeError, ValueError):
            bound = {}
        fields = dict(kwargs)
        fields.update({k: v for k, v in bound.items() if k not in ("args", "kwargs")})
        length = _cache_length(fields.get("past_key_values", fields.get("past_key_value")))
        offset = fields.get("cache_position")
        nonzero_offset = (offset is not None and torch.as_tensor(offset).numel() > 0
                          and int(torch.as_tensor(offset).reshape(-1)[0]) != 0)
        if length is None or length > 0 or nonzero_offset:
            self.skipped_cached_passes += 1
            return
        value = fields.get("input_ids")
        if value is None:
            value = fields.get("inputs_embeds")
        if value is None and args:
            value = args[0]
        if not isinstance(value, torch.Tensor) or value.ndim not in (2, 3):
            raise ValueError("Cannot identify complete prompt input")
        batch, tokens = map(int, value.shape[:2])
        if batch <= 0 or tokens <= 0:
            raise ValueError("Prompt must have nonempty batch and sequence")
        positions = (tokens - 1,) if self.positions is None else self.positions
        if max(positions) >= tokens:
            raise ValueError("Selected position is outside the full prompt")
        mask = fields.get("attention_mask")
        if mask is not None:
            if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != (batch, tokens):
                raise ValueError("Expected a two-dimensional full-prompt attention mask")
            if not bool((mask[:, list(positions)] == 1).all()):
                raise ValueError("Selected position must be a real, unmasked token in every row")
        # Recheck mode/parameter metadata at the actual forward (not only construction).
        if _native_model(module) != self.device:
            raise ValueError("Model device changed after hook construction")
        if torch.is_autocast_enabled(self.device.type):
            raise ValueError("Group restoration requires autocast disabled")
        if self.device.type == "cuda" and (torch.backends.cuda.matmul.allow_tf32
                                           or torch.backends.cudnn.allow_tf32):
            raise ValueError("Group restoration requires CUDA matmul and cuDNN TF32 disabled")
        self._positions = positions
        self._window = (batch, tokens)
        self._claimed = True
        self.full_passes += 1
        self._prepare(batch, positions)

    def _prepare(self, batch, positions):
        pass

    def _model_post(self, module, args, output):
        self._normal_post_ran = True
        if self._window is None:
            return  # A cached call before the fresh prefill is never claimed.
        try:
            self._finish(output)
            self.completed_count += 1
        finally:
            self.close()
        self.removed_before_decode = True
        return

    def _model_error(self, module, args, output):
        # On success _model_post already removed all handles. On failure it did
        # not run, including when another post-hook failed with non-None output.
        # PyTorch's exception path iterates model._forward_hooks directly, so
        # retain our model-post handles until __exit__ rather than mutate that
        # OrderedDict and replace the original error. Other hooks go now.
        if self._normal_post_ran:
            return
        self._remove_handles(keep_post=True)
        self._window = None
        return

    def _install(self):
        raise NotImplementedError

    def _finish(self, output):
        raise NotImplementedError

    def validate_fired(self):
        if self.full_passes != 1 or self.completed_count != 1:
            raise RuntimeError("Expected exactly one successful fresh prefill")

    def __enter__(self):
        if self._used:
            raise RuntimeError("Use a fresh group hook context for each prefill")
        self._used = True
        try:
            self._handles.append(self.model.register_forward_pre_hook(self._model_pre, with_kwargs=True))
            self._post_handles = [
                self.model.register_forward_hook(self._model_post),
                self.model.register_forward_hook(self._model_error, always_call=True),
            ]
            self._handles.extend(self._post_handles)
            self._install()
        except BaseException:
            self.close()
            raise
        return self

    def _remove_handles(self, *, keep_post=False):
        for handle in reversed(self._handles):
            if not keep_post or handle not in self._post_handles:
                handle.remove()
        self._handles = [h for h in self._handles if keep_post and h in self._post_handles]

    def close(self):
        self._remove_handles()
        self._window = None

    def __exit__(self, kind, error, tb):
        self.close()
        if kind is None:
            self.validate_fired()
        return False

    def _z(self, args, width):
        if not args or not isinstance(args[0], torch.Tensor):
            raise ValueError("Native down_proj must receive a positional tensor")
        z = args[0]
        if tuple(z.shape) != (*self._window, width) or z.dtype != torch.float32:
            raise ValueError("Unexpected native FP32 gated-MLP input shape/dtype")
        if z.device != self.device:
            raise ValueError("Gated-MLP input is on the wrong device")
        return z


class GroupZPatch(_AutoPrefill):
    """Replace or add signed raw-z values at exactly the selected cells once.

    ``donor_values`` and ``delta`` are mutually exclusive detached CPU FP32
    tensors, with exact shape [batch, len(positions), len(channels)]; no broadcast
    or dtype conversion is permitted. Neither supplied means an exact no-op.
    Replacement assigns donor FP32 values directly (not via subtract/add).

    ``record`` stores detached CPU before/after values, per-[B,P] L2 norms, and
    optionally the projected delta vector. Projection uses the selected columns
    of native W_down; it describes linear geometry, not measured output changes,
    compute saved, memory capacity, or energy. Norms are evaluated in float64.
    """

    def __init__(self, model, block: int, positions: Sequence[int], channels: Sequence[int], *,
                 donor_values=None, delta=None, record_projected_vector: bool = True):
        super().__init__(model, positions)
        self.block = _integer(block, "block")
        self.channels = _integers(channels, "channels")
        if self.block >= len(self.layers):
            raise ValueError("block is outside the native decoder")
        self.down_proj = self.layers[self.block].mlp.down_proj
        if type(self.down_proj) is not nn.Linear:
            raise ValueError("Expected the native dense down_proj Linear")
        if max(self.channels) >= self.down_proj.in_features:
            raise ValueError("channel is outside the gated-MLP width")
        if donor_values is not None and delta is not None:
            raise ValueError("Supply donor_values OR delta, never both")
        if not isinstance(record_projected_vector, bool):
            raise ValueError("record_projected_vector must be boolean")  # noqa: TRY004 -- consistent invalid-configuration API.
        self.mode = "replace" if donor_values is not None else "add" if delta is not None else "identity"
        self._source = _source(donor_values, "donor_values") if donor_values is not None else (
            _source(delta, "delta") if delta is not None else None)
        self.record_projected_vector = record_projected_vector
        self.fired_count = 0
        self.record = {}

    def _prepare(self, batch, positions):
        expected = (batch, len(positions), len(self.channels))
        if self._source is not None and tuple(self._source.shape) != expected:
            raise ValueError(f"Patch source must have exact shape {expected}; no broadcasting")

    def _install(self):
        # prepend guarantees passive CoordinateCapture prehooks observe patched z,
        # even if they were installed before this patch. Other mutators must not
        # target the same z site in the same forward.
        self._handles.append(self.down_proj.register_forward_pre_hook(self._patch, prepend=True))

    def _patch(self, module, args):
        if self._window is None:
            return None
        if self.fired_count:
            raise RuntimeError("Selected down_proj fired more than once during one prefill")
        z = self._z(args, module.in_features)
        p = torch.tensor(self._positions, device=z.device, dtype=torch.long)
        c = torch.tensor(self.channels, device=z.device, dtype=torch.long)
        with torch.no_grad():
            before = z[:, p[:, None], c[None, :]]
            if not bool(torch.isfinite(before).all()):
                raise ValueError("Selected native raw-z contains nonfinite values")
            if self._source is None:
                after = before
            else:
                source = self._source.to(device=z.device)
                after = source if self.mode == "replace" else before + source
            if not bool(torch.isfinite(after).all()):
                raise ValueError("Patch would produce nonfinite raw-z")
            actual_delta = after - before
            projected = F.linear(actual_delta, module.weight[:, c])
            if not bool(torch.isfinite(projected).all()):
                raise ValueError("Projected patch delta is nonfinite")
            self.record = {
                "receiver_before": _cpu_copy(before),
                "receiver_after": _cpu_copy(after),
                "raw_delta_norm": actual_delta.double().norm(dim=-1).cpu(),
                "projected_delta_norm": projected.double().norm(dim=-1).cpu(),
            }
            if self.record_projected_vector:
                self.record["projected_delta_vector"] = _cpu_copy(projected)
            self.fired_count += 1
            if torch.equal(before, after):
                return None  # Identity preserves the original input tensor exactly.
            changed = z.clone()
            changed[:, p[:, None], c[None, :]] = after
            return (changed, *args[1:])

    def _finish(self, output):
        if self.fired_count != 1:
            raise RuntimeError("Group patch must fire exactly once in the fresh prefill")

    def validate_fired(self):
        super().validate_fired()
        if self.fired_count != 1:
            raise RuntimeError("Group patch did not fire exactly once")


class ProbeCollector(_AutoPrefill):
    """Capture selected full raw-z rows and the first next-token logits once.

    By default only the last physical prompt column is copied. Set explicit
    positions only for a separately specified diagnostic. z_by_block[block] is
    detached CPU FP32 [B,P,intermediate_size]; first_logits is detached CPU FP32
    [B,vocab_size]. Results remain available after context exit. This collector
    works both with use_cache=True and with a plain use_cache=False prefill.
    """

    def __init__(self, model, blocks: Sequence[int] = (7, 14), positions=None):
        super().__init__(model, positions)
        self.blocks = _integers(blocks, "blocks")
        if max(self.blocks) >= len(self.layers):
            raise ValueError("block is outside the native decoder")
        self.z_by_block = {}
        self.first_logits = None
        self.fired_count = 0

    def _install(self):
        for block in self.blocks:
            down = self.layers[block].mlp.down_proj
            if type(down) is not nn.Linear:
                raise ValueError("Expected native dense down_proj Linear")
            self._handles.append(down.register_forward_pre_hook(
                lambda module, args, block=block: self._capture(block, module, args)))

    def _capture(self, block, module, args):
        if self._window is None:
            return
        if block in self.z_by_block:
            raise RuntimeError("Probe down_proj fired more than once during one prefill")
        z = self._z(args, module.in_features)
        chosen = z[:, list(self._positions), :]
        if not bool(torch.isfinite(chosen).all()):
            raise ValueError("Probe raw-z contains nonfinite values")
        self.z_by_block[block] = _cpu_copy(chosen)
        return

    def _finish(self, output):
        if set(self.z_by_block) != set(self.blocks):
            raise RuntimeError("Not all requested raw-z blocks were captured")
        logits = getattr(output, "logits", None)
        if logits is None and isinstance(output, tuple) and output:
            logits = output[0]
        batch = self._window[0]
        if (not isinstance(logits, torch.Tensor) or logits.ndim != 3
                or logits.shape[0] != batch or logits.shape[1] == 0
                or logits.dtype != torch.float32 or logits.device != self.device):
            raise ValueError("Expected native FP32 causal-LM logits [B,T,V]")
        last = logits[:, -1, :]
        if not bool(torch.isfinite(last).all()):
            raise ValueError("First next-token logits contain nonfinite values")
        self.first_logits = _cpu_copy(last)
        self.fired_count += 1

    def validate_fired(self):
        super().validate_fired()
        if self.fired_count != 1:
            raise RuntimeError("Probe did not complete exactly once")


def channels_in_block(coordinates: Iterable[Sequence[int]], block: int) -> tuple[int, ...]:
    """Select a block's channels while preserving the reference coordinate order."""
    block = _integer(block, "block")
    pairs = tuple(tuple(pair) for pair in coordinates)
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("coordinates must be (block, channel) integer pairs")
        _integers((pair[0],), "coordinate block")
        _integers((pair[1],), "coordinate channel")
    if len(set(pairs)) != len(pairs):
        raise ValueError("coordinates must be unique")
    return tuple(int(channel) for layer, channel in pairs if layer == block)


def sample_control_masks(width: int, excluded: Iterable[int], count: int, *,
                         seeds: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """Deterministic masks outside H, with a separate local Python RNG per seed.

    Channels are sorted in each mask. Masks can overlap across seeds, but exact
    duplicate masks are rejected rather than silently reseeded. Seeds belong in
    the frozen runner plan, and must not be chosen based on measured outcomes.
    """
    width = _integer(width, "width", positive=True)
    count = _integer(count, "count", positive=True)
    excluded = _integers(excluded, "excluded", nonempty=False)
    seeds = _integers(seeds, "seeds")
    if excluded and max(excluded) >= width:
        raise ValueError("excluded channel is outside width")
    allowed = sorted(set(range(width)).difference(excluded))
    if count > len(allowed):
        raise ValueError("Not enough outside-H channels for the matched mask")
    masks = tuple(tuple(sorted(random.Random(seed).sample(allowed, count))) for seed in seeds)
    if len(set(masks)) != len(masks):
        raise ValueError("Seeds produced duplicate complete masks; specify distinct frozen seeds")
    return masks


def random_matched_delta(delta: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Gaussian direction with per-[B,P] raw-delta norm, then one FP32 cast.

    A local CPU torch.Generator leaves global RNG state unchanged. Matching is
    performed in float64 before the final FP32 cast; callers must report actual
    norms and W_down projections after native FP32 addition. Zero target norm
    produces exactly zero (not an invented random signal). No direction is
    resampled to improve its projected alignment or effect on model outputs.
    """
    source = _source(delta, "delta")
    seed = _integer(seed, "seed")
    target = source.double().norm(dim=-1, keepdim=True)
    if not bool((target > 0).any()):
        return torch.zeros_like(source)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    direction = torch.randn(source.shape, dtype=torch.float64, generator=generator)
    length = direction.norm(dim=-1, keepdim=True)
    if bool(((target > 0) & (length == 0)).any()):
        raise ValueError("Degenerate zero Gaussian direction; do not silently reseed")
    direction = direction / length.clamp_min(torch.finfo(torch.float64).tiny)
    return (direction * target).float()
