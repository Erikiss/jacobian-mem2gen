"""One-pass attention pulses and read-only activation capture for Qwen3.

Positions are zero-based columns of the complete input sequence, not relative
indices of a cached decode chunk. Selected positions must be real tokens in
every batch row. Qwen3 queries are scaled *after* q_norm and before RoPE, so
normalization cannot cancel the pulse. Scaling queries scales their attention
logits; it makes no claim about FLOPs, energy, or a computational bottleneck.

CoordinateCapture stores raw gated-MLP z at down_proj input, not CETT.
"""
from __future__ import annotations

from collections import defaultdict
import math
import inspect
from numbers import Integral
from typing import Iterable, Sequence

import torch


def _integers(values: Iterable[int], name: str, *, nonempty: bool = True) -> tuple[int, ...]:
    result = tuple(values)
    if nonempty and not result:
        raise ValueError(f"{name} must not be empty")
    if any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in result):
        raise ValueError(f"{name} must contain nonnegative integers")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(int(x) for x in result)


def _layers(model):
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None or not len(layers):
        raise ValueError("Expected Hugging Face decoder model.model.layers (or model.layers)")
    return layers


def _cache_length(cache) -> int | None:
    if cache is None:
        return 0
    getter = getattr(cache, "get_seq_length", None)
    if getter is not None:
        try:
            return int(getter())
        except (TypeError, ValueError, RuntimeError):
            return None
    if isinstance(cache, (tuple, list)):
        if not cache:
            return 0
        try:
            return int(cache[0][0].shape[-2])
        except (IndexError, AttributeError, TypeError):
            return None
    return None


class _OneFullPass:
    """Bind hooks to one complete, uncached outer model forward and clean them up."""

    def __init__(self, model, positions: Sequence[int]):
        self.model = model
        self.positions = _integers(positions, "positions")
        self._handles = []
        self._entered = False
        self._used = False
        self._claimed = False
        self._window = None
        self.full_passes = 0
        self.skipped_cached_passes = 0

    def _model_pre(self, module, args, kwargs):
        self._window = None
        if self._claimed:
            return
        # HF forwards allow both positional and keyword cache arguments.
        try:
            named = inspect.signature(module.forward).bind_partial(*args, **kwargs).arguments
        except (TypeError, ValueError):
            named = kwargs
        fields = dict(kwargs)
        fields.update({name: value for name, value in named.items() if name not in ("args", "kwargs")})
        cache = fields.get("past_key_values", fields.get("past_key_value"))
        length = _cache_length(cache)
        cache_position = fields.get("cache_position")
        nonzero_offset = (cache_position is not None and torch.as_tensor(cache_position).numel()
                          and int(torch.as_tensor(cache_position).reshape(-1)[0]) != 0)
        if length is None or length > 0 or nonzero_offset:
            self.skipped_cached_passes += 1
            return
        ids = fields.get("input_ids")
        embeds = fields.get("inputs_embeds")
        value = ids if ids is not None else embeds
        if value is None and args:
            value = args[0]
        if not isinstance(value, torch.Tensor) or value.ndim not in (2, 3):
            raise ValueError("Cannot identify full input sequence for absolute token positions")
        batch, tokens = value.shape[:2]
        if max(self.positions) >= tokens:
            raise ValueError(f"position {max(self.positions)} is outside the full sequence of {tokens} tokens")
        mask = fields.get("attention_mask")
        if mask is not None:
            if tuple(mask.shape) != (batch, tokens):
                raise ValueError("Expected a two-dimensional outer attention mask")
            if not bool(mask[:, list(self.positions)].bool().all()):
                raise ValueError("Selected absolute position is padding in at least one batch row")
        self._claimed = True
        self.full_passes += 1
        self._window = (int(batch), int(tokens))

    def _model_post(self, module, args, kwargs, output):
        self._window = None

    def _install(self):
        raise NotImplementedError

    def validate_fired(self):
        raise NotImplementedError

    def __enter__(self):
        if self._used:
            raise RuntimeError("Use a fresh hook context for each attempt")
        if self.model.training:
            raise ValueError("Attention pilot requires model.eval(); model mode is never changed by hooks")
        self._used = self._entered = True
        try:
            self._handles.append(self.model.register_forward_pre_hook(self._model_pre, with_kwargs=True))
            self._handles.append(self.model.register_forward_hook(self._model_post, with_kwargs=True, always_call=True))
            self._install()
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._window = None
        self._entered = False

    def __exit__(self, kind, error, tb):
        self.close()
        if kind is None:
            self.validate_fired()
        return False


class AttentionPulse(_OneFullPass):
    """Scale post-q_norm queries, or add a per-head matched random delta, once.

    Primary architecture is config.model_type='qwen3'. Qwen2's unnormalized
    q_proj output is supported only with explicit architecture='qwen2'. Random
    deltas match ||(beta-1) q|| independently for each selected token/query head
    before output-dtype rounding. actual_delta_norms reports rounding effects.
    A local CPU generator never reads or changes PyTorch's global RNG state.
    """

    def __init__(self, model, block: int, positions: Sequence[int], beta: float = 1.0,
                 mode: str = "scale", seed: int = 0, *, architecture: str = "qwen3"):
        super().__init__(model, positions)
        if not isinstance(beta, (int, float)) or not math.isfinite(beta) or not 0 <= beta <= 1:
            raise ValueError("beta must be finite and between zero and one")
        if mode not in ("scale", "random"):
            raise ValueError("mode must be 'scale' or 'random'")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.layers = _layers(model)
        self.block = _integers((block,), "block")[0]
        if self.block >= len(self.layers):
            raise ValueError("block is outside the decoder")
        if architecture not in ("qwen3", "qwen2") or getattr(model.config, "model_type", None) != architecture:
            raise ValueError("Unsupported architecture: Qwen3 is primary; Qwen2 requires explicit opt-in")
        self.architecture, self.mode, self.beta = architecture, mode, float(beta)
        self.seed = int(seed)
        attention = self.layers[self.block].self_attn
        self.heads = int(model.config.num_attention_heads)
        self.head_dim = int(getattr(attention, "head_dim", getattr(model.config, "head_dim", 0)))
        if self.heads <= 0 or self.head_dim <= 0:
            raise ValueError("Query-head shape is not declared")
        if architecture == "qwen3":
            self.hook_module = getattr(attention, "q_norm", None)
            self.hook_location = "self_attn.q_norm.output (B,T,H,D), before RoPE"
            if self.hook_module is None:
                raise ValueError("Qwen3 requires post-q_norm intervention; q_proj fallback is forbidden")
        else:
            if getattr(attention, "q_norm", None) is not None:
                raise ValueError("Qwen2 fallback expects no query normalization")
            self.hook_module = getattr(attention, "q_proj", None)
            self.hook_location = "self_attn.q_proj.output, explicit Qwen2 fallback"
            if self.hook_module is None:
                raise ValueError("Qwen2 query projection is absent")
        self.fired_count = 0
        self.requested_delta_norms = None
        self.actual_delta_norms = None
        self._generator = torch.Generator(device="cpu").manual_seed(self.seed)

    def _install(self):
        self._handles.append(self.hook_module.register_forward_hook(self._pulse))

    def _pulse(self, module, args, output):
        if self._window is None or self.fired_count:
            return None
        batch, tokens = self._window
        if not isinstance(output, torch.Tensor):
            raise ValueError("Query hook did not return a tensor")
        shape = (batch, tokens, self.heads, self.head_dim)
        if self.architecture == "qwen3":
            if tuple(output.shape) != shape:
                raise ValueError(f"Unsupported Qwen3 post-q_norm layout: expected {shape}, got {tuple(output.shape)}")
            query = output
        else:
            if tuple(output.shape) != (batch, tokens, self.heads * self.head_dim):
                raise ValueError("Unsupported Qwen2 q_proj output shape")
            query = output.reshape(shape)
        selected = query[:, list(self.positions), :, :]
        if not bool(torch.isfinite(selected).all()):
            raise ValueError("Nonfinite query in selected pulse positions")
        self.fired_count += 1
        norms = (1-self.beta) * selected.detach().double().cpu().norm(dim=-1)
        self.requested_delta_norms = norms.clone()
        if self.beta == 1:
            self.actual_delta_norms = torch.zeros_like(norms)
            return None  # Exact sham: preserve the original tensor, no multiplication or RNG draw.
        if self.mode == "scale":
            changed = selected * self.beta
        else:
            random = torch.randn(selected.shape, generator=self._generator, dtype=torch.float64, device="cpu")
            random = random / random.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float64).tiny)
            delta = (random * norms[..., None]).to(device=selected.device, dtype=selected.dtype)
            changed = selected + delta
        self.actual_delta_norms = (changed.detach().double().cpu()-selected.detach().double().cpu()).norm(dim=-1)
        result = query.clone()
        result[:, list(self.positions), :, :] = changed
        return result if self.architecture == "qwen3" else result.reshape(output.shape)

    def validate_fired(self):
        if self.fired_count != 1:
            raise RuntimeError(f"Attention pulse must fire once on a full/prefill pass; fired {self.fired_count}")
        return self.fired_count


class CoordinateCapture(_OneFullPass):
    """Read exact ordered down_proj-input coordinates and decoder residuals.

    values is CPU float32 [batch, positions, coordinates]; coordinate order is
    exactly the supplied order. residuals maps each requested decoder layer to
    CPU float32 [batch, positions, hidden_size]. No input/output tensor changes.
    This is raw z capture, not CETT or a causal-neuron importance estimate.
    """

    feature_kind = "raw gated-MLP activation z at down_proj input; not CETT"

    def __init__(self, model, coordinates: Sequence[tuple[int, int]], positions: Sequence[int],
                 residual_layers: Sequence[int] | None = None, *, expected_count: int = 286):
        super().__init__(model, positions)
        self.layers = _layers(model)
        if getattr(model.config, "model_type", None) != "qwen3":
            raise ValueError("Original H coordinates require the Qwen3 decoder architecture")
        if isinstance(expected_count, bool) or not isinstance(expected_count, Integral) or expected_count < 1:
            raise ValueError("expected_count must be a positive integer")
        self.coordinates = tuple(tuple(pair) for pair in coordinates)
        if len(self.coordinates) != expected_count or not self.coordinates:
            raise ValueError(f"Expected exactly {expected_count} ordered H coordinates")
        if len(set(self.coordinates)) != len(self.coordinates):
            raise ValueError("Duplicate H coordinates")
        self._by_layer = defaultdict(list)
        for output_index, pair in enumerate(self.coordinates):
            if len(pair) != 2:
                raise ValueError("Each coordinate must be (block, down_proj_input_index)")
            # A coordinate may have equal block/channel numbers; duplicates are
            # forbidden across coordinate pairs, not within a pair.
            if any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in pair):
                raise ValueError("Coordinate indices must be nonnegative integers")
            block, neuron = (int(x) for x in pair)
            if block >= len(self.layers):
                raise ValueError("Coordinate block is outside the decoder")
            down = getattr(getattr(self.layers[block], "mlp", None), "down_proj", None)
            if down is None or not hasattr(down, "in_features") or neuron >= down.in_features:
                raise ValueError("Coordinate channel is outside mlp.down_proj input")
            self._by_layer[block].append((output_index, neuron))
        self.residual_layers = _integers(residual_layers if residual_layers is not None else sorted(self._by_layer),
                                        "residual_layers", nonempty=False)
        if any(block >= len(self.layers) for block in self.residual_layers):
            raise ValueError("Residual layer is outside the decoder")
        self._values = None
        self.residuals = {}
        self._seen = set()
        self.fired_count = 0

    def _install(self):
        for block in self._by_layer:
            self._handles.append(self.layers[block].mlp.down_proj.register_forward_pre_hook(self._coordinate_hook(block)))
        for block in self.residual_layers:
            self._handles.append(self.layers[block].register_forward_hook(self._residual_hook(block)))

    def _coordinate_hook(self, block):
        def hook(module, args):
            if self._window is None:
                return None
            if block in self._seen:
                raise RuntimeError("down_proj was called twice in the selected full pass")
            z = args[0]
            batch, tokens = self._window
            if not isinstance(z, torch.Tensor) or z.ndim != 3 or tuple(z.shape[:2]) != (batch, tokens):
                raise ValueError("Unsupported down_proj input shape")
            pairs = self._by_layer[block]
            observed = z[:, list(self.positions), :][:, :, [channel for _, channel in pairs]].detach().float().cpu().clone()
            if not bool(torch.isfinite(observed).all()):
                raise ValueError("Nonfinite selected raw activation")
            if self._values is None:
                self._values = torch.empty((batch, len(self.positions), len(self.coordinates)), dtype=torch.float32)
            self._values[:, :, [index for index, _ in pairs]] = observed
            self._seen.add(block)
            self.fired_count = int(self._seen == set(self._by_layer))
            return None
        return hook

    def _residual_hook(self, block):
        def hook(module, args, output):
            if self._window is None:
                return None
            if block in self.residuals:
                raise RuntimeError("Decoder layer was called twice in the selected full pass")
            value = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(value, torch.Tensor) or value.ndim != 3 or tuple(value.shape[:2]) != self._window:
                raise ValueError("Unsupported decoder residual output shape")
            observed = value[:, list(self.positions)].detach().float().cpu().clone()
            if not bool(torch.isfinite(observed).all()):
                raise ValueError("Nonfinite selected decoder residual")
            self.residuals[block] = observed
            return None
        return hook

    @property
    def values(self):
        self.validate_fired()
        return self._values

    def validate_fired(self):
        if self.fired_count != 1 or self._seen != set(self._by_layer) or set(self.residuals) != set(self.residual_layers):
            raise RuntimeError("Raw-coordinate capture did not complete exactly one full/prefill pass")
        return self.fired_count
