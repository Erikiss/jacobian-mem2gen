"""Native Qwen3 residual suffix maps and matrix-free local derivatives.

A map is resid_post(source_block, p) -> resid_post(target_block, p), where p is
ONLY the last column of the original fixed prompt. Block numbers are zero-based;
outputs precede the model's final RMSNorm. Other source-token rows stay fixed.
This is an offset-zero, prompt-local Jacobian, not a corpus-averaged J-lens or a
cross-token/time transport. No model weights are loaded or changed here.

One no-cache HF prefill records the source residual and the exact native suffix
arguments (causal masks, position IDs, rotary cos/sin). Subsequent calls execute
only the original decoder layers source+1..target, with those arguments. Native
MATH SDPA, FP32 and frozen weights are required. No TransformerLens conversion,
sequence flattening, dense Jacobian, cache, optimizer, or inferred bottleneck.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import torch
import transformers
from torch.nn.attention import SDPBackend, sdpa_kernel

TRANSFORMERS_VERSION = '5.22.2'


class _TargetReached(Exception):
    """Private early stop after the baseline target decoder block."""


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f'{label} must be a nonnegative integer')
    return int(value)


def _hooks(model):
    return sum(len(module._forward_hooks)+len(module._forward_pre_hooks) for module in model.modules())


def _parameters(model):
    return tuple((id(p), p.data_ptr(), p._version, p.dtype, p.device, p.requires_grad)
                 for p in model.parameters())


def _model_device(model):
    cls = type(model)
    if (cls.__name__ != 'Qwen3ForCausalLM'
            or cls.__module__ != 'transformers.models.qwen3.modeling_qwen3'
            or getattr(model.config, 'model_type', None) != 'qwen3'):
        raise ValueError('Expected native Hugging Face Qwen3ForCausalLM')
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise ValueError(f'Expected transformers=={TRANSFORMERS_VERSION}')
    if any(module.training for module in model.modules()):
        raise ValueError('Expected model.eval(); the adapter never changes model mode')
    if getattr(model.config, '_attn_implementation', None) != 'sdpa':
        raise ValueError('Expected native SDPA implementation for the MATH diagnostic')
    params = tuple(model.parameters())
    if not params or any(p.requires_grad for p in params):
        raise ValueError('Model parameters must already be frozen')
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in params):
        raise ValueError('Native FP32 parameters required; no implicit dtype conversion')
    devices = {p.device for p in params}
    if len(devices) != 1 or next(iter(devices)).type not in ('cpu', 'cuda'):
        raise ValueError('Expected parameters on one CPU/CUDA device, without meta/offload')
    return next(iter(devices))


def _execution_guard(model, device, expected_parameters=None):
    if torch.is_inference_mode_enabled():
        raise ValueError('Disable inference_mode: local derivatives require ordinary autograd tensors')
    if _model_device(model) != device:
        raise ValueError('Model device changed after constructing the residual map')
    if _hooks(model):
        raise ValueError('Residual map requires no pre-existing model hooks')
    if torch.is_autocast_enabled(device.type):
        raise ValueError('Autocast must be disabled for native FP32 derivatives')
    if device.type == 'cuda' and (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
        raise ValueError('CUDA matmul and cuDNN TF32 must both be disabled')
    if expected_parameters is not None and _parameters(model) != expected_parameters:
        raise ValueError('Model parameters changed after constructing the residual map')


def _constant(value):
    """Private copies of native mask/RoPE arguments; never model/cache objects."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_constant(item) for item in value)
    if isinstance(value, list):
        return [_constant(item) for item in value]
    if isinstance(value, dict):
        return {key: _constant(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f'Unexpected native suffix argument type: {type(value).__name__}')


def _hidden(output, batch, tokens, width):
    value = output[0] if isinstance(output, (tuple, list)) else output
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (batch, tokens, width)
            or value.dtype != torch.float32):
        raise ValueError('Unexpected native FP32 decoder residual shape')
    return value


class ResidualMap:
    """Fixed-prompt native suffix map with [hidden_size] input/output vectors.

    __call__(vector) returns the nonlinear target vector with an autograd graph
    when vector requires gradients. jvp/vjp return detached local products at
    baseline_source by default; `at` can specify another source vector. Both use
    the SAME primal map and no parameter gradients. central_fd(v, step) computes
    [F(x+step*v)-F(x-step*v)]/(2*step), so step is relative to v's native scale.
    """

    def __init__(self, model, input_ids, source_block, target_block, *, attention_mask=None):
        self.model = model
        self.device = _model_device(model)
        _execution_guard(model, self.device)
        self.source_block = _integer(source_block, 'source_block')
        self.target_block = _integer(target_block, 'target_block')
        self.layers = model.model.layers
        if not self.source_block < self.target_block < len(self.layers):
            raise ValueError('Require 0 <= source_block < target_block < native layer count')
        if (not isinstance(input_ids, torch.Tensor) or input_ids.dtype != torch.long
                or input_ids.device != self.device or input_ids.ndim != 2
                or input_ids.shape[0] != 1 or input_ids.shape[1] == 0):
            raise ValueError('input_ids must be native-device int64 [1,T] with T>0')
        if not bool(((input_ids >= 0) & (input_ids < model.config.vocab_size)).all()):
            raise ValueError('input_ids contain out-of-vocabulary indices')
        self.input_ids = input_ids.detach().clone()
        self.tokens = int(input_ids.shape[1])
        self.position = self.tokens-1
        self.width = int(model.config.hidden_size)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if (not isinstance(attention_mask, torch.Tensor) or attention_mask.device != self.device
                or tuple(attention_mask.shape) != tuple(input_ids.shape)
                or attention_mask.requires_grad or not bool(((attention_mask == 0) | (attention_mask == 1)).all())
                or not bool((attention_mask[:, -1] == 1).all())):
            raise ValueError('attention_mask must be native-device binary [1,T], with the last token unmasked')
        self.attention_mask = attention_mask.detach().clone()
        self._fingerprint = _parameters(model)
        self._source_rows = None
        self._target = None
        self._calls = {}
        self._busy = False
        self._capture_baseline()

    @property
    def baseline_source(self):
        return self._source_rows[0, -1, :].detach().clone()

    @property
    def baseline_target(self):
        return self._target.detach().clone()

    def _capture_baseline(self):
        handles = []
        seen = set()

        def source_hook(module, args, output):
            if self._source_rows is not None:
                raise RuntimeError('Source decoder block fired twice')
            value = _hidden(output, 1, self.tokens, self.width)
            if not bool(torch.isfinite(value).all()):
                raise ValueError('Nonfinite native source residual')
            self._source_rows = value.detach().clone()

        def capture_call(index):
            def hook(module, args, kwargs):
                if index in seen or len(args) != 1:
                    raise ValueError('Unexpected repeated/positional native decoder call')
                if kwargs.get('past_key_values') is not None or kwargs.get('use_cache') is not False:
                    raise ValueError('Suffix capture must use_cache=False and past_key_values=None')
                if kwargs.get('position_embeddings') is None or kwargs.get('position_ids') is None:
                    raise ValueError('Native rotary/position arguments were not provided')
                self._calls[index] = _constant(kwargs)
                seen.add(index)
            return hook

        def target_hook(module, args, output):
            value = _hidden(output, 1, self.tokens, self.width)[0, -1, :]
            if not bool(torch.isfinite(value).all()):
                raise ValueError('Nonfinite native target residual')
            self._target = value.detach().clone()
            raise _TargetReached

        try:
            handles.append(self.layers[self.source_block].register_forward_hook(source_hook))
            for index in range(self.source_block+1, self.target_block+1):
                handles.append(self.layers[index].register_forward_pre_hook(capture_call(index), with_kwargs=True))
            handles.append(self.layers[self.target_block].register_forward_hook(target_hook))
            try:
                with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
                    self.model(input_ids=self.input_ids, attention_mask=self.attention_mask, use_cache=False)
            except _TargetReached:
                pass
        finally:
            for handle in reversed(handles):
                handle.remove()
        expected = set(range(self.source_block+1, self.target_block+1))
        if self._source_rows is None or self._target is None or seen != expected:
            raise RuntimeError('Native baseline did not capture the complete requested suffix')
        _execution_guard(self.model, self.device, self._fingerprint)

    def _vector(self, value, label):
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (self.width,)
                or value.device != self.device or value.dtype != torch.float32):
            raise ValueError(f'{label} must be native-device FP32 [{self.width}]; no implicit casts/broadcasting')
        if value.is_inference():
            raise ValueError(f'{label} must not be an inference-mode tensor')
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f'{label} must be finite')
        return value

    def __call__(self, vector):
        _execution_guard(self.model, self.device, self._fingerprint)
        vector = self._vector(vector, 'source vector')
        if self._busy:
            raise RuntimeError('ResidualMap calls must not overlap')
        self._busy = True
        try:
            with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
                hidden = self._source_rows.clone()
                hidden[0, -1, :] = vector
                for index in range(self.source_block+1, self.target_block+1):
                    hidden = self.layers[index](hidden, **self._calls[index])
                    hidden = _hidden(hidden, 1, self.tokens, self.width)
                target = hidden[0, -1, :]
                if not bool(torch.isfinite(target).all()):
                    raise ValueError('Nonfinite target residual')
                return target
        finally:
            self._busy = False

    def _point(self, at):
        return self.baseline_source if at is None else self._vector(at, 'linearization point').detach().clone()

    def jvp(self, direction, *, at=None):
        """Reverse-over-reverse Jv; no dense Jacobian and no parameter gradients."""
        _execution_guard(self.model, self.device, self._fingerprint)
        direction = self._vector(direction, 'direction').detach()
        with torch.enable_grad():
            point = self._point(at).requires_grad_(True)
            target = self(point)
            cotangent = torch.zeros_like(target, requires_grad=True)
            reverse = torch.autograd.grad(target, point, cotangent, create_graph=True)[0]
            result = torch.autograd.grad(reverse, cotangent, direction)[0]
        return result.detach()

    def vjp(self, cotangent, *, at=None):
        """J^T u at the same fixed-prompt primal map used by jvp."""
        _execution_guard(self.model, self.device, self._fingerprint)
        cotangent = self._vector(cotangent, 'cotangent').detach()
        with torch.enable_grad():
            point = self._point(at).requires_grad_(True)
            result = torch.autograd.grad(self(point), point, cotangent)[0]
        return result.detach()

    def central_fd(self, direction, step, *, at=None):
        """Central finite difference, with an explicit positive finite step."""
        _execution_guard(self.model, self.device, self._fingerprint)
        direction = self._vector(direction, 'direction').detach()
        if isinstance(step, bool) or not isinstance(step, Real) or not math.isfinite(step) or step <= 0:
            raise ValueError('Finite-difference step must be finite and positive')
        point = self._point(at)
        plus = self(point+float(step)*direction).detach()
        minus = self(point-float(step)*direction).detach()
        return (plus-minus)/(2*float(step))

    def fd(self, direction, step, *, at=None):
        """Alias for central_fd; step multiplies the complete supplied direction."""
        return self.central_fd(direction, step, at=at)

    def validate_jvp(self, direction, step, *, at=None, rtol=1e-2, atol=1e-5):
        """Explicit diagnostic at a supplied step; no adaptive epsilon search."""
        for value in (rtol, atol):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError('Validation tolerances must be finite and nonnegative')
        product = self.jvp(direction, at=at)
        finite_difference = self.central_fd(direction, step, at=at)
        error = (product-finite_difference).double().norm()
        scale = max(float(product.double().norm()), float(finite_difference.double().norm()), 1e-12)
        return {'passed': bool(torch.allclose(product, finite_difference, rtol=rtol, atol=atol)),
                'absolute_l2_error': float(error), 'relative_l2_error': float(error)/scale,
                'jvp_norm': float(product.double().norm()), 'fd_norm': float(finite_difference.double().norm()),
                'step': float(step), 'rtol': float(rtol), 'atol': float(atol),
                'source_block': self.source_block, 'target_block': self.target_block,
                'position': self.position, 'scope': 'original-prompt last-position to same-position residual derivative'}


def build_residual_map(model, input_ids, source_block, target_block, *, attention_mask=None):
    return ResidualMap(model, input_ids, source_block, target_block, attention_mask=attention_mask)
