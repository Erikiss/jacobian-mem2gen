"""Native Qwen3 temporal cache operations; no training or cache reconstruction.

Offsets count *consumed* continuation tokens. With a prompt of L tokens, offset
0 captures absolute L-1; offset k captures L+k-1 and the cache has length L+k.
A cache swap made after offset8 affects future forwards, not its existing logits.
Only HF DynamicCache containing full-attention DynamicLayer is supported.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
from dataclasses import dataclass
from numbers import Integral

import torch

from .interventions import AttentionPulse, CoordinateCapture, _integers


def _cache_layers(cache):
    from transformers.cache_utils import DynamicCache, DynamicLayer
    if type(cache) is not DynamicCache or getattr(cache, 'offloading', False):
        raise ValueError('Expected a non-offloaded native DynamicCache')
    if not cache.layers or any(type(layer) is not DynamicLayer for layer in cache.layers):
        raise ValueError('Only initialized full-attention DynamicLayer caches are supported')
    lengths = []
    for layer in cache.layers:
        k, v = layer.keys, layer.values
        if (not layer.is_initialized or not isinstance(k, torch.Tensor) or
                not isinstance(v, torch.Tensor) or k.ndim != 4 or k.shape != v.shape or
                k.dtype != torch.float32 or v.dtype != k.dtype or k.device != v.device or
                k.shape[0] != 1 or k.shape[-2] < 1):
            raise ValueError('Cache requires batch1 FP32 K/V [B,KV-heads,absolute-position,D]')
        lengths.append(int(k.shape[-2]))
    if len(set(lengths)) != 1 or int(cache.get_seq_length()) != lengths[0]:
        raise ValueError('Native cache layer lengths must be identical')
    return cache.layers


def cache_length(cache):
    return 0 if cache is None else int(_cache_layers(cache)[0].keys.shape[-2])


def clone_cache(cache):
    """Deep-copy native metadata and tensors; no legacy conversion or forward.

    Qwen3 can mutate a supplied cache even with use_cache=False. Every branch
    therefore receives a fresh clone, and the donor is never fed to a model.
    """
    if cache is None:
        return None
    layers = _cache_layers(cache)
    if any(x.requires_grad for l in layers for x in (l.keys, l.values)):
        raise ValueError('Temporal inference cache must be detached')
    result = copy.deepcopy(cache)
    for before, after in zip(layers, _cache_layers(result)):
        if before is after or before.keys.data_ptr() == after.keys.data_ptr() or before.values.data_ptr() == after.values.data_ptr():
            raise RuntimeError('Cache clone retained shared tensor storage')
    return result


def _sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def cache_manifest(cache, *, hash_tensors=True):
    """Small JSON evidence; full hashes are optional to avoid per-step transfers."""
    layers = _cache_layers(cache)
    records = []
    for index, layer in enumerate(layers):
        row = {'layer': index, 'type': type(layer).__name__, 'shape': list(layer.keys.shape),
               'sequence_length': int(layer.get_seq_length()), 'dtype': str(layer.keys.dtype),
               'device': str(layer.keys.device), 'is_initialized': bool(layer.is_initialized)}
        if hash_tensors:
            row.update(keys_sha256=_sha(layer.keys), values_sha256=_sha(layer.values))
        records.append(row)
    return {'cache_type': type(cache).__name__, 'sequence_length': cache_length(cache),
            'layers': records, 'hashes_included': bool(hash_tensors),
            'layout': '[batch, native KV heads, absolute token position, head dimension]',
            'key_semantics': 'Native normalized and RoPE-rotated keys; do not rotate again.'}


def swap_cache_positions(receiver, donor, *, layer_indices, absolute_positions):
    """Replace only specified native K/V slots in a fresh receiver clone.

    Both histories must contain the identical token/position sequence. This
    function validates lengths/layout; the caller must bind the token-ID hash.
    It does not alter already computed states or logits at the swap boundary.
    """
    receiver_layers, donor_layers = _cache_layers(receiver), _cache_layers(donor)
    blocks = _integers(layer_indices, 'layer_indices')
    positions = _integers(absolute_positions, 'absolute_positions')
    if len(receiver_layers) != len(donor_layers) or cache_length(receiver) != cache_length(donor):
        raise ValueError('Receiver and donor must have identical layer counts and sequence lengths')
    if max(blocks) >= len(receiver_layers) or max(positions) >= cache_length(receiver):
        raise ValueError('Requested cache layer or absolute position is out of range')
    for a, b in zip(receiver_layers, donor_layers):
        if a.keys.shape != b.keys.shape or a.keys.device != b.keys.device:
            raise ValueError('Donor and receiver cache layouts/devices must match')
    before = cache_manifest(receiver)
    donor_before = cache_manifest(donor)
    result = clone_cache(receiver)
    changes = []
    with torch.no_grad():
        for block in blocks:
            rr, dd = result.layers[block], donor_layers[block]
            for kind in ('keys', 'values'):
                dest, src = getattr(rr, kind), getattr(dd, kind)
                old = dest[..., list(positions), :].clone()
                replacement = src[..., list(positions), :]
                dest[..., list(positions), :] = replacement
                diff = replacement.double()-old.double()
                changes.append({'layer': block, 'kind': kind,
                                'changed_scalars': int(torch.count_nonzero(diff)),
                                'delta_norm': float(diff.norm()),
                                'replacement_sha256': _sha(replacement)})
    # Full-slot checks are intentionally strict and only used at swap boundaries.
    selected = set(positions)
    others = [p for p in range(cache_length(result)) if p not in selected]
    for block, (old, new) in enumerate(zip(receiver_layers, result.layers)):
        for kind in ('keys', 'values'):
            oldt, newt = getattr(old, kind), getattr(new, kind)
            if block not in blocks:
                equal = torch.equal(oldt, newt)
            else:
                equal = torch.equal(oldt[..., others, :], newt[..., others, :])
                equal = equal and torch.equal(newt[..., list(positions), :], getattr(donor_layers[block], kind)[..., list(positions), :])
            if not equal:
                raise RuntimeError('Cache swap modified an unintended slot')
    receiver_unchanged = before == cache_manifest(receiver)
    donor_unchanged = donor_before == cache_manifest(donor)
    if not receiver_unchanged or not donor_unchanged:
        raise RuntimeError('A donor/receiver snapshot was mutated during cache swap')
    return result, {'layer_indices': list(blocks), 'absolute_positions': list(positions),
                    'receiver_before': before, 'donor': donor_before,
                    'result': cache_manifest(result), 'changes': changes,
                    'donor_unchanged': donor_unchanged, 'receiver_unchanged': receiver_unchanged,
                    'untouched_slots_exact': True,
                    'timing': 'Swap changes future forwards only; existing boundary logits/states are unchanged.'}


class AbsoluteCapture(CoordinateCapture):
    """One forward capture supporting cached chunks with absolute token indices."""

    def __init__(self, model, coordinates, positions, stats, *, residual_layers=(13,14,15,17,18,19), weight_norms=None):
        super().__init__(model, coordinates, positions, residual_layers,
                         expected_count=len(coordinates))
        self.absolute_positions = tuple(self.positions)
        self.stats = stats
        self.weight_norms = weight_norms
        self.cett = self.denominator = None
        self._cett_seen = set()
        self.input_offset = None
        self.input_tokens = None
        self.mu = torch.as_tensor(stats['refw']['c_mu'], dtype=torch.float32)
        self.sd = torch.as_tensor(stats['refw']['c_sd'], dtype=torch.float32)
        if (self.mu.shape != (len(coordinates),) or self.sd.shape != self.mu.shape or
                not bool(torch.isfinite(self.mu).all() and torch.isfinite(self.sd).all()) or bool((self.sd < 0).any())):
            raise ValueError('Fixed REF-W must be finite and match the ordered coordinates')
        self.sd = self.sd.clamp_min(1e-6)

    def _model_pre(self, module, args, kwargs):
        if self._claimed:
            raise RuntimeError('AbsoluteCapture supports exactly one model forward')
        ids = kwargs.get('input_ids', args[0] if args else None)
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError('Absolute capture expects batch1 input_ids')
        offset = cache_length(kwargs.get('past_key_values'))
        tokens = int(ids.shape[1])
        positions = [p-offset for p in self.absolute_positions]
        if min(positions) < 0 or max(positions) >= tokens:
            raise ValueError('Absolute capture position is outside this cached input chunk')
        expected = torch.arange(offset,offset+tokens,device=ids.device).unsqueeze(0)
        position_ids = kwargs.get('position_ids')
        if position_ids is not None and not torch.equal(position_ids, expected):
            raise ValueError('Position IDs must match contiguous absolute cache positions')
        mask = kwargs.get('attention_mask')
        if mask is not None and (tuple(mask.shape)!=(1,offset+tokens) or not bool(mask.bool().all())):
            raise ValueError('Only unpadded contiguous causal sequences are supported')
        self.positions = tuple(positions)
        self.input_offset, self.input_tokens = offset,tokens
        self._claimed = True
        self.full_passes += 1
        self._window = (1,tokens)

    def _install(self):
        super()._install()
        for block,pairs in self._by_layer.items():
            down = self.layers[block].mlp.down_proj
            columns = [channel for _,channel in pairs]
            weights = (down.weight.detach().float().norm(dim=0)[columns] if self.weight_norms is None
                       else torch.as_tensor(self.weight_norms[block],device=down.weight.device,dtype=torch.float32))
            if tuple(weights.shape)!=(len(columns),) or not bool(torch.isfinite(weights).all()):
                raise ValueError('Precomputed weight norms must follow selected coordinate order per layer')

            def tap(module,args,output,block=block,pairs=pairs,columns=columns,weights=weights):
                if self._window is None:
                    return
                if block in self._cett_seen:
                    raise RuntimeError('CETT layer captured twice')
                z = args[0][:,list(self.positions),:][:,:,columns].abs().float()
                den = output.norm(dim=-1)[:,list(self.positions)].float()
                contribution = z*weights/den.clamp_min(1e-9)[...,None]
                if not bool(torch.isfinite(contribution).all() and torch.isfinite(den).all()):
                    raise ValueError('Nonfinite CETT/denominator')
                if self.cett is None:
                    self.cett = torch.empty((1,len(self.positions),len(self.coordinates)),dtype=torch.float32)
                    self.denominator = torch.empty_like(self.cett)
                order = [index for index,_ in pairs]
                self.cett[:,:,order] = contribution.detach().cpu()
                self.denominator[:,:,order] = den[...,None].expand_as(contribution).detach().cpu()
                self._cett_seen.add(block)
                return
            self._handles.append(down.register_forward_hook(tap))

    def validate_fired(self):
        super().validate_fired()
        if self._cett_seen != set(self._by_layer):
            raise RuntimeError('CETT capture incomplete')
        return self.fired_count

    def arrays(self):
        self.validate_fired()
        result={'raw_z':self.values,'cett':self.cett,'zcett':(self.cett-self.mu)/self.sd,
                'down_output_norm_raw':self.denominator}
        result.update({f'residual_{block}':value for block,value in self.residuals.items()})
        return result


@dataclass
class CacheStep:
    cache: object
    logits: torch.Tensor
    arrays: dict
    evidence: dict


def forward_capture(model, input_ids, coordinates, stats, *, cache=None, capture_positions=None,
                    residual_layers=(13,14,15,17,18,19), pulse=None, residual_patch=None,
                    weight_norms=None, capture=True):
    """One native FP32 SDPA-MATH forward on a cloned cache, with passive capture.

    ``logits`` is CPU FP32 [1,captured positions,vocab]. ``arrays`` contains CPU
    FP32 tensors. ``pulse`` uses existing block/beta/mode/seed fields and is only
    permitted for uncached prefill/rebuild. ``pulse['positions']`` explicitly
    fixes the original prompt position during full-prefix rebuild parity;
    otherwise the default is the last input token. ``capture=False`` skips passive hooks.
    ``residual_patch`` is a fixed {block, delta:[H]} added
    at captured current-token positions; hooks end after this single forward.
    """
    if model.training or getattr(model.config,'model_type',None)!='qwen3':
        raise ValueError('Temporal capture requires model.eval() and original Qwen3 architecture')
    if getattr(model.config,'_attn_implementation',None)!='sdpa':
        raise ValueError('Temporal capture requires native SDPA with forced MATH backend')
    device = next(model.parameters()).device
    if any(p.is_floating_point() and (p.dtype!=torch.float32 or p.device!=device) for p in model.parameters()):
        raise ValueError('All floating parameters must be FP32 on one device')
    if device.type=='cuda' and torch.backends.cuda.matmul.allow_tf32:
        raise ValueError('TF32 must be disabled')
    ids=torch.as_tensor(input_ids,device=device)
    if ids.ndim==1:
        ids=ids.unsqueeze(0)
    if (ids.dtype!=torch.long or ids.ndim!=2 or ids.shape[0]!=1 or ids.shape[1]<1 or
            bool((ids<0).any()) or bool((ids>=model.config.vocab_size).any())):
        raise ValueError('input_ids must be nonempty batch1 integer vocabulary tokens')
    start=cache_length(cache); tokens=int(ids.shape[1]); end=start+tokens
    positions=_integers(capture_positions if capture_positions is not None else [end-1],'capture_positions')
    if min(positions)<start or max(positions)>=end:
        raise ValueError('Capture positions must fall within current chunk')
    if pulse is not None and cache is not None:
        raise ValueError('Attention pulse is prefill-only; cached continuations must be unpulsed')
    pulse_positions=[]
    if pulse is not None:
        pulse_positions=list(_integers(pulse.get('positions',[tokens-1]),'pulse positions'))
        if len(pulse_positions)!=1 or pulse_positions[0]>=tokens:
            raise ValueError('Temporal pulse requires one original prompt position within the uncached input')
    if cache is not None and len(cache.layers)!=len(model.model.layers):
        raise ValueError('Cache layer count differs from model')
    branch=clone_cache(cache)
    patch_count=0;patch_evidence=None
    if residual_patch is not None:
        block=residual_patch['block']
        if isinstance(block,bool) or not isinstance(block,Integral) or not 0<=block<len(model.model.layers):
            raise ValueError('Residual patch block outside model')
        delta=torch.as_tensor(residual_patch['delta'],device=device,dtype=torch.float32).detach()
        if tuple(delta.shape)!=(model.config.hidden_size,) or not bool(torch.isfinite(delta).all()):
            raise ValueError('Residual delta must be one finite fixed hidden-width vector')
        local=[p-start for p in positions]

        def patch(module,args,output):
            nonlocal patch_count,patch_evidence
            if patch_count:
                raise RuntimeError('Residual patch fired twice')
            value=output[0] if isinstance(output,(tuple,list)) else output
            result=value.clone(); old=value[:,local,:]
            result[:,local,:]=old+delta
            actual=result[:,local,:].double()-old.double()
            patch_count+=1
            patch_evidence={'block':int(block),'absolute_positions':list(positions),'requested_delta_norm':float(delta.double().norm()),
                            'actual_delta_norms':actual.norm(dim=-1).cpu().tolist(),
                            'rounding_error_norm':float((actual-delta.double()).norm()),'fired_count':patch_count}
            if isinstance(output,tuple):
                return (result,*output[1:])
            if isinstance(output,list):
                return [result,*output[1:]]
            return result
    pulse_hook=None
    with torch.no_grad(),torch.autocast(device_type=device.type,enabled=False),torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        with contextlib.ExitStack() as stack:
            if residual_patch is not None:
                handle=model.model.layers[block].register_forward_hook(patch)
                stack.callback(handle.remove)  # Installed first: residual capture observes the patched output.
            capture_hook=(stack.enter_context(AbsoluteCapture(model,coordinates,positions,stats,
                residual_layers=residual_layers,weight_norms=weight_norms)) if capture else None)
            if pulse is not None:
                pulse_hook=stack.enter_context(AttentionPulse(model,pulse['block'],pulse_positions,
                    beta=pulse.get('beta',1/1.08),mode=pulse.get('mode','scale'),seed=pulse.get('seed',0)))
            output=model(input_ids=ids,attention_mask=torch.ones((1,end),dtype=torch.long,device=device),
                         position_ids=torch.arange(start,end,device=device).unsqueeze(0),past_key_values=branch,use_cache=True)
        if cache_length(output.past_key_values)!=end:
            raise RuntimeError('Returned native cache has wrong consumed-token length')
        if residual_patch is not None and patch_count!=1:
            raise RuntimeError('Residual patch did not fire exactly once')
        logits=output.logits[:,[p-start for p in positions],:].detach().float().cpu().clone()
        if not bool(torch.isfinite(logits).all()):
            raise ValueError('Nonfinite temporal logits')
    evidence={'cache_input_length':start,'consumed_input_tokens':tokens,'cache_output_length':end,
              'absolute_capture_positions':list(positions),'local_capture_positions':[p-start for p in positions],
              'logit_predicts_absolute_positions':[p+1 for p in positions],
              'incoming_cache_cloned':cache is not None,'capture_fired':0 if capture_hook is None else capture_hook.fired_count,
              'pulse_fired':0 if pulse_hook is None else pulse_hook.fired_count,
              'pulse_absolute_positions':pulse_positions,
              'pulse_requested_delta_norms':None if pulse_hook is None else pulse_hook.requested_delta_norms.tolist(),
              'pulse_actual_delta_norms':None if pulse_hook is None else pulse_hook.actual_delta_norms.tolist(),
              'residual_patch':patch_evidence,'backend':'SDPA MATH','dtype':'float32','use_cache':True}
    return CacheStep(output.past_key_values,logits,{} if capture_hook is None else capture_hook.arrays(),evidence)
