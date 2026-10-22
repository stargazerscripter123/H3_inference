"""Ulysses sequence-parallel forward for MiniMax-H3.

The packed sequence is split by rows across ranks. Every per-token op (patch
proj, AdaLN, RoPE, MLP, out_proj) is row-local and needs no communication; only
attention does, via two all-to-all exchanges that trade the head dim for the
sequence dim. Attention math is therefore unchanged: each rank runs complete
attention over the full sequence for 56/P of the heads.

Row splits are deliberately uneven (base + remainder) instead of padding the
sequence, so no attention mask is needed and results stay numerically equivalent
to the single-GPU path.

Targets the MiniMax H3 API introduced by ComfyUI commit 8d534945.
sp_forward mirrors MiniMaxH3Model._forward and must be re-checked when it changes.
"""

# BEGIN reviewed SP2 activation amax imports
from contextlib import nullcontext
from contextvars import ContextVar
import threading
import types
# END reviewed SP2 activation amax imports

import logging
import os
import time

import torch
import torch.distributed as dist

import comfy.ldm.common_dit
import comfy.model_management
import comfy.model_prefetch
import comfy.quant_ops
from comfy.ldm.minimax.model import (
    AUDIO_COND_TIMESTEP,
    VISUAL_COND_TIMESTEP,
    PackedLayout,
    mask_row_values,
    pack_audio,
    patchify_video,
    rope_rotation_table,
    time_shift_sigma,
    unpack_audio,
    unpatchify_video,
)
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

# BEGIN reviewed block0 diagnostic trace
BLOCK0_TRACE_REQUEST = "_minimax_sp_block0_trace_request"
BLOCK0_TRACE_RESULT = "_minimax_sp_block0_trace_result"
BLOCK0_TRACE_POINTS = tuple(name + suffix for name in
    ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
    for suffix in (".pre", ".post")) + tuple(f"block{index}.post" for index in range(50))
HEAD_METADATA_DTYPES = ("torch.float32", "torch.bfloat16", "torch.float16",
                        "torch.float8_e4m3fn", "torch.float8_e5m2")
HEAD_METADATA_QUANT = ("nvfp4", "fp8_e4m3fn", "fp8_e5m2", "mxfp8")
HEAD_METADATA_LAYOUTS = ("TensorCoreNVFP4Layout", "TensorCoreFP8Layout", "TensorCoreMXFP8Layout")


def stored_head_metadata(final_layer):
    """Read attributes only: never materialize/cast/hash a lazy head weight."""
    def known(value, allowed):
        return value if value in allowed else "other"
    result = []
    for name in ("video_out", "audio_out"):
        head = getattr(final_layer, name)
        weight, bias = head.weight, head.bias
        quant, layout = getattr(head, "quant_format", None), getattr(head, "layout_type", None)
        result.append({
            "head": name,
            "weight_type": known(type(weight).__name__, ("Parameter", "Tensor", "QuantizedTensor")),
            "weight_dtype": known(str(weight.dtype), HEAD_METADATA_DTYPES),
            "weight_shape": list(weight.shape),
            "weight_device_type": known(weight.device.type, ("cpu", "cuda", "meta")),
            "weight_is_meta": bool(weight.is_meta),
            "bias_dtype": None if bias is None else known(str(bias.dtype), HEAD_METADATA_DTYPES),
            "bias_shape": None if bias is None else list(bias.shape),
            "quant_format": None if quant is None else known(quant, HEAD_METADATA_QUANT),
            "layout_type": None if layout is None else known(layout, HEAD_METADATA_LAYOUTS),
            "full_precision_mm": bool(getattr(head, "_full_precision_mm", False)),
            "has_v": hasattr(head, "_v"), "has_prefetch": hasattr(head, "_prefetch"),
            "weight_function_count": len(getattr(head, "weight_function", ())),
            "bias_function_count": len(getattr(head, "bias_function", ())),
        })
    return result


class Block0Trace:
    """Diagnostic CPU samples only; hooks never change a model result.

    Block0 projections plus all 50 block exits copy seven global rows (all features) per SP rank immediately,
    before native in-place RoPE/residual updates can change those rows. The
    native oracle samples the union of both ranks' rows. No tensor is logged.
    """
    def __init__(self, dit, *, rank, start, stop, rows=None):
        self.dit, self.rank, self.start, self.stop = dit, rank, start, stop
        count = stop - start
        self.rows = (sorted({start + i * (count - 1) // 6 for i in range(7)})
                     if rows is None and count > 0 else list(rows or []))
        self.entries, self.errors, self.handles = {}, [], []
        self.head_metadata = None
        self.head_errors = []
        self.entered = False

    def _capture(self, point, tensor):
        try:
            if point in self.entries:
                raise RuntimeError("duplicate_trace_point")
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
                raise RuntimeError("trace_requires_matrix")
            if tensor.shape[0] != self.stop - self.start or not self.rows:
                raise RuntimeError("trace_row_extent_mismatch")
            if self.rows != sorted(set(self.rows)) or any(
                    row < self.start or row >= self.stop for row in self.rows):
                raise RuntimeError("trace_global_rows_invalid")
            local_rows = [row - self.start for row in self.rows]
            samples = tensor[local_rows].detach().to(device="cpu").clone()
            minimum, maximum = torch.aminmax(tensor.detach())
            self.entries[point] = {
                "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "rows": list(self.rows), "samples": samples,
                "amax": max(abs(minimum.item()), abs(maximum.item())),
                "finite": bool(torch.isfinite(minimum)) and bool(torch.isfinite(maximum)),
            }
        except Exception as exc:
            self.errors.append(type(exc).__name__)

    def __enter__(self):
        if self.entered:
            raise RuntimeError("block0_trace_reentry")
        self.entered = True
        try:
            if len(self.dit.blocks) != 50:
                raise RuntimeError("layer_trace_requires_50_blocks")
            block = self.dit.blocks[0]
            for name, module in (("attn.qkv_proj", block.attn.qkv_proj),
                                 ("attn.out_proj", block.attn.out_proj),
                                 ("mlp.fc1", block.mlp.fc1), ("mlp.fc2", block.mlp.fc2)):
                def before(_module, inputs, point=name + ".pre"):
                    self._capture(point, inputs[0] if inputs else None)
                def after(_module, _inputs, output, point=name + ".post"):
                    self._capture(point, output)
                self.handles.append(module.register_forward_pre_hook(before))
                self.handles.append(module.register_forward_hook(after))
            for index, block in enumerate(self.dit.blocks):
                self.handles.append(block.register_forward_hook(
                    lambda _module, _inputs, output, point=f"block{index}.post": self._capture(point, output)))
            if self.rank == -1:
                self.handles.append(self.dit.final_layer.register_forward_pre_hook(
                    lambda module, _inputs: self.capture_head_metadata(module)))
        except Exception as exc:
            self.errors.append(type(exc).__name__)
            self._remove()
        return self

    def _remove(self):
        while self.handles:
            handle = self.handles.pop()
            try:
                handle.remove()
            except Exception as exc:
                self.errors.append(type(exc).__name__)

    def __exit__(self, _kind, _value, _traceback):
        self._remove()
        return False

    def packet(self):
        return {"rank": self.rank, "start": self.start, "stop": self.stop,
                "rows": list(self.rows), "errors": list(self.errors),
                "entries": self.entries, "head_metadata": self.head_metadata,
                "head_errors": list(self.head_errors)}

    def capture_head_metadata(self, final_layer):
        try:
            if self.head_metadata is not None:
                raise RuntimeError("head_metadata_duplicate_capture")
            self.head_metadata = stored_head_metadata(final_layer)
        except Exception as exc:
            self.head_errors.append(type(exc).__name__)


def block0_trace_rows(packets):
    """Validate rank/window identity before installing native-oracle hooks."""
    if not isinstance(packets, list) or len(packets) != 2:
        raise RuntimeError("block0_trace_requires_two_rank_packets")
    cursor = 0
    rows = []
    for rank, packet in enumerate(packets):
        if (not isinstance(packet, dict) or packet.get("rank") != rank
                or packet.get("start") != cursor or packet.get("stop", cursor) <= cursor
                or packet.get("errors") or set(packet.get("entries", {})) != set(BLOCK0_TRACE_POINTS)):
            raise RuntimeError("block0_trace_rank_packet_invalid")
        stop = packet["stop"]
        expected = sorted({cursor + i * (stop - cursor - 1) // 6 for i in range(7)})
        if packet.get("rows") != expected:
            raise RuntimeError("block0_trace_sample_rows_invalid")
        rows.extend(expected)
        cursor = stop
    return rows, cursor


def block0_trace_summary(packets, native):
    """Sampled-row parity, not full-layer parity; return scalars/metadata only."""
    rows, total = block0_trace_rows(packets)
    if (native.get("errors") or native.get("start") != 0 or native.get("stop") != total
            or native.get("rows") != rows or set(native.get("entries", {})) != set(BLOCK0_TRACE_POINTS)):
        raise RuntimeError("block0_trace_native_packet_invalid")
    result = []
    for point in BLOCK0_TRACE_POINTS:
        expected = native["entries"][point]
        if (expected["rows"] != rows or expected["shape"][0] != total
                or list(expected["samples"].shape) != [len(rows), expected["shape"][1]]):
            raise RuntimeError("block0_trace_native_sample_shape_invalid")
        global_amax = max(packet["entries"][point]["amax"] for packet in packets)
        for packet in packets:
            actual = packet["entries"][point]
            if (actual["rows"] != packet["rows"]
                    or actual["shape"] != [packet["stop"] - packet["start"], expected["shape"][1]]
                    or actual["dtype"] != expected["dtype"]
                    or list(actual["samples"].shape) != [len(packet["rows"]), expected["shape"][1]]):
                raise RuntimeError("block0_trace_sp_sample_shape_invalid")
            selected = expected["samples"][[rows.index(row) for row in packet["rows"]]].float()
            sampled = actual["samples"].float()
            delta = (sampled - selected).abs().max().item()
            denominator = max(selected.abs().max().item(), 1e-12)
            result.append({"rank": packet["rank"], "point": point,
                           "shape": actual["shape"], "dtype": actual["dtype"],
                           "sampled_rows": len(packet["rows"]), "global_rows": actual["rows"],
                           "finite": actual["finite"] and expected["finite"],
                           "max_abs": delta, "relative": delta / denominator,
                           "sp_local_amax": actual["amax"], "sp_global_amax": global_amax,
                           "native_global_amax": expected["amax"]})
    return result


def head_metadata_summary(packets, native):
    """Six closed-schema records; these are not executed/cast weight values."""
    result = []
    expected_keys = {"head", "weight_type", "weight_dtype", "weight_shape", "weight_device_type",
                     "weight_is_meta", "bias_dtype", "bias_shape", "quant_format", "layout_type",
                     "full_precision_mm", "has_v", "has_prefetch", "weight_function_count", "bias_function_count"}
    for packet, phase, rank in [(packets[0], "sp2", 0), (packets[1], "sp2", 1), (native, "native", -1)]:
        heads = packet.get("head_metadata")
        if packet.get("rank") != rank or packet.get("head_errors") or not isinstance(heads, list) or len(heads) != 2:
            raise RuntimeError("head_metadata_packet_invalid")
        for head, name, width in zip(heads, ("video_out", "audio_out"), (3072, 1024)):
            if (not isinstance(head, dict) or set(head) != expected_keys or head["head"] != name
                    or head["weight_shape"] != [width, 5376] or head["bias_shape"] != [width]
                    or head["weight_type"] not in ("Parameter", "Tensor", "QuantizedTensor", "other")
                    or head["weight_dtype"] not in (*HEAD_METADATA_DTYPES, "other")
                    or head["bias_dtype"] not in (*HEAD_METADATA_DTYPES, "other")
                    or head["weight_device_type"] not in ("cpu", "cuda", "meta", "other")
                    or head["quant_format"] not in (None, *HEAD_METADATA_QUANT, "other")
                    or head["layout_type"] not in (None, *HEAD_METADATA_LAYOUTS, "other")
                    or any(type(head[key]) is not bool for key in ("weight_is_meta", "full_precision_mm", "has_v", "has_prefetch"))
                    or any(type(head[key]) is not int or head[key] < 0 for key in ("weight_function_count", "bias_function_count"))):
                raise RuntimeError("head_metadata_fields_invalid")
            result.append({"schema_version": 1, "phase": phase, "rank": rank,
                           "scope": "stored_runtime_metadata", "execution_weight_verified": False, **head})
    return result
# END reviewed block0 diagnostic trace

# BEGIN reviewed SP2 activation amax scope
_SP_NVFP4_TICKET = ContextVar('minimax_sp_nvfp4_activation_ticket', default=None)
_SP_NVFP4_LOCK = threading.Lock()
_ABSENT = object()


class SPNVFP4AmaxScope:
    """Use immediately around all 50 SP blocks, never around the native oracle.

The classmethod shim delegates unchanged unless this thread is currently inside
one of this scope's 200 guarded module forwards. A dynamic input consumes its
ticket before calling native quantize. Weight scale='recalculate', stochastic
weight requantization, and calibrated/fixed input scales are never synchronized.

Required integration preconditions (not claimed validated by this draft):
Comfy b1a247c9; kitchen 0.2.39 NVFP4 linear_input_act returns NotImplemented;
no concurrent model forward; no training/LoRA/function patches; exactly 50
blocks; nonempty local row shards. Same scope must run on every SP rank.
"""
    def __init__(self, dit, ctx, *, torch_module, dist_module, quant_ops):
        self.dit, self.ctx = dit, ctx
        self.torch, self.dist, self.quant_ops = torch_module, dist_module, quant_ops
        self.layout = quant_ops.TensorCoreNVFP4Layout
        self.saved = []
        self.calls = {}
        self.syncs = {}
        self.entered = False

    def _inventory(self):
        if self.ctx.world != 2 or self.ctx.seq_len < self.ctx.world or self.ctx.local <= 0:
            raise RuntimeError('sp_amax_requires_two_nonempty_row_shards')
        if self.dist.get_world_size(self.ctx.group) != self.ctx.world:
            raise RuntimeError('sp_amax_group_size_mismatch')
        if len(self.dit.blocks) != 50:
            raise RuntimeError('sp_amax_requires_50_blocks')
        targets = []
        for index, block in enumerate(self.dit.blocks):
            for name, module in (('attn.qkv_proj',block.attn.qkv_proj),
                                 ('attn.out_proj',block.attn.out_proj),
                                 ('mlp.fc1',block.mlp.fc1),('mlp.fc2',block.mlp.fc2)):
                if (getattr(module,'quant_format',None) != 'nvfp4'
                    or getattr(module,'layout_type',None) != 'TensorCoreNVFP4Layout'
                    or getattr(module,'_full_precision_mm',False)
                    or getattr(module,'comfy_force_cast_weights',False)
                    or getattr(module,'weight_function',())
                    or getattr(module,'bias_function',())):
                    raise RuntimeError('sp_amax_target_quantization_guard_failed')
                input_scale = getattr(module,'input_scale',None)
                if input_scale is not None and not (isinstance(input_scale,self.torch.Tensor)
                        or type(input_scale) in (int,float)):
                    raise RuntimeError('sp_amax_unsupported_module_input_scale')
                targets.append((f'blocks.{index}.{name}',module))
        if len({id(module) for _,module in targets}) != 200:
            raise RuntimeError('sp_amax_target_alias_rejected')
        return targets

    def __enter__(self):
        if self.entered or self.saved or getattr(self,'locked',False):
            raise RuntimeError('sp_amax_scope_reentry_rejected')
        self.thread_id = threading.get_ident()
        # Rank-local inventory failure still participates in the same preflight.
        error = None
        targets = []
        signature = []
        try:
            targets = self._inventory()
            signature = [(name,getattr(module,'input_scale',None) is None,
                          module.in_features,module.out_features) for name,module in targets]
            descriptor = self.layout.__dict__.get('quantize')
            if not isinstance(descriptor,classmethod):
                raise RuntimeError('sp_amax_quantize_descriptor_changed')
            if not _SP_NVFP4_LOCK.acquire(blocking=False):
                raise RuntimeError('sp_amax_concurrent_scope_rejected')
            self.locked = True
        except Exception as exc:
            error = type(exc).__name__
            signature = []
        replies = [None] * self.ctx.world
        try:
            self.dist.all_gather_object(replies, {'error':error,'signature':signature}, group=self.ctx.group)
            if error or any(reply != {'error':None,'signature':signature} for reply in replies):
                raise RuntimeError('sp_amax_rank_inventory_mismatch')
            self.descriptor = descriptor
            self.original_quantize = descriptor.__get__(None,self.layout)
            self.targets = targets
            self.dynamic = {name:getattr(module,'input_scale',None) is None for name,module in targets}
            for name,module in targets:
                original = module.forward
                saved = module.__dict__.get('forward',_ABSENT)
                self.saved.append((module,saved))
                self.calls[name] = self.syncs[name] = 0
                module.forward = types.MethodType(self._forward_wrapper(name,original),module)
            scope = self
            def quantize(cls,tensor,scale=None,stochastic_rounding=0,inplace_ops=False):
                ticket = _SP_NVFP4_TICKET.get()
                if (threading.get_ident() == scope.thread_id
                    and ticket is not None and ticket['scope'] is scope and scale is None
                    and stochastic_rounding == 0):
                    if ticket['consumed']:
                        raise RuntimeError('sp_amax_second_dynamic_quantization_rejected')
                    if tensor.dim() != 2 or tensor.shape[0] != scope.ctx.local:
                        raise RuntimeError('sp_amax_input_row_scope_changed')
                    ticket['consumed'] = True
                    # Native amax is computed before padding, in input dtype.
                    # MAX on that dtype preserves native division/rounding order.
                    maximum = scope.torch.amax(tensor.abs())
                    scope.dist.all_reduce(maximum,op=scope.dist.ReduceOp.MAX,group=scope.ctx.group)
                    scale = maximum / (scope.quant_ops.ck.float_utils.F8_E4M3_MAX
                                       * scope.quant_ops.ck.float_utils.F4_E2M1_MAX)
                    scope.syncs[ticket['name']] += 1
                return scope.original_quantize(tensor,scale=scale,
                    stochastic_rounding=stochastic_rounding,inplace_ops=inplace_ops)
            self.layout.quantize = classmethod(quantize)
            self.entered = True
            return self
        except BaseException:
            self._restore()
            raise

    def _forward_wrapper(self,name,original):
        def forward(module,input,*args,**kwargs):
            if threading.get_ident() != self.thread_id:
                return original(input,*args,**kwargs)
            self.calls[name] += 1
            if self.calls[name] != 1:
                raise RuntimeError('sp_amax_repeated_target_forward_rejected')
            # Preserve fixed input_scale exactly; do not reinterpret floats/tensors.
            if not self.dynamic[name]:
                return original(input,*args,**kwargs)
            if (getattr(module,'input_scale',None) is not None or input.requires_grad
                or isinstance(input,self.quant_ops.QuantizedTensor)):
                raise RuntimeError('sp_amax_dynamic_inference_guard_changed')
            ticket = {'scope':self,'name':name,'consumed':False}
            token = _SP_NVFP4_TICKET.set(ticket)
            try:
                output = original(input,*args,**kwargs)
                if not ticket['consumed']:
                    raise RuntimeError('sp_amax_input_quantization_not_observed')
                return output
            finally:
                _SP_NVFP4_TICKET.reset(token)
        return forward

    def _restore(self):
        # Restore descriptors, not bound-function substitutes.
        if self.entered:
            self.layout.quantize = self.descriptor
            self.entered = False
        for module,saved in reversed(self.saved):
            if saved is _ABSENT:
                delattr(module,'forward')
            else:
                module.forward = saved
        self.saved.clear()
        if getattr(self,'locked',False):
            self.locked = False
            _SP_NVFP4_LOCK.release()

    def __exit__(self,exc_type,exc,tb):
        try:
            if exc_type is None:
                expected = {name:int(self.dynamic[name]) for name,_ in self.targets}
                if any(n != 1 for n in self.calls.values()) or self.syncs != expected:
                    raise RuntimeError('sp_amax_forward_sync_counts_mismatch')
        finally:
            self._restore()
        return False

    def counts(self):
        return {'target_forwards':sum(self.calls.values()),'activation_syncs':sum(self.syncs.values()),
                'per_block':[sum(n for name,n in self.syncs.items() if name.startswith(f'blocks.{i}.'))
                             for i in range(50)]}

# END reviewed SP2 activation amax scope

PROFILE_OPS = bool(os.environ.get("MINIMAX_SP_PROFILE_OPS"))
_prof_acc = {}


class region:
    """Accumulate wall time per named region; only active under MINIMAX_SP_PROFILE_OPS."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        if PROFILE_OPS:
            torch.cuda.synchronize()
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if PROFILE_OPS:
            torch.cuda.synchronize()
            _prof_acc[self.name] = _prof_acc.get(self.name, 0.0) + time.perf_counter() - self.t0
        return False


def prof_report(total):
    rows = sorted(_prof_acc.items(), key=lambda kv: -kv[1])
    body = "  ".join(f"{k}={v * 1000:.0f}ms" for k, v in rows)
    logging.info(f"[minimax_sp][ops] step={total * 1000:.0f}ms  {body}")
    _prof_acc.clear()


def validate_transformer_options(transformer_options):
    unsupported = [key for key in ("patches_replace", "patches") if transformer_options.get(key)]
    unsupported.extend(key for key in transformer_options
                       if "attention" in key.lower() and key not in unsupported)
    if unsupported:
        raise RuntimeError(f"MiniMax SP does not support transformer {', '.join(unsupported)}; remove those patches or use the single-GPU model")
    if transformer_options.get("multigpu_thread_device") is not None or transformer_options.get("multigpu_clones"):
        raise RuntimeError("MiniMax SP cannot be combined with ComfyUI threaded MultiGPU clones; disable one of the two MultiGPU modes")


class SPContext:
    """Row-sharding descriptor for one forward pass."""

    def __init__(self, rank, world, group, seq_len):
        self.rank = rank
        self.world = world
        self.group = group
        self.seq_len = seq_len
        base, rem = divmod(seq_len, world)
        self.splits = [base + (1 if r < rem else 0) for r in range(world)]
        self.start = sum(self.splits[:rank])
        self.stop = self.start + self.splits[rank]

    @property
    def local(self):
        return self.splits[self.rank]


def a2a_gather_seq_scatter_heads(t, ctx):
    """[s_local, H, D] -> [S_total, H/P, D]."""
    if ctx.world == 1:
        # Diagnostic mirror: preserve fresh-buffer semantics without a new PG.
        return t.contiguous().clone()
    s_local, heads, dim = t.shape
    p = ctx.world
    hp = heads // p
    chunk = hp * dim
    send = t.reshape(s_local, p, chunk).transpose(0, 1).contiguous().view(-1)
    out = torch.empty(ctx.seq_len * chunk, dtype=t.dtype, device=t.device)
    dist.all_to_all_single(
        out, send,
        output_split_sizes=[n * chunk for n in ctx.splits],
        input_split_sizes=[s_local * chunk] * p,
        group=ctx.group,
    )
    return out.view(ctx.seq_len, hp, dim)


def a2a_scatter_seq_gather_heads(t, ctx):
    """[S_total, (H/P)*D] -> [s_local, H*D]"""
    if ctx.world == 1:
        return t.contiguous().clone()
    chunk = t.shape[1]
    p = ctx.world
    s_local = ctx.local
    out = torch.empty(p * s_local * chunk, dtype=t.dtype, device=t.device)
    dist.all_to_all_single(
        out, t.contiguous().view(-1),
        output_split_sizes=[s_local * chunk] * p,
        input_split_sizes=[n * chunk for n in ctx.splits],
        group=ctx.group,
    )
    return out.view(p, s_local, chunk).transpose(0, 1).reshape(s_local, p * chunk)


def run_optimized_attention(attn, q, k, v, heads, transformer_options):
    out = optimized_attention(
        AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v),
        heads, preferred_attention=attn.comfy_attention, mask=None, skip_reshape=True,
        transformer_options=transformer_options)
    if not isinstance(out, torch.Tensor):
        raise TypeError("ComfyUI optimized_attention must return a tensor")
    return out


def sp_attention(attn, x, rope_freqs, ctx, transformer_options):
    s = x.shape[0]
    heads, dim = attn.heads, attn.head_dim
    with region("qkv_proj+qknorm+rope"):
        q, k, v = attn.qkv_proj(x).split(heads * dim, dim=-1)
        v = v.view(s, heads, dim)
        if rope_freqs is not None:
            q = q.view(1, s, heads, dim)
            k = k.view(1, s, heads, dim)
            qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
            kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
            rot = rope_freqs.shape[-3] * 2
            if comfy.model_management.in_training:
                q, k = comfy.quant_ops.ck.rms_rope_split_half(
                    q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
            else:
                comfy.quant_ops.ck.rms_rope_split_half_(
                    q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
            q = q[0]
            k = k[0]
        else:
            q = attn.q_norm(q.view(s, heads, dim))
            k = attn.k_norm(k.view(s, heads, dim))

    hp = heads // ctx.world
    with region("a2a_qkv"):
        q = a2a_gather_seq_scatter_heads(q, ctx)
        k = a2a_gather_seq_scatter_heads(k, ctx)
        v = a2a_gather_seq_scatter_heads(v, ctx)
    with region("attn"):
        out = run_optimized_attention(
            attn, q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0), hp, transformer_options)
    with region("a2a_out"):
        out = a2a_scatter_seq_gather_heads(out.squeeze(0), ctx)
    with region("out_proj"):
        return attn.out_proj(out)


def sp_block(block, x, t_emb, mod_segments, rope_freqs, ctx, transformer_options):
    def attention(h, rope_freqs=None, transformer_options=None):
        return sp_attention(block.attn, h, rope_freqs, ctx, transformer_options)

    with region("block"):
        return block(x, t_emb, mod_segments, rope_freqs,
                     transformer_options=transformer_options, attention=attention)


def shard_segments(segments, start, stop):
    """Clip global (start, stop, row) triples to a rank's row window, in local coords."""
    out = []
    for a, b, row in segments:
        lo, hi = max(a, start), min(b, stop)
        if hi > lo:
            out.append((lo - start, hi - start, row[lo-a:hi-a] if torch.is_tensor(row) else row))
    return out


def _gather_rows(local, counts, ctx, out_dim, device):
    """Padded all_gather of variable-length row blocks, concatenated in rank order."""
    maxn = max(counts)
    buf = torch.zeros(maxn, out_dim, dtype=torch.float32, device=device)
    if local is not None:
        buf[:local.shape[0]] = local
    if ctx.world == 1:
        return buf[:counts[0]].clone()
    gathered = torch.empty(ctx.world * maxn, out_dim, dtype=torch.float32, device=buf.device)
    dist.all_gather_into_tensor(gathered, buf, group=ctx.group)
    gathered = gathered.view(ctx.world, maxn, out_dim)
    return torch.cat([gathered[r, :counts[r]] for r in range(ctx.world) if counts[r] > 0], dim=0)


def sp_forward(dit, x, timestep, context, transformer_options, minimax_payload, rank, world, group,
               denoise_mask=None, audio_denoise_mask=None):
    """Sequence-parallel mirror of MiniMaxH3Model._forward.

    Returns [video_velocity, audio_velocity] on rank 0, None elsewhere.
    """
    validate_transformer_options(transformer_options)
    t_fwd = time.perf_counter()
    video_x, audio_x = x[0], x[1]
    orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, dit.patch_size)
    if video_x.shape[0] != 1:
        raise ValueError("MiniMax H3 supports batch size 1")
    payload = minimax_payload or {}
    device = video_x.device
    dtype = context.dtype

    latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    audio_t = audio_x.shape[-1]
    text_len = context.shape[1]
    layout = payload.get("layout")
    if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
        layout = PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                              keyframes=payload.get("keyframes"), refs=payload.get("refs"))
    transformer_options["minimax_h3_layout"] = layout

    shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", dit.sigma_shift_video))
    shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", dit.sigma_shift_audio))
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))

    vis_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
    aud_aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
    seg_t = {"text": t_v, "video": t_v, "audio": t_a,
             "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),
             "cond_audio": max(t_a, aud_aug), "ref_audio": max(t_a, aud_aug)}
    # Current ComfyUI labels masked rows at their own stream sigma.
    video_rows_t = audio_rows_t = None
    if denoise_mask is not None:
        m = mask_row_values(denoise_mask[0, 0].float(), latent_t, lat_h, lat_w)
        if m is not None:
            rows_t = (1.0 - m * sigma_v.to(m.device)).clamp(max=max(t_v, VISUAL_COND_TIMESTEP))
            if rows_t.unique().numel() == 1:
                seg_t["video"] = float(rows_t[0])
            else:
                video_rows_t = rows_t
    if audio_denoise_mask is not None:
        m = audio_denoise_mask[0, 0].float().reshape(-1)
        if not bool((m >= 1.0 - 1e-3).all()):
            rows_t = (1.0 - m * (1.0 - t_a)).clamp(max=max(t_a, AUDIO_COND_TIMESTEP))
            if rows_t.unique().numel() == 1:
                seg_t["audio"] = float(rows_t[0])
            else:
                audio_rows_t = rows_t
    unique_t = sorted({t_v, t_a} | {seg_t[k] for _, _, k in layout.segments}
                      | (set(video_rows_t.unique().tolist()) if video_rows_t is not None else set())
                      | (set(audio_rows_t.unique().tolist()) if audio_rows_t is not None else set()))
    t_row = {t: i for i, t in enumerate(unique_t)}
    seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "cond_audio": 2, "ref_audio": 2}

    def rows_to_mod_index(rows_t, tag):
        levels = rows_t.unique()
        base = torch.tensor([t_row[v] * 3 + tag for v in levels.tolist()], dtype=torch.long, device=rows_t.device)
        return base[torch.searchsorted(levels, rows_t)]

    text_tags = payload.get("text_token_tags")
    mod_segments = []
    for a, b, kind in layout.segments:
        row_base = t_row[seg_t[kind]] * 3
        if kind == "text" and text_tags is not None:
            tags = text_tags.view(-1).tolist()
            run_start = 0
            for i in range(1, b - a + 1):
                if i == b - a or tags[i] != tags[run_start]:
                    mod_segments.append((a + run_start, a + i, row_base + int(tags[run_start])))
                    run_start = i
        elif kind == "video" and video_rows_t is not None:
            mod_segments.append((a, b, rows_to_mod_index(video_rows_t, 0)))
        elif kind == "audio" and audio_rows_t is not None:
            mod_segments.append((a, b, rows_to_mod_index(audio_rows_t, 2)))
        else:
            mod_segments.append((a, b, row_base + seg_tag[kind]))

    video_seg = next((a, b, t_row[seg_t["video"]]) for a, b, k in layout.segments if k == "video")
    audio_seg = next((a, b, t_row[seg_t["audio"]]) for a, b, k in layout.segments if k == "audio")
    if video_rows_t is not None:
        video_seg = (*video_seg[:2], rows_to_mod_index(video_rows_t, 0) // 3)
    if audio_rows_t is not None:
        audio_seg = (*audio_seg[:2], rows_to_mod_index(audio_rows_t, 0) // 3)

    img_update = layout.img_update.to(device)
    audio_update = layout.audio_update.to(device)
    video_rows = patchify_video(video_x.to(torch.float32), dit.patch_size)
    audio_rows = pack_audio(audio_x.to(torch.float32))
    cond_video_rows = dit._cond_video_rows(payload, device)
    cond_audio_rows = dit._cond_audio_rows(payload, device)

    all_video_rows = video_rows
    if cond_video_rows is not None:
        all_video_rows = torch.empty(img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device)
        all_video_rows[~img_update] = cond_video_rows
        all_video_rows[img_update] = video_rows
    all_audio_rows = audio_rows
    if cond_audio_rows is not None:
        all_audio_rows = torch.empty(audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device)
        all_audio_rows[~audio_update] = cond_audio_rows
        all_audio_rows[audio_update] = audio_rows

    video_embed = dit.video_patch_proj(all_video_rows).to(dtype)
    audio_embed = dit.audio_patch_proj(all_audio_rows).to(dtype)
    text_states = context[0]
    if text_states.shape[-1] != dit.hidden_size:
        text_states = dit.token_refiner(dit.condition_proj(text_states),
                                        transformer_options=transformer_options)

    ctx = SPContext(rank, world, group, layout.seq_len)
    # embedding assembly is row-local and cheap; every rank builds only its own window
    h = torch.empty(ctx.local, dit.hidden_size, dtype=dtype, device=device)
    voff = aoff = 0
    for a, b, kind in layout.segments:
        n = b - a
        lo, hi = max(a, ctx.start), min(b, ctx.stop)
        if hi > lo:
            if kind == "text":
                h[lo - ctx.start:hi - ctx.start] = text_states[lo - a:hi - a]
            elif kind in ("cond", "ref_img", "video"):
                h[lo - ctx.start:hi - ctx.start] = video_embed[voff + (lo - a):voff + (hi - a)]
            else:
                h[lo - ctx.start:hi - ctx.start] = audio_embed[aoff + (lo - a):aoff + (hi - a)]
        if kind in ("cond", "ref_img", "video"):
            voff += n
        elif kind != "text":
            aoff += n

    t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
    if dit.use_adaln_curves:
        table = comfy.model_management.cast_to(dit.adaln_t_table, device=device)
        pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
        t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
    else:
        t_emb = dit.time_embedder(t_vals).to(dtype)


    rope_freqs = rope_rotation_table(dit.rope_freqs(layout.position_ids, device), dtype)
    local_segments = shard_segments(mod_segments, ctx.start, ctx.stop)
    rope_for_blocks = rope_freqs[:, ctx.start:ctx.stop].contiguous()

    if PROFILE_OPS:
        torch.cuda.synchronize()
        _prof_acc["pre"] = time.perf_counter() - t_fwd

    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(dit.blocks), device, transformer_options)
    # BEGIN reviewed block0 trace scope
    trace = (Block0Trace(dit, rank=rank, start=ctx.start, stop=ctx.stop)
             if ctx.world == 2 and transformer_options.get(BLOCK0_TRACE_REQUEST) is True else None)
    # END reviewed block0 trace scope
    # BEGIN reviewed SP2 activation amax loop
    scale_context = (SPNVFP4AmaxScope(dit, ctx, torch_module=torch, dist_module=dist,
                                    quant_ops=comfy.quant_ops) if ctx.world == 2 else nullcontext())
    with scale_context as scale_scope, (trace if trace is not None else nullcontext()):
        for i, block in enumerate(dit.blocks):
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, malloc_scope="block")
            transformer_options["block_index"] = i
            h = sp_block(block, h, t_emb, local_segments, rope_for_blocks, ctx, transformer_options)
    if ctx.world == 2 and os.environ.get("MINIMAX_SP_VERIFY") == "1":
        counts = scale_scope.counts()
        logging.info("[minimax_sp][amax_sync] rank=%d target_forwards=%d activation_syncs=%d per_block=%s",
                     rank, counts["target_forwards"], counts["activation_syncs"], counts["per_block"])
    # END reviewed SP2 activation amax loop
    comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None, malloc_scope="block")

    def counts_for(seg):
        a, b, _ = seg
        out = []
        s = 0
        for n in ctx.splits:
            lo, hi = max(a, s), min(b, s + n)
            out.append(max(0, hi - lo))
            s += n
        return out

    with region("final+gather"):
        def local_seg(seg):
            a, b, row = seg
            lo, hi = max(a, ctx.start), min(b, ctx.stop)
            if hi <= lo:
                return (0, 0, row[:0] if torch.is_tensor(row) else row)
            return (lo-ctx.start, hi-ctx.start, row[lo-a:hi-a] if torch.is_tensor(row) else row)

        if trace is not None:
            trace.capture_head_metadata(dit.final_layer)
        v_local, a_local = dit.final_layer(h, t_emb, local_seg(video_seg), local_seg(audio_seg),
                                          sigma_v, transformer_options.get("sample_sigmas"), (shift_v, shift_a))

        v = _gather_rows(v_local, counts_for(video_seg), ctx, v_local.shape[-1], device)
        a = _gather_rows(a_local, counts_for(audio_seg), ctx, a_local.shape[-1], device)
    # BEGIN reviewed block0 trace gather
    if trace is not None:
        packets = [None] * ctx.world
        # Only verification's first SP2 forward enters this one extra collective.
        # Samples stay internal, after all normal SP output collectives complete.
        dist.all_gather_object(packets, trace.packet(), group=ctx.group)
        if rank == 0:
            transformer_options[BLOCK0_TRACE_RESULT] = packets
    # END reviewed block0 trace gather
    if PROFILE_OPS and rank == 0:
        prof_report(time.perf_counter() - t_fwd)
    if rank != 0:
        return None

    video_out = unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, dit.latents_dim, dit.patch_size)
    video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
    audio_out = unpack_audio(a)
    return [-video_out.to(video_x.dtype), -audio_out.to(audio_x.dtype)]
