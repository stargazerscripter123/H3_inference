"""CPU shape/numeric tests of the actual gather function; no model/media/GPU IO.

NumPy stands in for CPU tensor storage and all_gather, so these tests do not
claim NCCL, CUDA, NVFP4, full-forward parity, or runtime lifecycle validation.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AFTER = ROOT / "custom_nodes/minimax_sp_pdd/minimax_sp/sp_forward.py"
CONTRACT = ROOT / "sp-node-contract.json"


AMAX_REVIEWED_BLOCKS = {'imports': ('# BEGIN reviewed SP2 activation amax imports\n', '# END reviewed SP2 activation amax imports\n\n', '1b89b51415c462249567b03f9fd53c023b77076555b0b14dd95dc0cbf35b9a56'), 'scope': ('# BEGIN reviewed SP2 activation amax scope\n', '# END reviewed SP2 activation amax scope\n\n', '3d826580e83279db44b64a125ac8d50af10bc4eae65529e417a0f9e78c170eba'), 'loop': ('    # BEGIN reviewed SP2 activation amax loop\n', '    # END reviewed SP2 activation amax loop\n', '47af9e1d2d2d2eeacb2eb9d6e33394e2cfe7584587e0a62e61094fe514ad5675')}
AMAX_ORIGINAL_LOOP = '    for i, block in enumerate(dit.blocks):\n        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block, malloc_scope="block")\n        transformer_options["block_index"] = i\n        h = sp_block(block, h, t_emb, local_segments, rope_for_blocks, ctx, transformer_options)\n'


PROJECTION_REPLAY_REVERSE_EDITS = {'sp_forward.py': [(106, 435, 'def nvfp4_scale_sha256(block_scale, *, logical_rows, logical_cols,\n                       global_start, global_stop, tensor_global_start=0,\n                       chunk_limit_bytes=16777216):\n    """Return a canonical logical-window SHA; never dequantize or move all scales.\n\n    Layout inverse follows comfy-kitchen v0.2.39 float_utils.from_blocked,\n    commit e40e76d9c544b3f74c3b9ea923f7bc3d45a96372, lines 329--352.\n    FP8 bytes, including signed zero/NaN encodings, remain unchanged.\n    """\n    integers = (logical_rows, logical_cols, global_start, global_stop,\n                tensor_global_start, chunk_limit_bytes)\n    if (any(type(value) is not int for value in integers)\n            or logical_rows <= 0 or logical_cols <= 0\n            or tensor_global_start < 0\n            or not 0 < chunk_limit_bytes <= 16777216\n            or not tensor_global_start <= global_start < global_stop <= tensor_global_start + logical_rows\n            or sys.byteorder != "little"):\n        raise RuntimeError("nvfp4_scale_hash_window_or_limit_invalid")\n    padded_rows = ((logical_rows + 127) // 128) * 128\n    padded_cols = ((logical_cols + 3) // 4) * 4\n    tile_bytes = 128 * padded_cols\n    if (not isinstance(block_scale, torch.Tensor) or block_scale.ndim != 2\n            or block_scale.dtype != torch.float8_e4m3fn\n            or tuple(block_scale.shape) != (padded_rows, padded_cols)\n            or not block_scale.is_contiguous()\n            or tile_bytes > chunk_limit_bytes):\n        raise RuntimeError("nvfp4_scale_hash_physical_layout_invalid")\n\n    header = {"schema": "minimax-sp-layer-sha256-v1", "dtype": "torch.uint8",\n              "shape": [global_stop - global_start, logical_cols],\n              "global_start": global_start, "global_stop": global_stop,\n              "byte_order": "little", "order": "C"}\n    digest = hashlib.sha256()\n    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\\n")\n    local_start = global_start - tensor_global_start\n    local_stop = global_stop - tensor_global_start\n    storage = block_scale.view(torch.uint8)\n    copied_bytes = 0\n    for tile_start in range((local_start // 128) * 128, local_stop, 128):\n        tile = storage[tile_start:tile_start + 128].detach().to(\n            device="cpu", non_blocking=False, copy=True,\n            memory_format=torch.contiguous_format)\n        # One complete 128-row physical tile contains all its column blocks.\n        # Do the inverse only after the bounded synchronous copy to CPU.\n        step1 = tile.reshape(-1, 32, 16)\n        step2 = step1.reshape(-1, 32, 4, 4).transpose(1, 2)\n        step3 = step2.reshape(1, padded_cols // 4, 4, 32, 4)\n        step4 = step3.reshape(1, padded_cols // 4, 128, 4)\n        unblocked = step4.permute(0, 2, 1, 3).reshape(128, padded_cols)\n        first = max(local_start, tile_start) - tile_start\n        last = min(local_stop, tile_start + 128) - tile_start\n        selected = unblocked[first:last, :logical_cols].contiguous()\n        array = selected.numpy()\n        view = memoryview(array).cast("B")\n        try:\n            expected = (last - first) * logical_cols\n            if view.nbytes != expected or view.nbytes > chunk_limit_bytes:\n                raise RuntimeError("nvfp4_scale_hash_chunk_size_invalid")\n            digest.update(view)\n            copied_bytes += view.nbytes\n        finally:\n            view.release()\n            del view, array, selected, unblocked, step4, step3, step2, step1, tile\n    if copied_bytes != (global_stop - global_start) * logical_cols:\n        raise RuntimeError("nvfp4_scale_hash_byte_count_invalid")\n    return digest.hexdigest()\n\n# BEGIN reviewed projection M replay diagnostic\nREPLAY_MINIMUM_FREE_BYTES = 1024 * 1024 * 1024\nREPLAY_ERROR_CODES = ("input_guard_failed", "quantized_guard_failed", "memory_budget_insufficient",\n    "quantized_capture_failed", "replay_execution_failed", "output_compare_failed", "hook_install_failed",\n    "duplicate_target_call", "hook_restore_failed")\n\n\nclass ProjectionReplayError(RuntimeError):\n    pass\n\n\ndef _replay_require(condition, code):\n    if not condition:\n        raise ProjectionReplayError(code)\n\n\ndef _replay_digest(metadata):\n    return hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")).hexdigest()\n\n\ndef _replay_quantized_identity(value, *, tensor_global_start, windows):\n    """Hash actual cast QT storage/parameters, excluding padding from input windows."""\n    params, data = value._params, value._qdata\n    shape = list(params.orig_shape)\n    _replay_require(value._layout_cls == "TensorCoreNVFP4Layout" and len(shape) == 2\n        and str(params.orig_dtype) == "torch.bfloat16" and not params.transposed\n        and all(type(n) is int and n > 0 for n in shape) and shape[1] % 16 == 0\n        and str(data.dtype) == "torch.uint8" and data.is_contiguous()\n        and list(data.shape) == [((shape[0] + 15) // 16) * 16, shape[1] // 2]\n        and params.scale.numel() == 1 and str(params.scale.dtype) == "torch.float32",\n        "quantized_guard_failed")\n    scale_sha = streamed_tensor_sha256(params.scale.reshape(1, 1), tensor_global_start=0,\n                                      global_start=0, global_stop=1)["sha256"]\n    items = []\n    for start, stop in windows:\n        packed = streamed_tensor_sha256(data[:shape[0], :shape[1] // 2], tensor_global_start=tensor_global_start,\n                                        global_start=start, global_stop=stop)["sha256"]\n        scales = nvfp4_scale_sha256(params.block_scale, logical_rows=shape[0], logical_cols=shape[1] // 16,\n                                   tensor_global_start=tensor_global_start, global_start=start, global_stop=stop)\n        items.append({"qdata_sha256": packed, "block_scale_sha256": scales, "scale_sha256": scale_sha})\n    return {"shape": shape, "padded_rows": int(data.shape[0]), "dtype": "torch.bfloat16",\n            "layout": "TensorCoreNVFP4Layout", "windows": items}\n\n\ndef _replay_weight_bias(weight, bias):\n    info = _replay_quantized_identity(weight, tensor_global_start=0, windows=[(0, 5376)])\n    _replay_require(info["shape"] == [5376, 7168], "quantized_guard_failed")\n    if bias is None:\n        bias_info = {"present": False}\n    else:\n        _replay_require(list(bias.shape) == [5376] and str(bias.dtype) in ("torch.bfloat16", "torch.float32"),\n                        "quantized_guard_failed")\n        bias_info = {"present": True, "shape": [5376], "dtype": str(bias.dtype),\n            "sha256": streamed_tensor_sha256(bias.reshape(1, 5376), tensor_global_start=0,\n                                             global_start=0, global_stop=1)["sha256"]}\n    return _replay_digest(info), _replay_digest(bias_info)\n\n\ndef _replay_compare_outputs(full, shard, *, global_start, global_stop):\n    """Complete-window numbers; all CPU comparison scratch is row bounded."""\n    _replay_require(list(full.shape)[1:] == [5376] and list(shard.shape) == [global_stop - global_start, 5376]\n                    and str(full.dtype) == str(shard.dtype) == "torch.bfloat16", "output_compare_failed")\n    # At most five FP32 row buffers fit in the 16 MiB scratch budget.\n    chunk_rows = max(1, TRACE_HASH_CHUNK_BYTES // (5376 * 4 * 5))\n    maximum = reference_amax = 0.0\n    unequal = 0\n    finite = True\n    for offset in range(0, global_stop - global_start, chunk_rows):\n        count = min(chunk_rows, global_stop - global_start - offset)\n        a = full[global_start + offset:global_start + offset + count].detach().to(\n            device="cpu", dtype=torch.float32, non_blocking=False, copy=True, memory_format=torch.contiguous_format)\n        b = shard[offset:offset + count].detach().to(\n            device="cpu", dtype=torch.float32, non_blocking=False, copy=True, memory_format=torch.contiguous_format)\n        finite = finite and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())\n        unequal += int(torch.ne(a, b).sum().item())\n        if finite:\n            maximum = max(maximum, float(torch.max(torch.abs(a - b)).item()))\n            reference_amax = max(reference_amax, float(torch.max(torch.abs(a)).item()))\n        del a, b\n    full_sha = streamed_tensor_sha256(full, tensor_global_start=0,\n                                     global_start=global_start, global_stop=global_stop)["sha256"]\n    shard_sha = streamed_tensor_sha256(shard, tensor_global_start=global_start,\n                                      global_start=global_start, global_stop=global_stop)["sha256"]\n    return {"native_output_sha256": full_sha, "replay_output_sha256": shard_sha,\n            "output_bits_match": full_sha == shard_sha, "finite": finite,\n            "max_abs": maximum if finite else None,\n            "relative": maximum / max(reference_amax, 1e-6) if finite else None,\n            "unequal_elements": unequal, "compared_elements": (global_stop - global_start) * 5376}\n\n\nclass ProjectionReplay:\n    """One actual target call observed per scope; native alone replays two row windows.\n\n    Wrapping _forward stays inside the original CastBiasWeightContext. It does not\n    change backend dispatch; a QT argument does not prove a particular CUDA kernel.\n    """\n    def __init__(self, module, *, rank, start, stop, windows):\n        self.module, self.rank, self.start, self.stop = module, rank, start, stop\n        self.windows = list(windows)\n        self.input = self.capture = None\n        self.replays = []\n        self.call_count = 0\n        self.error_code = None\n        self.handles = []\n        self.installed = False\n        self.free_bytes = self.reserved_bytes = self.allocated_bytes = None\n\n    def __enter__(self):\n        try:\n            _replay_require(self.module.in_features == 7168 and self.module.out_features == 5376\n                and getattr(self.module, "pre_quant_scale", None) is None\n                and getattr(self.module, "input_scale", None) is None\n                and getattr(self.module, "quant_format", None) == "nvfp4"\n                and getattr(self.module, "layout_type", None) == "TensorCoreNVFP4Layout"\n                and not getattr(self.module, "_full_precision_mm", False)\n                and not getattr(self.module, "comfy_force_cast_weights", False)\n                and not getattr(self.module, "weight_function", ()) and not getattr(self.module, "bias_function", ()),\n                "input_guard_failed")\n            self.thread = threading.get_ident()\n            self.previous = vars(self.module).get("_forward", _ABSENT)\n            self.original = self.module._forward\n            def before(_module, inputs):\n                self.input = inputs[0] if inputs else None\n            self.handles.append(self.module.register_forward_pre_hook(before))\n            def wrapped(qinput, weight, bias):\n                self.call_count += 1\n                try:\n                    result = self.original(qinput, weight, bias)\n                    self._observe(qinput, weight, bias, result)\n                    return result\n                finally:\n                    self.input = None\n            self.wrapped = wrapped\n            self.module._forward = wrapped\n            self.installed = True\n        except ProjectionReplayError as exc:\n            self.error_code = exc.args[0]\n            self.__exit__(None, None, None)\n        except Exception:\n            self.error_code = "hook_install_failed"\n            self.__exit__(None, None, None)\n        return self\n\n    def _observe(self, qinput, weight, bias, full_output):\n        try:\n            _replay_require(self.call_count == 1, "duplicate_target_call")\n            source = self.input\n            _replay_require(threading.get_ident() == self.thread and source is not None\n                and list(source.shape) == [self.stop - self.start, 7168]\n                and str(source.dtype) == "torch.bfloat16" and source.is_contiguous()\n                and not source.requires_grad, "input_guard_failed")\n            inputs = _replay_quantized_identity(qinput, tensor_global_start=self.start, windows=self.windows)\n            _replay_require(inputs["shape"] == list(source.shape), "quantized_guard_failed")\n            weight_sha, bias_sha = _replay_weight_bias(weight, bias)\n            self.capture = {"input": inputs, "weight": weight_sha, "bias": bias_sha}\n        except ProjectionReplayError as exc:\n            self.error_code = exc.args[0]\n            return\n        except Exception:\n            self.error_code = "quantized_capture_failed"\n            return\n        if self.rank != -1:\n            return\n        try:\n            _replay_require(self.start == 0 and len(self.windows) == 2\n                and self.windows[0][0] == 0 and self.windows[0][1] == self.windows[1][0]\n                and self.windows[1][1] == self.stop, "input_guard_failed")\n            torch.cuda.synchronize(source.device)\n            self.free_bytes = int(torch.cuda.mem_get_info(source.device)[0])\n            self.reserved_bytes = int(torch.cuda.memory_reserved(source.device))\n            self.allocated_bytes = int(torch.cuda.memory_allocated(source.device))\n            _replay_require(self.free_bytes >= REPLAY_MINIMUM_FREE_BYTES, "memory_budget_insufficient")\n            for rank, (start, stop) in enumerate(self.windows):\n                qshard = shard_output = None\n                try:\n                    xshard = source[start:stop]\n                    _replay_require(xshard.is_contiguous(), "input_guard_failed")\n                    qshard = type(qinput).from_float(xshard, qinput._layout_cls, scale=qinput._params.scale)\n                    replay_input = _replay_quantized_identity(qshard, tensor_global_start=start, windows=[(start, stop)])\n                    # _forward is called directly, so replay cannot recurse through the module\'s hooks.\n                    original_weight, original_bias = weight, bias\n                    shard_output = self.original(qshard, original_weight, original_bias)\n                    _replay_require(original_weight is weight and original_bias is bias\n                        and torch.equal(qshard._params.scale, qinput._params.scale), "quantized_guard_failed")\n                    numbers = _replay_compare_outputs(full_output, shard_output, global_start=start, global_stop=stop)\n                    self.replays.append({"schema_version": 1, "block": 34, "point": "attn.out_proj", "rank": rank,\n                        "global_start": start, "global_stop": stop, "logical_rows": stop - start,\n                        "padded_rows": replay_input["padded_rows"], "full_logical_rows": self.stop,\n                        "full_padded_rows": inputs["padded_rows"], "input_width": 7168, "output_width": 5376,\n                        "dtype": "torch.bfloat16", "layout": "TensorCoreNVFP4Layout", "scale_reused": True,\n                        "weight_object_reused": original_weight is weight, "bias_object_reused": original_bias is bias,\n                        "backend_verified": False, "native_input": inputs["windows"][rank],\n                        "replay_input": replay_input["windows"][0],\n                        "input_match": inputs["windows"][rank] == replay_input["windows"][0], **numbers})\n                finally:\n                    del qshard, shard_output\n        except ProjectionReplayError as exc:\n            self.error_code = exc.args[0]\n        except Exception:\n            self.error_code = "replay_execution_failed"\n\n    def __exit__(self, _kind, _value, _traceback):\n        self.input = None\n        if self.installed:\n            try:\n                _replay_require(self.module._forward is self.wrapped, "hook_restore_failed")\n                if self.previous is _ABSENT:\n                    delattr(self.module, "_forward")\n                else:\n                    self.module._forward = self.previous\n                self.installed = False\n            except Exception:\n                self.error_code = "hook_restore_failed"\n        while self.handles:\n            try:\n                self.handles.pop().remove()\n            except Exception:\n                self.error_code = "hook_restore_failed"\n        return False\n\n    def packet(self):\n        code = self.error_code\n        if code is None and (self.call_count != 1 or self.capture is None or (self.rank == -1 and len(self.replays) != 2)):\n            code = "quantized_capture_failed"\n        status = "complete" if code is None else ("budget_insufficient" if code == "memory_budget_insufficient" else "error")\n        return {"capture": self.capture, "replays": list(self.replays), "status": {\n            "schema_version": 1, "block": 34, "point": "attn.out_proj", "phase": "native" if self.rank == -1 else "sp2",\n            "rank": self.rank, "status": status, "error_code": code, "call_count": self.call_count,\n            "captured": self.capture is not None, "replay_count": len(self.replays), "backend_verified": False,\n            "minimum_free_bytes": REPLAY_MINIMUM_FREE_BYTES, "free_bytes": self.free_bytes,\n            "reserved_bytes": self.reserved_bytes, "allocated_bytes": self.allocated_bytes}}\n\n\ndef projection_replay_summary(packets, native):\n    """Only metadata leaves the hook; incomplete replay never promotes the original parity gate."""\n    actual, replays, statuses = [], [], []\n    records = [packet.get("projection_replay", {}) for packet in [*packets, native]]\n    for record in records:\n        statuses.append(record["status"])\n    reference = records[2].get("capture")\n    if reference is not None:\n        for rank, packet in enumerate(packets):\n            captured = records[rank].get("capture")\n            if captured is None:\n                continue\n            sp_input = captured["input"]["windows"][0]\n            native_input = reference["input"]["windows"][rank]\n            actual.append({"schema_version": 1, "block": 34, "point": "attn.out_proj", "rank": rank,\n                "global_start": packet["start"], "global_stop": packet["stop"], "input_width": 7168,\n                "output_width": 5376, "dtype": "torch.bfloat16", "layout": "TensorCoreNVFP4Layout",\n                "sp_padded_rows": captured["input"]["padded_rows"], "native_padded_rows": reference["input"]["padded_rows"],\n                "sp_input": sp_input, "native_input": native_input, "input_match": sp_input == native_input,\n                "sp_weight": captured["weight"], "native_weight": reference["weight"],\n                "weight_match": captured["weight"] == reference["weight"],\n                "sp_bias": captured["bias"], "native_bias": reference["bias"],\n                "bias_match": captured["bias"] == reference["bias"], "backend_verified": False})\n    replays = records[2].get("replays", [])\n    complete = len(actual) == 2 and len(replays) == 2 and all(row["status"] == "complete" for row in statuses)\n    return {"actual": actual, "replays": replays, "statuses": statuses, "complete": complete}\n# END reviewed projection M replay diagnostic\n\n\n', ''), (482, 483, '        self.replay = None\n', ''), (521, 524, '            self.replay = ProjectionReplay(block.attn.out_proj, rank=self.rank, start=self.start,\n                                           stop=self.stop, windows=self.hash_windows)\n            self.replay.__enter__()\n', ''), (545, 547, '        if self.replay is not None:\n            self.replay.__exit__(None, None, None)\n', ''), (563, 565, '                "hash_errors": list(self.hash_errors), "projection_hashes": self.projection_hashes,\n                "projection_replay": None if self.replay is None else self.replay.packet()}\n', '                "hash_errors": list(self.hash_errors), "projection_hashes": self.projection_hashes}\n')], 'sp_group.py': [(386, 395, '                        replay_summary = spf.projection_replay_summary(packets, native_trace.packet())\n                        for row in replay_summary["statuses"]:\n                            logging.info("[minimax_sp][projection_replay_status] %s", json.dumps(row, sort_keys=True, allow_nan=False))\n                        for row in replay_summary["actual"]:\n                            logging.info("[minimax_sp][projection_actual] %s", json.dumps(row, sort_keys=True, allow_nan=False))\n                        for row in replay_summary["replays"]:\n                            logging.info("[minimax_sp][projection_replay] %s", json.dumps(row, sort_keys=True, allow_nan=False))\n                        if not replay_summary["complete"]:\n                            trace_error = "ProjectionReplayIncomplete"\n', '')]}
PROJECTION_REPLAY_PRIOR_SHA = {'sp_forward.py': '9c9dd6617aa7e392d40d573307a7e00c69fcb32477871f770e81a3cf53db5bcf', 'sp_group.py': 'e17a20698d52f87df9c3712a290f496a5f321eae7091665477a9bbda58f9c2ed'}

def reverse_projection_replay(source, name='sp_forward.py'):
    lines=source.splitlines(keepends=True)
    for lo,hi,after,before in reversed(PROJECTION_REPLAY_REVERSE_EDITS[name]):
        if ''.join(lines[lo:hi])!=after:
            raise AssertionError('projection replay reviewed block digest/lines changed')
        lines[lo:hi]=before.splitlines(keepends=True)
    source=''.join(lines)
    if hashlib.sha256(source.encode()).hexdigest()!=PROJECTION_REPLAY_PRIOR_SHA[name]:
        raise AssertionError('projection replay reviewed block digest differs from deployed projection source')
    return source


PROJECTION_HASH_REVERSE_EDITS = {'sp_forward.py': [(54, 55, 'TRACE_PROJECTION_BLOCK = 34\n', 'TRACE_PROJECTION_BLOCK = 36\n'), (137, 138, '    Block34 projections plus all 50 block exits copy seven global rows (all features) per SP rank immediately,\n', '    Block36 projections plus all 50 block exits copy seven global rows (all features) per SP rank immediately,\n'), (151, 152, '        self.projection_hashes = {}\n', ''), (174, 180, '            try:\n                destination = self.full_hashes if point.startswith("block") else self.projection_hashes\n                destination[point] = [streamed_tensor_sha256(tensor, tensor_global_start=self.start,\n                    global_start=start, global_stop=stop) for start, stop in self.hash_windows]\n            except Exception as exc:\n                self.hash_errors.append(type(exc).__name__)\n', '            if point.startswith("block") and point.endswith(".post"):\n                try:\n                    self.full_hashes[point] = [streamed_tensor_sha256(tensor, tensor_global_start=self.start,\n                        global_start=start, global_stop=stop) for start, stop in self.hash_windows]\n                except Exception as exc:\n                    self.hash_errors.append(type(exc).__name__)\n'), (228, 229, '                "hash_errors": list(self.hash_errors), "projection_hashes": self.projection_hashes}\n', '                "hash_errors": list(self.hash_errors)}\n'), (293, 294, 'def _trace_hash_summary(packets, native, *, points, field, projection=False):\n', 'def layer_hash_summary(packets, native):\n'), (296, 296, '', '    points = {f"block{index}.post" for index in range(50)}\n'), (297, 298, '        if packet.get("hash_errors") or set(packet.get(field, {})) != set(points):\n', '        if packet.get("hash_errors") or set(packet.get("full_hashes", {})) != points:\n'), (304, 306, '    for index, point in enumerate(points):\n        expected = native[field][point]\n', '    for index in range(50):\n        point = f"block{index}.post"\n        expected = native["full_hashes"][point]\n'), (309, 310, '            actual = packet[field][point]\n', '            actual = packet["full_hashes"][point]\n'), (333, 335, '            result.append({"schema_version": 1, "block": TRACE_PROJECTION_BLOCK if projection else index,\n                           "rank": rank, **({"point": point} if projection else {}),\n', '            result.append({"schema_version": 1, "block": index, "rank": rank,\n'), (339, 349, '\n\ndef layer_hash_summary(packets, native):\n    return _trace_hash_summary(packets, native, points=tuple(f"block{index}.post" for index in range(50)),\n                               field="full_hashes")\n\n\ndef projection_hash_summary(packets, native):\n    return _trace_hash_summary(packets, native, points=BLOCK0_TRACE_POINTS[:8],\n                               field="projection_hashes", projection=True)\n', '')], 'sp_group.py': [(380, 383, '                        projection_hashes = spf.projection_hash_summary(packets, native_trace.packet())\n                        for row in projection_hashes:\n                            logging.info("[minimax_sp][projection_hash] %s", json.dumps(row, sort_keys=True, allow_nan=False))\n', '')]}
PROJECTION_HASH_PRIOR_SHA = {'sp_forward.py': '3c0f0d1f82a8ae0f077c937c81f16fd87be38b82dd65ccecdb95b5c6eb4e3973', 'sp_group.py': '53f9ed0dc0c4b3c61d829446b3b15d5fce5d22cb3df16fa02a4ea317cabb78b3'}

def reverse_projection_hash(source, name='sp_forward.py'):
    source=reverse_projection_replay(source,name)
    lines=source.splitlines(keepends=True)
    for lo,hi,after,before in reversed(PROJECTION_HASH_REVERSE_EDITS[name]):
        if ''.join(lines[lo:hi])!=after:
            raise AssertionError('projection hash reviewed block digest/lines changed')
        lines[lo:hi]=before.splitlines(keepends=True)
    source=''.join(lines)
    if hashlib.sha256(source.encode()).hexdigest()!=PROJECTION_HASH_PRIOR_SHA[name]:
        raise AssertionError('projection hash reviewed block digest differs from deployed fullhash source')
    return source


FULLHASH_REVERSE_EDITS = {'sp_forward.py': [(24, 26, 'import hashlib\nimport json\n', ''), (27, 28, 'import sys\n', ''), (54, 56, 'TRACE_PROJECTION_BLOCK = 36\nTRACE_HASH_CHUNK_BYTES = 16 * 1024 * 1024\n', ''), (63, 104, '\n\ndef streamed_tensor_sha256(tensor, *, tensor_global_start, global_start, global_stop,\n                          chunk_limit_bytes=TRACE_HASH_CHUNK_BYTES):\n    """Hash original scalar bits in logical C order, one bounded CPU copy at a time."""\n    if (sys.byteorder != "little" or tensor.ndim != 2\n            or type(chunk_limit_bytes) is not int or not 0 < chunk_limit_bytes <= TRACE_HASH_CHUNK_BYTES\n            or not tensor_global_start <= global_start < global_stop <= tensor_global_start + tensor.shape[0]):\n        raise RuntimeError("layer_hash_window_or_byte_order_invalid")\n    width = tensor.shape[1]\n    row_bytes = width * tensor.element_size()\n    if row_bytes <= 0 or row_bytes > chunk_limit_bytes:\n        raise RuntimeError("layer_hash_row_exceeds_chunk_limit")\n    rows_per_chunk = chunk_limit_bytes // row_bytes\n    header = {"schema": "minimax-sp-layer-sha256-v1", "dtype": str(tensor.dtype),\n              "shape": [global_stop - global_start, width], "global_start": global_start,\n              "global_stop": global_stop, "byte_order": "little", "order": "C"}\n    digest = hashlib.sha256()\n    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\\n")\n    copied_bytes = chunks = 0\n    for start in range(global_start - tensor_global_start, global_stop - tensor_global_start, rows_per_chunk):\n        stop = min(start + rows_per_chunk, global_stop - tensor_global_start)\n        chunk = tensor[start:stop].detach().to(device="cpu", non_blocking=False, copy=True,\n                                               memory_format=torch.contiguous_format)\n        # View the original bits before NumPy; NumPy need not understand BF16.\n        array = chunk.view(torch.uint8).numpy()\n        view = memoryview(array).cast("B")\n        try:\n            expected_bytes = (stop - start) * row_bytes\n            if view.nbytes != expected_bytes or view.nbytes > chunk_limit_bytes:\n                raise RuntimeError("layer_hash_cpu_chunk_size_invalid")\n            digest.update(view)\n            copied_bytes += view.nbytes\n            chunks += 1\n        finally:\n            view.release()\n            del view, array, chunk\n    if copied_bytes != (global_stop - global_start) * row_bytes:\n        raise RuntimeError("layer_hash_byte_count_invalid")\n    return {**header, "tensor_bytes": copied_bytes, "chunk_limit_bytes": chunk_limit_bytes,\n            "chunks": chunks, "sha256": digest.hexdigest()}\n', ''), (137, 138, '    Block36 projections plus all 50 block exits copy seven global rows (all features) per SP rank immediately,\n', '    Block0 projections plus all 50 block exits copy seven global rows (all features) per SP rank immediately,\n'), (141, 142, '    def __init__(self, dit, *, rank, start, stop, rows=None, hash_windows=None):\n', '    def __init__(self, dit, *, rank, start, stop, rows=None):\n'), (149, 151, '        self.hash_windows = list(hash_windows) if hash_windows is not None else [(start, stop)]\n        self.full_hashes, self.hash_errors = {}, []\n', ''), (173, 179, '            if point.startswith("block") and point.endswith(".post"):\n                try:\n                    self.full_hashes[point] = [streamed_tensor_sha256(tensor, tensor_global_start=self.start,\n                        global_start=start, global_stop=stop) for start, stop in self.hash_windows]\n                except Exception as exc:\n                    self.hash_errors.append(type(exc).__name__)\n', ''), (189, 190, '            block = self.dit.blocks[TRACE_PROJECTION_BLOCK]\n', '            block = self.dit.blocks[0]\n'), (226, 228, '                "head_errors": list(self.head_errors), "full_hashes": self.full_hashes,\n                "hash_errors": list(self.hash_errors)}\n', '                "head_errors": list(self.head_errors)}\n'), (282, 283, '            result.append({"rank": packet["rank"], "point": point, "projection_block": TRACE_PROJECTION_BLOCK,\n', '            result.append({"rank": packet["rank"], "point": point,\n'), (289, 338, '    return result\n\n\ndef layer_hash_summary(packets, native):\n    """Compare each SP window to the same native window, never concatenate digests."""\n    _, total = block0_trace_rows(packets)\n    points = {f"block{index}.post" for index in range(50)}\n    for packet in [*packets, native]:\n        if packet.get("hash_errors") or set(packet.get("full_hashes", {})) != points:\n            raise RuntimeError("layer_hash_packet_incomplete")\n    if native.get("start") != 0 or native.get("stop") != total:\n        raise RuntimeError("layer_hash_native_extent_invalid")\n    result = []\n    keys = {"schema", "dtype", "shape", "global_start", "global_stop", "byte_order", "order",\n            "tensor_bytes", "chunk_limit_bytes", "chunks", "sha256"}\n    for index in range(50):\n        point = f"block{index}.post"\n        expected = native["full_hashes"][point]\n        if not isinstance(expected, list) or len(expected) != 2:\n            raise RuntimeError("layer_hash_native_windows_invalid")\n        for rank, packet in enumerate(packets):\n            actual = packet["full_hashes"][point]\n            if not isinstance(actual, list) or len(actual) != 1:\n                raise RuntimeError("layer_hash_sp_windows_invalid")\n            actual, reference = actual[0], expected[rank]\n            for value in (actual, reference):\n                item_size = {"torch.bfloat16": 2, "torch.float16": 2, "torch.float32": 4,\n                             "torch.float64": 8}.get(value.get("dtype"))\n                if (set(value) != keys or value["schema"] != "minimax-sp-layer-sha256-v1"\n                        or value["global_start"] != packet["start"] or value["global_stop"] != packet["stop"]\n                        or value["shape"] != [packet["stop"] - packet["start"], native["entries"][point]["shape"][1]]\n                        or value["dtype"] != native["entries"][point]["dtype"]\n                        or value["byte_order"] != "little" or value["order"] != "C"\n                        or value["chunk_limit_bytes"] != TRACE_HASH_CHUNK_BYTES\n                        or type(value["tensor_bytes"]) is not int or value["tensor_bytes"] <= 0\n                        or item_size is None or value["tensor_bytes"] != value["shape"][0] * value["shape"][1] * item_size\n                        or type(value["chunks"]) is not int or value["chunks"] <= 0\n                        or not isinstance(value["sha256"], str) or len(value["sha256"]) != 64\n                        or any(c not in "0123456789abcdef" for c in value["sha256"])):\n                    raise RuntimeError("layer_hash_fields_invalid")\n                rows_per_chunk = TRACE_HASH_CHUNK_BYTES // (value["shape"][1] * item_size)\n                if rows_per_chunk < 1 or value["chunks"] != (value["shape"][0] + rows_per_chunk - 1) // rows_per_chunk:\n                    raise RuntimeError("layer_hash_chunk_count_invalid")\n            if any(actual[key] != reference[key] for key in keys - {"sha256"}):\n                raise RuntimeError("layer_hash_window_metadata_mismatch")\n            result.append({"schema_version": 1, "block": index, "rank": rank,\n                           **{key: actual[key] for key in keys - {"sha256", "schema"}},\n                           "sp_sha256": actual["sha256"], "native_sha256": reference["sha256"],\n                           "match": actual["sha256"] == reference["sha256"]})\n', '')], 'sp_group.py': [(362, 364, '                        native_trace = spf.Block0Trace(dit, rank=-1, start=0, stop=total, rows=rows,\n                            hash_windows=[(packet["start"], packet["stop"]) for packet in packets])\n', '                        native_trace = spf.Block0Trace(dit, rank=-1, start=0, stop=total, rows=rows)\n'), (377, 380, '                        hash_summaries = spf.layer_hash_summary(packets, native_trace.packet())\n                        for row in hash_summaries:\n                            logging.info("[minimax_sp][layer_hash] %s", json.dumps(row, sort_keys=True, allow_nan=False))\n', '')]}
FULLHASH_PRIOR_SHA = {'sp_forward.py': 'e4bd4a8fb0385639946313ad6f7eb3fe4a79825e90ab715aa37d63186c2aa68c', 'sp_group.py': 'e46b6b0062cc376041a4faa3970b1374899f00846c3502651c384fa8f60010df'}

def reverse_fullhash_trace(source, name='sp_forward.py'):
    source=reverse_projection_hash(source,name)
    lines=source.splitlines(keepends=True)
    for lo,hi,after,before in reversed(FULLHASH_REVERSE_EDITS[name]):
        if ''.join(lines[lo:hi])!=after:
            raise AssertionError('fullhash reviewed block digest/lines changed')
        lines[lo:hi]=before.splitlines(keepends=True)
    source=''.join(lines)
    if hashlib.sha256(source.encode()).hexdigest()!=FULLHASH_PRIOR_SHA[name]:
        raise AssertionError('fullhash reviewed block digest differs from deployed layer source')
    return source


LAYER_TRACE_REVERSE_EDITS = {'sp_forward.py': [(53, 86, '    for suffix in (".pre", ".post")) + tuple(f"block{index}.post" for index in range(50))\nHEAD_METADATA_DTYPES = ("torch.float32", "torch.bfloat16", "torch.float16",\n                        "torch.float8_e4m3fn", "torch.float8_e5m2")\nHEAD_METADATA_QUANT = ("nvfp4", "fp8_e4m3fn", "fp8_e5m2", "mxfp8")\nHEAD_METADATA_LAYOUTS = ("TensorCoreNVFP4Layout", "TensorCoreFP8Layout", "TensorCoreMXFP8Layout")\n\n\ndef stored_head_metadata(final_layer):\n    """Read attributes only: never materialize/cast/hash a lazy head weight."""\n    def known(value, allowed):\n        return value if value in allowed else "other"\n    result = []\n    for name in ("video_out", "audio_out"):\n        head = getattr(final_layer, name)\n        weight, bias = head.weight, head.bias\n        quant, layout = getattr(head, "quant_format", None), getattr(head, "layout_type", None)\n        result.append({\n            "head": name,\n            "weight_type": known(type(weight).__name__, ("Parameter", "Tensor", "QuantizedTensor")),\n            "weight_dtype": known(str(weight.dtype), HEAD_METADATA_DTYPES),\n            "weight_shape": list(weight.shape),\n            "weight_device_type": known(weight.device.type, ("cpu", "cuda", "meta")),\n            "weight_is_meta": bool(weight.is_meta),\n            "bias_dtype": None if bias is None else known(str(bias.dtype), HEAD_METADATA_DTYPES),\n            "bias_shape": None if bias is None else list(bias.shape),\n            "quant_format": None if quant is None else known(quant, HEAD_METADATA_QUANT),\n            "layout_type": None if layout is None else known(layout, HEAD_METADATA_LAYOUTS),\n            "full_precision_mm": bool(getattr(head, "_full_precision_mm", False)),\n            "has_v": hasattr(head, "_v"), "has_prefetch": hasattr(head, "_prefetch"),\n            "weight_function_count": len(getattr(head, "weight_function", ())),\n            "bias_function_count": len(getattr(head, "bias_function", ())),\n        })\n    return result\n', '    for suffix in (".pre", ".post")) + ("block0.post",)\n'), (91, 92, '    Block0 projections plus all 50 block exits copy seven global rows (all features) per SP rank immediately,\n', '    Each point copies seven global rows (all features) per SP rank immediately,\n'), (101, 103, '        self.head_metadata = None\n        self.head_errors = []\n', ''), (133, 135, '            if len(self.dit.blocks) != 50:\n                raise RuntimeError("layer_trace_requires_50_blocks")\n', ''), (145, 151, '            for index, block in enumerate(self.dit.blocks):\n                self.handles.append(block.register_forward_hook(\n                    lambda _module, _inputs, output, point=f"block{index}.post": self._capture(point, output)))\n            if self.rank == -1:\n                self.handles.append(self.dit.final_layer.register_forward_pre_hook(\n                    lambda module, _inputs: self.capture_head_metadata(module)))\n', '            self.handles.append(block.register_forward_hook(\n                lambda _module, _inputs, output: self._capture("block0.post", output)))\n'), (171, 181, '                "entries": self.entries, "head_metadata": self.head_metadata,\n                "head_errors": list(self.head_errors)}\n\n    def capture_head_metadata(self, final_layer):\n        try:\n            if self.head_metadata is not None:\n                raise RuntimeError("head_metadata_duplicate_capture")\n            self.head_metadata = stored_head_metadata(final_layer)\n        except Exception as exc:\n            self.head_errors.append(type(exc).__name__)\n', '                "entries": self.entries}\n'), (234, 261, '    return result\n\n\ndef head_metadata_summary(packets, native):\n    """Six closed-schema records; these are not executed/cast weight values."""\n    result = []\n    expected_keys = {"head", "weight_type", "weight_dtype", "weight_shape", "weight_device_type",\n                     "weight_is_meta", "bias_dtype", "bias_shape", "quant_format", "layout_type",\n                     "full_precision_mm", "has_v", "has_prefetch", "weight_function_count", "bias_function_count"}\n    for packet, phase, rank in [(packets[0], "sp2", 0), (packets[1], "sp2", 1), (native, "native", -1)]:\n        heads = packet.get("head_metadata")\n        if packet.get("rank") != rank or packet.get("head_errors") or not isinstance(heads, list) or len(heads) != 2:\n            raise RuntimeError("head_metadata_packet_invalid")\n        for head, name, width in zip(heads, ("video_out", "audio_out"), (3072, 1024)):\n            if (not isinstance(head, dict) or set(head) != expected_keys or head["head"] != name\n                    or head["weight_shape"] != [width, 5376] or head["bias_shape"] != [width]\n                    or head["weight_type"] not in ("Parameter", "Tensor", "QuantizedTensor", "other")\n                    or head["weight_dtype"] not in (*HEAD_METADATA_DTYPES, "other")\n                    or head["bias_dtype"] not in (*HEAD_METADATA_DTYPES, "other")\n                    or head["weight_device_type"] not in ("cpu", "cuda", "meta", "other")\n                    or head["quant_format"] not in (None, *HEAD_METADATA_QUANT, "other")\n                    or head["layout_type"] not in (None, *HEAD_METADATA_LAYOUTS, "other")\n                    or any(type(head[key]) is not bool for key in ("weight_is_meta", "full_precision_mm", "has_v", "has_prefetch"))\n                    or any(type(head[key]) is not int or head[key] < 0 for key in ("weight_function_count", "bias_function_count"))):\n                raise RuntimeError("head_metadata_fields_invalid")\n            result.append({"schema_version": 1, "phase": phase, "rank": rank,\n                           "scope": "stored_runtime_metadata", "execution_weight_verified": False, **head})\n', ''), (807, 809, '        if trace is not None:\n            trace.capture_head_metadata(dit.final_layer)\n', '')], 'sp_group.py': [(376, 379, '                        head_summaries = spf.head_metadata_summary(packets, native_trace.packet())\n                        for row in head_summaries:\n                            logging.info("[minimax_sp][head_metadata] %s", json.dumps(row, sort_keys=True, allow_nan=False))\n', '')]}
LAYER_TRACE_PRIOR_SHA = {'sp_forward.py': '191008ba795c38ec50603941a3a3fd64d967e27de0659fecc558aee11e2dcdb0', 'sp_group.py': 'e524cdc22d6e4d9353044a113bc34b433eb0a885a4df6b78e8500ad0dfc70d7f'}

def reverse_layer_trace(source, name='sp_forward.py'):
    source=reverse_fullhash_trace(source,name)
    lines=source.splitlines(keepends=True)
    for lo,hi,after,before in reversed(LAYER_TRACE_REVERSE_EDITS[name]):
        if ''.join(lines[lo:hi])!=after:
            raise AssertionError('layer trace reviewed block digest/lines changed')
        lines[lo:hi]=before.splitlines(keepends=True)
    source=''.join(lines)
    if hashlib.sha256(source.encode()).hexdigest()!=LAYER_TRACE_PRIOR_SHA[name]:
        raise AssertionError('layer trace reviewed block digest differs from deployed block0 source')
    return source


BLOCK0_REVIEWED_BLOCKS = {'class': ('# BEGIN reviewed block0 diagnostic trace\n', '# END reviewed block0 diagnostic trace\n\n', '7ada8a396f9a99192ae927268f808bcf44b425d365ae2fd048a2bd2804387326'), 'scope': ('    # BEGIN reviewed block0 trace scope\n', '    # END reviewed block0 trace scope\n', '0d062e08e33d1df72cc597dfba34d186ca444531f11cab7ed8f2909b2434a26b'), 'gather': ('    # BEGIN reviewed block0 trace gather\n', '    # END reviewed block0 trace gather\n', '7cca78719d17d2ae116fea9feca89bbbd90d6f3fda4a270d68ff063abbfcb2c8')}

def reverse_block0_trace(source):
    source = reverse_layer_trace(source)
    for start,stop,digest in BLOCK0_REVIEWED_BLOCKS.values():
        if source.count(start)!=1 or source.count(stop)!=1:
            raise AssertionError('block0 trace reviewed markers changed')
        lo=source.index(start);hi=source.index(stop,lo)+len(stop)
        if hashlib.sha256(source[lo:hi].encode()).hexdigest()!=digest:
            raise AssertionError('block0 trace reviewed block digest changed')
        source=source[:lo]+source[hi:]
    old='    with scale_context as scale_scope, (trace if trace is not None else nullcontext()):\n'
    if source.count(old)!=1:
        raise AssertionError('block0 trace reviewed loop changed')
    source=source.replace(old,'    with scale_context as scale_scope:\n',1)
    if hashlib.sha256(source.encode()).hexdigest()!='c64d8912fd2b99df1c02ebc4ed76d48ec221dae31c3746cf96ec5cc63349c2b0':
        raise AssertionError('reviewed block digest: block0 reversal differs from prior global-amax source')
    return source


def reverse_amax_scope(source):
    """Reverse only reviewed exact imports/scope/loop; retain original upstream SHA."""
    source = reverse_block0_trace(source)
    for name,(start,stop,digest) in AMAX_REVIEWED_BLOCKS.items():
        if source.count(start)!=1 or source.count(stop)!=1:
            raise AssertionError('SP2 amax reviewed block markers changed')
        lo=source.index(start);hi=source.index(stop,lo)+len(stop)
        if hashlib.sha256(source[lo:hi].encode()).hexdigest()!=digest:
            raise AssertionError('SP2 amax reviewed block digest changed')
        source=source[:lo]+(AMAX_ORIGINAL_LOOP if name=='loop' else '')+source[hi:]
    return source


WORLD1_BRANCHES = (
    "    if ctx.world == 1:\n"
    "        # Diagnostic mirror: preserve fresh-buffer semantics without a new PG.\n"
    "        return t.contiguous().clone()\n",
    "    if ctx.world == 1:\n"
    "        return t.contiguous().clone()\n",
    "    if ctx.world == 1:\n"
    "        return buf[:counts[0]].clone()\n",
)


def reverse_world1_branches(source):
    """Remove only the three exact reviewed no-collective identity branches."""
    for branch in WORLD1_BRANCHES:
        if source.count(branch) != 1:
            raise AssertionError("world1 diagnostic branch differs from reviewed source")
        source = source.replace(branch, "", 1)
    return source


def original_source():
    """Reverse reviewed world1 branches, two widths, and one collective API."""
    source = reverse_world1_branches(reverse_amax_scope(AFTER.read_text()))
    for local, attribute in (("v_local", "video_out"), ("a_local", "audio_out")):
        before = f"{local}.shape[-1], device)"
        after = f"dit.final_layer.{attribute}.out_features, device)"
        if source.count(before) != 1:
            raise AssertionError("local patch no longer contains the two reviewed width expressions")
        source = source.replace(before, after, 1)
    patched_api = "dist.all_gather_into_tensor(gathered, buf, group=ctx.group)"
    if source.count(patched_api) != 1:
        raise AssertionError("local patch no longer contains the single reviewed API replacement")
    source = source.replace(patched_api, "dist.all_gather_single(gathered, buf, group=ctx.group)", 1)
    contract = json.loads(CONTRACT.read_text())
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != contract["upstream_files_sha256"]["minimax_sp/sp_forward.py"]:
        raise AssertionError("reverse patch differs from the pinned public upstream source")
    return source


BEFORE = original_source()



class Array:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)
        self.shape = self.data.shape
        self.device = "cpu"

    def __getitem__(self, index):
        return Array(self.data[index])

    def __setitem__(self, index, value):
        self.data[index] = value.data if isinstance(value, Array) else value

    def view(self, *shape):
        return Array(self.data.reshape(shape))

    def contiguous(self):
        return self

    def clone(self):
        return Array(self.data.copy())


class TorchCPU:
    float32 = np.float32

    @staticmethod
    def zeros(*shape, **kwargs):
        return Array(np.zeros(shape, dtype=np.float32))

    @staticmethod
    def empty(*shape, **kwargs):
        return Array(np.empty(shape, dtype=np.float32))

    @staticmethod
    def cat(rows, dim=0):
        return Array(np.concatenate([row.data for row in rows], axis=dim))


def tree(path):
    return ast.parse(path.read_text() if isinstance(path, Path) else path)


def width_expression(source, name):
    for node in ast.walk(source):
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == name and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "_gather_rows"):
            return node.value.args[3]
    raise AssertionError("gather call absent")


def run_gather(source, stream, shards, heads, *, legacy_api=False):
    """Execute the real gather AST with each rank's actual returned output width."""
    width = shards[0].shape[1]
    dit = SimpleNamespace(final_layer=SimpleNamespace(
        video_out=SimpleNamespace(out_features=96 * heads),
        audio_out=SimpleNamespace(out_features=32 * heads)))
    counts = [s.shape[0] for s in shards]
    maxn = max(counts)
    actual = []
    for rank, local in enumerate(shards):
        def collective(output, send, group):
            if group != "cpu-test-only":
                raise AssertionError("gather did not preserve the selected process group")
            expected_send = np.zeros((maxn, width), dtype=np.float32)
            expected_send[:local.shape[0]] = local
            np.testing.assert_array_equal(send.data, expected_send)
            padded = []
            for shard in shards:
                part = np.zeros((maxn, width), dtype=np.float32)
                part[:shard.shape[0]] = shard
                padded.append(part)
            output.data[:] = np.concatenate(padded, axis=0)

        # Default models the installed torch.distributed API: it has
        # all_gather_into_tensor and does not have all_gather_single.
        # Legacy is an explicit oracle only for comparing upstream numerics.
        api = "all_gather_single" if legacy_api else "all_gather_into_tensor"
        namespace = {"torch": TorchCPU, "dist": SimpleNamespace(**{api: collective})}
        fn = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "_gather_rows")
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "<actual-gather>", "exec"), namespace)
        expr = width_expression(source, stream)
        out_dim = eval(compile(ast.Expression(expr), "<actual-width>", "eval"), {},
                       {"dit": dit, "v_local": Array(local), "a_local": Array(local)})
        context = SimpleNamespace(world=len(shards), rank=rank, group="cpu-test-only")
        actual.append(namespace["_gather_rows"](Array(local), counts, context, out_dim, "cpu").data)
    return actual


class PDDGatherTests(unittest.TestCase):
    def test_actual_32_head_shapes_fail_before_and_preserve_values_after(self):
        for name, width in (("v", 96), ("a", 32)):
            shards = [np.arange(n * width, dtype=np.float32).reshape(n, width) + rank * 1000
                      for rank, n in enumerate((2, 3))]
            with self.subTest(stream=name), self.assertRaises(ValueError):
                run_gather(tree(BEFORE), name, shards, heads=32, legacy_api=True)
            for output in run_gather(tree(AFTER), name, shards, heads=32):
                np.testing.assert_array_equal(output, np.concatenate(shards))

    def test_empty_local_stream_rank_preserves_output(self):
        for name, width in (("v", 96), ("a", 32)):
            for counts in ((0, 3), (3, 0)):
                shards = [np.arange(n * width, dtype=np.float32).reshape(n, width) for n in counts]
                for output in run_gather(tree(AFTER), name, shards, heads=32):
                    np.testing.assert_array_equal(output, np.concatenate(shards))

    def test_single_head_shape_and_values_unchanged(self):
        for name, width in (("v", 96), ("a", 32)):
            shards = [np.ones((n, width), dtype=np.float32) * i for i, n in enumerate((1, 4))]
            before = run_gather(tree(BEFORE), name, shards, heads=1, legacy_api=True)
            after = run_gather(tree(AFTER), name, shards, heads=1)
            for old, new in zip(before, after):
                np.testing.assert_array_equal(old, new)

    def test_into_tensor_only_runtime_fails_before_and_succeeds_after(self):
        # Single-head data isolates the API regression from the PDD width bug.
        for name, width in (("v", 96), ("a", 32)):
            shards = [np.arange(n * width, dtype=np.float32).reshape(n, width) + rank * 1000
                      for rank, n in enumerate((2, 3))]
            with self.subTest(stream=name), self.assertRaisesRegex(AttributeError, "all_gather_single"):
                run_gather(tree(BEFORE), name, shards, heads=1)
            for output in run_gather(tree(AFTER), name, shards, heads=1):
                np.testing.assert_array_equal(output, np.concatenate(shards))

    def test_only_reviewed_width_api_and_world1_identity_changes(self):
        before = tree(BEFORE)
        after = tree(reverse_world1_branches(reverse_amax_scope(AFTER.read_text())))
        for name in ("v", "a"):
            old = width_expression(before, name)
            replacement = width_expression(after, name)
            class Replace(ast.NodeTransformer):
                def visit_Attribute(self, node):
                    return copy.deepcopy(replacement) if node is old else self.generic_visit(node)
            before = Replace().visit(before)
        changed = []
        class ReplaceAPI(ast.NodeTransformer):
            def visit_Attribute(self, node):
                if isinstance(node.value, ast.Name) and node.value.id == "dist" and node.attr == "all_gather_single":
                    changed.append(node)
                    node.attr = "all_gather_into_tensor"
                return self.generic_visit(node)
        before = ReplaceAPI().visit(before)
        self.assertEqual(len(changed), 1)
        self.assertEqual(ast.dump(before), ast.dump(after))
        contract = json.loads(CONTRACT.read_text())
        digest = hashlib.sha256(AFTER.read_bytes()).hexdigest()
        self.assertEqual(digest, contract["files"]["minimax_sp/sp_forward.py"])
        reverse_layer_trace((AFTER.parent / "sp_group.py").read_text(), "sp_group.py")
        self.assertEqual(digest, contract["local_patch"]["patched_sha256"])


class World1IdentityTests(unittest.TestCase):
    def actual_helper(self, name):
        function = next(n for n in tree(AFTER).body
                        if isinstance(n, ast.FunctionDef) and n.name == name)
        # Empty dist deliberately has no collective API. A world1 regression
        # into any distributed call fails instead of being silently mocked.
        namespace = {"torch": TorchCPU, "dist": SimpleNamespace()}
        exec(compile(ast.Module(body=[function], type_ignores=[]),
                     "<actual-world1-helper>", "exec"), namespace)
        return namespace[name]

    def test_actual_world1_a2a_is_exact_fresh_buffer_without_pg(self):
        ctx = SimpleNamespace(world=1)
        for name, shape in (("a2a_gather_seq_scatter_heads", (7, 56, 128)),
                            ("a2a_scatter_seq_gather_heads", (7, 56 * 128))):
            with self.subTest(helper=name):
                source = Array(np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
                result = self.actual_helper(name)(source, ctx)
                np.testing.assert_array_equal(result.data, source.data)
                self.assertEqual(result.shape, shape)
                self.assertFalse(np.shares_memory(result.data, source.data))
                result.data.flat[0] = -999
                self.assertEqual(source.data.flat[0], 0)

    def test_actual_world1_gather_preserves_pdd_width_and_empty_rows(self):
        helper = self.actual_helper("_gather_rows")
        ctx = SimpleNamespace(world=1)
        for width in (96, 32):
            for rows in (0, 7):
                with self.subTest(width=width, rows=rows):
                    source = Array(np.arange(rows * width, dtype=np.float32).reshape(rows, width))
                    result = helper(source, [rows], ctx, width, "cpu")
                    np.testing.assert_array_equal(result.data, source.data)
                    self.assertEqual(result.shape, (rows, width))
                    self.assertEqual(result.data.dtype, np.dtype(np.float32))
                    self.assertFalse(np.shares_memory(result.data, source.data))

    def test_world1_branch_tamper_cannot_pass_upstream_reverse_contract(self):
        source = AFTER.read_text()
        with self.assertRaisesRegex(AssertionError, "world1 diagnostic branch"):
            reverse_world1_branches(source.replace("return t.contiguous().clone()",
                                                  "return t", 1))


"""Execute candidate parity/forward AST with CPU arrays and mocked collectives.

No model/media/source-payload loading, CUDA, distributed process, or network.
These checks prove dispatch/gate behavior only, not NVFP4 numerical equivalence.
"""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

GROUP_SOURCE = Path(__file__).resolve().parents[1] / 'custom_nodes/minimax_sp_pdd/minimax_sp/sp_group.py'


class ParityArray:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)
        self.device = 'cpu'

    def float(self):
        return self

    def __sub__(self, other):
        return ParityArray(self.data - other.data)

    def abs(self):
        return ParityArray(np.abs(self.data))

    def max(self):
        return self.data.max()


class ParityTorchCPU:
    cuda = SimpleNamespace(synchronize=lambda:None)

    @staticmethod
    def isfinite(value):
        return np.isfinite(value.data)


def parity_functions(namespace):
    tree = ast.parse(GROUP_SOURCE.read_text())
    parity = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'native_parity_metrics')
    group = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SPGroup')
    forward = next(n for n in group.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    module = ast.Module(body=[parity, forward], type_ignores=[])
    exec(compile(module, str(GROUP_SOURCE), 'exec'), namespace)
    return namespace['native_parity_metrics'], namespace['forward']


def parity_fixture(*, verify=True, world2_good=False, mirror_good=True, mirror_error=False, profile=False):
    events = []
    native = [ParityArray([1., -2.]), ParityArray([3., -4.])]
    sp2 = native if world2_good else [ParityArray([1.5, -2.]), ParityArray([3.8, -4.])]
    mirror = native if mirror_good else [ParityArray([1.25, -2.]), ParityArray([3.4, -4.])]

    class Dit:
        def _forward(self, *args, **kwargs):
            events.append('native')
            return native

    def dispatch(dit, x, timestep, context, options, payload, rank, world, group, denoise_mask, audio_denoise_mask):
        events.append(('sp', world))
        if world == 1:
            if (rank, group) != (0, None):
                raise AssertionError('world1 must not adopt the existing tensor group')
            if mirror_error:
                raise RuntimeError('mirror_execution_failed_fixture')
            return mirror
        if (rank, world, group) != (0, 2, 'owned-tensor-pg'):
            raise AssertionError('original world2 dispatch changed')
        if options.get('_minimax_sp_block0_trace_request'):
            options['_minimax_sp_block0_trace_result'] = [
                {'start':0,'stop':1}, {'start':1,'stop':2}]
        return sp2

    class Trace:
        def __init__(self,*a,**kw): pass
        def __enter__(self): return self
        def __exit__(self,*a): return False
        def packet(self): return {}
    log = Mock(spec=logging)
    def logged(fmt, *args):
        events.append(('log', fmt % args))
    log.info.side_effect = logged
    destroy = Mock(side_effect=lambda: events.append('destroy'))
    owner = SimpleNamespace(check_alive=Mock(), patches_ready=True, verify_pending=verify,
        world=2, control_pg='owned-control-pg', tensor_pg='owned-tensor-pg',
        operation_id=0, n_forward=0, destroy=destroy)
    dist = Mock()
    ns = {'torch':ParityTorchCPU, 'logging':log,
        'os':SimpleNamespace(environ={'MINIMAX_SP_PROFILE':'1'} if profile else {}),
        'time':SimpleNamespace(perf_counter=lambda:1.), 'dist':dist,
        'nullcontext':__import__('contextlib').nullcontext, 'json':json,
        'spf':SimpleNamespace(validate_transformer_options=Mock(), sp_forward=dispatch,
            BLOCK0_TRACE_REQUEST='_minimax_sp_block0_trace_request', BLOCK0_TRACE_RESULT='_minimax_sp_block0_trace_result',
            block0_trace_rows=Mock(return_value=([0,1],2)),
            Block0Trace=Trace,
            block0_trace_summary=Mock(return_value=[]), head_metadata_summary=Mock(return_value=[]),
            layer_hash_summary=Mock(return_value=[]), projection_hash_summary=Mock(return_value=[]),
            projection_replay_summary=Mock(return_value={"statuses":[],"actual":[],"replays":[],"complete":True})),
        'build_meta':lambda *args:{}, 'tensor_meta':lambda value:None,
        'forward_status':lambda meta:{}, 'validate_forward_status':lambda *args:None,
        'ordered_tensors':lambda *args:[],
        'comfy':SimpleNamespace(model_management=SimpleNamespace(InterruptProcessingException=type('Interrupt', (Exception,), {})))}
    parity, forward = parity_functions(ns)
    args = (owner, Dit(), [ParityArray([0]), ParityArray([0])], None, None, {}, {}, None, None)
    return SimpleNamespace(parity=parity, forward=forward, args=args, owner=owner,
        native=native, events=events, dist=dist, log=log, namespace=ns)


class MirrorDiagnosticTests(unittest.TestCase):
    def test_failed_sp2_records_all_four_stream_metrics_before_original_gate(self):
        f = parity_fixture(profile=True)
        with self.assertRaisesRegex(RuntimeError, 'video differs from native H3'):
            f.forward(*f.args)
        self.assertEqual([e for e in f.events if e == 'native' or isinstance(e, tuple) and e[0]=='sp'],
                         [('sp', 2), 'native', ('sp', 1)])
        metrics = [e[1] for e in f.events if isinstance(e, tuple) and e[0]=='log' and 'max_abs=' in e[1]]
        self.assertEqual(len(metrics), 4)
        self.assertTrue(metrics[0].startswith('[minimax_sp][verify] video '))
        self.assertTrue(metrics[1].startswith('[minimax_sp][verify] audio '))
        self.assertTrue(metrics[2].startswith('[minimax_sp][verify_mirror_world1] video '))
        self.assertTrue(metrics[3].startswith('[minimax_sp][verify_mirror_world1] audio '))
        self.assertTrue(all('finite=1' in row for row in metrics))
        self.assertIn(('log', '[minimax_sp][verify_summary] world2_pass=0 world1_pass=1 threshold=0.001'), f.events)
        self.assertEqual(f.events[-1], 'destroy')
        self.assertEqual(f.owner.n_forward, 0)
        self.assertFalse(any(isinstance(e, tuple) and e[0]=='log' and '[profile] step' in e[1]
                             for e in f.events))
        f.owner.destroy.assert_called_once_with()
        # No mirror control-plane broadcast: only the original request envelope.
        self.assertEqual(f.dist.broadcast_object_list.call_count, 1)
        self.assertEqual(f.dist.all_gather_object.call_count, 1)

    def test_passed_sp2_does_not_run_mirror_or_change_return(self):
        f = parity_fixture(world2_good=True)
        result = f.forward(*f.args)
        self.assertIs(result, f.native)
        self.assertNotIn(('sp', 1), f.events)
        self.assertFalse(f.owner.verify_pending)
        f.owner.destroy.assert_not_called()

    def test_verification_disabled_does_not_run_native_or_mirror(self):
        f = parity_fixture(verify=False)
        f.forward(*f.args)
        self.assertNotIn('native', f.events)
        self.assertNotIn(('sp', 1), f.events)
        self.assertEqual([e for e in f.events if isinstance(e, tuple) and e[0]=='sp'], [('sp', 2)])
        f.owner.destroy.assert_not_called()

    def test_failed_mirror_still_cannot_accept_failed_sp2(self):
        f = parity_fixture(mirror_good=False)
        with self.assertRaisesRegex(RuntimeError, 'video differs from native H3'):
            f.forward(*f.args)
        self.assertIn(('log', '[minimax_sp][verify_summary] world2_pass=0 world1_pass=0 threshold=0.001'), f.events)
        f.owner.destroy.assert_called_once_with()

    def test_mirror_exception_destroys_owned_group_without_retry(self):
        f = parity_fixture(mirror_error=True)
        with self.assertRaisesRegex(RuntimeError, 'mirror_execution_failed_fixture'):
            f.forward(*f.args)
        self.assertEqual(f.events.count(('sp', 1)), 1)
        f.owner.destroy.assert_called_once_with()
        original_metrics=[e for e in f.events if isinstance(e, tuple) and e[0]=='log' and '[verify]' in e[1]]
        self.assertEqual(len(original_metrics), 2)

    def test_nonfinite_both_streams_recorded_and_strict_threshold_unchanged(self):
        f = parity_fixture()
        rows = f.parity([ParityArray([np.nan]),ParityArray([np.inf])],[ParityArray([1.]),ParityArray([1.])])
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(not row['finite'] and not row['passed'] for row in rows))
        self.assertTrue(f.parity([ParityArray([1.0005]),ParityArray([1.])],[ParityArray([1.]),ParityArray([1.])])[0]['passed'])
        self.assertFalse(f.parity([ParityArray([1.002]),ParityArray([1.])],[ParityArray([1.]),ParityArray([1.])])[0]['passed'])
        with self.assertRaisesRegex(RuntimeError, 'both video and audio'):
            f.parity([ParityArray([1.])], [ParityArray([1.]),ParityArray([1.])])



# BEGIN GLOBAL NVFP4 AMAX CPU CONTRACT TESTS
# Append before the existing unittest.main() guard in test_minimax_sp_pdd.py.
# Reads actual integrated source AST, never imports Comfy/Torch or model weights.
import contextvars as _amax_contextvars
import threading as _amax_threading
import types as _amax_types

_GLOBAL_AMAX_SOURCE = Path(__file__).resolve().parents[1] / 'custom_nodes/minimax_sp_pdd/minimax_sp/sp_forward.py'
_GLOBAL_AMAX_QUANTIZE_FILE_SHA = '37f0054dfbd8430524ac2c73dafd25b8ad4eabc490a2b7490eeebde8dffbc749'
# Native method from ComfyUI b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c,
# comfy/quant_ops.py:151-180. Fake packing tests the scale contract, not FP4 kernels.
_GLOBAL_AMAX_NATIVE_QUANTIZE = '@classmethod\ndef quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False):\n    if tensor.dim() != 2:\n        raise ValueError(f"NVFP4 requires 2D tensor, got {tensor.dim()}D")\n\n    orig_dtype = tensor.dtype\n    orig_shape = tuple(tensor.shape)\n\n    if scale is None or (isinstance(scale, str) and scale == "recalculate"):\n        scale = torch.amax(tensor.abs()) / (ck.float_utils.F8_E4M3_MAX * ck.float_utils.F4_E2M1_MAX)\n\n    if not isinstance(scale, torch.Tensor):\n        scale = torch.tensor(scale)\n    scale = scale.to(device=tensor.device, dtype=torch.float32)\n\n    padded_shape = cls.get_padded_shape(orig_shape)\n    needs_padding = padded_shape != orig_shape\n\n    if stochastic_rounding > 0:\n        qdata, block_scale = comfy.float.stochastic_round_quantize_nvfp4_by_block(tensor, scale, pad_16x=needs_padding, seed=stochastic_rounding)\n    else:\n        qdata, block_scale = ck.quantize_nvfp4(tensor, scale, pad_16x=needs_padding)\n\n    params = cls.Params(\n        scale=scale,\n        orig_dtype=orig_dtype,\n        orig_shape=orig_shape,\n        block_scale=block_scale,\n    )\n    return qdata, params\n'
_GLOBAL_AMAX_METHOD_SHA = 'a99a89e2bd61e1bcb5faab9be5f479d29772dceea9e50d5944959b75ca4011c9'


def _load_global_amax_scope_ast():
    source = ast.parse(_GLOBAL_AMAX_SOURCE.read_text())
    constants = {'_SP_NVFP4_TICKET', '_SP_NVFP4_LOCK', '_ABSENT'}
    nodes = [node for node in source.body if
             (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets))
             or (isinstance(node, ast.ClassDef) and node.name == 'SPNVFP4AmaxScope')]
    if len(nodes) != 4:
        raise AssertionError('expected exactly three scope globals and the actual scope class')
    namespace = {'ContextVar': _amax_contextvars.ContextVar, 'threading': _amax_threading, 'types': _amax_types}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), '<actual-integrated-SP-amax-scope>', 'exec'), namespace)
    return SimpleNamespace(**namespace)

class GlobalAmaxTensor:
    requires_grad=False
    def __init__(self,value):self.value=np.array(value,dtype=np.float32);self.dtype='float32';self.device='cpu'
    @property
    def shape(self):return self.value.shape
    def dim(self):return self.value.ndim
    def abs(self):return GlobalAmaxTensor(np.abs(self.value))
    def __truediv__(self,value):return GlobalAmaxTensor(self.value/value)
    def to(self,**kwargs):return self


class GlobalAmaxDist:
    ReduceOp=SimpleNamespace(MAX='max')
    def __init__(self,global_max=64):self.reductions=0;self.global_max=global_max;self.inventory_calls=0
    def get_world_size(self,group):return 2
    def all_gather_object(self,output,value,group):
        self.inventory_calls+=1;output[:]=[value,value]
    def all_reduce(self,value,op,group):
        assert op=='max';self.reductions+=1
        value.value=np.maximum(value.value,np.float32(self.global_max))


def global_amax_fixture():
    # Fixed native scalar oracle + actual inlined implementation under test.
    m = _load_global_amax_scope_ast()
    assert hashlib.sha256(_GLOBAL_AMAX_NATIVE_QUANTIZE.encode()).hexdigest() == _GLOBAL_AMAX_METHOD_SHA
    fn = ast.parse(_GLOBAL_AMAX_NATIVE_QUANTIZE).body[0]
    calls=[]
    def pack(tensor,scale,**kwargs):
        calls.append({'scale':float(scale.value),'shape':tensor.shape,**kwargs})
        return tensor,None
    torch=SimpleNamespace(Tensor=GlobalAmaxTensor,float32='float32',amax=lambda t:GlobalAmaxTensor(np.max(t.value)),tensor=GlobalAmaxTensor)
    ck=SimpleNamespace(float_utils=SimpleNamespace(F8_E4M3_MAX=448.,F4_E2M1_MAX=6.),quantize_nvfp4=pack)
    namespace={'torch':torch,'ck':ck,'comfy':SimpleNamespace(float=SimpleNamespace(stochastic_round_quantize_nvfp4_by_block=pack))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<pinned-quantize-AST>','exec'),namespace)
    Layout=type('Layout',(),{'quantize':namespace['quantize'],'Params':SimpleNamespace,
      'get_padded_shape':staticmethod(lambda shape:tuple((n+15)//16*16 for n in shape))})
    class QuantizedTensor:pass
    ops=SimpleNamespace(TensorCoreNVFP4Layout=Layout,QuantizedTensor=QuantizedTensor,ck=ck)
    class Module:
        quant_format='nvfp4';layout_type='TensorCoreNVFP4Layout'
        _full_precision_mm=False;comfy_force_cast_weights=False
        weight_function=[];bias_function=[];in_features=16;out_features=16
        def forward(self,input):
            result=Layout.quantize(input,scale=getattr(self,'input_scale',None))
            if getattr(self,'weight_requant',False):Layout.quantize(GlobalAmaxTensor([[256]]),scale='recalculate')
            if getattr(self,'fail',False):raise RuntimeError('synthetic_forward_failure')
            return result
    blocks=[SimpleNamespace(attn=SimpleNamespace(qkv_proj=Module(),out_proj=Module()),mlp=SimpleNamespace(fc1=Module(),fc2=Module())) for _ in range(50)]
    dit=SimpleNamespace(blocks=blocks);targets=[x for b in blocks for x in (b.attn.qkv_proj,b.attn.out_proj,b.mlp.fc1,b.mlp.fc2)]
    ctx=SimpleNamespace(world=2,seq_len=6,local=3,group='owned-test-group')
    dist=GlobalAmaxDist()
    scope=m.SPNVFP4AmaxScope(dit,ctx,torch_module=torch,dist_module=dist,quant_ops=ops)
    return scope,targets,Layout,dist,calls


class GlobalAmaxScopeTests(unittest.TestCase):
    def test_global_scale_matches_unsharded_native_and_200_syncs(self):
        scope,targets,layout,dist,calls=global_amax_fixture();descriptor=layout.__dict__['quantize']
        native=layout.quantize(GlobalAmaxTensor([[64]*16]*6))[1].scale.value
        with scope:
            for module in targets:
                self.assertEqual(module.forward(GlobalAmaxTensor([[1]*16]*3))[1].scale.value,native)
        self.assertEqual(dist.reductions,200)
        self.assertEqual(scope.counts()['per_block'],[4]*50)
        self.assertIs(layout.__dict__['quantize'],descriptor)
        self.assertTrue(all('forward' not in module.__dict__ for module in targets))
        before=dist.reductions;layout.quantize(GlobalAmaxTensor([[1]*16]*6))
        self.assertEqual(dist.reductions,before) # rank0 native oracle after scope

    def test_fixed_tensor_and_float_scales_unchanged_weight_requant_not_synchronized(self):
        scope,targets,layout,dist,calls=global_amax_fixture()
        fixed=GlobalAmaxTensor(0.5);targets[0].input_scale=fixed;targets[1].input_scale=0.25
        targets[2].weight_requant=True
        with scope:
            for module in targets:module.forward(GlobalAmaxTensor([[1]*16]*3))
        self.assertEqual(dist.reductions,198)
        self.assertIs(targets[0].input_scale,fixed);self.assertEqual(targets[1].input_scale,0.25)
        self.assertEqual(calls[0]['scale'],0.5);self.assertEqual(calls[1]['scale'],0.25)
        self.assertAlmostEqual(calls[3]['scale'],256/2688,places=7)

    def test_exception_restores_existing_forward_descriptor_and_context(self):
        scope,targets,layout,dist,calls=global_amax_fixture();descriptor=layout.__dict__['quantize']
        original=targets[0].forward;targets[0].forward=original;targets[0].fail=True
        with self.assertRaisesRegex(RuntimeError,'synthetic_forward_failure'):
            with scope:targets[0].forward(GlobalAmaxTensor([[1]*16]*3))
        self.assertIs(targets[0].__dict__['forward'],original)
        self.assertIs(layout.__dict__['quantize'],descriptor)
        self.assertIsNone(scope._forward_wrapper.__globals__['_SP_NVFP4_TICKET'].get())
        self.assertFalse(scope._forward_wrapper.__globals__['_SP_NVFP4_LOCK'].locked())

    def test_other_thread_and_unrelated_quantize_do_not_reduce(self):
        scope,targets,layout,dist,calls=global_amax_fixture()
        with self.assertRaisesRegex(RuntimeError,'end_scope'):
            with scope:
                thread=_amax_threading.Thread(target=lambda:layout.quantize(GlobalAmaxTensor([[2]*16]*3)))
                thread.start();thread.join()
                thread=_amax_threading.Thread(target=lambda:targets[0].forward(GlobalAmaxTensor([[2]*16]*3)))
                thread.start();thread.join()
                layout.quantize(GlobalAmaxTensor([[2]*16]*3))
                self.assertEqual(dist.reductions,0)
                self.assertEqual(sum(scope.calls.values()),0)
                raise RuntimeError('end_scope')

    def test_zero_scale_and_padding_follow_native(self):
        scope,targets,layout,dist,calls=global_amax_fixture();dist.global_max=0
        native=layout.quantize(GlobalAmaxTensor([[0]*16]*3))[1].scale.value
        with scope:
            for module in targets:self.assertEqual(module.forward(GlobalAmaxTensor([[0]*16]*3))[1].scale.value,native)
        self.assertEqual(float(native),0)
        self.assertTrue(all(call['pad_16x'] for call in calls))

    def test_uniform_inventory_rejects_wrong_scope_without_patching(self):
        scope,targets,layout,dist,calls=global_amax_fixture();descriptor=layout.__dict__['quantize']
        scope.ctx.local=0
        with self.assertRaisesRegex(RuntimeError,'rank_inventory_mismatch'):
            with scope:pass
        self.assertEqual(dist.reductions,0)
        self.assertIs(layout.__dict__['quantize'],descriptor)
        self.assertTrue(all('forward' not in module.__dict__ for module in targets))


    def test_inventory_signature_failure_releases_lock_and_participates_in_ballot(self):
        scope,targets,layout,dist,calls=global_amax_fixture()
        descriptor=layout.__dict__['quantize']
        class BrokenInventory(type(targets[0])):
            @property
            def out_features(self):
                raise AttributeError('synthetic_missing_out_features')
        targets[0].__class__=BrokenInventory
        with self.assertRaisesRegex(RuntimeError,'rank_inventory_mismatch'):
            with scope:pass
        self.assertEqual(dist.inventory_calls,1)
        self.assertFalse(scope._forward_wrapper.__globals__['_SP_NVFP4_LOCK'].locked())
        self.assertIs(layout.__dict__['quantize'],descriptor)
        self.assertTrue(all('forward' not in module.__dict__ for module in targets))

    def test_same_scope_reentry_does_not_restore_outer_scope(self):
        scope,targets,layout,dist,calls=global_amax_fixture()
        descriptor=layout.__dict__['quantize']
        with scope:
            active_descriptor=layout.__dict__['quantize']
            with self.assertRaises(RuntimeError):scope.__enter__()
            self.assertIs(layout.__dict__['quantize'],active_descriptor)
            for module in targets:module.forward(GlobalAmaxTensor([[1]*16]*3))
        self.assertEqual(dist.reductions,200)
        self.assertIs(layout.__dict__['quantize'],descriptor)

    def test_copy_context_child_quantize_does_not_consume_main_ticket(self):
        scope,targets,layout,dist,calls=global_amax_fixture()
        original=targets[0].forward
        observations=[]
        def with_context_copy(input):
            child_context=_amax_contextvars.copy_context()
            errors=[]
            def child():
                try:
                    result=child_context.run(layout.quantize,GlobalAmaxTensor([[2]*16]*3))
                    observations.append(float(result[1].scale.value))
                except BaseException as error:
                    errors.append(type(error).__name__)
            thread=_amax_threading.Thread(target=child)
            thread.start();thread.join()
            self.assertEqual(errors,[])
            self.assertEqual(dist.reductions,0)
            return original(input)
        targets[0].forward=with_context_copy
        with scope:
            for module in targets:module.forward(GlobalAmaxTensor([[1]*16]*3))
        self.assertAlmostEqual(observations[0],2/2688,places=8)
        self.assertEqual(dist.reductions,200)
        self.assertEqual(scope.counts()['per_block'],[4]*50)

# END GLOBAL NVFP4 AMAX CPU CONTRACT TESTS


class GlobalAmaxIntegrationTests(unittest.TestCase):
    def test_reviewed_scope_reversal_is_exact_previous_source_and_rejects_tamper(self):
        source = AFTER.read_text()
        self.assertEqual(hashlib.sha256(reverse_amax_scope(source).encode()).hexdigest(),
                         '51e98f82d164e5c6bf447eaf095ce1dfae0ca4ace10037f695d9f88226142ff3')
        for old,new in (('scope.dist.ReduceOp.MAX','scope.dist.ReduceOp.SUM'),
                        ('if ctx.world == 2 else nullcontext()', 'if ctx.world >= 1 else nullcontext()')):
            with self.subTest(change=old), self.assertRaisesRegex(AssertionError,'reviewed block digest'):
                reverse_amax_scope(source.replace(old,new,1))

    def test_actual_block_loop_scopes_world2_only_and_restores_before_count_log(self):
        function = next(n for n in ast.parse(AFTER.read_text()).body
                        if isinstance(n,ast.FunctionDef) and n.name=='sp_forward')
        index = next(i for i,n in enumerate(function.body) if isinstance(n,ast.Assign)
                     and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='scale_context')
        body = function.body[index:index+3]
        self.assertEqual([type(n).__name__ for n in body],['Assign','With','If'])
        for world in (1,2):
            with self.subTest(world=world):
                events=[]
                class Scope:
                    def __init__(self,*args,**kwargs):events.append('construct')
                    def __enter__(self):events.append('enter');return self
                    def __exit__(self,*args):events.append('restore');return False
                    def counts(self):
                        events.append('counts')
                        return {'target_forwards':200,'activation_syncs':200,'per_block':[4]*50}
                def block(*args):events.append('block');return args[1]+1
                namespace={'SPNVFP4AmaxScope':Scope,'nullcontext':__import__('contextlib').nullcontext,
                    'dit':SimpleNamespace(blocks=list(range(50))),'ctx':SimpleNamespace(world=world),
                    'torch':object(),'dist':object(),'rank':0,'prefetch_queue':None,'device':'cpu','trace':None,
                    'comfy':SimpleNamespace(quant_ops=object(),model_prefetch=SimpleNamespace(prefetch_queue_pop=lambda *a,**k:None)),
                    'transformer_options':{},'h':0,'t_emb':None,'local_segments':None,'rope_for_blocks':None,
                    'sp_block':block,'os':SimpleNamespace(environ={'MINIMAX_SP_VERIFY':'1'}),
                    'logging':SimpleNamespace(info=lambda *args:events.append('log'))}
                exec(compile(ast.Module(body=body,type_ignores=[]),'<actual-SP-block-loop>','exec'),namespace)
                self.assertEqual(namespace['h'],50)
                if world==1:
                    self.assertEqual(events,['block']*50)
                else:
                    self.assertEqual(events,['construct','enter']+['block']*50+['restore','counts','log'])


"""Portable CPU test fragment for the real block-0 diagnostic trace AST.

Append before the existing unittest.main guard in test_minimax_sp_pdd.py.
NumPy models storage/hooks only: these tests do not validate CUDA, NVFP4,
distributed collectives, full-row parity, or any generated media.
"""
import ast as _b0t_ast
from contextlib import nullcontext as _b0t_nullcontext
import json as _b0t_json
from pathlib import Path as _B0TPath
from types import SimpleNamespace as _B0TNamespace
import unittest as _b0t_unittest

import numpy as _b0t_np


def _b0t_source_path():
    if "AFTER" in globals():
        return _B0TPath(globals()["AFTER"])
    return (_B0TPath(__file__).resolve().parent / "comfy-omni-deployment" /
            "custom_nodes/minimax_sp_pdd/minimax_sp/sp_forward.py")


class _B0TTensor:
    def __init__(self, value):
        self.value = _b0t_np.asarray(value, dtype=_b0t_np.float32)
        self.dtype = "torch.float32"

    @property
    def shape(self):
        return self.value.shape

    @property
    def ndim(self):
        return self.value.ndim

    def __getitem__(self, item):
        return _B0TTensor(self.value[item])

    def detach(self):
        return self

    def to(self, *, device, **kwargs):
        if device != "cpu":
            raise AssertionError("test only permits CPU snapshots")
        return _B0TTensor(self.value.copy(order="C")) if kwargs.get("copy") else self

    def element_size(self):
        return self.value.dtype.itemsize

    def view(self, dtype):
        if dtype != "uint8":
            raise AssertionError("hash must view original bits as uint8")
        return _B0TNamespace(numpy=lambda: self.value.view(_b0t_np.uint8))

    def clone(self):
        return _B0TTensor(self.value.copy())

    def abs(self):
        return _B0TTensor(abs(self.value))

    def max(self):
        return _B0TTensor(self.value.max())

    def item(self):
        return float(self.value.item())

    def float(self):
        return self

    def all(self):
        return bool(self.value.all())

    def __sub__(self, other):
        return _B0TTensor(self.value - other.value)


class _B0TModule:
    def __init__(self, function=None):
        self.function = function or (lambda tensor, *args, **kwargs: tensor)
        self.before, self.after = [], []
        self.reject_registration = False

    def _register(self, target, hook):
        if self.reject_registration:
            raise ValueError("private exception text must not enter a packet")
        target.append(hook)
        return _B0TNamespace(remove=lambda: target.remove(hook))

    def register_forward_pre_hook(self, hook):
        return self._register(self.before, hook)

    def register_forward_hook(self, hook):
        return self._register(self.after, hook)

    def __call__(self, tensor, *args, **kwargs):
        for hook in tuple(self.before):
            hook(self, (tensor, *args))
        output = self.function(tensor, *args, **kwargs)
        for hook in tuple(self.after):
            hook(self, (tensor, *args), output)
        return output


def _b0t_model():
    modules = [_B0TModule(lambda tensor, factor=factor:
                         _B0TTensor(tensor.value * factor))
               for factor in (2, 3, 4, 5)]

    def block_forward(tensor, *args, **kwargs):
        for module in modules:
            tensor = module(tensor)
        return tensor

    block = _B0TModule(block_forward)
    block.attn = _B0TNamespace(qkv_proj=modules[0], out_proj=modules[1])
    block.mlp = _B0TNamespace(fc1=modules[2], fc2=modules[3])
    exits = [_B0TModule() for _ in range(49)]
    final = _B0TModule()
    for name, width in (("video_out",3072),("audio_out",1024)):
        weight = _B0TNamespace(dtype="torch.float32",shape=(width,5376),
                               device=_B0TNamespace(type="meta"),is_meta=True)
        bias = _B0TNamespace(dtype="torch.float32",shape=(width,))
        setattr(final,name,_B0TNamespace(weight=weight,bias=bias))
    return _B0TNamespace(blocks=exits[:34] + [block] + exits[34:],final_layer=final), modules + [block] + exits + [final]


def _b0t_load():
    parsed = _b0t_ast.parse(_b0t_source_path().read_text())
    symbols = {"Block0Trace", "block0_trace_rows", "block0_trace_summary", "sp_block",
               "stored_head_metadata", "head_metadata_summary", "streamed_tensor_sha256", "layer_hash_summary", "_trace_hash_summary", "projection_hash_summary", "ProjectionReplay", "ProjectionReplayError",
               "_replay_require", "_replay_digest", "_replay_quantized_identity", "_replay_weight_bias",
               "_replay_compare_outputs", "projection_replay_summary", "nvfp4_scale_sha256"}
    selected = [node for node in parsed.body if
                isinstance(node, (_b0t_ast.FunctionDef, _b0t_ast.ClassDef))
                and node.name in symbols or
                isinstance(node, _b0t_ast.Assign) and any(
                    isinstance(target, _b0t_ast.Name)
                    and (target.id in ("BLOCK0_TRACE_POINTS", "TRACE_PROJECTION_BLOCK", "TRACE_HASH_CHUNK_BYTES")
                         or target.id.startswith("HEAD_METADATA_") or target.id.startswith("REPLAY_")) for target in node.targets)]
    namespace = {"torch": _B0TNamespace(
        Tensor=_B0TTensor, uint8="uint8", contiguous_format="contiguous",
        aminmax=lambda tensor: (_B0TTensor(tensor.value.min()),
                                _B0TTensor(tensor.value.max())),
        isfinite=lambda tensor: _B0TTensor(_b0t_np.isfinite(tensor.value))),
        "hashlib":hashlib,"json":json,"sys":__import__('sys'), "threading":__import__('threading'), "_ABSENT":object(),
        "region": lambda _name: _b0t_nullcontext(),
        "sp_attention": lambda *args, **kwargs: None}
    exec(compile(_b0t_ast.Module(body=selected, type_ignores=[]),
                 str(_b0t_source_path()), "exec"), namespace)
    return _B0TNamespace(**namespace)


class Block0TraceMetadataTests(_b0t_unittest.TestCase):
    def setUp(self):
        self.api = _b0t_load()

    def _packet(self, rank, start, stop, rows=None, hash_windows=None):
        model, modules = _b0t_model()
        tensor = _B0TTensor(_b0t_np.arange(start * 3, stop * 3).reshape(-1, 3))
        with self.api.Block0Trace(model, rank=rank, start=start, stop=stop,
                                  rows=rows, hash_windows=hash_windows) as trace:
            # Exercise actual sp_block AST, including block.__call__ hooks.
            for block in model.blocks:
                tensor = self.api.sp_block(block, tensor, None, None, None,
                                          _B0TNamespace(), {})
            if rank == -1:
                model.final_layer(tensor)
        if rank != -1:
            trace.capture_head_metadata(model.final_layer)
        self.assertTrue(all(not module.before and not module.after for module in modules))
        return trace.packet()

    def _packets(self):
        packets = [self._packet(0, 0, 10), self._packet(1, 10, 19)]
        rows, total = self.api.block0_trace_rows(packets)
        return packets, self._packet(-1, 0, total, rows, [(p['start'],p['stop']) for p in packets])

    def test_real_sp_block_hook_route_and_global_row_alignment(self):
        packets, native = self._packets()
        result = self.api.block0_trace_summary(packets, native)
        self.assertEqual(len(result), 116)
        self.assertTrue(all(row["max_abs"] == 0 and row["relative"] == 0 for row in result))
        self.assertTrue(all(row["sampled_rows"] == 7 for row in result))
        self.assertGreater(min(packets[1]["rows"]), max(packets[0]["rows"]))
        encoded = _b0t_json.dumps(result, allow_nan=False)
        self.assertNotIn("samples", encoded)
        self.assertNotIn("private exception", encoded)

    def test_snapshot_does_not_alias_inplace_source_and_all_features_retained(self):
        model, _ = _b0t_model()
        tensor = _B0TTensor(_b0t_np.arange(60).reshape(10, 6))
        trace = self.api.Block0Trace(model, rank=0, start=0, stop=10)
        trace._capture("attn.qkv_proj.post", tensor)
        snapshot = trace.packet()["entries"]["attn.qkv_proj.post"]["samples"]
        saved = snapshot.value.copy()
        tensor.value[:] = -999
        _b0t_np.testing.assert_array_equal(snapshot.value, saved)
        self.assertEqual(snapshot.shape, (7, 6))

    def test_partial_hook_registration_failure_cleans_every_installed_hook(self):
        model, modules = _b0t_model()
        modules[1].reject_registration = True
        with self.api.Block0Trace(model, rank=0, start=0, stop=10) as trace:
            self.assertEqual(trace.packet()["errors"], ["ValueError"])
            self.assertTrue(all(not module.before and not module.after for module in modules))
        self.assertEqual(trace.handles, [])

    def test_body_exception_propagates_and_removes_hooks(self):
        model, modules = _b0t_model()
        with self.assertRaisesRegex(RuntimeError, "body_failed"):
            with self.api.Block0Trace(model, rank=0, start=0, stop=10):
                raise RuntimeError("body_failed")
        self.assertTrue(all(not module.before and not module.after for module in modules))

    def test_missing_point_errors_duplicate_rank_and_row_mismatch_rejected(self):
        mutations = [lambda packets: packets[0]["entries"].pop("mlp.fc2.post"),
                     lambda packets: packets[1].update(rank=0),
                     lambda packets: packets[1].update(errors=["RuntimeError"]),
                     lambda packets: packets[1]["rows"].__setitem__(0, 0),
                     lambda packets: packets[1]["entries"]["mlp.fc1.pre"].update(dtype="other")]
        for mutate in mutations:
            with self.subTest(mutation=mutations.index(mutate)):
                packets, native = self._packets()
                mutate(packets)
                with self.assertRaises(RuntimeError):
                    self.api.block0_trace_summary(packets, native)

    def test_numeric_difference_at_correct_point_rank_and_finite_not_promoted(self):
        packets, native = self._packets()
        entry = packets[1]["entries"]["attn.qkv_proj.post"]
        entry["samples"].value[0, 0] += 8
        entry["finite"] = False
        result = self.api.block0_trace_summary(packets, native)
        changed = [row for row in result if row["max_abs"] != 0]
        self.assertEqual(len(changed), 1)
        self.assertEqual((changed[0]["rank"], changed[0]["point"]),
                         (1, "attn.qkv_proj.post"))
        self.assertEqual(changed[0]["max_abs"], 8)
        self.assertFalse(changed[0]["finite"])

    def test_later_block_exit_difference_and_missing_final_exit_are_localized(self):
        packets, native = self._packets()
        packets[1]["entries"]["block37.post"]["samples"].value[0, 0] += 16
        result = self.api.block0_trace_summary(packets, native)
        changed = [row for row in result if row["max_abs"] != 0]
        self.assertEqual([(row["rank"], row["point"], row["max_abs"]) for row in changed],
                         [(1, "block37.post", 16)])
        packets[0]["entries"].pop("block49.post")
        with self.assertRaisesRegex(RuntimeError, 'rank_packet_invalid'):
            self.api.block0_trace_summary(packets, native)

    def test_late_hook_registration_failure_cleans_all_prior_layer_hooks(self):
        model, modules = _b0t_model()
        model.blocks[49].reject_registration = True
        with self.api.Block0Trace(model, rank=0, start=0, stop=10) as trace:
            self.assertEqual(trace.packet()["errors"], ["ValueError"])
            self.assertTrue(all(not module.before and not module.after for module in modules))
        self.assertEqual(trace.handles, [])

    def test_stored_heads_are_attributes_only_and_six_closed_schema_records(self):
        packets, native = self._packets()
        packets[1]["head_metadata"][0]["weight_dtype"] = "torch.bfloat16"
        result = self.api.head_metadata_summary(packets,native)
        self.assertEqual(len(result),6)
        self.assertEqual([(row["phase"],row["rank"],row["head"]) for row in result],
                         [(phase,rank,head) for phase,rank in (("sp2",0),("sp2",1),("native",-1))
                          for head in ("video_out","audio_out")])
        self.assertTrue(all(row["execution_weight_verified"] is False for row in result))
        self.assertTrue(all(row["scope"] == "stored_runtime_metadata" for row in result))
        self.assertEqual(result[2]["weight_dtype"],"torch.bfloat16")
        self.assertNotIn("sha256",_b0t_json.dumps(result,allow_nan=False))
        # SimpleNamespace fixtures have no tensor math/copy APIs: capture above
        # would fail if it tried detach/to/cpu/hash or an extra cast context.
        packets[1]["head_metadata"][0]["weight_type"] = "private_unknown_string"
        with self.assertRaisesRegex(RuntimeError,'head_metadata_fields_invalid'):
            self.api.head_metadata_summary(packets,native)

    def test_head_capture_error_preserves_layer_samples_but_rejects_head_evidence(self):
        packets, native = self._packets()
        packets[1]["head_errors"] = ["AttributeError"]
        self.assertEqual(len(self.api.block0_trace_summary(packets,native)),116)
        with self.assertRaisesRegex(RuntimeError,'head_metadata_packet_invalid'):
            self.api.head_metadata_summary(packets,native)




class Block0TraceGroupGateTests(unittest.TestCase):
    def test_trace_packet_failure_cannot_accept_passing_native_parity(self):
        f = parity_fixture(world2_good=True)
        f.namespace['spf'].block0_trace_rows.side_effect = RuntimeError('invalid_packet_fixture')
        with self.assertRaisesRegex(RuntimeError, 'diagnostic evidence incomplete'):
            f.forward(*f.args)
        f.owner.destroy.assert_called_once()
        self.assertTrue(f.owner.verify_pending)
        self.assertNotIn(('sp',1),f.events)
        self.assertEqual(f.args[5],{})

    def test_trace_summary_failure_keeps_original_failed_gate_and_world1(self):
        f = parity_fixture()
        f.namespace['spf'].block0_trace_summary.side_effect = RuntimeError('summary_fixture')
        with self.assertRaisesRegex(RuntimeError, 'video differs from native H3'):
            f.forward(*f.args)
        self.assertIn(('sp',1),f.events)
        f.owner.destroy.assert_called_once()
        self.assertEqual(f.args[5],{})

    def test_formal_path_keeps_original_options_and_has_no_trace_oracle(self):
        f = parity_fixture(verify=False)
        dispatch = Mock(wraps=f.namespace['spf'].sp_forward)
        f.namespace['spf'].sp_forward = dispatch
        f.forward(*f.args)
        self.assertIs(dispatch.call_args.args[4], f.args[5])
        f.namespace['spf'].block0_trace_rows.assert_not_called()
        f.namespace['spf'].block0_trace_summary.assert_not_called()
        self.assertEqual(f.args[5],{})



class _HashRawTensor:
    """CPU fixture preserving FP16/BF16 raw storage; no tensor math conversion."""
    def __init__(self, values, dtype, tracker=None, copied=False):
        self.values = np.asarray(values)
        self.dtype = dtype
        self.tracker = tracker if tracker is not None else {"copies": [], "alive": 0, "peak": 0}
        self.copied = copied
        if copied:
            self.tracker["copies"].append(self.values.nbytes)
            self.tracker["alive"] += 1
            self.tracker["peak"] = max(self.tracker["peak"], self.tracker["alive"])

    def __del__(self):
        if getattr(self, "copied", False):
            self.tracker["alive"] -= 1

    @property
    def shape(self): return self.values.shape
    @property
    def ndim(self): return self.values.ndim
    def element_size(self): return self.values.dtype.itemsize
    def __getitem__(self, key): return _HashRawTensor(self.values[key], self.dtype, self.tracker)
    def detach(self): return self

    def to(self, **kwargs):
        if kwargs != {"device":"cpu", "non_blocking":False, "copy":True, "memory_format":"contiguous"}:
            raise AssertionError("hash copy must be explicit synchronous C-order CPU copy")
        return _HashRawTensor(self.values.copy(order="C"), self.dtype, self.tracker, copied=True)

    def view(self, dtype):
        if dtype != "uint8":
            raise AssertionError("original scalar bits must be viewed as bytes before NumPy")
        return SimpleNamespace(numpy=lambda:self.values.view(np.uint8))


class FullLayerHashTests(unittest.TestCase):
    def setUp(self): self.api = _b0t_load()

    def test_window_header_and_digest_are_independent_of_rank_phase_and_chunk_size(self):
        values = np.arange(105,dtype=np.float32).reshape(21,5)
        source = _HashRawTensor(values,"torch.float32")
        small = self.api.streamed_tensor_sha256(source,tensor_global_start=0,
                    global_start=3,global_stop=17,chunk_limit_bytes=40)
        large = self.api.streamed_tensor_sha256(source,tensor_global_start=0,
                    global_start=3,global_stop=17,chunk_limit_bytes=100)
        shard = self.api.streamed_tensor_sha256(_HashRawTensor(values[3:17],"torch.float32"),
                    tensor_global_start=3,global_start=3,global_stop=17,chunk_limit_bytes=40)
        self.assertEqual(small["sha256"],large["sha256"])
        self.assertEqual(small,shard)
        self.assertEqual(small["shape"],[14,5])
        self.assertEqual((small["chunks"],large["chunks"]),(7,3))
        self.assertEqual(small["tensor_bytes"],280)
        self.assertLessEqual(max(source.tracker["copies"]),100)
        self.assertEqual(source.tracker["peak"],1)
        self.assertEqual(source.tracker["alive"],0)

    def test_noncontiguous_logical_rows_match_contiguous_without_full_copy(self):
        view = np.arange(96,dtype=np.float32).reshape(12,8)[::2,::2]
        self.assertFalse(view.flags.c_contiguous)
        source = _HashRawTensor(view,"torch.float32")
        kwargs = dict(tensor_global_start=7,global_start=7,global_stop=13,chunk_limit_bytes=32)
        got = self.api.streamed_tensor_sha256(source,**kwargs)
        expected = self.api.streamed_tensor_sha256(_HashRawTensor(view.copy(),"torch.float32"),**kwargs)
        self.assertEqual(got,expected)
        self.assertEqual(source.tracker["copies"],[32,32,32])
        self.assertEqual(source.tracker["peak"],1)

    def test_float16_and_bfloat16_original_bits_match_manual_canonical_digest(self):
        bits=np.array([[0,32768,1,32767],[16256,65472,12345,45678]],dtype=np.uint16)
        for dtype,values in (("torch.bfloat16",bits),("torch.float16",bits.view(np.float16))):
            with self.subTest(dtype=dtype):
                got=self.api.streamed_tensor_sha256(_HashRawTensor(values,dtype),
                    tensor_global_start=9,global_start=9,global_stop=11,chunk_limit_bytes=8)
                header={key:got[key] for key in ("schema","dtype","shape","global_start","global_stop","byte_order","order")}
                h=hashlib.sha256(json.dumps(header,sort_keys=True,separators=(",",":")).encode("ascii")+b"\n")
                h.update(memoryview(bits.view(np.uint8)).cast("B"))
                self.assertEqual(got["sha256"],h.hexdigest())
                self.assertEqual((got["tensor_bytes"],got["chunks"]),(16,2))

    def test_change_outside_sample_rows_is_detected_by_full_hash_and_snapshot_is_immediate(self):
        model,_=_b0t_model()
        source=_B0TTensor(np.arange(30,dtype=np.float32).reshape(10,3))
        altered=_B0TTensor(source.value.copy())
        altered.value.view(np.uint32)[2,0]^=1
        a=self.api.Block0Trace(model,rank=0,start=0,stop=10)
        b=self.api.Block0Trace(model,rank=0,start=0,stop=10)
        a._capture("block0.post",source); b._capture("block0.post",altered)
        self.assertNotIn(2,a.rows)
        np.testing.assert_array_equal(a.entries["block0.post"]["samples"].value,
                                      b.entries["block0.post"]["samples"].value)
        first=a.full_hashes["block0.post"][0]["sha256"]
        self.assertNotEqual(first,b.full_hashes["block0.post"][0]["sha256"])
        source.value[:]=-100
        self.assertEqual(a.full_hashes["block0.post"][0]["sha256"],first)

    def test_hash_summary_has_100_bound_rows_and_mismatch_is_observation(self):
        fixtures=Block0TraceMetadataTests();fixtures.setUp()
        packets,native=fixtures._packets()
        rows=self.api.layer_hash_summary(packets,native)
        self.assertEqual(len(rows),100)
        self.assertTrue(all(row["match"] for row in rows))
        self.assertEqual([(row["block"],row["rank"]) for row in rows],
                         [(block,rank) for block in range(50) for rank in range(2)])
        numeric=self.api.block0_trace_summary(packets,native)
        self.assertTrue(all(row["projection_block"]==34 for row in numeric))
        packets[1]["full_hashes"]["block2.post"][0]["sha256"]="0"*64
        rows=self.api.layer_hash_summary(packets,native)
        self.assertEqual([(r["block"],r["rank"]) for r in rows if not r["match"]],[(2,1)])
        self.assertNotIn("samples",json.dumps(rows,allow_nan=False))

    def test_missing_hash_wrong_window_dtype_shape_and_counts_fail_closed(self):
        fixtures=Block0TraceMetadataTests();fixtures.setUp()
        mutations=[lambda p:p[0]["full_hashes"].pop("block49.post"),
                   lambda p:p[1]["full_hashes"]["block0.post"][0].update(global_start=0),
                   lambda p:p[0]["full_hashes"]["block0.post"][0].update(dtype="torch.float16"),
                   lambda p:p[0]["full_hashes"]["block0.post"][0].update(shape=[9,3]),
                   lambda p:p[0]["full_hashes"]["block0.post"][0].update(chunks=2),
                   lambda p:p[0]["full_hashes"]["block0.post"][0].update(tensor_bytes=1),
                   lambda p:p[0].update(hash_errors=["RuntimeError"])]
        for index,mutation in enumerate(mutations):
            with self.subTest(index=index):
                packets,native=fixtures._packets();mutation(packets)
                with self.assertRaises(RuntimeError):self.api.layer_hash_summary(packets,native)

    def test_hash_capture_failure_retains_error_code_cleans_hooks_and_cannot_pass_gate(self):
        model,modules=_b0t_model()
        with self.api.Block0Trace(model,rank=0,start=0,stop=10,hash_windows=[(0,11)]) as trace:
            tensor=_B0TTensor(np.ones((10,3)))
            for block in model.blocks:tensor=block(tensor)
        self.assertEqual(trace.errors,[])
        self.assertEqual(len(trace.hash_errors),58)
        self.assertEqual(set(trace.hash_errors),{"RuntimeError"})
        self.assertTrue(all(not m.before and not m.after for m in modules))
        f=parity_fixture(world2_good=True)
        f.namespace["spf"].layer_hash_summary.side_effect=RuntimeError("hash_fixture")
        with self.assertRaisesRegex(RuntimeError,"diagnostic evidence incomplete"):f.forward(*f.args)
        f.owner.destroy.assert_called_once()
        self.assertTrue(f.owner.verify_pending)
        self.assertEqual(f.args[5],{})

    def test_oversize_row_invalid_cap_and_foreign_byte_order_are_rejected(self):
        source=_HashRawTensor(np.zeros((2,5),dtype=np.float32),"torch.float32")
        for cap in (0,True,19,16*1024*1024+1):
            with self.subTest(cap=cap),self.assertRaises(RuntimeError):
                self.api.streamed_tensor_sha256(source,tensor_global_start=0,
                    global_start=0,global_stop=2,chunk_limit_bytes=cap)
        namespace=self.api.streamed_tensor_sha256.__globals__
        original=namespace["sys"]
        try:
            namespace["sys"]=SimpleNamespace(byteorder="big")
            with self.assertRaisesRegex(RuntimeError,"byte_order_invalid"):
                self.api.streamed_tensor_sha256(source,tensor_global_start=0,global_start=0,global_stop=2)
        finally:namespace["sys"]=original
        self.assertEqual(source.tracker["copies"],[])


class ProjectionHashTests(unittest.TestCase):
    def setUp(self):
        self.api=_b0t_load()
        self.fixtures=Block0TraceMetadataTests();self.fixtures.setUp()

    def test_sixteen_projection_rows_and_layer_summary_exactly_preserved(self):
        packets,native=self.fixtures._packets()
        rows=self.api.projection_hash_summary(packets,native)
        self.assertEqual(len(rows),16)
        self.assertEqual([(r["point"],r["rank"]) for r in rows],
                         [(point,rank) for point in self.api.BLOCK0_TRACE_POINTS[:8] for rank in (0,1)])
        self.assertTrue(all(r["match"] and r["block"]==34 for r in rows))
        layer=self.api.layer_hash_summary(packets,native)
        previous=ast.parse(reverse_projection_hash(AFTER.read_text()))
        function=next(n for n in previous.body if isinstance(n,ast.FunctionDef) and n.name=="layer_hash_summary")
        namespace=dict(self.api.layer_hash_summary.__globals__)
        exec(compile(ast.Module(body=[function],type_ignores=[]),"prior-layer-summary","exec"),namespace)
        self.assertEqual(layer,namespace["layer_hash_summary"](packets,native))
        self.assertEqual(len(layer),100)
        self.assertTrue(all(set(r)==set(layer[0])|{"point"} for r in rows))
        self.assertNotIn("samples",json.dumps(rows,allow_nan=False))

    def test_projection_snapshot_detects_unsampled_bit_without_layer_hash_mutation(self):
        model,_=_b0t_model()
        source=_B0TTensor(np.arange(30,dtype=np.float32).reshape(10,3))
        changed=_B0TTensor(source.value.copy());changed.value.view(np.uint32)[2,0]^=1
        traces=[self.api.Block0Trace(model,rank=0,start=0,stop=10) for _ in range(2)]
        for trace,tensor in zip(traces,(source,changed)):trace._capture("mlp.fc1.post",tensor)
        np.testing.assert_array_equal(traces[0].entries["mlp.fc1.post"]["samples"].value,
                                      traces[1].entries["mlp.fc1.post"]["samples"].value)
        self.assertNotEqual(traces[0].projection_hashes["mlp.fc1.post"][0]["sha256"],
                            traces[1].projection_hashes["mlp.fc1.post"][0]["sha256"])
        self.assertTrue(all(not t.full_hashes for t in traces))
        saved=traces[0].projection_hashes["mlp.fc1.post"][0]["sha256"]
        source.value[:]=0
        self.assertEqual(traces[0].projection_hashes["mlp.fc1.post"][0]["sha256"],saved)

    def test_missing_projection_wrong_dtype_window_and_hash_errors_rejected(self):
        mutations=[lambda p:p[0]["projection_hashes"].pop("mlp.fc2.post"),
            lambda p:p[1]["projection_hashes"]["attn.qkv_proj.pre"][0].update(global_start=0),
            lambda p:p[0]["projection_hashes"]["mlp.fc1.post"][0].update(dtype="torch.float16"),
            lambda p:p[0]["projection_hashes"]["mlp.fc2.pre"][0].update(shape=[10,4]),
            lambda p:p[0].update(hash_errors=["RuntimeError"])]
        for index,mutation in enumerate(mutations):
            with self.subTest(index=index):
                packets,native=self.fixtures._packets();mutation(packets)
                with self.assertRaises(RuntimeError):self.api.projection_hash_summary(packets,native)
        packets,native=self.fixtures._packets()
        packets[0]["projection_hashes"]["attn.out_proj.pre"][0]["sha256"]="0"*64
        rows=self.api.projection_hash_summary(packets,native)
        self.assertEqual([(r["point"],r["rank"]) for r in rows if not r["match"]],[("attn.out_proj.pre",0)])
        self.assertTrue(all(r["match"] for r in self.api.layer_hash_summary(packets,native)))

    def test_projection_evidence_failure_cannot_accept_good_parity_and_formal_is_unchanged(self):
        f=parity_fixture(world2_good=True)
        f.namespace["spf"].projection_hash_summary.side_effect=RuntimeError("projection_fixture")
        with self.assertRaisesRegex(RuntimeError,"diagnostic evidence incomplete"):f.forward(*f.args)
        f.owner.destroy.assert_called_once()
        self.assertTrue(f.owner.verify_pending)
        self.assertEqual(f.args[5],{})
        f=parity_fixture(verify=False)
        f.forward(*f.args)
        f.namespace["spf"].projection_hash_summary.assert_not_called()
        f.namespace["spf"].layer_hash_summary.assert_not_called()
        self.assertEqual(f.args[5],{})

"""CPU NumPy layout fixtures only; no Torch/GPU/media is imported or read."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


class Tensor:
    copies = []

    def __init__(self, array, dtype="fp8", device="cuda"):
        self.array, self.dtype, self.device = array, dtype, device

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def shape(self):
        return self.array.shape

    def is_contiguous(self):
        return self.array.flags.c_contiguous

    def view(self, dtype):
        if dtype != "u8":
            raise AssertionError("only bit-preserving uint8 view is expected")
        return Tensor(self.array, dtype, self.device)

    def __getitem__(self, index):
        return Tensor(self.array[index], self.dtype, self.device)

    def detach(self):
        return self

    def to(self, *, device, non_blocking, copy, memory_format):
        if (device, non_blocking, copy, memory_format) != ("cpu", False, True, "C"):
            raise AssertionError("copy must be synchronous CPU contiguous")
        self.copies.append(self.array.nbytes)
        return Tensor(np.array(self.array, copy=True, order="C"), self.dtype, "cpu")

    def reshape(self, *shape):
        if self.device != "cpu":
            raise AssertionError("inverse layout must run only on CPU")
        return Tensor(self.array.reshape(*shape), self.dtype, self.device)

    def transpose(self, first, second):
        return Tensor(self.array.swapaxes(first, second), self.dtype, self.device)

    def permute(self, *axes):
        return Tensor(self.array.transpose(*axes), self.dtype, self.device)

    def contiguous(self):
        return Tensor(np.ascontiguousarray(self.array), self.dtype, self.device)

    def numpy(self):
        if self.device != "cpu":
            raise AssertionError("no direct GPU NumPy conversion")
        return self.array


FAKE_TORCH = types.SimpleNamespace(Tensor=Tensor, uint8="u8", float8_e4m3fn="fp8", contiguous_format="C")
_scale_node=next(n for n in ast.parse(AFTER.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=="nvfp4_scale_sha256")
_scale_namespace={"hashlib":hashlib,"json":json,"sys":sys,"torch":FAKE_TORCH}
exec(compile(ast.Module(body=[_scale_node],type_ignores=[]),str(AFTER),"exec"),_scale_namespace)
helper=types.SimpleNamespace(**_scale_namespace)


def blocked(logical):
    """Independent direct CUDA layout offsets, not the helper's reshape inverse."""
    rows, cols = logical.shape
    padded_rows, padded_cols = (rows + 127) // 128 * 128, (cols + 3) // 4 * 4
    flat = np.full(padded_rows * padded_cols, 251, dtype=np.uint8)
    r = np.arange(rows, dtype=np.int64)[:, None]
    c = np.arange(cols, dtype=np.int64)[None, :]
    offsets = ((r // 128) * (padded_cols // 4) + c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4
    flat[offsets] = logical
    return Tensor(flat.reshape(padded_rows, padded_cols))


def expected(logical, start, stop, tensor_start=0):
    values = np.ascontiguousarray(logical[start - tensor_start:stop - tensor_start])
    header = {"schema": "minimax-sp-layer-sha256-v1", "dtype": "torch.uint8",
              "shape": list(values.shape), "global_start": start, "global_stop": stop,
              "byte_order": "little", "order": "C"}
    digest = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n")
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


class ScaleHashTests(unittest.TestCase):
    def setUp(self):
        Tensor.copies = []

    def test_independent_swizzle_oracle_and_padding(self):
        values = np.arange(259 * 7, dtype=np.uint16).reshape(259, 7).astype(np.uint8)
        actual = helper.nvfp4_scale_sha256(blocked(values), logical_rows=259, logical_cols=7,
                                           global_start=31, global_stop=258)
        self.assertEqual(actual, expected(values, 31, 258))
        self.assertEqual(Tensor.copies, [128 * 8] * 3)

    def test_actual_22584_boundary_full_and_shard(self):
        rows, cols, split = 45167, 448, 22584
        r, c = np.arange(rows, dtype=np.uint32)[:, None], np.arange(cols, dtype=np.uint32)[None, :]
        values = ((r * 17 + c * 29 + r // 128) % 256).astype(np.uint8)
        full, local = blocked(values), blocked(values[split:])
        a = helper.nvfp4_scale_sha256(full, logical_rows=rows, logical_cols=cols,
                                     global_start=split, global_stop=rows)
        b = helper.nvfp4_scale_sha256(local, logical_rows=rows - split, logical_cols=cols,
                                     global_start=split, global_stop=rows, tensor_global_start=split)
        self.assertEqual(a, b)
        self.assertEqual(a, expected(values, split, rows))
        self.assertEqual(max(Tensor.copies), 57344)
        self.assertTrue(all(size <= 16777216 for size in Tensor.copies))
        # One byte far from the seven diagnostic sample rows must be detected.
        changed = values[split:].copy()
        changed[123, 77] ^= 1
        self.assertNotEqual(b, helper.nvfp4_scale_sha256(blocked(changed),
            logical_rows=rows - split, logical_cols=cols, global_start=split,
            global_stop=rows, tensor_global_start=split))

    def test_global_coordinates_are_bound_without_affecting_values(self):
        values = np.arange(137 * 5, dtype=np.uint16).reshape(137, 5).astype(np.uint8)
        hashes = []
        for origin in (0, 22584):
            actual = helper.nvfp4_scale_sha256(blocked(values), logical_rows=137, logical_cols=5,
                global_start=origin + 9, global_stop=origin + 130, tensor_global_start=origin)
            self.assertEqual(actual, expected(values, origin + 9, origin + 130, origin))
            hashes.append(actual)
        self.assertNotEqual(*hashes)

    def test_reject_invalid_shape_dtype_window_and_limit_before_copy(self):
        values = np.zeros((129, 7), dtype=np.uint8)
        base = dict(logical_rows=129, logical_cols=7, global_start=0, global_stop=129)
        invalid = [dict(global_start=-1), dict(global_stop=130), dict(global_stop=0),
                   dict(logical_rows=True), dict(logical_cols=0), dict(chunk_limit_bytes=1023),
                   dict(chunk_limit_bytes=16777217), dict(tensor_global_start=1)]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                helper.nvfp4_scale_sha256(blocked(values), **{**base, **changes})
        for tensor in (Tensor(np.zeros((128, 8), dtype=np.uint8)),
                       Tensor(np.zeros((256, 8), dtype=np.uint8), dtype="u8"),
                       Tensor(np.zeros((256, 16), dtype=np.uint8)[:, ::2])):
            with self.assertRaises(RuntimeError):
                helper.nvfp4_scale_sha256(tensor, **base)
        self.assertEqual(Tensor.copies, [])

    def test_exact_tile_limit_and_digest_independent_of_limit(self):
        values = np.arange(131 * 7, dtype=np.uint16).reshape(131, 7).astype(np.uint8)
        args = dict(logical_rows=131, logical_cols=7, global_start=5, global_stop=130)
        a = helper.nvfp4_scale_sha256(blocked(values), **args, chunk_limit_bytes=1024)
        b = helper.nvfp4_scale_sha256(blocked(values), **args)
        self.assertEqual(a, b)
        self.assertEqual(a, expected(values, 5, 130))




class _ReplayNumber(_B0TTensor):
    def __init__(self, value, dtype="torch.bfloat16"):
        super().__init__(value);self.dtype=dtype
    def __getitem__(self,item):return _ReplayNumber(self.value[item],self.dtype)
    def to(self,*,device,**kwargs):
        if device!="cpu":raise AssertionError("CPU only")
        return _ReplayNumber(self.value.copy(order="C"),kwargs.get("dtype",self.dtype))
    def sum(self):return _ReplayNumber(self.value.sum(),self.dtype)


class ReplayOutputAndGateTests(unittest.TestCase):
    def test_full_output_comparison_counts_every_value_and_finite_failure_is_explicit(self):
        api=_b0t_load();namespace=api._replay_compare_outputs.__globals__
        namespace["torch"]=SimpleNamespace(float32="torch.float32",uint8="uint8",contiguous_format="contiguous",
            ne=lambda a,b:_ReplayNumber(np.not_equal(a.value,b.value)),
            isfinite=lambda a:_ReplayNumber(np.isfinite(a.value)),
            abs=lambda a:_ReplayNumber(np.abs(a.value)),max=lambda a:_ReplayNumber(np.max(a.value)))
        full=_ReplayNumber(np.full((9,5376),2.0,np.float32));shard=_ReplayNumber(full.value[4:].copy())
        shard.value[2,4511]=3.5
        result=api._replay_compare_outputs(full,shard,global_start=4,global_stop=9)
        self.assertEqual((result["max_abs"],result["relative"],result["unequal_elements"]),(1.5,.75,1))
        self.assertEqual(result["compared_elements"],5*5376)
        self.assertTrue(result["finite"]);self.assertFalse(result["output_bits_match"])
        shard.value[0,0]=np.nan
        result=api._replay_compare_outputs(full,shard,global_start=4,global_stop=9)
        self.assertFalse(result["finite"]);self.assertIsNone(result["max_abs"]);self.assertIsNone(result["relative"])
        self.assertEqual(result["unequal_elements"],2)
        json.dumps(result,allow_nan=False)

    def test_actual_quantized_identity_excludes_padding_and_binds_scale_and_window(self):
        api=_b0t_load();namespace=api._replay_quantized_identity.__globals__
        namespace["nvfp4_scale_sha256"]=Mock(return_value="7"*64)
        class Data(_HashRawTensor):
            def is_contiguous(self):return self.values.flags.c_contiguous
            def numel(self):return self.values.size
            def reshape(self,*shape):return Data(self.values.reshape(*shape),self.dtype)
        params=SimpleNamespace(orig_shape=(5,16),orig_dtype="torch.bfloat16",transposed=False,
            scale=Data(np.array([2],np.float32),"torch.float32"),block_scale=object())
        data=Data(np.arange(16*8,dtype=np.uint8).reshape(16,8),"torch.uint8")
        qt=SimpleNamespace(_params=params,_qdata=data,_layout_cls="TensorCoreNVFP4Layout")
        kwargs=dict(tensor_global_start=100,windows=[(100,105)])
        first=api._replay_quantized_identity(qt,**kwargs)
        self.assertEqual((first["shape"],first["padded_rows"]),([5,16],16))
        data.values[5:]=0
        self.assertEqual(first,api._replay_quantized_identity(qt,**kwargs))
        data.values[3,2]^=1
        changed=api._replay_quantized_identity(qt,**kwargs)
        self.assertNotEqual(first["windows"][0]["qdata_sha256"],changed["windows"][0]["qdata_sha256"])
        self.assertEqual(first["windows"][0]["scale_sha256"],changed["windows"][0]["scale_sha256"])
        params.transposed=True
        with self.assertRaisesRegex(RuntimeError,"quantized_guard_failed"):api._replay_quantized_identity(qt,**kwargs)

    def test_budget_incomplete_logs_metadata_but_cannot_accept_original_parity(self):
        f=parity_fixture(world2_good=True)
        f.namespace["spf"].projection_replay_summary.return_value={"statuses":[{"status":"budget_insufficient"}],
            "actual":[],"replays":[],"complete":False}
        with self.assertRaisesRegex(RuntimeError,"diagnostic evidence incomplete"):f.forward(*f.args)
        f.namespace["spf"].head_metadata_summary.assert_called_once()
        f.namespace["spf"].layer_hash_summary.assert_called_once()
        f.namespace["spf"].projection_hash_summary.assert_called_once()
        self.assertTrue(any(isinstance(e,tuple) and e[0]=="log" and "[projection_replay_status]" in e[1] for e in f.events))
        self.assertNotIn(("sp",1),f.events)
        self.assertTrue(f.owner.verify_pending);f.owner.destroy.assert_called_once()
        self.assertEqual(f.args[5],{})

"""Appendable CPU lifecycle fixtures; no Torch import, tensors, GPU, or media.

The production ProjectionReplay AST executes unchanged. Only identity hashing and
numeric output comparison are replaced at their interfaces; these tests do not
claim NVFP4 arithmetic, CUDA memory availability, or GPU replay correctness.
"""
import ast as _pr_ast
import hashlib as _pr_hashlib
import json as _pr_json
from pathlib import Path as _PRPath
import threading as _pr_threading
from types import SimpleNamespace as _PRNamespace
import unittest as _pr_unittest


def _pr_source_path():
    here = _PRPath(__file__).resolve()
    core = here.with_name("replay_core.py")
    if core.is_file():
        return core
    return here.parents[1] / "custom_nodes/minimax_sp_pdd/minimax_sp/sp_forward.py"


class _PRFloatRows:
    """Shape-only input view; slicing never allocates element storage."""
    dtype = "torch.bfloat16"
    device = "cuda:0"
    requires_grad = False

    def __init__(self, rows=45167, origin=0, width=7168):
        self.shape, self.origin = (rows, width), origin

    def is_contiguous(self):
        return True

    def __getitem__(self, item):
        if not isinstance(item, slice) or item.step is not None:
            raise AssertionError("fixture only supports a contiguous row window")
        start, stop, _ = item.indices(self.shape[0])
        return type(self)(stop - start, self.origin + start, self.shape[1])


class _PRHarness:
    def __init__(self, *, free_bytes=2**31, fail_call=None, instance_forward=False):
        self.quantizations, self.calls, self.comparisons = [], [], []
        self.identity_calls, self.weight_calls, self.cuda_calls = [], [], []
        self.scale, self.weight, self.bias = object(), object(), object()
        self.fail_call = fail_call
        owner = self

        class QuantizedInput:
            def __init__(self, source, scale):
                self.source = source
                self._layout_cls = "TensorCoreNVFP4Layout"
                self._params = _PRNamespace(scale=scale)

            @classmethod
            def from_float(cls, source, layout, *, scale):
                assert layout == "TensorCoreNVFP4Layout" and type(layout) is str
                owner.quantizations.append((source, layout, scale))
                return cls(source, scale)

        class Module:
            in_features, out_features = 7168, 5376
            quant_format, layout_type = "nvfp4", "TensorCoreNVFP4Layout"
            input_scale = pre_quant_scale = None

            def __init__(self):
                self.hooks, self.removed = [], 0

            def register_forward_pre_hook(self, hook):
                self.hooks.append(hook)
                def remove():
                    self.hooks.remove(hook)
                    self.removed += 1
                return _PRNamespace(remove=remove)

            def _forward(self, qinput, weight, bias):
                owner.calls.append((qinput, weight, bias))
                if len(owner.calls) == owner.fail_call:
                    raise RuntimeError("fixture_forward_failed")
                return _PRNamespace(shape=(qinput.source.shape[0], 5376), call=len(owner.calls))

            def __call__(self, source):
                for hook in tuple(self.hooks):
                    hook(self, (source,))
                qinput = QuantizedInput.from_float(source, self.layout_type, scale=owner.scale)
                return self._forward(qinput, owner.weight, owner.bias)

        self.module = Module()
        if instance_forward:
            bound = self.module._forward
            self.module._forward = lambda q, w, b: bound(q, w, b)
        self.previous = vars(self.module).get("_forward")
        def cuda_call(name, value, device):
            self.cuda_calls.append((name, device))
            return value
        fake_torch = _PRNamespace(equal=lambda a, b: a is b, cuda=_PRNamespace(
            synchronize=lambda device: cuda_call("synchronize", None, device),
            mem_get_info=lambda device: cuda_call("mem_get_info", (free_bytes, 2**35), device),
            memory_reserved=lambda device: cuda_call("memory_reserved", 2**20, device),
            memory_allocated=lambda device: cuda_call("memory_allocated", 2**19, device)))
        path = _pr_source_path()
        selected = {"ProjectionReplay", "ProjectionReplayError", "_replay_require", "_replay_digest",
                    "_replay_quantized_identity", "_replay_weight_bias", "_replay_compare_outputs"}
        tree = _pr_ast.parse(path.read_text(encoding="utf-8"))
        nodes = [node for node in tree.body if isinstance(node, (_pr_ast.ClassDef, _pr_ast.FunctionDef))
                 and node.name in selected]
        assert {node.name for node in nodes} == selected
        ns = {"hashlib": _pr_hashlib, "json": _pr_json, "threading": _pr_threading,
              "torch": fake_torch, "_ABSENT": object(), "REPLAY_MINIMUM_FREE_BYTES": 2**30}
        exec(compile(_pr_ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
        def identity(qinput, *, tensor_global_start, windows):
            self.identity_calls.append((qinput, tensor_global_start, tuple(windows)))
            assert qinput.source.origin == tensor_global_start
            def digest(start, stop):
                value = _pr_hashlib.sha256(f"{start}:{stop}:7168".encode()).hexdigest()
                return {"qdata_sha256": value, "block_scale_sha256": value, "scale_sha256": "a" * 64}
            return {"shape": list(qinput.source.shape),
                    "padded_rows": (qinput.source.shape[0] + 15) // 16 * 16,
                    "dtype": "torch.bfloat16", "layout": "TensorCoreNVFP4Layout",
                    "windows": [digest(start, stop) for start, stop in windows]}
        def weight_bias(weight, bias):
            assert weight is self.weight and bias is self.bias
            self.weight_calls.append((weight, bias))
            return "b" * 64, "c" * 64
        def compare(full, shard, *, global_start, global_stop):
            assert full.shape == (45167, 5376) and shard.shape == (global_stop - global_start, 5376)
            self.comparisons.append((full, shard, global_start, global_stop))
            return {"native_output_sha256": "d" * 64, "replay_output_sha256": "d" * 64,
                    "output_bits_match": True, "finite": True, "max_abs": 0.0,
                    "relative": 0.0, "unequal_elements": 0,
                    "compared_elements": (global_stop - global_start) * 5376}
        ns.update(_replay_quantized_identity=identity, _replay_weight_bias=weight_bias,
                  _replay_compare_outputs=compare)
        self.scope = ns["ProjectionReplay"](self.module, rank=-1, start=0, stop=45167,
                                            windows=[(0, 22584), (22584, 45167)])


class ProjectionReplayLifecycleTests(_pr_unittest.TestCase):
    def assert_restored(self, harness):
        self.assertEqual(harness.module.hooks, [])
        self.assertFalse(harness.scope.installed)
        self.assertIsNone(harness.scope.input)
        if harness.previous is None:
            self.assertNotIn("_forward", vars(harness.module))
        else:
            self.assertIs(harness.module._forward, harness.previous)

    def test_native_once_then_two_windows_reuse_actual_weight_bias_scale(self):
        h = _PRHarness()
        with h.scope:
            result = h.module(_PRFloatRows())
        self.assertEqual(result.call, 1)
        self.assertEqual([call[0].source.shape[0] for call in h.calls], [45167, 22584, 22583])
        self.assertEqual([call[0].source.origin for call in h.calls], [0, 0, 22584])
        self.assertTrue(all(w is h.weight and b is h.bias for _, w, b in h.calls))
        self.assertTrue(all(scale is h.scale for _, _, scale in h.quantizations))
        self.assertEqual(len(h.quantizations), 3)
        self.assertEqual([(x[2], x[3]) for x in h.comparisons], [(0, 22584), (22584, 45167)])
        self.assertEqual(h.scope.packet()["status"]["call_count"], 1)
        self.assertEqual(h.scope.packet()["status"]["status"], "complete")
        self.assert_restored(h)

    def test_insufficient_free_memory_preserves_native_capture_without_replay(self):
        h = _PRHarness(free_bytes=2**30 - 1)
        with h.scope:
            result = h.module(_PRFloatRows())
        packet = h.scope.packet()
        self.assertEqual(result.call, 1)
        self.assertEqual(len(h.calls), 1)
        self.assertEqual(len(h.quantizations), 1)
        self.assertEqual(h.comparisons, [])
        self.assertIsNotNone(packet["capture"])
        self.assertEqual(packet["status"]["status"], "budget_insufficient")
        self.assertEqual(packet["status"]["error_code"], "memory_budget_insufficient")
        self.assertEqual(packet["status"]["free_bytes"], 2**30 - 1)
        self.assert_restored(h)

    def test_second_shard_failure_restores_prior_instance_forward(self):
        h = _PRHarness(fail_call=3, instance_forward=True)
        with h.scope:
            result = h.module(_PRFloatRows())
        packet = h.scope.packet()
        self.assertEqual(result.call, 1)
        self.assertEqual(len(h.calls), 3)
        self.assertEqual(len(packet["replays"]), 1)
        self.assertEqual(packet["status"]["status"], "error")
        self.assertEqual(packet["status"]["error_code"], "replay_execution_failed")
        self.assert_restored(h)

    def test_nondefault_input_scaling_guards_install_no_hook(self):
        for attribute, value in (("pre_quant_scale", object()), ("input_scale", object()),
                                 ("input_scale", 1.0)):
            with self.subTest(attribute=attribute, kind=type(value).__name__):
                h = _PRHarness()
                setattr(h.module, attribute, value)
                with h.scope:
                    self.assertFalse(h.scope.installed)
                    self.assertEqual(h.module.hooks, [])
                self.assertEqual(h.module.removed, 0)
                self.assertEqual(h.calls, [])
                self.assertEqual(h.scope.packet()["status"]["error_code"], "input_guard_failed")
                self.assert_restored(h)

    def test_original_forward_exception_propagates_and_cleans_hook(self):
        h = _PRHarness(fail_call=1)
        with self.assertRaisesRegex(RuntimeError, "fixture_forward_failed"):
            with h.scope:
                h.module(_PRFloatRows())
        self.assertEqual(len(h.calls), 1)
        self.assertEqual(h.identity_calls, [])
        self.assertEqual(h.cuda_calls, [])
        self.assertEqual(h.module.removed, 1)
        self.assertEqual(h.scope.packet()["status"]["status"], "error")
        self.assert_restored(h)

    def test_packet_is_json_metadata_and_never_claims_backend_verified(self):
        h = _PRHarness()
        with h.scope:
            h.module(_PRFloatRows())
        packet = h.scope.packet()
        decoded = _pr_json.loads(_pr_json.dumps(packet, allow_nan=False))
        self.assertEqual(decoded, packet)
        self.assertFalse(packet["status"]["backend_verified"])
        self.assertTrue(all(row["backend_verified"] is False for row in packet["replays"]))
        self.assertTrue(all(row["input_match"] is True for row in packet["replays"]))
        self.assertEqual([row["padded_rows"] for row in packet["replays"]], [22592, 22592])
        self.assertEqual([row["full_padded_rows"] for row in packet["replays"]], [45168, 45168])
        self.assert_restored(h)



if __name__ == "__main__":
    unittest.main()
