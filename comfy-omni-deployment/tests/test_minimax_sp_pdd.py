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


def reverse_amax_scope(source):
    """Reverse only reviewed exact imports/scope/loop; retain original upstream SHA."""
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
        return sp2

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
        'spf':SimpleNamespace(validate_transformer_options=Mock(), sp_forward=dispatch),
        'build_meta':lambda *args:{}, 'tensor_meta':lambda value:None,
        'forward_status':lambda meta:{}, 'validate_forward_status':lambda *args:None,
        'ordered_tensors':lambda *args:[],
        'comfy':SimpleNamespace(model_management=SimpleNamespace(InterruptProcessingException=type('Interrupt', (Exception,), {})))}
    parity, forward = parity_functions(ns)
    args = (owner, Dit(), [ParityArray([0]), ParityArray([0])], None, None, {}, {}, None, None)
    return SimpleNamespace(parity=parity, forward=forward, args=args, owner=owner,
        native=native, events=events, dist=dist, log=log)


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
                    'torch':object(),'dist':object(),'rank':0,'prefetch_queue':None,'device':'cpu',
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


if __name__ == "__main__":
    unittest.main()
