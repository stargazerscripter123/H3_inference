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
    source = reverse_world1_branches(AFTER.read_text())
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
        after = tree(reverse_world1_branches(AFTER.read_text()))
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



if __name__ == "__main__":
    unittest.main()
