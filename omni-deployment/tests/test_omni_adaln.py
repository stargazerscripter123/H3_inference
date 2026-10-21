"""Regression checks for the checkpoint's FP32 key / BF16 projection contract."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_omni_adaln import inspect_table


class Tensor:
    ndim = 2
    dtype = "torch.bfloat16"
    shape = (2, 96768)
    finite = True

    def is_floating_point(self):
        return True


class Finite:
    def __init__(self, value):
        self.value = value

    def all(self):
        return self

    def item(self):
        return self.value


class Torch:
    def __init__(self, table):
        self.table = table

    def load(self, path, **kwargs):
        return self.table

    def is_tensor(self, value):
        return isinstance(value, Tensor)

    def isfinite(self, value):
        return Finite(value.finite)


class AdaLNContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "table.pt"
        self.path.write_bytes(b"mock table for checksum only")
        self.table = {block: {repr(((float(i), 0.0), (2, 5376), "torch.float32")): Tensor()
                             for i in range(1, 9)} for block in range(50)}

    def test_fp32_key_with_bf16_modulation_is_valid(self):
        result = inspect_table(self.path, Torch(self.table))
        self.assertEqual(result["tensors"], 400)
        self.assertEqual(result["clock_entries"][0]["embedding_dtype"], "torch.float32")
        self.assertEqual(result["clock_entries"][0]["modulation_dtype"], "torch.bfloat16")

    def test_missing_block_is_rejected(self):
        del self.table[49]
        with self.assertRaisesRegex(ValueError, "blocks 0 through 49"):
            inspect_table(self.path, Torch(self.table))

    def test_inconsistent_clock_keys_are_rejected(self):
        block = self.table[49]
        key = next(iter(block))
        block[repr(((99.0, 0.0), (2, 5376), "torch.float32"))] = block.pop(key)
        with self.assertRaisesRegex(ValueError, "different clock entries"):
            inspect_table(self.path, Torch(self.table))

    def test_wrong_projection_width_is_rejected(self):
        next(iter(self.table[0].values())).shape = (2, 5376)
        with self.assertRaisesRegex(ValueError, "tensor shape"):
            inspect_table(self.path, Torch(self.table))

    def test_nonfinite_projection_is_rejected(self):
        next(iter(self.table[0].values())).finite = False
        with self.assertRaisesRegex(ValueError, "non-finite modulation"):
            inspect_table(self.path, Torch(self.table))


if __name__ == "__main__":
    unittest.main()
