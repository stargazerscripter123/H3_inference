"""Patch identity failure cases, using temporary text files only."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/omni_reference_patch.py"
SPEC = importlib.util.spec_from_file_location("omni_reference_patch_under_test", SCRIPT)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


class ReferencePatchIdentityTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="omni-patch-identity-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.patch = self.root / "change.patch"
        self.patch.write_text("fixture patch text\n")
        self.manifest = self.root / "manifest.json"
        self.metadata = {
            "schema_version": 1, "upstream_commit": "pinned-test-commit",
            "patch_sha256": helper.sha256(self.patch),
            "files": {name: {
                "before_sha256": hashlib.sha256(b"before\n").hexdigest(),
                "after_sha256": hashlib.sha256(b"after\n").hexdigest(),
            } for name in ("one.py", "two.py")},
        }
        self.manifest.write_text(json.dumps(self.metadata))
        for name in self.metadata["files"]:
            (self.source / name).write_text("before\n")
        for target, value in (("MANIFEST", self.manifest), ("PATCH", self.patch)):
            manager = patch.object(helper, target, value)
            manager.start()
            self.addCleanup(manager.stop)
        manager = patch.object(helper.subprocess, "check_output", return_value="pinned-test-commit\n")
        self.git_head = manager.start()
        self.addCleanup(manager.stop)

    def test_original_allowed_but_non_native_requires_patched_source(self):
        self.assertEqual(helper.inspect_patch(self.source)["state"], "native")
        with self.assertRaisesRegex(RuntimeError, "explicit"):
            helper.inspect_patch(self.source, required=True)

    def test_exact_patched_source_and_identity_accepted(self):
        for name in self.metadata["files"]:
            (self.source / name).write_text("after\n")
        state = helper.inspect_patch(self.source, required=True)
        self.assertEqual(state["state"], "patched")
        self.assertEqual(state["patch_sha256"], self.metadata["patch_sha256"])
        self.assertEqual(state["manifest_sha256"], helper.sha256(self.manifest))
        self.assertEqual(state["files"], {name: self.metadata["files"][name]["after_sha256"]
                                         for name in self.metadata["files"]})

    def test_partially_patched_source_rejected(self):
        (self.source / "one.py").write_text("after\n")
        with self.assertRaisesRegex(RuntimeError, "partially patched"):
            helper.inspect_patch(self.source)

    def test_unknown_source_hash_rejected_without_overwrite(self):
        target = self.source / "one.py"
        target.write_text("user changes\n")
        with self.assertRaisesRegex(RuntimeError, "modified"):
            helper.inspect_patch(self.source)
        self.assertEqual(target.read_text(), "user changes\n")

    def test_changed_patch_rejected_before_git_or_source_checks(self):
        self.patch.write_text("tampered fixture patch\n")
        with self.assertRaisesRegex(RuntimeError, "manifest/hash mismatch"):
            helper.inspect_patch(self.source)
        self.git_head.assert_not_called()

    def test_wrong_upstream_commit_rejected(self):
        self.git_head.return_value = "another-commit\n"
        with self.assertRaisesRegex(RuntimeError, "pinned FastVideo commit"):
            helper.inspect_patch(self.source)


if __name__ == "__main__":
    unittest.main()
