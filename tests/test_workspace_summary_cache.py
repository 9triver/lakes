from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from lake_workbench.workspaces import WorkspaceStore
from lake_workbench.workspaces import store as store_module


class WorkspaceSummaryCacheTests(unittest.TestCase):
    def setUp(self):
        self.store = WorkspaceStore({"chongqing": SimpleNamespace()}, root=Path("unused"))
        self.manifest = Mock()
        self.modified = 1
        self.manifest.stat.side_effect = lambda: SimpleNamespace(st_mtime_ns=self.modified, st_size=100)
        self.manifest.exists.return_value = True
        self.store.workspace_logical_patch_manifest = Mock(return_value=self.manifest)
        self.store._workspace_record = Mock(return_value={"id": "default"})
        self.rows = [
            {"logical_patch_id": "one", "site_id": "site1", "include": "true"},
            {"logical_patch_id": "two", "site_id": "site2", "include": "false"},
        ]

    def test_repeated_summary_and_members_do_not_reread_large_csv(self):
        with patch.object(store_module, "read_csv_records", return_value=self.rows) as reader:
            for _ in range(3):
                summary = self.store._summary({"id": "default"})
                self.assertEqual(summary["selected_patch_count"], 1)
                self.assertEqual(summary["site_count"], 1)
                self.assertEqual(self.store.members("default"), {("chongqing", "one")})
        reader.assert_called_once_with(self.manifest)

    def test_external_export_and_review_invalidate_cached_counts(self):
        with patch.object(store_module, "read_csv_records", side_effect=lambda _: [dict(row) for row in self.rows]) as reader:
            self.store._summary({"id": "default"})
            self.rows[1]["include"] = "true"
            self.modified += 1
            summary = self.store._summary({"id": "default"})
            self.assertEqual(summary["selected_patch_count"], 2)
            self.assertEqual(summary["site_count"], 2)
            self.rows[0]["review_status"] = "excluded"
            self.modified += 1
            summary = self.store._summary({"id": "default"})
            self.assertEqual(summary["selected_patch_count"], 1)
            self.assertEqual(summary["site_count"], 1)
        self.assertEqual(reader.call_count, 3)


if __name__ == "__main__":
    unittest.main()
