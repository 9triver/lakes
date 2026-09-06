import unittest

from lake_workbench.workspaces.membership import (
    find_patch_conflicts,
    included_members,
)


class WorkspaceMembershipTests(unittest.TestCase):
    def test_review_status_takes_precedence_over_legacy_include_flag(self) -> None:
        rows = [
            {"logical_patch_id": "included", "include": "false", "review_status": "included"},
            {"logical_patch_id": "excluded", "include": "true", "review_status": "excluded"},
        ]

        self.assertEqual(included_members(rows, "gansu"), {("gansu", "included")})

    def test_conflict_requires_same_image_and_spatial_overlap(self) -> None:
        canonical = {
            "existing": {
                "image_fingerprint": "image-a",
                "bounds_left": "0",
                "bounds_bottom": "0",
                "bounds_right": "2",
                "bounds_top": "2",
            },
            "incoming": {
                "image_fingerprint": "image-a",
                "bounds_left": "0",
                "bounds_bottom": "0",
                "bounds_right": "2",
                "bounds_top": "2",
            },
            "different-image": {
                "image_fingerprint": "image-b",
                "bounds_left": "0",
                "bounds_bottom": "0",
                "bounds_right": "2",
                "bounds_top": "2",
            },
        }

        conflicts = find_patch_conflicts(canonical, ["existing"], ["incoming", "different-image"])

        self.assertEqual([item["patch_id"] for item in conflicts], ["incoming"])

if __name__ == "__main__":
    unittest.main()
