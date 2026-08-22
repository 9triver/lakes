from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from lake_workbench.jobs import TrainingManager, _release_cuda_memory


class TrainingManagerTests(unittest.TestCase):
    def test_dataset_source_is_forwarded_to_summary(self) -> None:
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            manager = TrainingManager(
                "all",
                workspace_id="workspace",
                model_root=Path(directory),
                dataset_summary=lambda *args: calls.append(args) or {},
                runner=lambda *args, **kwargs: {},
                persisted_job_loader=lambda *args: None,
                parse_epochs=lambda value, default: default,
            )

            job = manager.create({"dataset_config_id": "resize256_v1", "dataset_source": "global"})

            self.assertEqual(calls[0], ("all", "resize256_v1", "global"))
            manager.cancel(job["job_id"])

    def test_cuda_memory_is_released_after_failed_training(self) -> None:
        def fail(*args, **kwargs):
            raise RuntimeError("training failed")

        with tempfile.TemporaryDirectory() as directory:
            manager = TrainingManager(
                "all",
                model_root=Path(directory),
                dataset_summary=lambda *args: {},
                runner=fail,
                persisted_job_loader=lambda *args: None,
                parse_epochs=lambda value, default: default,
            )
            manager.jobs["job"] = {"options": {}}
            manager.cancel_events["job"] = threading.Event()

            with patch("lake_workbench.jobs._release_cuda_memory") as release:
                manager._run("job")

            release.assert_called_once_with()
            self.assertEqual(manager.jobs["job"]["status"], "failed")
            self.assertNotIn("job", manager.cancel_events)

    def test_release_cuda_memory_collects_before_emptying_cache(self) -> None:
        calls = []
        with (
            patch("lake_workbench.jobs.gc.collect", side_effect=lambda: calls.append("gc")),
            patch("torch.cuda.is_initialized", return_value=True),
            patch("torch.cuda.empty_cache", side_effect=lambda: calls.append("cuda")),
        ):
            _release_cuda_memory()

        self.assertEqual(calls, ["gc", "cuda"])


if __name__ == "__main__":
    unittest.main()
