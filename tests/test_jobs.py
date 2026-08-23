from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from lake_workbench.jobs import TrainingManager, TrainingRunActiveError, _release_cuda_memory


class TrainingManagerTests(unittest.TestCase):
    def test_auto_run_name_is_recorded_on_job_and_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "lake_workbench.jobs.time.strftime", return_value="20260823_162605"
        ):
            manager = TrainingManager(
                "all",
                workspace_id="workspace",
                model_root=Path(directory),
                dataset_summary=lambda *args: {},
                runner=lambda *args, **kwargs: {},
                persisted_job_loader=lambda *args: None,
                parse_epochs=lambda value, default: default,
            )

            job = manager.create({"model_type": "unet"})

            self.assertEqual(job["run_name"], "unet_20260823_162605")
            self.assertEqual(job["options"]["run_name"], job["run_name"])
            manager.cancel(job["job_id"])

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

    def test_delete_removes_completed_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = TrainingManager(
                "all",
                workspace_id="workspace",
                model_root=root,
                dataset_summary=lambda *args: {},
                runner=lambda *args, **kwargs: {},
                persisted_job_loader=lambda *args: None,
                parse_epochs=lambda value, default: default,
            )
            run_dir = root / "all" / "finished_run"
            run_dir.mkdir(parents=True)
            (run_dir / "best.pt").write_bytes(b"weights")
            manager.jobs["finished"] = {
                "job_id": "finished",
                "run_name": "finished_run",
                "status": "completed",
            }

            result = manager.delete("finished")

            self.assertEqual(result, {"job_id": "finished", "run_name": "finished_run", "deleted": True})
            self.assertFalse(run_dir.exists())
            self.assertIsNone(manager.get("finished"))

    def test_delete_rejects_active_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = TrainingManager(
                "all",
                model_root=Path(directory),
                dataset_summary=lambda *args: {},
                runner=lambda *args, **kwargs: {},
                persisted_job_loader=lambda *args: None,
                parse_epochs=lambda value, default: default,
            )
            manager.jobs["running"] = {
                "job_id": "running",
                "run_name": "running_run",
                "status": "running",
            }

            with self.assertRaises(TrainingRunActiveError):
                manager.delete("running")

    def test_delete_rejects_run_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = TrainingManager(
                "all",
                model_root=Path(directory),
                dataset_summary=lambda *args: {},
                runner=lambda *args, **kwargs: {},
                persisted_job_loader=lambda *args: None,
                parse_epochs=lambda value, default: default,
            )
            manager.jobs["unsafe"] = {
                "job_id": "unsafe",
                "run_name": "../outside",
                "status": "failed",
            }

            with self.assertRaises(ValueError):
                manager.delete("unsafe")

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

    def test_system_exit_from_training_is_recorded_as_failed(self) -> None:
        def stop_before_training(*args, **kwargs):
            raise SystemExit("no included training patches found")

        with tempfile.TemporaryDirectory() as directory:
            manager = TrainingManager(
                "all",
                model_root=Path(directory),
                dataset_summary=lambda *args: {},
                runner=stop_before_training,
                persisted_job_loader=lambda *args: None,
                parse_epochs=lambda value, default: default,
            )
            job = manager.create({})
            job_id = job["job_id"]
            manager._run(job_id)

            self.assertEqual(manager.jobs[job_id]["status"], "failed")
            self.assertEqual(manager.jobs[job_id]["message"], "no included training patches found")

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
