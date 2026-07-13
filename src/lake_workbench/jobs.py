"""Background job managers for downloads, patch export, and model training."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lake_workbench.sentinel.download import download_copernicus_product


JobRunner = Callable[..., dict]


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class DownloadManager:
    def __init__(self, catalog: Any) -> None:
        self.catalog = catalog
        self._lock = threading.Lock()
        self.jobs: dict[str, dict] = {}

    def create(self, product: dict) -> dict:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "job_id": job_id,
            "status": "queued",
            "message": "排队中",
            "progress": 0,
            "downloaded_bytes": 0,
            "total_bytes": int(product.get("content_length") or 0),
            "product": product,
            "created_at": _timestamp(),
            "updated_at": _timestamp(),
        }
        with self._lock:
            self.jobs[job_id] = job
        threading.Thread(target=self._run, args=(job_id,), daemon=True).start()
        return dict(job)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    def _update(self, job_id: str, **updates) -> None:
        with self._lock:
            job = self.jobs[job_id]
            job.update(updates)
            job["updated_at"] = _timestamp()

    def _run(self, job_id: str) -> None:
        product = self.get(job_id)["product"]
        try:
            self._update(job_id, status="authenticating", message="连接 Copernicus")

            def progress(done: int, total: int) -> None:
                pct = int(done * 100 / total) if total else 0
                self._update(
                    job_id,
                    status="downloading",
                    message=f"下载中 {pct}%",
                    progress=pct,
                    downloaded_bytes=done,
                    total_bytes=total,
                )

            safe_dir, tci_path = download_copernicus_product(
                product,
                self.catalog.region.sentinel_download_dir,
                progress=progress,
            )
            self._update(job_id, status="indexing", message="登记本地影像", progress=100)
            row = self.catalog.register_downloaded_product(product, safe_dir, tci_path)
            self._update(job_id, status="completed", message="下载完成", progress=100, result=row)
        except Exception as exc:  # noqa: BLE001 - surfaced to the local UI.
            self._update(job_id, status="failed", message=f"{type(exc).__name__}: {exc}")


class PatchExportManager:
    def __init__(
        self,
        catalog: Any | None = None,
        catalogs: dict[str, Any] | None = None,
        *,
        exporter: Callable[[str, dict], dict],
    ) -> None:
        self.catalog = catalog
        self.catalogs = catalogs or {}
        self.exporter = exporter
        self._lock = threading.Lock()
        self.jobs: dict[str, dict] = {}

    def create(self, options: dict) -> dict:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "job_id": job_id,
            "status": "queued",
            "message": "排队中",
            "progress": 0,
            "created_at": _timestamp(),
            "updated_at": _timestamp(),
            "options": options,
        }
        with self._lock:
            self.jobs[job_id] = job
        threading.Thread(target=self._run, args=(job_id,), daemon=True).start()
        return dict(job)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    def _update(self, job_id: str, **updates) -> None:
        with self._lock:
            job = self.jobs[job_id]
            job.update(updates)
            job["updated_at"] = _timestamp()

    def _run(self, job_id: str) -> None:
        options = self.get(job_id)["options"]
        try:
            self._update(job_id, status="running", message="生成 patch 中", progress=10)
            if self.catalogs:
                results = []
                total = max(1, len(self.catalogs))
                for index, region_key in enumerate(self.catalogs, start=1):
                    self._update(job_id, message=f"生成 {region_key} patch 中", progress=max(10, int(index * 80 / total)))
                    results.append(self.exporter(region_key, options))
                result = {
                    "region": "all",
                    "regions": results,
                    "samples": sum(item.get("samples", 0) for item in results),
                    "patches": sum(item.get("patches", 0) for item in results),
                }
            else:
                result = self.exporter(self.catalog.region.key, options)
            self._update(
                job_id,
                status="completed",
                message=f"生成完成：{result.get('patches', 0)} 个 patch",
                progress=100,
                result=result,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the local UI.
            self._update(job_id, status="failed", message=f"{type(exc).__name__}: {exc}")


class TrainingManager:
    def __init__(
        self,
        scope: str,
        *,
        model_root: Path,
        dataset_summary: Callable[[str], dict],
        runner: Callable[..., dict],
        persisted_job_loader: Callable[[str, Path], dict | None],
        parse_epochs: Callable[[Any, int], int],
    ) -> None:
        self.scope = scope
        self.model_root = model_root
        self.dataset_summary = dataset_summary
        self.runner = runner
        self.persisted_job_loader = persisted_job_loader
        self.parse_epochs = parse_epochs
        self._lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.cancel_events: dict[str, threading.Event] = {}
        self._load_persisted_jobs()

    @property
    def model_dir(self) -> Path:
        return self.model_root / self.scope

    def create(self, options: dict) -> dict:
        job_id = uuid.uuid4().hex[:12]
        cancel_event = threading.Event()
        job = {
            "job_id": job_id,
            "scope": self.scope,
            "status": "queued",
            "message": "排队中",
            "progress": 0,
            "epoch": 0,
            "epochs": self.parse_epochs(options.get("epochs"), 30),
            "history": [],
            "options": options,
            "dataset": self.dataset_summary(self.scope),
            "created_at": _timestamp(),
            "updated_at": _timestamp(),
        }
        with self._lock:
            self.jobs[job_id] = job
            self.cancel_events[job_id] = cancel_event
        threading.Thread(target=self._run, args=(job_id,), daemon=True).start()
        return dict(job)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    def list(self) -> dict:
        with self._lock:
            jobs = [dict(job) for job in self.jobs.values()]
        jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return {"scope": self.scope, "dataset": self.dataset_summary(self.scope), "items": jobs}

    def cancel(self, job_id: str) -> dict | None:
        with self._lock:
            event = self.cancel_events.get(job_id)
            job = self.jobs.get(job_id)
            if job is None:
                return None
            if event is not None:
                event.set()
            if job.get("status") in {"queued", "running", "configured"}:
                job.update(status="cancel_requested", message="正在取消", updated_at=_timestamp())
            return dict(job)

    def _update(self, job_id: str, **updates) -> None:
        with self._lock:
            job = self.jobs[job_id]
            job.update(updates)
            job["updated_at"] = _timestamp()

    def _progress(self, job_id: str, payload: dict) -> None:
        updates = {
            "status": payload.get("status", "running"),
            "message": payload.get("message", ""),
            "progress": payload.get("progress", 0),
        }
        for key in ("epoch", "epochs", "record", "best_iou", "output_dir", "config", "result"):
            if key in payload:
                updates[key] = payload[key]
        if "record" in payload:
            current = self.get(job_id) or {}
            history = list(current.get("history") or [])
            history.append(payload["record"])
            updates["history"] = history
        self._update(job_id, **updates)

    def _run(self, job_id: str) -> None:
        job = self.get(job_id)
        options = job["options"]
        cancel_event = self.cancel_events[job_id]
        try:
            self._update(job_id, status="running", message="准备训练数据", progress=2)
            result = self.runner(
                self.scope,
                options,
                progress_callback=lambda payload: self._progress(job_id, payload),
                cancel_event=cancel_event,
            )
            if result.get("status") == "cancelled":
                self._update(job_id, status="cancelled", message="训练已取消", progress=100, result=result)
            else:
                self._update(job_id, status="completed", message="训练完成", progress=100, result=result)
        except Exception as exc:  # noqa: BLE001 - surfaced to the local UI.
            self._update(job_id, status="failed", message=f"{type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                self.cancel_events.pop(job_id, None)

    def _load_persisted_jobs(self) -> None:
        if not self.model_dir.exists():
            return
        paths = sorted(self.model_dir.glob("*/config.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        for config_path in paths:
            job = self.persisted_job_loader(self.scope, config_path.parent)
            if job:
                self.jobs[job["job_id"]] = job
