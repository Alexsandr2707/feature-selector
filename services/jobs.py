"""One subprocess at a time with reserved inputs and restart-visible job state."""

import logging
import secrets
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from fastapi import HTTPException

from services.catalog import Catalog
from services.common import StateFiles, atomic_json, identifier, now, read_json

logger = logging.getLogger(__name__)


class Jobs:
    """Persistent executor whose unresolved publications retain their input reservations."""

    def __init__(self, settings: dict[str, Any]) -> None:
        """Recover previous jobs without treating a Storage outage as missing results.

        Parameters:
            settings (dict[str, Any]): State directory and Storage address.

        Returns:
            None.
        """
        self.settings = settings
        self.states = StateFiles(Path(settings["state_dir"]) / "jobs")
        self.work = Path(settings["state_dir"]) / "work"
        self.work.mkdir(parents=True, exist_ok=True)

        self.lock = threading.RLock()
        self.process: subprocess.Popen[bytes] | None = None
        self.thread: threading.Thread | None = None
        self.active: str | None = None
        self.closing = False
        self.catalog = Catalog(settings["storage_url"], min(settings["timeout"], 3))

        self._recover()

    def _reconcile(self, job: dict[str, Any], final_status: str, error: str) -> None:
        """Resolve publication before declaring failure or releasing protected inputs.

        Parameters:
            job (dict[str, Any]): Persisted job whose worker has stopped.
            final_status (str): failed or interrupted if no published result exists.
            error (str): Original execution error, retained during Storage outages.

        Returns:
            None: Saves completed, terminal failure or retryable reconciling state.
        """
        key, kind = job["id"], job["result_kind"]
        try:
            try:
                self.catalog.get(kind, key)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 404:
                    raise

                request_path = self.work / key / "request.json"
                request = read_json(request_path) if request_path.exists() else {}

                try:
                    self.catalog.discard(kind, key, request.get("reservation_token"))
                except httpx.HTTPStatusError as cleanup:
                    if cleanup.response.status_code == 409:
                        # A response can be lost after the server commits publication.
                        self.catalog.get(kind, key)
                    elif cleanup.response.status_code != 404:
                        raise
                    else:
                        job.pop("pending_status", None)
                        job.pop("reconciliation_error", None)
                        job.update(status=final_status, error=error, finished_at=now())
                        self.states.put(job)
                        return
                else:
                    job.pop("pending_status", None)
                    job.pop("reconciliation_error", None)
                    job.update(status=final_status, error=error, finished_at=now())
                    self.states.put(job)
                    return

            job.update(status="completed", result_id=key, error=None, finished_at=now())
            job.pop("pending_status", None)
            job.pop("reconciliation_error", None)
        except Exception as exc:
            # Keep capabilities and dependency protection until Storage can be checked again.
            logger.exception("Could not reconcile job %s", key)
            job.update(
                status="reconciling",
                pending_status=final_status,
                error=error,
                reconciliation_error=str(exc),
            )

        self.states.put(job)

    def _recover(self) -> None:
        """Retry orphaned and unresolved jobs while no child belongs to this controller.

        Returns:
            None: Keeps the single executor occupied until reconciliation succeeds.
        """
        with self.lock:
            if self.process is not None:
                return

            self.active = None
            for job in self.states.list():
                if job["status"] not in {"running", "reconciling"}:
                    continue

                self._reconcile(
                    job,
                    job.get("pending_status", "interrupted"),
                    job.get("error") or "Executor restarted before completion",
                )
                if job["status"] == "reconciling":
                    self.active = job["id"]

    def _reserve(self, key: str, action: str, payload: dict[str, Any], token: str) -> None:
        """Protect published inputs by creating the result draft before child startup.

        Parameters:
            key (str): New job and result ID.
            action (str): collect, train or predict.
            payload (dict[str, Any]): Validated submission data.
            token (str): Private reservation capability.

        Returns:
            None: Reserves all referenced inputs atomically in Storage.
        """
        if action == "collect":
            kind, name, references = "datasets", payload["name"], {}
        elif action == "train":
            model = payload["model"]
            kind, name = "runs", model["name"]
            references = {"dataset_id": model["references"]["dataset_id"], "model_id": model["id"]}
        elif action == "predict":
            run = self.catalog.get("runs", payload["run_id"])
            kind, name = "predictions", run["name"]
            references = {"dataset_id": run["references"]["dataset_id"], "run_id": run["id"]}
        else:
            raise ValueError("Unknown job action")

        self.catalog.reserve(kind, key, name, references, token)

    def start(self, action: str, payload: dict[str, Any], result_kind: str) -> dict[str, Any]:
        """Reserve inputs and launch one child independently of the HTTP request lifetime.

        Parameters:
            action (str): collect, train or predict.
            payload (dict[str, Any]): JSON execution arguments.
            result_kind (str): Storage collection for the completed result.

        Returns:
            dict[str, Any]: Running job; running or unresolved submissions cause HTTP 409.
        """
        with self.lock:
            self._recover()
            if self.closing or self.active:
                raise HTTPException(409, {"code": "busy", "active_job_id": self.active})

            key = uuid4().hex
            directory = (self.work / key).resolve()
            directory.mkdir()
            token = secrets.token_urlsafe(32)
            job = {
                "id": key,
                "action": action,
                "status": "running",
                "created_at": now(),
                "result_kind": result_kind,
                "result_id": None,
                "error": None,
            }

            atomic_json(
                directory / "request.json",
                {
                    "job_id": key,
                    "action": action,
                    "payload": payload,
                    "settings": self.settings,
                    "directory": str(directory),
                    "reservation_token": token,
                },
            )
            self.states.put(job)
            self.active = key

            try:
                self._reserve(key, action, payload, token)
                with (directory / "worker.log").open("wb") as log:
                    self.process = subprocess.Popen(
                        [sys.executable, "-m", "services.worker", str(directory / "request.json")],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
            except Exception as exc:
                self._reconcile(job, "failed", str(exc))
                self.active = key if job["status"] == "reconciling" else None
                raise

            self.thread = threading.Thread(target=self._monitor, args=(key,), daemon=True)
            self.thread.start()
            return job

    def _monitor(self, key: str) -> None:
        """Reconcile the worker outcome even if its result file is missing or malformed.

        Parameters:
            key (str): Active job ID.

        Returns:
            None: Releases the process slot, retaining it only for unresolved publication.
        """
        process = self.process
        if process is None:
            return

        code = process.wait()
        with self.lock:
            job = self.states.get(key)
            try:
                outcome = self.work / key / "outcome.json"
                result = read_json(outcome) if outcome.exists() else {}
                if not isinstance(result, dict):
                    raise TypeError("Worker outcome must be a JSON object")
                error = result.get("error", f"Worker exited with code {code}; see worker log")
            except (OSError, ValueError, TypeError) as exc:
                error = f"Cannot read worker outcome: {exc}"

            try:
                self._reconcile(job, "interrupted" if self.closing else "failed", error)
            finally:
                self.process = None
                self.active = key if job["status"] in {"running", "reconciling"} else None

    def list(self) -> list[dict[str, Any]]:
        """List jobs after retrying pending publication checks.

        Returns:
            list[dict[str, Any]]: Public job states without reservation capabilities.
        """
        self._recover()
        return self.states.list()

    def get(self, key: str) -> dict[str, Any]:
        """Read status and the latest execution stage, retrying unresolved publication.

        Parameters:
            key (str): Job ID.

        Returns:
            dict[str, Any]: Public state with progress details if available.
        """
        self._recover()

        value = self.states.get(identifier(key))
        progress = self.work / key / "progress.json"
        value["progress"] = read_json(progress) if progress.exists() else {}
        return value

    def events(self, key: str) -> dict[str, Any]:
        """Read a bounded tail of worker logs and numerical JSONL events.

        Parameters:
            key (str): Existing job ID.

        Returns:
            dict[str, Any]: Recent lines, tolerating a partially written event.
        """
        self.states.get(key)
        directory = self.work / identifier(key)
        lines: list[str] = []

        for path in [directory / "worker.log", *directory.glob("artifacts/*/*/events.jsonl")]:
            if not path.exists():
                continue
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 20000))
                lines.extend(stream.read().decode("utf-8", errors="replace").splitlines())

        return {"lines": lines[-100:]}

    def close(self) -> None:
        """Terminate the active child before the service exits.

        Returns:
            None: Waits briefly, then kills an unresponsive child.
        """
        with self.lock:
            self.closing = True
            process = self.process

        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        if self.thread:
            self.thread.join(timeout=5)
