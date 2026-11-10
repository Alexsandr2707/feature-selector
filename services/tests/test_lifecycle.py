"""Regression checks for reservations, uncertain publication and CSV schema preservation."""

import csv
import io
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import httpx
import pandas as pd

from services.catalog import Catalog
from services.collector import read_frame, table
from services.common import StateFiles, atomic_json
from services.demo import wait
from services.jobs import Jobs
from services.storage import REQUIRED, create_app
from services.tests.test_services import serve


def missing() -> httpx.HTTPStatusError:
    """Construct an explicit missing-object response for recovery checks.

    Returns:
        httpx.HTTPStatusError: HTTP 404 with an attached request.
    """
    response = httpx.Response(404, request=httpx.Request("GET", "http://storage/missing"))
    return httpx.HTTPStatusError("Missing object", request=response.request, response=response)


class LifecycleTests(unittest.TestCase):
    """Verify failure paths that successful end-to-end scenarios cannot exercise."""

    def test_demo_waits_for_reconciliation(self) -> None:
        """Keep the demonstration polling while Storage publication is unresolved.

        Returns:
            None: Does not report a retryable reconciliation state as execution failure.
        """
        remote = Mock()
        remote.json.side_effect = [
            {"status": "running"},
            {"status": "reconciling"},
            {"status": "completed", "result_id": "published"},
        ]

        with patch("services.demo.time.sleep"):
            result = wait(remote, "/jobs", {"id": "job"}, 10)

        self.assertEqual(result, "published")
        self.assertEqual(remote.json.call_count, 3)

    def test_reserved_inputs_survive_launch_failure(self) -> None:
        """Protect train and prediction inputs before launch, then release on launch failure.

        Returns:
            None: Checks reservation capabilities, public CRUD and failed-start cleanup.
        """
        for action in ("train", "predict"):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = {"state_dir": str(root / "storage"), "max_file_bytes": 64}
                with serve(create_app(config)) as url:
                    catalog = Catalog(url)
                    path = root / "artifact"
                    path.write_bytes(b"fixture")
                    catalog.publish(
                        "datasets",
                        "data",
                        "Data",
                        {},
                        {},
                        {name: path for name in REQUIRED["datasets"]},
                    )
                    model = catalog.publish(
                        "models",
                        "model",
                        "Model",
                        {},
                        {"dataset_id": "data"},
                        {},
                    )

                    if action == "train":
                        payload = {"model": model}
                        kind, input_kind, input_id = "runs", "models", "model"
                    else:
                        catalog.publish(
                            "runs",
                            "run",
                            "Run",
                            {},
                            {"dataset_id": "data", "model_id": "model"},
                            {name: path for name in REQUIRED["runs"]},
                        )
                        payload = {"run_id": "run", "split": "test"}
                        kind, input_kind, input_id = "predictions", "runs", "run"

                    jobs = Jobs({"state_dir": str(root / "jobs"), "storage_url": url, "timeout": 2})

                    def refuse_launch(*args: Any, **kwargs: Any) -> None:
                        """Inspect the reservation before simulating subprocess failure.

                        Parameters:
                            *args (Any): Popen positional arguments.
                            **kwargs (Any): Popen keyword arguments.

                        Returns:
                            None: Raises after checking all public mutation paths.
                        """
                        draft = catalog.json("GET", f"/drafts/{kind}")[0]
                        self.assertTrue(draft["reserved"])
                        self.assertEqual(draft["job_id"], jobs.active)
                        self.assertNotIn("token", repr(draft))

                        for method, route, options in (
                            ("DELETE", f"/objects/{input_kind}/{input_id}", {}),
                            ("DELETE", f"/objects/{kind}/{draft['id']}", {}),
                            ("DELETE", f"/drafts/{kind}/{draft['id']}", {}),
                            ("POST", f"/objects/{kind}/{draft['id']}/publish", {}),
                            (
                                "PUT",
                                f"/objects/{kind}/{draft['id']}/files/rogue",
                                {"content": b"x"},
                            ),
                            (
                                "PATCH",
                                f"/drafts/{kind}/{draft['id']}",
                                {"json": {"name": "Hijack"}},
                            ),
                        ):
                            response = httpx.request(method, url + route, timeout=2, **options)
                            self.assertEqual(response.status_code, 409, response.text)

                        raise OSError("Simulated process startup failure")

                    try:
                        with patch("services.jobs.subprocess.Popen", side_effect=refuse_launch):
                            with self.assertRaisesRegex(OSError, "Simulated"):
                                jobs.start(action, payload, kind)

                        self.assertEqual(catalog.json("GET", f"/drafts/{kind}"), [])
                        self.assertIsNone(jobs.active)
                        self.assertEqual(jobs.list()[0]["status"], "failed")
                        catalog.request("DELETE", f"/objects/{input_kind}/{input_id}")
                    finally:
                        jobs.close()

    def test_legacy_missing_result_becomes_interrupted(self) -> None:
        """Read an old running job without a request, token or ownership fields.

        Returns:
            None: Distinguishes confirmed HTTP 404 from a temporary Storage outage.
        """
        with tempfile.TemporaryDirectory() as temporary:
            StateFiles(Path(temporary) / "jobs").put(
                {"id": "old", "created_at": "2024", "status": "running", "result_kind": "runs"}
            )

            with (
                patch("services.jobs.Catalog.get", side_effect=missing()),
                patch("services.jobs.Catalog.discard", side_effect=missing()),
            ):
                jobs = Jobs({"state_dir": temporary, "storage_url": "http://storage", "timeout": 1})
                self.assertEqual(jobs.get("old")["status"], "interrupted")
                self.assertIsNone(jobs.active)
                jobs.close()

    def test_lost_publication_response_is_completed(self) -> None:
        """Prefer the committed Storage object over a worker's timeout error.

        Returns:
            None: Reconciles a failed worker after successful server publication.
        """
        with tempfile.TemporaryDirectory() as temporary:
            jobs = Jobs({"state_dir": temporary, "storage_url": "http://storage", "timeout": 1})
            jobs.states.put(
                {"id": "lost", "created_at": "2024", "status": "running", "result_kind": "runs"}
            )
            atomic_json(jobs.work / "lost" / "outcome.json", {"error": "Publication timed out"})
            jobs.catalog = Mock(spec=Catalog)
            jobs.catalog.get.return_value = {"id": "lost"}
            jobs.process = Mock(wait=Mock(return_value=1))
            jobs.active = "lost"

            jobs._monitor("lost")

            self.assertEqual(jobs.get("lost")["status"], "completed")
            jobs.catalog.discard.assert_not_called()
            self.assertIsNone(jobs.active)
            jobs.close()

    def test_cleanup_outage_preserves_retryable_state(self) -> None:
        """Retry reserved-draft cleanup without losing the original worker failure.

        Returns:
            None: Clears pending diagnostics and releases the slot after Storage recovers.
        """
        with tempfile.TemporaryDirectory() as temporary:
            jobs = Jobs({"state_dir": temporary, "storage_url": "http://storage", "timeout": 1})
            jobs.states.put(
                {"id": "broken", "created_at": "2024", "status": "running", "result_kind": "runs"}
            )
            atomic_json(jobs.work / "broken" / "outcome.json", {"error": "Numerical failure"})
            atomic_json(jobs.work / "broken" / "request.json", {"reservation_token": "private"})
            jobs.catalog = Mock(spec=Catalog)
            jobs.catalog.get.side_effect = missing()
            jobs.catalog.discard.side_effect = httpx.ConnectError("Storage unavailable")
            jobs.process = Mock(wait=Mock(return_value=1))
            jobs.active = "broken"

            jobs._monitor("broken")

            state = jobs.states.get("broken")
            self.assertEqual(state["status"], "reconciling")
            self.assertEqual(state["error"], "Numerical failure")
            self.assertEqual(jobs.active, "broken")

            jobs.catalog.discard.side_effect = None
            state = jobs.get("broken")

            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["error"], "Numerical failure")
            self.assertNotIn("pending_status", state)
            self.assertNotIn("reconciliation_error", state)
            self.assertNotIn("private", repr(state))
            self.assertIsNone(jobs.active)
            jobs.close()

    def test_malformed_outcome_releases_slot(self) -> None:
        """Treat unreadable worker diagnostics as a failed execution requiring reconciliation.

        Returns:
            None: Avoids a permanently busy executor after a malformed outcome file.
        """
        with tempfile.TemporaryDirectory() as temporary:
            jobs = Jobs({"state_dir": temporary, "storage_url": "http://storage", "timeout": 1})
            jobs.states.put(
                {"id": "broken", "created_at": "2024", "status": "running", "result_kind": "runs"}
            )
            directory = jobs.work / "broken"
            directory.mkdir()
            (directory / "outcome.json").write_text("incomplete")
            jobs.catalog = Mock(spec=Catalog)
            jobs.catalog.get.side_effect = missing()
            jobs.process = Mock(wait=Mock(return_value=1))
            jobs.active = "broken"

            jobs._monitor("broken")

            self.assertEqual(jobs.get("broken")["status"], "failed")
            self.assertIn("Cannot read worker outcome", jobs.get("broken")["error"])
            self.assertIsNone(jobs.active)
            jobs.close()

    def test_timestamp_feature_roundtrip(self) -> None:
        """Preserve feature names when one or more candidate timestamp names already exist.

        Returns:
            None: Checks CSV loading, batch encoding and the collected CSV schema.
        """
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.csv"
            path.write_text("time,timestamp,timestamp_,a\n2024-01-01,1,2,3\n")

            original = read_frame(path)
            batch = table(original)

            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(batch["columns"])
            writer.writerows(batch["rows"])

            collected = pd.read_csv(io.StringIO(output.getvalue()), index_col=0)
            self.assertEqual(batch["columns"][0], "timestamp__")
            self.assertEqual(collected.columns.tolist(), original.columns.tolist())
            self.assertEqual(collected.iloc[0].tolist(), original.iloc[0].tolist())


if __name__ == "__main__":
    unittest.main()
