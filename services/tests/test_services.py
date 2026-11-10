"""Real HTTP and subprocess checks for the service boundary and unchanged ML core."""

import csv
import io
import json
import os
import socket
import tempfile
import threading
import time
import unittest
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import httpx
import numpy as np
import pandas as pd
import uvicorn
import yaml
from fastapi.testclient import TestClient

from services.catalog import Catalog
from services.collector import create_app as collector_app
from services.collector import read_frame, table
from services.common import API, atomic_json
from services.jobs import Jobs
from services.master import create_app as master_app
from services.ml import create_app as ml_app
from services.storage import create_app as storage_app
from services.templates import resolve_template


@contextmanager
def serve(app: Any) -> Iterator[str]:
    """Run a real local HTTP server on a reserved ephemeral socket.

    Parameters:
        app (Any): ASGI application.

    Returns:
        Iterator[str]: API base URL during the server lifetime.
    """
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    address = f"http://127.0.0.1:{sock.getsockname()[1]}{API}"

    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()

    try:
        deadline = time.monotonic() + 15
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("Test server did not start")
            time.sleep(0.01)

        yield address
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        sock.close()
        if thread.is_alive():
            raise RuntimeError("Test server did not stop")


def fixture(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write deterministic CSVs, including sparse targets outside the feature interval.

    Parameters:
        root (Path): Temporary directory.

    Returns:
        tuple: Collector settings and ML template settings.
    """
    times = pd.date_range("2024-01-01", periods=180, freq="h")
    x = np.arange(len(times), dtype=float)
    features = pd.DataFrame({"a": x / 20, "b": np.sin(x / 6)}, index=times)
    features.iloc[20, 1] = np.nan
    features.iloc[::-1].to_csv(root / "features.csv")

    target_times = pd.date_range(times[0] - pd.Timedelta(hours=3), periods=62, freq="3h")
    targets = pd.DataFrame({"y": np.arange(62) / 8}, index=target_times)
    targets.to_csv(root / "target.csv")

    base = {
        "data": {
            "features_path": "unused",
            "target_path": "unused",
            "target_column": "y",
            "freq": "1h",
        },
        "split": {"train_fraction": 0.6, "validation_fraction": 0.2},
        "run": {"seed": 0},
        "pipeline": [
            {"name": "scale", "type": "standard"},
            {"name": "future", "type": "align_target", "params": {"horizon": 1}},
            {"name": "windows", "type": "windows", "params": {"lookback": 5}},
            {"name": "model", "type": "ridge", "params": {"alpha": 1}},
        ],
    }

    templates = {}
    for model in ("ridge", "gru"):
        config = json.loads(json.dumps(base))
        params = {"run_seed": 0, "horizon": 1, "lookback": 5}
        config["pipeline"][-1] = {"name": "model", "type": model, "params": {}}
        if model == "ridge":
            params["alpha"] = 1.0
        else:
            params.update(model_seed=0, epochs=1, batch_size=8, hidden_size=4, learning_rate=0.01)

        path = root / f"{model}.yaml"
        path.write_text(yaml.safe_dump(config))
        templates[model] = {
            "name": model,
            "source_id": "uzk",
            "config_path": str(path),
            "parameters": params,
        }

    collector = {
        "state_dir": str(root / "collector"),
        "max_batch_size": 1000,
        "sources": {
            "uzk": {
                "features_path": str(root / "features.csv"),
                "target_path": str(root / "target.csv"),
                "target_column": "y",
                "freq": "1h",
            }
        },
    }
    return collector, {"templates": templates}


class BoundaryTests(unittest.TestCase):
    """Draft lifecycle, explicit errors and template isolation."""

    def test_storage_lifecycle(self) -> None:
        """Protect unpublished and referenced data while preserving completed artifacts.

        Returns:
            None: Checks CRUD, checksum and restart visibility.
        """
        with tempfile.TemporaryDirectory() as temporary:
            config = {"state_dir": temporary, "max_file_bytes": 64}
            with TestClient(storage_app(config), raise_server_exceptions=False) as client:
                path = f"{API}/objects/datasets"
                draft = client.post(path, json={"name": "input"}).json()
                key = draft["id"]

                self.assertEqual(client.get(path).json(), [])
                self.assertEqual(client.get(f"{path}/{key}").status_code, 404)
                self.assertEqual(client.post(f"{path}/{key}/publish").status_code, 422)

                artifacts = {
                    "features.csv.upload": b"previous artifact",
                    "features.csv": b"timestamp,x\n0,2\n",
                    "target.csv": b"timestamp,y\n0,1\n",
                }
                for name, content in artifacts.items():
                    upload = client.put(f"{path}/{key}/files/{name}", content=content)
                    self.assertEqual(upload.status_code, 200)
                    self.assertEqual(len(upload.json()["sha256"]), 64)

                self.assertEqual(client.post(f"{path}/{key}/publish").status_code, 200)
                for name, content in artifacts.items():
                    download = client.get(f"{path}/{key}/files/{name}")
                    self.assertEqual(download.status_code, 200)
                    self.assertEqual(download.content, content)
                self.assertEqual(
                    client.put(f"{path}/{key}/files/features.csv", content=b"x").status_code, 409
                )

                model = client.post(
                    f"{API}/objects/models",
                    json={
                        "name": "model",
                        "references": {"dataset_id": key},
                    },
                ).json()
                client.post(f"{API}/objects/models/{model['id']}/publish")
                self.assertEqual(client.delete(f"{path}/{key}").status_code, 409)
                self.assertEqual(
                    client.patch(f"{path}/{key}", json={"name": "renamed"}).status_code, 200
                )

                self.assertEqual(
                    client.put(f"{path}/{key}/files/manifest.json", content=b"x").status_code, 422
                )
                self.assertEqual(
                    client.put(f"{path}/{key}/files/large", content=b"x" * 65).status_code, 413
                )

            # Published v1 manifests predate reservation fields and remain readable.
            manifest_path = Path(temporary) / "objects" / "datasets" / key / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.pop("reserved")
            manifest.pop("job_id")
            atomic_json(manifest_path, manifest)

            with TestClient(storage_app(config)) as restarted:
                self.assertEqual(restarted.get(path).json()[0]["name"], "renamed")
                restarted.delete(f"{API}/objects/models/{model['id']}")
                self.assertEqual(restarted.delete(f"{path}/{key}").status_code, 204)

    def test_batches_and_templates(self) -> None:
        """Preserve sparse targets and reject unsupported template fields.

        Returns:
            None: Checks disjoint, repeatable batches and unchanged source YAML.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collector, ml = fixture(root)
            before = (root / "gru.yaml").read_bytes()

            config = resolve_template(ml, "gru", {"epochs": 2})
            self.assertEqual(config["pipeline"][-1]["params"]["patience"], 2)
            self.assertEqual((root / "gru.yaml").read_bytes(), before)

            with self.assertRaises(ValueError):
                resolve_template(ml, "gru", {"device": "cuda"})
            with self.assertRaises(TypeError):
                resolve_template(ml, "gru", {"epochs": True})

            with TestClient(collector_app(collector), raise_server_exceptions=False) as client:
                route = f"{API}/sources/uzk/batches"
                first = client.get(route, params={"limit": 17}).json()
                self.assertEqual(first, client.get(route, params={"limit": 17}).json())

                rows, targets, offset = [], [], 0
                while True:
                    part = client.get(route, params={"offset": offset, "limit": 17}).json()
                    rows.extend(part["features"]["rows"])
                    targets.extend(part["targets"]["rows"])
                    if part["done"]:
                        break
                    offset = part["next_offset"]

                self.assertEqual(len(rows), 180)
                self.assertEqual(len({row[0] for row in rows}), 180)
                self.assertEqual(len(targets), 60)
                self.assertEqual(len({row[0] for row in targets}), 60)
                self.assertIsNone(rows[20][2])
                self.assertEqual(client.get(route, params={"offset": 181}).status_code, 422)

                collector["sources"]["missing"] = {
                    **collector["sources"]["uzk"],
                    "features_path": "/missing.csv",
                }
                self.assertFalse(client.get(f"{API}/sources").json()[1]["available"])

    def test_interrupted_recovery(self) -> None:
        """Retain retryable recovery when Storage cannot confirm whether a result exists.

        Returns:
            None: Checks restart state without a live worker.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atomic_json(
                root / "jobs" / "orphan.json",
                {
                    "id": "orphan",
                    "created_at": "2024",
                    "status": "running",
                    "result_kind": "runs",
                },
            )
            settings = {
                "state_dir": temporary,
                "storage_url": "http://127.0.0.1:1",
                "timeout": 0.1,
            }

            with patch("services.jobs.Catalog.get", side_effect=httpx.ConnectError("Offline")):
                jobs = Jobs(settings)
                self.assertEqual(jobs.get("orphan")["status"], "reconciling")
                self.assertEqual(jobs.active, "orphan")
                jobs.close()

            with patch("services.jobs.Catalog.get", return_value={"id": "orphan"}):
                restarted = Jobs(settings)
                self.assertEqual(restarted.get("orphan")["status"], "completed")
                self.assertIsNone(restarted.active)
                restarted.close()

    def test_shutdown_interrupts_child(self) -> None:
        """Stop a real child and release its reserved draft during orderly shutdown.

        Returns:
            None: Checks shutdown and a subsequent idle controller restart.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            storage_settings = {"state_dir": str(root / "storage"), "max_file_bytes": 64}
            with serve(storage_app(storage_settings)) as url:
                settings = {
                    "state_dir": str(root / "jobs"),
                    "storage_url": url,
                    "timeout": 2,
                    "collector_url": "http://127.0.0.1:1",
                    "batch_size": 1,
                }
                jobs = Jobs(settings)

                submitted = jobs.start(
                    "collect", {"name": "Interrupted", "source_id": "uzk"}, "datasets"
                )
                jobs.close()

                self.assertEqual(jobs.get(submitted["id"])["status"], "interrupted")
                self.assertEqual(Catalog(url).json("GET", "/drafts/datasets"), [])

                restarted = Jobs(settings)
                self.assertIsNone(restarted.active)
                self.assertEqual(restarted.get(submitted["id"])["status"], "interrupted")
                restarted.close()

    def test_gateway_unavailable_dependency(self) -> None:
        """Expose an unavailable ML endpoint as a handled dependency error.

        Returns:
            None: Checks public error status and independent service reporting.
        """
        with tempfile.TemporaryDirectory() as temporary:
            settings = {
                "state_dir": temporary,
                "timeout": 0.1,
                "storage_url": "http://127.0.0.1:1",
                "collector_url": "http://127.0.0.1:1",
                "ml_url": "http://127.0.0.1:1",
            }
            with TestClient(master_app(settings)) as client:
                response = client.get(f"{API}/templates")
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()["code"], "dependency_unavailable")

                status = client.get(f"{API}/status").json()
                self.assertEqual(status["ml"]["status"], "unavailable")


class ScenarioTests(unittest.TestCase):
    """Run the four scenarios through real gateway HTTP and child processes."""

    def test_complete_flow(self) -> None:
        """Collect, train two models, evaluate explicitly and compare validation scores.

        Returns:
            None: Checks compatibility with direct core prediction and persistent artifacts.
        """
        from feature_selector.artifacts import load_run
        from feature_selector.data import load_data

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            collector, ml = fixture(root)
            storage_config = {"state_dir": str(root / "storage"), "max_file_bytes": 10000000}

            storage_url = stack.enter_context(serve(storage_app(storage_config)))
            collector_url = stack.enter_context(serve(collector_app(collector)))
            ml.update(state_dir=str(root / "ml"), storage_url=storage_url, timeout=10)
            ml_url = stack.enter_context(serve(ml_app(ml)))

            master_config = {
                "state_dir": str(root / "master"),
                "storage_url": storage_url,
                "collector_url": collector_url,
                "ml_url": ml_url,
                "timeout": 10,
                "batch_size": 17,
            }
            master_url = stack.enter_context(serve(master_app(master_config)))
            client = stack.enter_context(httpx.Client(base_url=master_url, timeout=15))

            def request(method: str, path: str, **kwargs: Any) -> Any:
                """Send a scenario request and require success.

                Parameters:
                    method (str): HTTP method.
                    path (str): Gateway path.
                    **kwargs (Any): HTTPX request arguments.

                Returns:
                    Any: Response JSON.
                """
                response = client.request(method, master_url + path, **kwargs)
                response.raise_for_status()
                return response.json()

            def wait(job: dict[str, Any], route: str = "/jobs") -> dict[str, Any]:
                """Await a bounded background job and surface worker errors.

                Parameters:
                    job (dict[str, Any]): Submitted job.
                    route (str): Gateway job collection.

                Returns:
                    dict[str, Any]: Completed state.
                """
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline:
                    state = request("GET", f"{route}/{job['id']}")
                    if state["status"] not in {"running", "reconciling"}:
                        self.assertEqual(state["status"], "completed", state)
                        return state
                    time.sleep(0.1)

                self.fail(f"Job timed out: {job}")

            collected = wait(
                request(
                    "POST",
                    "/collections",
                    json={
                        "source_id": "uzk",
                        "name": "fixture",
                    },
                ),
                "/collections",
            )
            dataset_id = collected["result_id"]
            self.assertEqual(request("GET", "/drafts/datasets"), [])

            abandoned = Catalog(storage_url).json(
                "POST",
                "/objects/models",
                json={"name": "Abandoned", "references": {"dataset_id": dataset_id}},
            )
            draft_id = abandoned["id"]
            self.assertEqual(request("GET", "/drafts/models")[0]["id"], draft_id)
            self.assertEqual(request("GET", f"/drafts/models/{draft_id}")["reserved"], False)
            discarded = client.delete(master_url + f"/drafts/models/{draft_id}")
            self.assertEqual(discarded.status_code, 204)

            content = client.get(
                master_url + f"/objects/datasets/{dataset_id}/files/features.csv"
            ).text
            pd.testing.assert_frame_equal(
                pd.read_csv(io.StringIO(content), index_col=0),
                pd.read_csv(root / "features.csv", index_col=0).sort_index(),
                check_names=False,
            )
            self.assertEqual(len(request("GET", f"/datasets/{dataset_id}/preview")["rows"]), 20)

            runs = []
            for template in ("ridge", "gru"):
                model = request(
                    "POST",
                    "/models",
                    json={
                        "name": "m" * 200 if template == "ridge" else template,
                        "dataset_id": dataset_id,
                        "template_id": template,
                    },
                )
                job = request("POST", f"/models/{model['id']}/train")
                # Fresh child imports numerical dependencies before a tiny fit can finish.
                busy = client.post(master_url + f"/models/{model['id']}/train")
                self.assertEqual(busy.status_code, 409)

                run_id = wait(job)["result_id"]
                run = request("GET", f"/objects/runs/{run_id}")
                self.assertNotIn("test", run["metadata"]["metrics"])
                self.assertFalse(run["metadata"]["runtime"]["test_evaluated"])
                runs.append(run_id)

                prediction_id = wait(
                    request(
                        "POST",
                        f"/runs/{run_id}/predict",
                        json={
                            "split": "test",
                        },
                    )
                )["result_id"]
                report = pd.read_csv(
                    io.StringIO(
                        client.get(
                            master_url
                            + (f"/objects/predictions/{prediction_id}/files/predictions.csv")
                        ).text
                    )
                )

                pickle_path = root / f"{template}.pkl"
                pickle_path.write_bytes(
                    client.get(master_url + (f"/objects/runs/{run_id}/files/run.pkl")).content
                )
                loaded = load_run(pickle_path)
                direct = loaded.predict(
                    load_data(root / "features.csv", root / "target.csv", "y"), "test"
                )
                np.testing.assert_allclose(report["y_pred"], direct["y_pred"], rtol=1e-6)

                prediction = request("GET", f"/objects/predictions/{prediction_id}")
                self.assertLessEqual(len(prediction["name"]), 200)
                self.assertEqual(prediction["metadata"]["run_name"], model["name"])
                metrics = prediction["metadata"]["metrics"]
                self.assertGreater(metrics["n"], 0)
                self.assertEqual(len(request("GET", f"/models/{model['id']}")["runs"]), 1)

            compared = request("GET", "/comparison", params=[("run_ids", key) for key in runs])
            self.assertEqual(len(compared["rows"]), 2)
            self.assertEqual(
                client.delete(master_url + f"/objects/datasets/{dataset_id}").status_code, 409
            )
            self.assertEqual(request("GET", "/status")["ml"]["status"], "ok")

            # Fail an otherwise valid submission after numerical preparation, then recover.
            bad_model = request(
                "POST",
                "/models",
                json={
                    "name": "oversized batch",
                    "dataset_id": dataset_id,
                    "template_id": "gru",
                    "parameters": {"batch_size": 1024},
                },
            )
            failed = request("POST", f"/models/{bad_model['id']}/train")
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                state = request("GET", f"/jobs/{failed['id']}")
                if state["status"] not in {"running", "reconciling"}:
                    break
                time.sleep(0.1)

            self.assertEqual(state["status"], "failed", state)
            self.assertIn("batch_size", state["error"])
            self.assertTrue(request("GET", f"/jobs/{failed['id']}/events")["lines"])

            # Exercise all four GUI pages and verify reruns never submit duplicate jobs.
            from streamlit.testing.v1 import AppTest

            gui_config = root / "visualization.yaml"
            gui_config.write_text(
                yaml.safe_dump(
                    {
                        "master_url": master_url,
                        "timeout": 15,
                        "refresh_seconds": 2,
                    }
                )
            )

            with patch.dict(os.environ, {"SERVICE_CONFIG": str(gui_config)}):
                ui_path = Path(__file__).resolve().parents[1] / "visualization.py"
                ui = AppTest.from_file(str(ui_path), default_timeout=30).run()
                self.assertEqual(list(ui.exception), [])
                self.assertEqual(list(ui.error), [])
                self.assertEqual(ui.sidebar.selectbox[0].value, 180)

                ui.sidebar.selectbox[0].set_value(0).run()
                self.assertEqual(list(ui.exception), [])
                self.assertEqual(ui.sidebar.selectbox[0].value, 0)

                ui.sidebar.radio[0].set_value("Training").run()
                self.assertEqual(ui.sidebar.selectbox[0].value, 0)
                self.assertEqual(list(ui.exception), [])
                self.assertEqual(list(ui.error), [])

                before = len(request("GET", "/jobs"))
                next(
                    button for button in ui.button if button.label == "Create model and train"
                ).click().run()
                self.assertEqual(list(ui.exception), [])
                self.assertEqual(list(ui.error), [])
                self.assertEqual(len(request("GET", "/jobs")), before + 1)

                ui.run()
                self.assertEqual(len(request("GET", "/jobs")), before + 1)
                wait(request("GET", "/jobs")[-1])

                ui.sidebar.radio[0].set_value("Prediction").run()
                self.assertEqual(list(ui.exception), [])
                self.assertEqual(list(ui.error), [])
                self.assertGreater(len(ui.get("plotly_chart")), 0)

                ui.sidebar.radio[0].set_value("Comparison").run()
                ui.multiselect[0].set_value(runs).run()
                self.assertEqual(list(ui.exception), [])
                self.assertEqual(list(ui.error), [])
                self.assertGreater(len(ui.dataframe), 0)
                self.assertGreater(len(ui.get("plotly_chart")), 0)


if __name__ == "__main__":
    unittest.main()
