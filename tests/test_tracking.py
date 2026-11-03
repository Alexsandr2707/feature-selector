"""Durable telemetry agrees with numerical history and TensorBoard coordinates."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from feature_selector.models.base import TorchModelRunner
from feature_selector.tracking import RunTracker, StepContext
from feature_selector import run_experiment
from feature_selector.experiment import _log_final_metrics
from support import write_config


class TrackingTests(unittest.TestCase):
    def test_same_timestamp_in_different_experiments_keeps_logs_separate(self) -> None:
        """Isolate simultaneous trackers whose directories share a timestamp name.

        Returns:
            None: Checks file logs and events for cross-experiment messages.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first" / "2026-11-03_14-05-09"
            second = root / "second" / first.name
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            tracker = RunTracker(first)
            self.addCleanup(tracker.close)
            other = RunTracker(second)
            self.addCleanup(other.close)
            self.assertIsNot(tracker.logger, other.logger)
            tracker.record("pipeline", "run_started", {"marker": "first_only"})
            other.record("pipeline", "run_started", {"marker": "second_only"})
            for directory, own, foreign in (
                (first, "first_only", "second_only"),
                (second, "second_only", "first_only"),
            ):
                for filename in ("run.log", "events.jsonl"):
                    text = (directory / filename).read_text(encoding="utf-8")
                    self.assertIn(own, text)
                    self.assertNotIn(foreign, text)

    def test_runs_log_final_results_without_extra_tensorboard_sessions(self) -> None:
        """Publish aggregate saved scores and a table alongside existing training events.

        Returns:
            None.
        """
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        models = [
            ("ridge", {}),
            ("gru", {"batch_size": 1, "epochs": 1, "hidden_size": 2}),
            ("conv_autoencoder", {"batch_size": 1, "epochs": 1,
                                  "hidden_channels": 2, "latent_channels": 1}),
            ("conv_autoencoder", {"batch_size": 1, "epochs": 1, "hidden_channels": 2,
                                  "latent_channels": 1, "compression": "features"}),
            ("gru_autoencoder", {"batch_size": 1, "epochs": 1,
                                 "hidden_channels": 2, "latent_channels": 1}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for kind, params in models:
                with self.subTest(model=kind, params=params):
                    spec = write_config(Path(directory))
                    if kind.endswith("autoencoder"):
                        spec["data"].pop("target_path")
                        spec["data"].pop("target_column")
                        spec["pipeline"] = [spec["pipeline"][0], spec["pipeline"][2]]
                        spec["pipeline"].append({"name": "model", "type": kind, "params": params})
                    else:
                        spec["pipeline"][-1] = {"name": "model", "type": kind, "params": params}
                    run = run_experiment(spec)
                    tensorboard = run.artifact_dir / "tensorboard"
                    self.assertFalse(any(path.is_dir() for path in tensorboard.iterdir()))
                    records = [json.loads(line) for line in
                               (run.artifact_dir / "events.jsonl").read_text().splitlines()]
                    self.assertFalse(any(record["event"] == "comparison" for record in records))
                    events = EventAccumulator(str(tensorboard)).Reload()
                    self.assertFalse(any(tag.startswith("_hparams_/")
                                         for tag in events.Tags()["tensors"]))
                    self.assertIn("results/summary/text_summary", events.Tags()["tensors"])
                    if kind != "ridge":
                        self.assertIn("pipeline/model/train/batch/loss", events.Tags()["scalars"])
                        self.assertNotIn("pipeline/model/train/epoch/loss",
                                         events.Tags()["scalars"])
                        self.assertIn("pipeline/model/validation/epoch/loss",
                                      events.Tags()["scalars"])
                    metrics = json.loads((run.artifact_dir / "metrics.json").read_text())
                    self.assertIn("train", metrics)
                    self.assertIn("validation", metrics)
                    for split, report in metrics.items():
                        scores = report.get("overall", report)
                        for name in ("mae", "rmse", "r2", "pearson", "n"):
                            tag = f"results/{split}/{name}"
                            if scores[name] is None:
                                self.assertNotIn(tag, events.Tags()["scalars"])
                            else:
                                scalar, = events.Scalars(tag)
                                self.assertEqual(scalar.step, 0)
                                np.testing.assert_allclose(scalar.value, scores[name], rtol=1e-6)
                    self.assertFalse(any(tag.startswith("results/test/")
                                         for tag in events.Tags()["scalars"]))
                    table, = events.Tensors("results/summary/text_summary")
                    markdown = table.tensor_proto.string_val[0].decode()
                    self.assertIn("| train |", markdown)
                    self.assertIn("| validation |", markdown)
                    self.assertNotIn("per_feature", markdown)

    def test_final_results_skip_undefined_scores_and_missing_splits(self) -> None:
        """Keep unavailable scores readable without writing nonfinite scalars or test results.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracker = RunTracker(root)
            try:
                _log_final_metrics(tracker, {
                    "train": {"mae": 2.0, "rmse": 3.0, "r2": float("nan"),
                              "pearson": None, "n": 1, "notes": {"r2": "constant target"}},
                    "validation": {},
                })
            finally:
                tracker.close()
            events = EventAccumulator(str(root / "tensorboard")).Reload()
            self.assertEqual(set(events.Tags()["scalars"]), {
                "results/train/mae", "results/train/rmse", "results/train/n",
            })
            table, = events.Tensors("results/summary/text_summary")
            markdown = table.tensor_proto.string_val[0].decode()
            self.assertIn("| train | 2 | 3 | — | — | 1 |", markdown)
            self.assertNotIn("validation", markdown)
            self.assertNotIn("test", markdown)

    def test_batches_epochs_and_tensorboard_agree(self) -> None:
        """Keep epoch train means in JSONL and publish batch and validation losses.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracker = RunTracker(root)
            context = StepContext(tracker).child("model")
            network = torch.nn.Linear(1, 1)
            runner = TorchModelRunner(network, torch.optim.SGD(network.parameters(), lr=0), None,
                                      "cpu")
            X = np.arange(5, dtype=np.float32).reshape(-1, 1)
            try:
                runner.fit(X, X.ravel(), (X[:2], X[:2].ravel()), epochs=2, batch_size=3,
                           patience=5, random_state=0, context=context)
                context.log_metadata({"selected": [1, 2]})
                context.child("nested").log_metadata({"selected": [3]})
                self.assertEqual(tracker.metadata["pipeline/model"]["selected"], [1, 2])
            finally:
                tracker.close()
            events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
            metrics = [event["data"] for event in events if event["event"] == "metrics"]
            batches = [event for event in metrics if event["batch"] is not None]
            self.assertEqual([event["step"] for event in batches], [1, 2])
            for epoch in runner.history_:
                expected = sum(batch["samples"] * batch["train_loss"] for batch in epoch["batches"])
                self.assertAlmostEqual(epoch["train_loss"], expected / 3)
                actual = [e for e in metrics if e["split"] == "validation" and
                          e["epoch"] == epoch["epoch"]]
                self.assertEqual(actual[0]["values"]["loss"], epoch["validation_loss"])
                train = [e for e in metrics if e["split"] == "train" and
                         e["batch"] is None and e["epoch"] == epoch["epoch"]]
                self.assertEqual(len(train), 1)
                self.assertEqual(train[0]["values"]["loss"], epoch["train_loss"])
                self.assertEqual(train[0]["samples"], 3)
            tensorboard = EventAccumulator(str(root / "tensorboard")).Reload()
            self.assertNotIn("pipeline/model/train/epoch/loss", tensorboard.Tags()["scalars"])
            self.assertIn("pipeline/model/validation/epoch/loss", tensorboard.Tags()["scalars"])
            self.assertNotIn("pipeline/model train epoch=", (root / "run.log").read_text())
            self.assertIn("pipeline/model validation epoch=", (root / "run.log").read_text())
            scalar = tensorboard.Scalars("pipeline/model/train/batch/loss")
            self.assertEqual([event.step for event in scalar], [1, 2])
            np.testing.assert_allclose([event.value for event in scalar],
                                       [event["values"]["loss"] for event in batches], rtol=1e-6)

    def test_interruption_preserves_completed_batches(self) -> None:
        """Keep batch events even if a subsequent optimizer step raises.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracker = RunTracker(root)
            network = torch.nn.Linear(1, 1)
            optimizer = torch.optim.SGD(network.parameters(), lr=0.01)
            runner = TorchModelRunner(network, optimizer, None, "cpu")
            original = optimizer.step
            calls = 0
            def interrupted() -> None:
                """Fail on the second optimizer update.

                Returns:
                    None.
                """
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("interrupted")
                original()
            X = np.ones((5, 1), dtype=np.float32)
            try:
                with patch.object(optimizer, "step", interrupted), self.assertRaises(RuntimeError):
                    runner.fit(X, X.ravel(), epochs=2, batch_size=2, patience=3, random_state=0,
                               context=StepContext(tracker).child("model"))
            finally:
                tracker.close()
            records = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["data"]["samples"], 2)

    def test_step_payload_cannot_change_event_scope(self) -> None:
        """Keep path and event identity outside user-supplied diagnostic fields.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracker = RunTracker(root)
            try:
                StepContext(tracker).child("a").log_event("custom", {"path": "pipeline/b"})
            finally:
                tracker.close()
            record = json.loads((root / "events.jsonl").read_text())
            self.assertEqual(record["path"], "pipeline/a")
