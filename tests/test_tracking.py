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
from support import write_config


class TrackingTests(unittest.TestCase):
    def test_runs_write_only_training_tensorboard_events(self) -> None:
        """Keep reports on disk without generating final-metric or HParams sessions.

        Returns:
            None.
        """
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        models = [
            ("ridge", {}),
            ("gru", {"epochs": 1, "hidden_size": 2}),
            ("conv_autoencoder", {"epochs": 1, "hidden_channels": 2, "latent_channels": 1}),
            ("conv_autoencoder", {"epochs": 1, "hidden_channels": 2,
                                  "latent_channels": 1, "compression": "features"}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for kind, params in models:
                with self.subTest(model=kind, params=params):
                    spec = write_config(Path(directory))
                    if kind == "conv_autoencoder":
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
                    self.assertFalse(any(tag.startswith("results/")
                                         for tag in events.Tags()["scalars"]))
                    if kind != "ridge":
                        self.assertIn("pipeline/model/train/batch/loss", events.Tags()["scalars"])
                        self.assertIn("pipeline/model/validation/epoch/loss",
                                      events.Tags()["scalars"])
                    metrics = json.loads((run.artifact_dir / "metrics.json").read_text())
                    self.assertIn("train", metrics)
                    self.assertIn("validation", metrics)

    def test_batches_epochs_and_tensorboard_agree(self) -> None:
        """Emit exact batch sizes, weighted means and validation loss each epoch.

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
            self.assertEqual([event["step"] for event in batches], [1, 2, 3, 4])
            for epoch in runner.history_:
                expected = sum(batch["samples"] * batch["train_loss"] for batch in epoch["batches"])
                self.assertAlmostEqual(epoch["train_loss"], expected / 5)
                actual = [e for e in metrics if e["split"] == "validation" and
                          e["epoch"] == epoch["epoch"]]
                self.assertEqual(actual[0]["values"]["loss"], epoch["validation_loss"])
            tensorboard = EventAccumulator(str(root / "tensorboard")).Reload()
            scalar = tensorboard.Scalars("pipeline/model/train/batch/loss")
            self.assertEqual([event.step for event in scalar], [1, 2, 3, 4])
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
