"""End-to-end execution, original-unit predictions and durable run artifacts."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sklearn.preprocessing import StandardScaler

from feature_selector import load_config, run_experiment
from feature_selector.artifacts import load_run
from feature_selector.data import load_data, load_features
from feature_selector.models.ridge import WindowRidge
from feature_selector.pipeline.adapters import ModelStep, TargetScale
from feature_selector.pipeline.temporal import Windows
from support import write_config


class ExperimentTests(unittest.TestCase):
    def test_train_scaling_predictions_reload_and_input_ownership(self) -> None:
        """Fit on train only and reuse the exact preparation path after reload.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = write_config(root, sparse=True)
            before = deepcopy(spec)
            run = run_experiment(spec)
            self.assertEqual(spec, before)
            self.assertAlmostEqual(run.pipeline["scaler"].estimator.mean_[0], 17.5)
            self.assertNotIn("test", run.predictions)
            self.assertFalse(run.metadata["test_evaluated"])
            self.assertTrue(run.predictions["validation"].y_true.isna().any())
            source = load_data(**spec["data"])
            expected = run.predict(source, "test")
            loaded = load_run(run.artifact_dir)
            np.testing.assert_array_equal(expected.y_pred, loaded.predict(source, "test").y_pred)
            self.assertLess(run.metrics["validation"]["mae"], 3)
            for name in ("events.jsonl", "metadata.json", "config.yaml", "history.json", "run.pkl"):
                self.assertTrue((run.artifact_dir / name).is_file())

    def test_unlabelled_prediction_has_no_metric_requirement(self) -> None:
        """Keep forecasts available without a target file at inference.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            spec = write_config(Path(directory))
            run = run_experiment(spec)
            report = run.predict(load_features(spec["data"]["features_path"]), "test")
            self.assertTrue(report.y_true.isna().all())
            self.assertTrue(np.isfinite(report.y_pred).all())

    def test_test_not_used_even_for_automatic_prediction(self) -> None:
        """Do not invoke the numerical model on test during experiment execution.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            spec = write_config(Path(directory))
            original = WindowRidge.predict
            shapes = []
            def observe(model: WindowRidge, values: np.ndarray) -> np.ndarray:
                """Record prediction count before delegation.

                Parameters:
                    model (WindowRidge): Fitted estimator.
                    values (np.ndarray): Prediction inputs.

                Returns:
                    np.ndarray: Delegated predictions.
                """
                shapes.append(len(values))
                return original(model, values)
            with patch.object(WindowRidge, "predict", observe):
                run = run_experiment(spec)
            self.assertEqual(shapes, [43])
            self.assertEqual(sum(len(frame) for frame in run.predictions.values()), 43)

    def test_disabled_validation_and_test(self) -> None:
        """Support train-only and zero-validation configurations.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            spec = write_config(Path(directory))
            spec["split"] = {"train_fraction": 1, "validation_fraction": 0}
            run = run_experiment(spec)
            self.assertEqual(set(run.metrics), {"train"})
            self.assertEqual(set(run.predictions), {"train"})

    def test_failure_keeps_original_error_and_writes_context(self) -> None:
        """Preserve algorithm errors and step names in failure diagnostics.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = write_config(root)
            spec["pipeline"][-1]["params"] = {"alpha": -2}
            with self.assertRaises(ValueError) as caught:
                run_experiment(spec)
            self.assertIn("pipeline/model", " ".join(caught.exception.__notes__))
            run_dir = next((root / "runs").iterdir())
            self.assertTrue((run_dir / "failure.json").is_file())
            self.assertIn("Traceback", (run_dir / "run.log").read_text())
            lines = (run_dir / "events.jsonl").read_text().splitlines()
            events = [json.loads(line) for line in lines]
            self.assertEqual(events[-1]["event"], "run_failed")

    def test_cli_uses_new_config(self) -> None:
        """Execute the same public entrypoint from a YAML configuration.

        Returns:
            None.
        """
        import yaml

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = write_config(root)
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump(spec))
            result = subprocess.run([sys.executable, "-m", "feature_selector", "run", str(path)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("validation", json.loads(result.stdout)["metrics"])

    def test_python_steps_config_roundtrip(self) -> None:
        """Reload saved Python steps and nested estimators as unfitted constructors.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = write_config(root)
            spec["pipeline"][2] = ("windows", Windows(5))
            spec["pipeline"][3] = ("target_scaler", TargetScale(StandardScaler()))
            spec["pipeline"][4] = ("model", ModelStep(WindowRidge(alpha=3)))
            first = run_experiment(spec)
            config = load_config(first.artifact_dir / "config.yaml")
            repeated = run_experiment(config)
            self.assertEqual(repeated.pipeline["windows"].lookback, 5)
            self.assertEqual(repeated.model.alpha, 3)
            for split in ("train", "validation"):
                np.testing.assert_array_equal(first.predictions[split].y_pred,
                                              repeated.predictions[split].y_pred)

    def test_nonfinite_prediction_fails_the_run(self) -> None:
        """Persist a failure instead of accepting a model with invalid predictions.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = write_config(root)
            with patch.object(WindowRidge, "predict", return_value=np.full(43, np.nan)):
                with self.assertRaisesRegex(ValueError, "finite"):
                    run_experiment(spec)
            run_dir = next((root / "runs").iterdir())
            self.assertTrue((run_dir / "failure.json").is_file())
            self.assertFalse((run_dir / "run.pkl").exists())
