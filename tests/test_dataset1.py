"""DataSet1 conversion and held-out experiment contracts."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from feature_selector.artifacts import load_run
from feature_selector.config import ExperimentConfig
from feature_selector.data import load_data
from scripts.dataset1 import QUALITY_COLUMNS, TARGET, evaluate, prepare


class Dataset1Tests(unittest.TestCase):
    def test_preparation_preserves_values_and_excludes_quality(self) -> None:
        """Parse day-first dates and preserve ordered process columns.

        Returns:
            None: Checks exported values, metadata and exclusion of all quality signals.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            times = pd.date_range("2018-02-01", periods=6, freq="1h")
            labs = [TARGET, "lab_rvp", "lab_70", "lab_ar"]
            columns = ["sensor_z", *QUALITY_COLUMNS, "sensor_a", *labs]
            frame = pd.DataFrame(np.arange(60).reshape(6, 10), index=times, columns=columns)
            source, metadata = root / "source.csv", root / "source.json"
            frame.to_csv(source, sep=";", date_format="%d.%m.%Y %H:%M:%S")
            metadata.write_text(json.dumps({"LabName": dict(zip(QUALITY_COLUMNS, labs))}))
            prepare(source, metadata, root / "prepared")
            data = load_data(root / "prepared/features.csv", root / "prepared/target.csv", TARGET)
            pd.testing.assert_index_equal(data.X.index, times)
            self.assertEqual(list(data.X.columns), ["sensor_z", "sensor_a"])
            np.testing.assert_array_equal(data.X, frame[["sensor_z", "sensor_a"]])
            np.testing.assert_array_equal(data.y, frame[TARGET])
            report = json.loads((root / "prepared/preparation.json").read_text())
            self.assertEqual(report["rows"], 6)
            self.assertEqual(len(report["inputs"]["source"]["sha256"]), 64)
            self.assertEqual(set(report["excluded_columns"]), set(QUALITY_COLUMNS) | set(labs))

            frame.iloc[[0, 2, 3]].to_csv(
                source, sep=";", date_format="%d.%m.%Y %H:%M:%S"
            )
            with self.assertRaisesRegex(ValueError, "complete hourly grid"):
                prepare(source, metadata, root / "invalid")
            self.assertFalse((root / "invalid").exists())

    def test_gru_test_reports_and_train_only_reference(self) -> None:
        """Persist test reports and train-only scaling after a short GRU fit.

        Returns:
            None: Checks chronological windows, validation isolation and restored artifacts.
        """
        from feature_selector.models.gru import GRURegressor

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            times = pd.date_range("2018-01-01", periods=100, freq="1h")
            values = np.arange(100, dtype=float)
            pd.DataFrame({"sensor": values}, index=times).to_csv(root / "features.csv")
            pd.DataFrame({TARGET: values}, index=times).to_csv(root / "target.csv")
            config = ExperimentConfig.from_dict({
                "data": {"features_path": str(root / "features.csv"),
                         "target_path": str(root / "target.csv"), "target_column": TARGET},
                "pipeline": [
                    {"name": "scale", "type": "standard"},
                    {"name": "align", "type": "align_target", "params": {"horizon": 1}},
                    {"name": "windows", "type": "windows", "params": {"lookback": 48}},
                    {"name": "target_scaler", "type": "target_scale"},
                    {"name": "model", "type": "gru", "params": {
                        "epochs": 1, "hidden_size": 2, "batch_size": 8}},
                ],
                "split": {"train_fraction": 0.6, "validation_fraction": 0.2},
                "run": {"output_dir": str(root / "runs")},
            })
            original_fit = GRURegressor.fit
            with patch.object(GRURegressor, "fit", autospec=True, side_effect=original_fit) as fit:
                run = evaluate(config)
            self.assertEqual(fit.call_count, 1)
            self.assertEqual(fit.call_args.args[1].shape, (12, 48, 1))
            validation = fit.call_args.kwargs["validation_data"]
            self.assertEqual(validation[0].shape, (20, 48, 1))
            # Known target values identify exactly which timestamps entered training/validation.
            np.testing.assert_allclose(
                run.pipeline["target_scaler"].scaler.inverse_transform(validation[1].reshape(-1, 1)).ravel(),
                values[60:80],
            )
            self.assertAlmostEqual(run.pipeline["target_scaler"].scaler.mean_[0], values[48:60].mean())
            restored = load_run(run.artifact_dir)
            self.assertTrue(restored.metadata["test_evaluated"])
            self.assertEqual(restored.metrics["test"]["n"], 20)
            test = restored.predictions["test"]
            self.assertTrue(np.isfinite(test.y_pred).all())
            self.assertEqual(int(test.y_true.isna().sum()), 1)
            self.assertTrue((test.target_time - test.origin_time == pd.Timedelta("1h")).all())
            self.assertEqual(pd.Timestamp(test.target_time.iloc[0]), times[80])
            data = load_data(root / "features.csv", root / "target.csv", TARGET)
            pd.testing.assert_frame_equal(test, restored.predict(data, split="test"))
            saved = pd.read_csv(run.artifact_dir / "predictions_test.csv")
            np.testing.assert_allclose(saved.y_pred, test.y_pred)
            metrics = json.loads((run.artifact_dir / "metrics.json").read_text())
            self.assertEqual(metrics["test"], restored.metrics["test"])
            baseline = json.loads((run.artifact_dir / "baseline_metrics.json").read_text())
            self.assertEqual(baseline["value"], values[48:60].mean())
            self.assertEqual(baseline["metrics"]["test"]["n"], int(test.y_true.notna().sum()))


if __name__ == "__main__":
    unittest.main()
