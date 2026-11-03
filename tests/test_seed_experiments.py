"""Seed series freeze preprocessing and persist explicit test scores and individual runs."""

from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import torch
import yaml

from feature_selector.artifacts import load_run
from feature_selector.config import ExperimentConfig
from feature_selector.data import load_data
from feature_selector.evaluation import METRIC_NAMES, report_metrics
from feature_selector.experiment import run_seed_experiments
from feature_selector.models.gru import GRURegressor
from feature_selector.pipeline.adapters import ModelStep
from feature_selector.tracking import StepContext
from support import write_config


def gru_config(root: Path, *, sparse: bool = False) -> dict[str, Any]:
    """Create short CPU GRU settings and input files for a seed series.

    Parameters:
        root (Path): Existing temporary directory for inputs and run artifacts.
        sparse (bool): Observe targets only every fifth sample.

    Returns:
        dict[str, Any]: Two-epoch GRU experiment with feature and target scaling.
    """
    config = write_config(root, sparse=sparse)
    config["pipeline"][-1] = {
        "name": "model",
        "type": "gru",
        "params": {
            "hidden_size": 4,
            "epochs": 2,
            "batch_size": 2,
            "patience": 2,
            "scheduler": False,
            "random_state": 999,
        },
    }
    config["run"]["seed"] = 123
    return config


class SeedExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        """Use one CPU thread and restore global random generators after each test.

        Returns:
            None.
        """
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        self.addCleanup(random.setstate, random.getstate())
        self.addCleanup(np.random.set_state, np.random.get_state())
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())

    def test_saved_scores_models_predictions_and_frozen_preprocessing(self) -> None:
        """Fit scalers once, preserve identical training inputs and save all seed reports.

        Returns:
            None.
        """
        training: list[tuple[int, NDArray[Any], NDArray[Any], NDArray[Any]]] = []
        original_fit = GRURegressor.fit

        def observe_fit(
            model: GRURegressor,
            X: NDArray[Any],
            y: NDArray[Any],
            validation_data: tuple[NDArray[Any], NDArray[Any]] | None = None,
            *,
            context: StepContext | None = None,
        ) -> GRURegressor:
            """Capture train and validation inputs before the real training call.

            Parameters:
                model (GRURegressor): Seeded model under test.
                X (NDArray[Any]): Train windows.
                y (NDArray[Any]): Train targets in training units.
                validation_data (tuple[NDArray[Any], NDArray[Any]] | None): Validation pair.
                context (StepContext | None): Active model telemetry.

            Returns:
                GRURegressor: Trained model.
            """
            assert validation_data is not None
            training.append((model.random_state, X.copy(), y.copy(), validation_data[0].copy()))
            return original_fit(model, X, y, validation_data, context=context)

        with tempfile.TemporaryDirectory() as directory:
            config = gru_config(Path(directory), sparse=True)
            config["run"]["experiment_name"] = "gru_seeds"
            before = deepcopy(config)
            with (
                patch.object(
                    StandardScaler, "fit", autospec=True, side_effect=StandardScaler.fit
                ) as scaler_fit,
                patch.object(GRURegressor, "fit", observe_fit),
            ):
                result = run_seed_experiments(config, seeds=[7, 3, 11])
            self.assertEqual(config, before)
            self.assertEqual(result.artifact_dir.parent, Path(directory) / "runs" / "gru_seeds")
            self.assertEqual(scaler_fit.call_count, 2)
            self.assertEqual([entry[0] for entry in training], [7, 3, 11])
            for entry in training[1:]:
                for actual, expected in zip(entry[1:], training[0][1:], strict=True):
                    np.testing.assert_array_equal(actual, expected)

            per_seed = result.metrics_per_seed
            self.assertEqual(per_seed["seed"].tolist(), [7, 3, 11])
            pd.testing.assert_frame_equal(
                pd.read_csv(result.artifact_dir / "metrics_per_seed.csv"),
                per_seed,
                check_dtype=False,
            )
            pd.testing.assert_frame_equal(
                pd.read_csv(result.artifact_dir / "metrics_summary.csv"),
                result.metrics_summary,
                check_dtype=False,
            )
            summary = result.metrics_summary.set_index("metric")
            for metric in METRIC_NAMES:
                self.assertAlmostEqual(summary.loc[metric, "mean"], per_seed[metric].mean())
                self.assertAlmostEqual(summary.loc[metric, "std"], per_seed[metric].std(ddof=1))
                self.assertEqual(summary.loc[metric, "n_valid"], 3)
                self.assertEqual(summary.loc[metric, "n_total"], 3)

            source = load_data(**config["data"])
            predictions = []
            for row in per_seed.itertuples(index=False):
                run = load_run(row.artifact_dir)
                self.assertEqual(
                    run.artifact_dir.parent,
                    result.artifact_dir / "runs" / f"seed_{row.seed}",
                )
                self.assertEqual(run.config.run["experiment_name"], f"seed_{row.seed}")
                self.assertEqual(run.model.random_state, row.seed)
                self.assertEqual(run.config.run["seed"], 123)
                self.assertEqual(run.metadata["model_seed"], row.seed)
                self.assertTrue(run.metadata["test_evaluated"])
                self.assertEqual(run.pipeline.provenance_["model_seed"], row.seed)
                pd.testing.assert_frame_equal(run.predict(source, "test"), run.predictions["test"])
                self.assertEqual(report_metrics(run.predictions["test"]), run.metrics["test"])
                stored = json.loads((run.artifact_dir / "metrics.json").read_text())
                self.assertEqual(stored["test"], run.metrics["test"])
                self.assertTrue((run.artifact_dir / "predictions_test.csv").is_file())
                self.assertLess(row.n, len(run.predictions["test"]))
                predictions.append(run.predictions["test"]["y_pred"].to_numpy())
            self.assertFalse(np.array_equal(predictions[0], predictions[1]))
            manifest = json.loads((result.artifact_dir / "series.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["seeds"], [7, 3, 11])
            self.assertEqual(manifest["completed_seeds"], [7, 3, 11])
            self.assertEqual(manifest["std_ddof"], 1)

            # Saved runs must retain the fitted prefix even after its export is moved.
            prefix = result.artifact_dir / "preprocessing.pkl"
            prefix.rename(prefix.with_suffix(".moved"))
            reloaded = load_run(per_seed.iloc[-1]["artifact_dir"])
            np.testing.assert_array_equal(reloaded.predict(source, "test").y_pred, predictions[-1])

    def test_repeated_series_match_and_support_yaml_and_python_steps(self) -> None:
        """Repeat seeded predictions with YAML settings and Python estimator instances.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = gru_config(root)
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump(config))
            first = run_seed_experiments(path, seeds=[2, 5])
            model = GRURegressor(**config["pipeline"][-1]["params"])
            config["pipeline"][-1] = ("model", ModelStep(model))
            settings = ExperimentConfig.from_dict(config)
            second = run_seed_experiments(settings, seeds=[2, 5])
            self.assertEqual(model.random_state, 999)
            self.assertFalse(hasattr(model, "network_"))
            pd.testing.assert_frame_equal(first.metrics_summary, second.metrics_summary)
            for left, right in zip(
                first.metrics_per_seed.itertuples(),
                second.metrics_per_seed.itertuples(),
                strict=True,
            ):
                left_run, right_run = load_run(left.artifact_dir), load_run(right.artifact_dir)
                pd.testing.assert_frame_equal(
                    left_run.predictions["test"], right_run.predictions["test"]
                )

    def test_invalid_seeds_and_nonrandom_model_fail_before_creating_artifacts(self) -> None:
        """Reject ambiguous seed series and models that cannot vary initialization.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = gru_config(root)
            invalid_seeds: list[Any] = [[], [1, 1], [-1], [2**32], [True], [1.5], ["2"]]
            for seeds in invalid_seeds:
                with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                    run_seed_experiments(config, seeds=seeds)
            config["pipeline"][-1] = {"name": "model", "type": "ridge"}
            with self.assertRaisesRegex(ValueError, "random_state"):
                run_seed_experiments(config, seeds=[0])
            self.assertFalse((root / "runs").exists())

    def test_ensemble_seed_controls_member_initializations(self) -> None:
        """Apply each series seed to the ensemble's member seed sequence.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            config = gru_config(Path(directory))
            params = config["pipeline"][-1]["params"]
            del params["random_state"]
            config["pipeline"][-1] = {
                "name": "model",
                "type": "gru_ensemble",
                "params": {"n_estimators": 2, "gru_params": params},
            }
            result = run_seed_experiments(config, seeds=[2, 8])
            for row in result.metrics_per_seed.itertuples(index=False):
                run = load_run(row.artifact_dir)
                self.assertEqual(
                    [member.random_state for member in run.model.estimators_],
                    [row.seed, row.seed + 1],
                )
                self.assertIn("test", run.metrics)

    def test_missing_targets_and_test_split_fail_before_training(self) -> None:
        """Require labelled data and a test partition without invoking model training.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            config = gru_config(Path(directory))
            config["split"] = {"train_fraction": 1, "validation_fraction": 0}
            with patch.object(GRURegressor, "fit") as fit:
                with self.assertRaisesRegex(ValueError, "test split"):
                    run_seed_experiments(config, seeds=[0])
                del config["data"]["target_path"]
                del config["data"]["target_column"]
                with self.assertRaisesRegex(ValueError, "target data"):
                    run_seed_experiments(config, seeds=[0])
            fit.assert_not_called()

    def test_partial_series_survives_training_failure(self) -> None:
        """Keep the first seed's saved scores when the next model fails to train.

        Returns:
            None.
        """
        original_fit = GRURegressor.fit

        def fail_second(
            model: GRURegressor,
            X: NDArray[Any],
            y: NDArray[Any],
            validation_data: tuple[NDArray[Any], NDArray[Any]] | None = None,
            *,
            context: StepContext | None = None,
        ) -> GRURegressor:
            """Fail the second seed after a real successful first training run.

            Parameters:
                model (GRURegressor): Current seeded estimator.
                X (NDArray[Any]): Train windows.
                y (NDArray[Any]): Train targets.
                validation_data (tuple[NDArray[Any], NDArray[Any]] | None): Validation pair.
                context (StepContext | None): Active telemetry.

            Returns:
                GRURegressor: Trained first-seed model.
            """
            if model.random_state == 2:
                raise RuntimeError("Training interrupted")
            return original_fit(model, X, y, validation_data, context=context)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = gru_config(root)
            with patch.object(GRURegressor, "fit", fail_second):
                with self.assertRaisesRegex(RuntimeError, "Training interrupted"):
                    run_seed_experiments(config, seeds=[1, 2, 3])
            series_dir = next((root / "runs" / "default").iterdir())
            rows = pd.read_csv(series_dir / "metrics_per_seed.csv")
            self.assertEqual(rows["seed"].tolist(), [1])
            summary = pd.read_csv(series_dir / "metrics_summary.csv")
            self.assertTrue(summary["std"].isna().all())
            self.assertTrue((summary["n_total"] == 1).all())
            manifest = json.loads((series_dir / "series.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["failed_seed"], 2)
            self.assertEqual(manifest["completed_seeds"], [1])
            failure = json.loads((series_dir / "failure.json").read_text())
            self.assertEqual(failure["type"], "RuntimeError")
            self.assertEqual(load_run(rows.iloc[0]["artifact_dir"]).model.random_state, 1)

    def test_no_observed_test_targets_fail_instead_of_silently_skipping_seed(self) -> None:
        """Record unavailable test evaluation as a failed series with no completed scores.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = gru_config(root)
            targets = pd.read_csv(root / "target.csv", index_col=0)
            targets.iloc[48:] = np.nan
            targets.to_csv(root / "target.csv")
            with self.assertRaisesRegex(ValueError, "No observed usable test targets"):
                run_seed_experiments(config, seeds=[0])
            series_dir = next((root / "runs" / "default").iterdir())
            self.assertTrue(pd.read_csv(series_dir / "metrics_per_seed.csv").empty)
            self.assertTrue(pd.read_csv(series_dir / "metrics_summary.csv").empty)
            self.assertEqual(
                json.loads((series_dir / "series.json").read_text())["status"], "failed"
            )

    def test_default_ten_seeds_without_preprocessing_and_undefined_mape(self) -> None:
        """Run the default seed set for a raw-feature model and retain undefined scores.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = gru_config(root)
            config["pipeline"] = [
                ("model", ModelStep(RandomForestRegressor(n_estimators=2, max_depth=2)))
            ]
            targets = pd.read_csv(root / "target.csv", index_col=0)
            targets.iloc[48] = 0
            targets.to_csv(root / "target.csv")
            result = run_seed_experiments(config)
            self.assertEqual(result.metrics_per_seed["seed"].tolist(), list(range(10)))
            self.assertTrue(result.metrics_per_seed["mape"].isna().all())
            mape = result.metrics_summary.set_index("metric").loc["mape"]
            self.assertTrue(np.isnan(mape["mean"]))
            self.assertTrue(np.isnan(mape["std"]))
            self.assertEqual(mape["n_valid"], 0)
            self.assertEqual(mape["n_total"], 10)
            self.assertFalse((result.artifact_dir / "preprocessing.pkl").exists())
            self.assertIsNone(
                json.loads((result.artifact_dir / "series.json").read_text())["preprocessing_path"]
            )

    def test_disabled_validation_keeps_final_model_weights(self) -> None:
        """Evaluate test without introducing validation-dependent training requirements.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            config = gru_config(Path(directory))
            config["split"] = {"train_fraction": 0.8, "validation_fraction": 0}
            result = run_seed_experiments(config, seeds=[0])
            run = load_run(result.metrics_per_seed.iloc[0]["artifact_dir"])
            self.assertNotIn("validation", run.metrics)
            self.assertIsNone(run.model.best_epoch_)
            self.assertEqual(len(run.model.history_), 2)
            self.assertTrue(result.metrics_summary["std"].isna().all())
