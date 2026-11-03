"""Legacy target masks preserve the temporal grid and feature histories."""

from dataclasses import replace
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from feature_selector import run_experiment
from feature_selector.data import TemporalData
from feature_selector.factory import build_pipeline
from feature_selector.pipeline.data import PipelineData, prepare_data
from feature_selector.pipeline.filter import TargetFilter
from feature_selector.pipeline.adapters import TargetScale
from feature_selector.pipeline.temporal import AlignTarget, Windows
from feature_selector.tracking import RunTracker, StepContext


def weekly_data() -> PipelineData:
    """Create four weeks of daily targets with a single change between weekly means.

    Returns:
        PipelineData: Complete features and targets changing from zero to 100 halfway through.
    """
    times = pd.date_range("2024-01-01", periods=28, freq="1D")
    X = pd.DataFrame({"a": np.arange(28, dtype=float)}, index=times)
    y = pd.Series(np.repeat([0.0, 0.0, 100.0, 100.0], 7), index=times, name="target")
    source = TemporalData(X, y, pd.Series(False, index=times))
    return prepare_data(source, {}, frequency="1D")


class TargetFilterTests(unittest.TestCase):
    def test_known_legacy_mask_and_strict_threshold(self) -> None:
        """Mask the nearest weekly-change region with a strictly greater comparison.

        Returns:
            None.
        """
        data = weekly_data()
        result = TargetFilter(max_diff=30).transform(data)
        expected = np.zeros(28, dtype=bool)
        expected[17:24] = True
        np.testing.assert_array_equal(pd.isna(result.y), expected)
        np.testing.assert_array_equal(
            np.asarray(result.y)[~expected], np.asarray(data.y)[~expected]
        )
        unchanged = TargetFilter(max_diff=100).transform(data)
        np.testing.assert_array_equal(unchanged.y, data.y)

    def test_containers_metadata_and_missing_targets(self) -> None:
        """Keep Series identity metadata and array containers without filling missing targets.

        Returns:
            None.
        """
        data = weekly_data()
        original = data.y.copy()
        for targets in (data.y, np.asarray(data.y)):
            with self.subTest(container=type(targets)):
                result = TargetFilter(max_diff=30).transform(replace(data, y=targets))
                self.assertIs(result.X, data.X)
                self.assertIs(result.samples, data.samples)
                self.assertIs(result.temporal, data.temporal)
                self.assertEqual(result.feature_names, data.feature_names)
                self.assertEqual(len(result.y), len(data.y))
                self.assertIsInstance(result.y, type(targets))
                if isinstance(result.y, pd.Series):
                    pd.testing.assert_index_equal(result.y.index, data.X.index)
                    self.assertEqual(result.y.name, "target")
        pd.testing.assert_series_equal(data.y, original)
        sparse = np.asarray(data.y).copy()
        sparse[[0, 10, 21]] = np.nan
        result = TargetFilter(max_diff=30).transform(replace(data, y=sparse))
        self.assertTrue(np.isnan(np.asarray(result.y)[[0, 10, 21]]).all())
        unlabelled = replace(data, y=None)
        self.assertIs(TargetFilter().transform(unlabelled), unlabelled)

    def test_short_empty_and_missing_batches(self) -> None:
        """Leave batches without a computable change unmasked.

        Returns:
            None.
        """
        data = weekly_data()
        for positions in (np.arange(0), np.arange(1), np.arange(3)):
            with self.subTest(size=len(positions)):
                selected = data.select(positions)
                result = TargetFilter().transform(selected)
                np.testing.assert_array_equal(result.y, selected.y)
        missing = replace(data, y=np.full(28, np.nan))
        result = TargetFilter().transform(missing)
        self.assertTrue(pd.isna(result.y).all())

    def test_target_timestamps_after_alignment(self) -> None:
        """Associate shifted targets with their actual dates rather than feature dates.

        Returns:
            None.
        """
        data = weekly_data()
        before = AlignTarget(2).transform(TargetFilter(max_diff=30).transform(data))
        after = TargetFilter(max_diff=30).transform(AlignTarget(2).transform(data))
        np.testing.assert_allclose(before.y, after.y, equal_nan=True)
        self.assertIs(after.X, data.X)

    def test_filtered_targets_do_not_remove_feature_windows(self) -> None:
        """Retain windows with masked y for unlabelled prediction and downstream fit selection.

        Returns:
            None.
        """
        data = weekly_data()
        result = Windows(3).transform(TargetFilter(max_diff=30).transform(data))
        self.assertEqual(len(result.X), 26)
        self.assertEqual(int(pd.isna(result.y).sum()), 7)
        self.assertTrue(np.isfinite(result.X).all())
        self.assertEqual(int(result.valid(require_y=True).sum()), 19)

    def test_pipeline_cloning_reload_and_unlabelled_inference(self) -> None:
        """Serialize a stateless filter and preserve prediction inputs when references are absent.

        Returns:
            None.
        """
        pipeline = build_pipeline([
            {"name": "filter", "type": "filter", "params": {"max_diff": 30}},
            {"name": "windows", "type": "windows", "params": {"lookback": 3}},
            {"name": "model", "type": "ridge"},
        ])
        data = weekly_data()
        cloned = clone(pipeline)
        self.assertIsNot(cloned["filter"], pipeline["filter"])
        pipeline.fit(data)
        expected = pipeline.predict(data)
        unlabelled = pipeline.predict(replace(data, y=None))
        np.testing.assert_array_equal(expected.values, unlabelled.values)
        self.assertIsNone(unlabelled.truth)
        self.assertEqual(int(pd.isna(expected.truth).sum()), 7)
        restored = pickle.loads(pickle.dumps(pipeline))
        np.testing.assert_array_equal(restored.predict(data).values, expected.values)

    def test_yaml_filter_precedes_target_scaling_and_fit_selection(self) -> None:
        """Fit scaling and regression on cleaned targets and retain masked report rows.

        Returns:
            None.
        """
        data = weekly_data()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assert isinstance(data.X, pd.DataFrame)
            assert isinstance(data.y, pd.Series)
            data.X.to_csv(root / "features.csv")
            data.y.to_frame().to_csv(root / "targets.csv")
            settings = {
                "data": {"features_path": "features.csv", "target_path": "targets.csv",
                         "target_column": "target", "freq": "1D"},
                "split": {"train_fraction": 0.75, "validation_fraction": 0.15},
                "pipeline": [
                    {"name": "filter", "type": "filter", "params": {"max_diff": 30}},
                    {"name": "windows", "type": "windows", "params": {"lookback": 3}},
                    {"name": "target_scaler", "type": "target_scale"},
                    {"name": "model", "type": "ridge"},
                ],
                "run": {"output_dir": "runs"},
            }
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump(settings), encoding="utf-8")
            run = run_experiment(path)
            scaler_step = run.pipeline["target_scaler"]
            assert isinstance(scaler_step, TargetScale)
            assert isinstance(scaler_step.scaler, StandardScaler)
            np.testing.assert_allclose(scaler_step.scaler.mean_, [20])
            self.assertEqual(run.metrics["train"]["n"], 15)
            self.assertEqual(run.metrics["validation"]["n"], 1)
            self.assertEqual(len(run.predictions["validation"]), 4)
            self.assertEqual(int(run.predictions["validation"].y_true.isna().sum()), 3)
            self.assertTrue(np.isfinite(run.predictions["validation"].y_pred).all())
            self.assertEqual(run.metadata["steps"]["pipeline/filter"]["masked_targets"],
                             {"train": 4, "validation": 3, "test": 0})
            self.assertFalse(run.metadata["test_evaluated"])

    def test_metadata_counts_only_newly_masked_targets(self) -> None:
        """Record split-specific removal counts without target arrays or sibling metadata changes.

        Returns:
            None.
        """
        data = weekly_data()
        targets = np.asarray(data.y).copy()
        targets[21] = np.nan
        data = replace(data, y=targets)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracker = RunTracker(root)
            context = StepContext(tracker, operation="transform")
            try:
                context.child("other").log_metadata({"kept": True})
                result = TargetFilter(max_diff=30).transform(data, context=context.child("filter"))
            finally:
                tracker.close()
            removed = pd.notna(data.y) & pd.isna(result.y)
            expected = {
                split: int(np.count_nonzero(removed & data.samples["split"].eq(split).to_numpy()))
                for split in ("train", "validation", "test")
            }
            self.assertEqual(sum(expected.values()), 6)
            self.assertEqual(tracker.metadata["pipeline/filter"]["masked_targets"], expected)
            self.assertEqual(tracker.metadata["pipeline/other"], {"kept": True})
            records = [json.loads(line) for line in
                       (root / "events.jsonl").read_text().splitlines()]
            self.assertEqual(records[-1]["event"], "metadata")
            self.assertEqual(records[-1]["path"], "pipeline/filter")
            self.assertEqual(records[-1]["data"]["operation"], "transform")

    def test_invalid_thresholds(self) -> None:
        """Reject nonfinite and negative limits rather than silently disabling masks.

        Returns:
            None.
        """
        for limit in (-1, float("inf"), float("nan")):
            with self.subTest(max_diff=limit), self.assertRaises(ValueError):
                TargetFilter(max_diff=limit).transform(weekly_data())
