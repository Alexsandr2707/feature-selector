"""Temporal steps preserve sample alignment, gaps and sparse targets."""

from dataclasses import replace
import unittest

import numpy as np
import pandas as pd

from feature_selector.pipeline.temporal import AlignTarget, ExcludeIntervals, Windows
from support import sample_data


class TemporalStepTests(unittest.TestCase):
    def test_known_windows_and_future_alignment(self) -> None:
        """Assign windows by future target time and retain earlier split history.

        Returns:
            None.
        """
        data = sample_data(20)
        aligned = AlignTarget(2).transform(data)
        windows = Windows(4).transform(aligned)
        first = windows.select(np.array([0]))
        np.testing.assert_array_equal(first.X[0, :, 0], [0, 1, 2, 3])
        self.assertEqual(first.y[0], data.y.iloc[5])
        self.assertEqual(first.samples.iloc[0].target_time, data.X.index[5])
        validation = windows.split("validation")
        self.assertEqual(validation.samples.iloc[0].target_time, data.X.index[12])
        self.assertEqual(validation.samples.iloc[0].origin_time, data.X.index[10])
        np.testing.assert_array_equal(validation.X[0, :, 0], [7, 8, 9, 10])

    def test_sparse_targets_do_not_remove_windows(self) -> None:
        """Leave target selection to downstream algorithms.

        Returns:
            None.
        """
        windows = Windows(5).transform(sample_data(sparse=True))
        self.assertEqual(len(windows.X), 56)
        self.assertTrue(np.isnan(windows.y).any())
        without_y = Windows(5).transform(replace(sample_data(), y=None))
        self.assertIsNone(without_y.y)
        self.assertEqual(len(without_y.X), 56)

    def test_exclusions_are_idempotent_and_preserve_grid(self) -> None:
        """Mask rows without joining history across exclusions.

        Returns:
            None.
        """
        data = sample_data(20)
        step = ExcludeIntervals([[str(data.X.index[6]), str(data.X.index[7])]])
        excluded = step.transform(data)
        pd.testing.assert_frame_equal(excluded.X, step.transform(excluded).X)
        pd.testing.assert_index_equal(excluded.X.index, data.X.index)
        windows = Windows(3).transform(excluded)
        rejected = set(data.X.index[6:10])
        self.assertTrue(rejected.isdisjoint(windows.samples.origin_time))
        self.assertTrue(np.isfinite(data.X.to_numpy()).all())

    def test_exclusion_after_windows_checks_history(self) -> None:
        """Mark windows whose histories intersect excluded intervals.

        Returns:
            None.
        """
        data = sample_data(20)
        windows = Windows(3).transform(data)
        step = ExcludeIntervals([[str(data.X.index[6]), str(data.X.index[6])]])
        result = step.transform(windows)
        origins = result.samples.loc[result.samples.excluded, "origin_time"]
        self.assertEqual(origins.tolist(), data.X.index[6:9].tolist())

    def test_missing_features_do_not_compress_time(self) -> None:
        """Reject histories containing missing observations.

        Returns:
            None.
        """
        data = sample_data(20)
        X = data.X.copy()
        X.iloc[7] = np.nan
        windows = Windows(3).transform(replace(data, X=X))
        self.assertNotIn(data.X.index[9], windows.samples.origin_time.tolist())
        self.assertIn(data.X.index[10], windows.samples.origin_time.tolist())

    def test_future_prediction_without_target(self) -> None:
        """Retain final input rows when forecasting beyond the observed grid.

        Returns:
            None.
        """
        data = replace(sample_data(20), y=None)
        result = Windows(3).transform(AlignTarget(2).transform(data))
        self.assertEqual(result.samples.iloc[-1].target_time, data.X.index[-1] + pd.Timedelta('2h'))
        self.assertIsNone(result.y)

    def test_empty_windows_and_invalid_grid(self) -> None:
        """Preserve empty trailing shapes and reject compressed irregular input.

        Returns:
            None.
        """
        data = sample_data(20)
        self.assertEqual(Windows(30).transform(data).X.shape, (0, 30, 2))
        with self.assertRaises(ValueError):
            Windows(3).transform(data.select(np.array([0, 2, 3])))

    def test_target_exclusion_does_not_contaminate_feature_history(self) -> None:
        """Exclude future targets without masking otherwise valid origin features.

        Returns:
            None.
        """
        data = sample_data(20)
        step = ExcludeIntervals([[str(data.X.index[8]), str(data.X.index[8])]])
        excluded = step.transform(AlignTarget(2).transform(data))
        self.assertTrue(np.isfinite(excluded.X.iloc[6]).all())
        self.assertTrue(excluded.samples.iloc[6].excluded)
        windows = Windows(3).transform(excluded)
        self.assertIn(data.X.index[7], windows.samples.origin_time.tolist())
        self.assertNotIn(data.X.index[6], windows.samples.origin_time.tolist())
