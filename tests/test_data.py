"""Feature-only CSV loading and shared temporal exclusion contracts."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from feature_selector.data import (
    FeatureData,
    TemporalData,
    load_data,
    load_features,
)


class FeatureDataTests(unittest.TestCase):
    """Check regular-grid loading without requiring a target series."""

    def setUp(self) -> None:
        """Create temporary feature data with an hourly UTC index.

        Returns:
            None.
        """
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.index = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
        self.features = pd.DataFrame({
            "a": np.arange(6, dtype=float), "b": np.arange(6, dtype=float) * 2,
        }, index=self.index)
        self.features.to_csv(self.root / "X.csv")

    def test_feature_only_grid_preserves_gaps(self) -> None:
        """Sort timestamps and restore absent rows without interpolating values.

        Returns:
            None.
        """
        features = self.features.drop(self.index[2]).iloc[::-1].copy()
        features.loc[self.index[1], "b"] = np.nan
        features.to_csv(self.root / "X.csv")
        data = load_features(self.root / "X.csv")
        expected = self.features.copy()
        expected.loc[self.index[2]] = np.nan
        expected.loc[self.index[1], "b"] = np.nan
        self.assertIsInstance(data, FeatureData)
        self.assertFalse(hasattr(data, "y"))
        assert_frame_equal(data.X, expected)
        assert_series_equal(data.excluded, pd.Series(False, index=self.index, name="excluded"))

    def test_forecast_and_reconstruction_share_feature_grid(self) -> None:
        """Keep forecast target alignment while sharing feature grid validation.

        Returns:
            None.
        """
        target_index = pd.date_range(self.index[0] - pd.Timedelta(hours=1), periods=8, freq="h")
        target = pd.Series(np.arange(8, dtype=float), index=target_index, name="target")
        target.drop(self.index[2]).to_csv(self.root / "y.csv")
        reconstruction = load_features(self.root / "X.csv")
        forecast = load_data(self.root / "X.csv", self.root / "y.csv", "target")
        self.assertIsInstance(forecast, TemporalData)
        assert_frame_equal(forecast.X, reconstruction.X)
        assert_series_equal(forecast.excluded, reconstruction.excluded)
        assert_series_equal(forecast.y, target.drop(self.index[2]).reindex(self.index))

    def test_fixed_frequency_and_grid_alignment(self) -> None:
        """Reject nonfixed or misaligned grids and support finer sampling intervals.

        Returns:
            None.
        """
        path = self.root / "X.csv"
        for freq in ("h", pd.Timedelta(hours=1), pd.offsets.Hour()):
            with self.subTest(freq=freq):
                assert_frame_equal(load_features(path, freq=freq).X, self.features)
        finer = load_features(path, freq="30min")
        self.assertEqual(len(finer.X), 11)
        self.assertTrue(finer.X.iloc[1::2].isna().all().all())
        for freq in ("0h", "-1h", "ME", "invalid"):
            with self.subTest(freq=freq), self.assertRaisesRegex(ValueError, "fixed duration"):
                load_features(path, freq=freq)
        off_grid = self.features.copy()
        off_grid.index = self.index[:2].append(
            pd.DatetimeIndex([self.index[2] + pd.Timedelta(minutes=30)])
        ).append(self.index[3:])
        off_grid.to_csv(path)
        with self.assertRaisesRegex(ValueError, "Feature timestamps do not align"):
            load_features(path)
        self.features.iloc[:2].to_csv(path)
        with self.assertRaisesRegex(ValueError, "At least three"):
            load_features(path)

    def test_invalid_feature_csv(self) -> None:
        """Apply existing CSV validation to feature-only datasets.

        Returns:
            None.
        """
        path = self.root / "X.csv"
        invalid = [
            pd.concat([self.features, self.features.iloc[:1]]),
            self.features.assign(a=np.inf),
            self.features.assign(a="not numeric"),
            self.features.iloc[:0],
        ]
        for frame in invalid:
            with self.subTest(shape=frame.shape):
                frame.to_csv(path)
                with self.assertRaises(ValueError):
                    load_features(path)
        duplicate_headers = self.features.copy()
        duplicate_headers.columns = ["a", "a"]
        duplicate_headers.to_csv(path)
        with self.assertRaisesRegex(ValueError, "unique feature column names"):
            load_features(path)
