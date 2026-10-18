"""CSV loading and regular, uncompressed temporal data."""

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

type TimestampInput = str | date | datetime | pd.Timestamp | np.datetime64 | int | float


@dataclass
class TemporalData:
    """Features, target and exclusion mask sharing a regular DatetimeIndex."""

    X: pd.DataFrame
    y: pd.Series
    excluded: pd.Series


@dataclass
class FeatureData:
    """Features and exclusion mask sharing a regular DatetimeIndex, without a target."""

    X: pd.DataFrame
    excluded: pd.Series


def _read(path: str | Path) -> pd.DataFrame:
    """Read numeric CSV data and sort it by unique timestamps; allow missing values.

    Parameters:
        path (str | Path): CSV with timestamps in the first column and unique data headers.

    Returns:
        pd.DataFrame: Nonempty numeric data with a sorted DatetimeIndex and no infinities.
    """
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream), [])

    # pandas renames duplicate CSV headers, so inspect them before parsing the table.
    if len(header) < 2 or len(set(header[1:])) != len(header[1:]):
        raise ValueError(f"{path}: expected unique feature column names")

    frame = pd.read_csv(path, index_col=0)
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="raise"))
    frame.index = timestamps

    if frame.empty or timestamps.hasnans or timestamps.has_duplicates:
        raise ValueError(f"{path}: expected nonempty data with unique, valid timestamps")
    if frame.columns.has_duplicates:
        raise ValueError(f"{path}: duplicate column names")

    frame = frame.sort_index().apply(pd.to_numeric, errors="raise")
    if np.isinf(frame.to_numpy(dtype=float)).any():
        raise ValueError(f"{path}: infinite values are not supported")

    return frame


def load_data(
    features_path: str | Path,
    target_path: str | Path,
    target_column: str,
    freq: str | pd.Timedelta | pd.DateOffset = "1h",
) -> TemporalData:
    """Align feature and target CSV data on a regular grid without filling missing values.

    Parameters:
        features_path (str | Path): Numeric feature CSV with timestamps in the first column.
        target_path (str | Path): Numeric target CSV with timestamps in the first column.
        target_column (str): Target column to select.
        freq (str | pd.Timedelta | pd.DateOffset): Positive fixed grid duration. Both inputs
            must share a timezone and align with the grid within the feature time range.

    Returns:
        TemporalData: At least three grid timestamps spanning the features, with missing
            rows restored as NaN and an initially false exclusion mask.
    """
    features = _read(features_path)
    targets = _read(target_path)
    if target_column not in targets:
        raise ValueError(f"Unknown target column: {target_column}")

    feature_times = pd.DatetimeIndex(features.index)
    target_times = pd.DatetimeIndex(targets.index)

    if feature_times.tz != target_times.tz:
        raise ValueError("Feature and target timezones must match")

    grid = _feature_grid(features, freq)
    first_time, last_time = grid[0], grid[-1]
    target = targets[target_column].loc[first_time:last_time]
    if not target.index.isin(grid).all():
        raise ValueError("Target timestamps do not align with the requested grid")

    excluded = pd.Series(False, index=grid, name="excluded")
    return TemporalData(X=features.reindex(grid), y=target.reindex(grid), excluded=excluded)


def load_features(
    features_path: str | Path,
    freq: str | pd.Timedelta | pd.DateOffset = "1h",
) -> FeatureData:
    """Load features on a regular temporal grid without requiring a target CSV.

    Parameters:
        features_path (str | Path): Numeric feature CSV with unique timestamps in the
            first column and unique feature headers; missing values are allowed.
        freq (str | pd.Timedelta | pd.DateOffset): Positive fixed grid duration;
            all feature timestamps must align with this grid.

    Returns:
        FeatureData: At least three grid timestamps spanning the features, with missing
            rows restored as NaN and an initially false exclusion mask.
    """
    features = _read(features_path)
    grid = _feature_grid(features, freq)
    excluded = pd.Series(False, index=grid, name="excluded")
    return FeatureData(X=features.reindex(grid), excluded=excluded)


def _feature_grid(
    features: pd.DataFrame, freq: str | pd.Timedelta | pd.DateOffset
) -> pd.DatetimeIndex:
    """Build a fixed grid spanning all feature timestamps without compressing gaps.

    Parameters:
        features (pd.DataFrame): Nonempty features with a sorted, unique DatetimeIndex.
        freq (str | pd.Timedelta | pd.DateOffset): Positive fixed duration compatible
            with every feature timestamp.

    Returns:
        pd.DatetimeIndex: At least three regularly spaced timestamps in the input timezone.
    """
    # Calendar offsets cannot describe a fixed sampling interval.
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
        duration = offset.nanos
    except ValueError as exc:
        raise ValueError("freq must be a positive fixed duration") from exc
    if duration <= 0:
        raise ValueError("freq must be a positive fixed duration")

    feature_times = pd.DatetimeIndex(features.index)
    first_time, last_time = feature_times.min(), feature_times.max()
    grid = pd.date_range(first_time, last_time, freq=offset)
    if len(grid) < 3:
        raise ValueError("At least three grid timestamps are required")

    if not feature_times.isin(grid).all():
        raise ValueError("Feature timestamps do not align with the requested grid")

    return grid
