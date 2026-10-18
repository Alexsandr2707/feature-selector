"""Small deterministic temporal inputs shared by architecture integration tests."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from feature_selector.data import FeatureData, TemporalData
from feature_selector.pipeline import PipelineData
from feature_selector.pipeline.data import prepare_data


def sample_data(size: int = 60, sparse: bool = False) -> PipelineData:
    """Create a regular two-feature grid with optional sparse external targets.

    Parameters:
        size (int): Number of hourly rows.
        sparse (bool): Observe every fifth target only.

    Returns:
        PipelineData: Raw grid split into train, validation and test.
    """
    times = pd.date_range("2024-01-01", periods=size, freq="1h")
    values = np.arange(size, dtype=float)
    X = pd.DataFrame({"a": values, "b": np.sin(values / 4)}, index=times)
    y = pd.Series(values * 2 + 1, index=times)
    if sparse:
        y.iloc[np.arange(size) % 5 != 0] = np.nan
    return prepare_data(TemporalData(X, y, pd.Series(False, index=times)), {})


def write_config(root: Path, *, sparse: bool = False) -> dict[str, Any]:
    """Write deterministic input CSVs and return a forecast configuration.

    Parameters:
        root (Path): Existing temporary directory.
        sparse (bool): Save sparse targets.

    Returns:
        dict[str, Any]: Current-schema Ridge experiment settings.
    """
    data = sample_data(sparse=sparse)
    data.X.to_csv(root / "features.csv")
    pd.DataFrame({"y": data.y}).to_csv(root / "target.csv")
    return {"data": {"features_path": str(root / "features.csv"),
                     "target_path": str(root / "target.csv"), "target_column": "y"},
            "pipeline": [{"name": "scaler", "type": "standard"},
                         {"name": "align", "type": "align_target", "params": {"horizon": 1}},
                         {"name": "windows", "type": "windows", "params": {"lookback": 5}},
                         {"name": "target_scaler", "type": "target_scale"},
                         {"name": "model", "type": "ridge"}],
            "run": {"output_dir": str(root / "runs")}}
