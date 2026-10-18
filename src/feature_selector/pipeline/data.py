"""Aligned values and temporal metadata passed through every pipeline step."""

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..data import FeatureData, TemporalData
from ..temporal import Boundaries

type Values = pd.DataFrame | NDArray[Any]
type Targets = pd.Series | NDArray[Any] | None


@dataclass(frozen=True)
class TemporalContext:
    """Sampling interval and fixed temporal boundaries shared by all representations."""

    frequency: pd.Timedelta
    boundaries: Boundaries


@dataclass(frozen=True)
class PipelineData:
    """Values with one metadata row per example; y may be missing or partly unobserved."""

    X: Values
    y: Targets
    samples: pd.DataFrame
    feature_names: tuple[Any, ...]
    temporal: TemporalContext

    def select(self, positions: NDArray[Any]) -> "PipelineData":
        """Select examples consistently across values, targets and metadata.

        Parameters:
            positions (NDArray[Any]): Boolean mask or integer positions in current order.

        Returns:
            PipelineData: Selected examples preserving their temporal context.
        """
        X = self.X.iloc[positions] if isinstance(self.X, pd.DataFrame) else self.X[positions]
        y = self.y
        if isinstance(y, pd.Series):
            y = y.iloc[positions]
        elif y is not None:
            y = y[positions]
        return replace(self, X=X, y=y, samples=self.samples.iloc[positions].reset_index(drop=True))

    def split(self, name: str) -> "PipelineData":
        """Select the examples assigned to one temporal split.

        Parameters:
            name (str): Train, validation or test.

        Returns:
            PipelineData: Examples in unchanged order, including missing values.
        """
        return self.select(self.samples["split"].to_numpy() == name)

    def valid(self, *, require_y: bool = False) -> NDArray[np.bool_]:
        """Identify usable numeric examples at an algorithm's input boundary.

        Parameters:
            require_y (bool): Require a finite external target for each example.

        Returns:
            NDArray[np.bool_]: Complete nonexcluded examples; missing targets are optional.
        """
        values = np.asarray(self.X)
        valid = (
            np.isfinite(values).reshape(len(values), -1).all(axis=1)
            if len(values)
            else (np.zeros(0, dtype=bool))
        )
        valid &= ~self.samples["excluded"].to_numpy(dtype=bool)
        if require_y:
            if self.y is None:
                return np.zeros(len(values), dtype=bool)
            valid &= np.isfinite(np.asarray(self.y))
        return valid


@dataclass(frozen=True)
class PredictionData:
    """Predictions and optional reference values aligned with sample metadata."""

    values: NDArray[Any]
    truth: NDArray[Any] | None
    samples: pd.DataFrame
    feature_names: tuple[Any, ...]
    kind: Literal["target", "features", "latent"]


def prepare_data(
    source: FeatureData | TemporalData,
    split: dict[str, Any] | Boundaries,
    frequency: str = "1h",
) -> PipelineData:
    """Attach sample metadata to a loaded regular temporal grid.

    Parameters:
        source (FeatureData | TemporalData): Loaded regular feature grid and optional targets.
        split (dict[str, Any] | Boundaries): Fraction/date settings or fitted boundaries.
        frequency (str): Fixed sampling interval matching the loaded grid.

    Returns:
        PipelineData: Raw row representation with aligned temporal metadata.
    """
    from ..temporal import split_boundaries

    times = pd.DatetimeIndex(source.X.index)
    boundaries = split if isinstance(split, Boundaries) else split_boundaries(times, **split)
    samples = pd.DataFrame(
        {
            "origin_time": times,
            "target_time": times,
            "history_start": times,
            "history_end": times,
            "split": boundaries.labels(times),
            "excluded": source.excluded.to_numpy(dtype=bool),
            "input_excluded": source.excluded.to_numpy(dtype=bool),
        }
    )
    return PipelineData(
        source.X,
        getattr(source, "y", None),
        samples,
        tuple(source.X.columns),
        TemporalContext(pd.Timedelta(frequency), boundaries),
    )
