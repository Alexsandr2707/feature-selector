"""Configured exclusions, target alignment and causal window construction."""

from dataclasses import replace

import numpy as np
import pandas as pd

from ..tracking import StepContext
from .base import Step
from .data import PipelineData, PredictionData


class ExcludeIntervals(Step):
    """Mask examples whose history or target intersects excluded intervals."""

    def __init__(self, intervals: list[list[str]] | None = None) -> None:
        """Store inclusive temporal intervals.

        Parameters:
            intervals (list[list[str]] | None): Inclusive timestamp pairs, or no exclusions.

        Returns:
            None.
        """
        self.intervals = intervals

    def transform(
        self,
        data: PipelineData,
        *,
        context: StepContext | None = None,
    ) -> PipelineData:
        """Mask exclusions without compressing the temporal grid or sample order.

        Parameters:
            data (PipelineData): Row or window representation with history boundaries.
            context (StepContext | None): Records affected example counts by split.

        Returns:
            PipelineData: Copied values with accumulated exclusion flags and NaN masks.
        """
        samples = data.samples.copy()
        mask = samples["excluded"].to_numpy(dtype=bool).copy()
        input_mask = samples["input_excluded"].to_numpy(dtype=bool).copy()
        for start, end in self.intervals or []:
            start, end = pd.Timestamp(start), pd.Timestamp(end)
            if start > end:
                raise ValueError("Excluded interval requires start <= end")
            history = (samples["history_start"] <= end) & (samples["history_end"] >= start)
            target = samples["target_time"].between(start, end)
            input_mask |= history.to_numpy()
            mask |= (history | target).to_numpy()
        samples["excluded"] = mask
        samples["input_excluded"] = input_mask
        X = data.X.astype(float).copy()
        if isinstance(X, pd.DataFrame):
            X.iloc[input_mask] = np.nan
        else:
            X[input_mask] = np.nan
        y = None if data.y is None else np.array(data.y, dtype=float, copy=True)
        if y is not None:
            y[mask] = np.nan
        if context:
            context.log_metadata(
                {
                    "excluded": {
                        name: int((mask & (samples["split"] == name)).sum())
                        for name in ("train", "validation", "test")
                    }
                }
            )
        return replace(data, X=X, y=y, samples=samples)


class AlignTarget(Step):
    """Align a future target with each row before feature selection or windowing."""

    def __init__(self, horizon: int = 1) -> None:
        """Store the target distance on the regular grid.

        Parameters:
            horizon (int): Nonnegative number of sampling intervals.

        Returns:
            None.
        """
        self.horizon = horizon

    def transform(
        self,
        data: PipelineData,
        *,
        context: StepContext | None = None,
    ) -> PipelineData:
        """Shift targets and exclusions, preserving the input rows and their history.

        Parameters:
            data (PipelineData): Regular rows before window construction.
            context (StepContext | None): Execution diagnostics.

        Returns:
            PipelineData: Future targets and split labels associated with each origin.
        """
        if self.horizon < 0 or not isinstance(data.X, pd.DataFrame):
            raise ValueError("align_target requires rows and a nonnegative horizon")
        samples = data.samples.copy()
        samples["target_time"] = samples["target_time"] + self.horizon * data.temporal.frequency
        target_times = pd.DatetimeIndex(samples["target_time"])
        samples["split"] = data.temporal.boundaries.labels(target_times)
        future_excluded = (
            pd.Series(samples["input_excluded"].to_numpy())
            .shift(-self.horizon, fill_value=False)
            .to_numpy(dtype=bool)
        )
        samples["excluded"] |= future_excluded
        y = (
            None
            if data.y is None
            else pd.Series(np.asarray(data.y)).shift(-self.horizon).to_numpy()
        )
        return replace(data, y=y, samples=samples)


class Windows(Step):
    """Collect complete feature histories while preserving missing external targets."""

    def __init__(self, lookback: int = 48) -> None:
        """Store causal history length.

        Parameters:
            lookback (int): Positive number of consecutive regular rows.

        Returns:
            None.
        """
        self.lookback = lookback

    def transform(
        self,
        data: PipelineData,
        *,
        context: StepContext | None = None,
    ) -> PipelineData:
        """Create finite windows and retain metadata for their final input rows.

        Parameters:
            data (PipelineData): DataFrame on a regular grid; y may be absent or sparse.
            context (StepContext | None): Receives rejection diagnostics by split.

        Returns:
            PipelineData: Three-dimensional windows, endpoint targets and temporal metadata.
        """
        if not isinstance(data.X, pd.DataFrame) or self.lookback < 1:
            raise ValueError("windows requires a DataFrame and positive lookback")
        times = pd.DatetimeIndex(data.X.index)
        regular = np.diff(times.as_unit("ns").asi8) == data.temporal.frequency.value  # type:ignore
        if len(times) > 1 and not regular.all():
            raise ValueError("windows requires a regular uncompressed grid")
        values = data.X.to_numpy(dtype=float)
        excluded = data.samples["excluded"].to_numpy(dtype=bool)
        input_excluded = data.samples["input_excluded"].to_numpy(dtype=bool)
        # History exclusions are measured at feature timestamps, not shifted target times.
        missing = ~np.isfinite(values).all(axis=1)
        positions: list[int] = []
        diagnostics: dict[str, dict[str, int]] = {}
        for position in range(len(values)):
            split = str(data.samples.iloc[position]["split"])
            counts = diagnostics.setdefault(
                split,
                {
                    "candidates": 0,
                    "insufficient_history": 0,
                    "excluded": 0,
                    "missing_features": 0,
                    "accepted": 0,
                },
            )
            counts["candidates"] += 1
            start = position - self.lookback + 1
            if start < 0:
                counts["insufficient_history"] += 1
            elif excluded[position] or input_excluded[start : position + 1].any():
                counts["excluded"] += 1
            elif missing[start : position + 1].any():
                counts["missing_features"] += 1
            else:
                counts["accepted"] += 1
                positions.append(position)
        indices = np.asarray(positions, dtype=np.intp)
        selected = data.select(indices)
        X = (
            np.stack([values[p - self.lookback + 1 : p + 1] for p in positions])
            if positions
            else (np.empty((0, self.lookback, values.shape[1])))
        )
        samples = selected.samples.copy()
        samples["history_start"] = times[indices - self.lookback + 1]
        samples["history_end"] = times[indices]
        if context:
            context.log_metadata({"windows": diagnostics})
        return replace(selected, X=X, samples=samples)

    def inverse_prediction(self, prediction: PredictionData) -> PredictionData:
        """Select reconstruction endpoints while leaving scalar predictions unchanged.

        Parameters:
            prediction (PredictionData): Scalar targets or reconstructed windows.

        Returns:
            PredictionData: Endpoint feature values for reconstruction reporting.
        """
        if prediction.kind != "features" or prediction.values.ndim != 3:
            return prediction
        truth = None if prediction.truth is None else prediction.truth[:, -1, :]
        return replace(prediction, values=prediction.values[:, -1, :], truth=truth)
