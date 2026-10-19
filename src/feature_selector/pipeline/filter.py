"""Legacy period-change filtering of external targets on their temporal grid."""

from dataclasses import replace

import numpy as np
import pandas as pd

from ..tracking import StepContext
from .base import Step
from .data import PipelineData, Targets


class TargetFilter(Step):
    """Mask targets using changes between neighboring period means, including future values."""

    def __init__(self, filter_freq: str = "1W", max_diff: float = 20.0) -> None:
        """Configure the legacy target filter without learning parameters.

        Parameters:
            filter_freq (str): Pandas resampling rule for period means, such as "1W".
            max_diff (float): Finite nonnegative limit on the absolute change between
                neighboring means, in the current target units.

        Returns:
            None.
        """
        self.filter_freq = filter_freq
        self.max_diff = max_diff

    def _validate_params(self) -> None:
        """Reject thresholds that cannot define a meaningful change limit.

        Returns:
            None.
        """
        if not np.isfinite(self.max_diff) or self.max_diff < 0:
            raise ValueError("max_diff must be finite and nonnegative")

    def transform(
        self,
        data: PipelineData,
        *,
        context: StepContext | None = None,
    ) -> PipelineData:
        """Mask y using nearest-interpolated absolute differences of period means.

        Parameters:
            data (PipelineData): Aligned scalar targets and target timestamps. Place after
                exclusions and before target alignment, windows and target scaling to
                filter the complete target grid in its original units. With y=None,
                the step passes data through. The supplied batch may span multiple splits.
            context (StepContext | None): Receives newly masked target counts by split.

        Returns:
            PipelineData: Targets above the change limit become NaN; rows, X and temporal
                metadata are retained. Series targets keep their index and name. The filter
                uses future values and recomputes statistics on each supplied batch.
        """
        self._validate_params()
        if data.y is None:
            return data

        # Target timestamps remain correct even if align_target has already shifted y.
        times = pd.DatetimeIndex(data.samples["target_time"])
        values = pd.Series(np.asarray(data.y, dtype=float), index=times)
        changes = (
            values.resample(self.filter_freq)
            .mean()
            .diff()
            .abs()
            .resample(data.temporal.frequency)
            .first()
            .interpolate(method="nearest")
        )
        # Resampling extends to period labels; only mask the original target positions.
        mask = changes.gt(self.max_diff).reindex(times, fill_value=False).to_numpy(dtype=bool)
        targets: Targets
        if isinstance(data.y, pd.Series):
            targets = data.y.mask(mask)
        else:
            targets = np.array(data.y, dtype=float, copy=True)
            targets[mask] = np.nan

        if context:
            removed = mask & values.notna().to_numpy()
            labels = data.samples["split"].to_numpy()
            context.log_metadata({
                "filter_freq": self.filter_freq,
                "max_diff": self.max_diff,
                "masked_targets": {
                    split: int(np.count_nonzero(removed & (labels == split)))
                    for split in ("train", "validation", "test")
                },
            })
        return replace(data, y=targets)
