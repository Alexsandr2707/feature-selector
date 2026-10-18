"""Fixed temporal split boundaries independent of pipeline representations."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .data import TimestampInput


@dataclass(frozen=True)
class Boundaries:
    """Inclusive split starts; equal starts omit validation, starts past the grid omit test."""

    validation_start: pd.Timestamp
    test_start: pd.Timestamp

    def labels(self, index: pd.DatetimeIndex) -> NDArray[np.str_]:
        """Assign each timestamp to its time range.

        Parameters:
            index (pd.DatetimeIndex): Timestamps with a timezone compatible with the boundaries.

        Returns:
            NDArray[np.str_]: Train, validation or test labels in input order.
        """
        return np.where(
            index < self.validation_start,
            "train",
            np.where(index < self.test_start, "validation", "test"),
        )


def split_boundaries(
    index: pd.DatetimeIndex,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    validation_start: TimestampInput | None = None,
    test_start: TimestampInput | None = None,
) -> Boundaries:
    """Choose time ranges, allowing zero validation or remaining test fractions.

    Parameters:
        index (pd.DatetimeIndex): Sorted, unique regular timestamps with at least two entries.
        train_fraction (float): Training fraction used when explicit starts are absent.
        validation_fraction (float): Nonnegative validation fraction. Train must be positive;
            their sum may equal one to omit test. Train may equal one to omit both.
        validation_start (TimestampInput | None): Validation start; provide both starts
            or neither. Explicit starts override fractions.
        test_start (TimestampInput | None): Test start, after validation starts and no
            later than the last timestamp. Both starts must match the index timezone.

    Returns:
        Boundaries: Split starts, with disabled ranges represented by equal or past-grid starts.
    """
    if (validation_start is None) != (test_start is None):
        raise ValueError("Provide both validation_start and test_start")

    if validation_start is None or test_start is None:
        if not (
            0 < train_fraction <= 1
            and 0 <= validation_fraction < 1
            and train_fraction + validation_fraction <= 1
        ):
            raise ValueError("Require train > 0, validation >= 0 and their sum <= 1")

        validation_position = int(len(index) * train_fraction)
        test_position = int(len(index) * (train_fraction + validation_fraction))
        if validation_position == 0 or (
            validation_fraction > 0 and test_position == validation_position
        ):
            raise ValueError("Not enough timestamps for the requested splits")

        end = index[-1] + (index[1] - index[0])
        validation_start = index[validation_position] if validation_position < len(index) else end
        test_start = index[test_position] if test_position < len(index) else end

        return Boundaries(validation_start, test_start)

    bounds = Boundaries(pd.Timestamp(validation_start), pd.Timestamp(test_start))

    if not index[0] < bounds.validation_start < bounds.test_start <= index[-1]:
        raise ValueError("Split boundaries must create three ordered, nonempty time ranges")

    return bounds
