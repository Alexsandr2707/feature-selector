"""Adapt sklearn Ridge regression to batches of fixed-shape temporal windows."""

from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted


class WindowRidge(RegressorMixin, BaseEstimator):
    """Flatten each input window and fit a single-target Ridge regressor.

    Accepts temporal windows or feature matrices. Trailing dimensions are
    flattened in C order and must match between training and prediction.
    """

    def __init__(
        self,
        alpha: float | NDArray[Any] = 1.0,
        fit_intercept: bool = True,
    ) -> None:
        """Configure regularization and intercept fitting.

        Parameters:
            alpha (float | NDArray[Any]): Nonnegative regularization strength.
                Ridge also accepts a one-element array for a single target.
            fit_intercept (bool): Whether to fit an intercept.

        Returns:
            None.
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(
        self,
        X: NDArray[Any],
        y: ArrayLike,
        validation_data: tuple[NDArray[Any], ArrayLike] | None = None,
    ) -> Self:
        """Fit Ridge using flattened windows and a single target per sample.

        Parameters:
            X (NDArray[Any]): Numeric input shaped (samples, *window_shape).
            y (ArrayLike): One numeric target per sample; flattened before fitting.
            validation_data (tuple[NDArray[Any], ArrayLike] | None): Ignored;
                accepted for compatibility with other experiment models.

        Returns:
            Self: Fitted estimator with an empty training history.
        """
        self.window_shape_ = X.shape[1:]
        self.model_ = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)

        # Preserve the time/feature order while turning each window into one row.
        flattened_windows = X.reshape(len(X), -1)
        targets = np.asarray(y).reshape(-1)
        self.model_.fit(flattened_windows, targets)

        # Ridge has no epochs; retain the shared experiment history interface.
        self.history_: list[dict[str, Any]] = []
        return self

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        """Predict one target per window using the fitted Ridge model.

        Parameters:
            X (NDArray[Any]): Numeric windows with the same trailing shape
                as the training input.

        Returns:
            NDArray[Any]: Predicted target values with shape (samples,).
        """
        check_is_fitted(self, "model_")
        if X.shape[1:] != self.window_shape_:
            raise ValueError("Prediction window shape must match fit")

        flattened_windows = X.reshape(len(X), -1)
        return self.model_.predict(flattened_windows)
