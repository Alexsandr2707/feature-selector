"""Stack independently initialized GRU predictions with validation-trained Ridge."""

from typing import Any, Self, TypedDict

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted

from ..tracking import StepContext
from .base import EpochHistory
from .gru import GRURegressor


class MemberHistory(TypedDict):
    """One GRU's initialization seed, selected epoch and unmodified epoch records."""

    estimator: int
    random_state: int
    best_epoch: int | None
    history: list[EpochHistory]


class GRUEnsembleRegressor(RegressorMixin, BaseEstimator):
    """Train GRUs on train windows and combine their validation predictions with Ridge."""

    estimators_: list[GRURegressor]
    meta_model_: Ridge
    window_shape_: tuple[int, int]
    history_: list[MemberHistory]

    def __init__(
        self,
        n_estimators: int = 10,
        gru_params: dict[str, Any] | None = None,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        random_state: int = 0,
    ) -> None:
        """Configure the GRU members and the validation-trained combination.

        Parameters:
            n_estimators (int): Positive number of sequentially trained GRUs.
            gru_params (dict[str, Any] | None): Shared GRURegressor constructor parameters;
                random_state is controlled by the ensemble and must be omitted.
                Set patience >= epochs to run all epochs while selecting validation weights.
            alpha (float): Finite nonnegative Ridge regularization strength.
            fit_intercept (bool): Whether Ridge fits an intercept.
            random_state (int): First member's initialization and shuffling seed;
                successive members increment it. Seeds must be accepted by PyTorch.

        Returns:
            None.
        """
        self.n_estimators = n_estimators
        self.gru_params = gru_params
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    def _validate_params(self) -> None:
        """Reject member counts that cannot define a nonempty ensemble.

        Returns:
            None.
        """
        if (
            isinstance(self.n_estimators, bool)
            or not isinstance(self.n_estimators, int)
            or self.n_estimators < 1
        ):
            raise ValueError("n_estimators must be a positive integer")

    def fit(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        validation_data: tuple[NDArray[Any], NDArray[Any]] | None = None,
        *,
        context: StepContext | None = None,
    ) -> Self:
        """Train GRUs on train only, then fit Ridge on all validation predictions.

        Parameters:
            X (NDArray[Any]): Finite nonempty windows shaped (samples, timesteps, features).
            y (NDArray[Any]): Finite targets shaped (samples,).
            validation_data (tuple[NDArray[Any], NDArray[Any]] | None): Required validation
                windows and scalar targets with the same trailing window shape as X.
                Used for GRU validation loss and checkpoint selection, then for Ridge fitting.
                None is rejected because Ridge requires validation predictions.
            context (StepContext | None): Receives separate telemetry for each member
                and Ridge, including combination coefficients and sample counts.

        Returns:
            Self: Fitted ensemble with estimators_, meta_model_, and one history_ entry
                per member containing its seed, epoch history and best epoch.
        """
        self._validate_params()
        if validation_data is None:
            raise ValueError("validation_data is required to train Ridge")
        valid_X, valid_y = validation_data
        if valid_X.shape[1:] != X.shape[1:]:
            raise ValueError("Validation window shape must match train")

        context = context or StepContext()
        context.log_metadata(
            {
                "n_estimators": self.n_estimators,
                "train_samples": len(X),
                "validation_samples": len(valid_X),
            }
        )
        estimators: list[GRURegressor] = []
        histories: list[MemberHistory] = []
        predictions: list[NDArray[Any]] = []
        params = {} if self.gru_params is None else self.gru_params
        for index in range(self.n_estimators):
            seed = self.random_state + index
            estimator = GRURegressor(**params, random_state=seed)
            member_context = context.child(f"gru_{index}")
            member_context.log_metadata(
                {
                    "random_state": seed,
                    "train_samples": len(X),
                    "validation_samples": len(valid_X),
                }
            )
            estimator.fit(
                X,
                y,
                validation_data=validation_data,
                context=member_context,
            )
            estimators.append(estimator)
            predictions.append(estimator.predict(valid_X))
            histories.append(
                {
                    "estimator": index,
                    "random_state": seed,
                    "best_epoch": estimator.best_epoch_,
                    "history": estimator.history_,
                }
            )

        # Columns keep the same member order during fitting and prediction.
        meta_X = np.column_stack(predictions)
        meta_model = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)
        meta_model.fit(meta_X, valid_y)
        context.child("ridge").log_metadata(
            {
                "train_samples": len(valid_X),
                "coefficients": meta_model.coef_,
                "intercept": meta_model.intercept_,
            }
        )

        # Publish fitted state together so a failed fit cannot expose a partial ensemble.
        self.estimators_ = estimators
        self.meta_model_ = meta_model
        self.history_ = histories
        self.window_shape_ = (X.shape[1], X.shape[2])
        return self

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        """Combine member predictions using the validation-trained Ridge coefficients.

        Parameters:
            X (NDArray[Any]): Finite nonempty windows with the trailing shape used in fit.

        Returns:
            NDArray[Any]: Target predictions shaped (samples,) in the training target units.
        """
        check_is_fitted(self, "meta_model_")
        if X.ndim != 3 or X.shape[1:] != self.window_shape_:
            raise ValueError("Prediction window shape must match fit")
        predictions = np.column_stack([estimator.predict(X) for estimator in self.estimators_])
        return self.meta_model_.predict(predictions)
