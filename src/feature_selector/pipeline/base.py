"""Uniform transformation contract and optional inverse prediction handling."""

from typing import Any, Self

from sklearn.base import BaseEstimator

from ..tracking import StepContext
from .data import PipelineData, PredictionData


class Step(BaseEstimator):
    """Base for steps; stateless fitting and unrelated prediction restoration are identities."""

    input_names_: tuple[Any, ...]

    def fit(
        self,
        train: PipelineData,
        validation: PipelineData | None = None,
        *,
        context: StepContext | None = None,
    ) -> Self:
        """Accept training data for a stateless transformation.

        Parameters:
            train (PipelineData): Current training representation, possibly with missing y.
            validation (PipelineData | None): Current validation representation.
            context (StepContext | None): Execution-scoped diagnostics.

        Returns:
            Self: Ready step.
        """
        return self

    def transform(
        self,
        data: PipelineData,
        *,
        context: StepContext | None = None,
    ) -> PipelineData:
        """Pass aligned values to the next step.

        Parameters:
            data (PipelineData): Current values and temporal metadata.
            context (StepContext | None): Execution-scoped diagnostics.

        Returns:
            PipelineData: Unchanged representation.
        """
        return data

    def inverse_prediction(self, prediction: PredictionData) -> PredictionData:
        """Preserve a prediction whose units this step does not change.

        Parameters:
            prediction (PredictionData): Predictions and reference in current units.

        Returns:
            PredictionData: Unchanged prediction.
        """
        return prediction
