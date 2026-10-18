"""Sequential fitting and transformation without model-specific orchestration."""

import pickle
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ..tracking import StepContext
from .base import Step
from .data import PipelineData, PredictionData


@dataclass
class TransformArtifact:
    """Fitted transform prefix and its input schema and training provenance."""

    pipeline: "ConfiguredPipeline"
    feature_names: tuple[Any, ...]
    frequency_ns: int
    provenance: dict[str, Any]
    format_version: int = 1


class ConfiguredPipeline(BaseEstimator):
    """Execute configured steps uniformly; only final prediction is a distinct operation."""

    input_names_: tuple[Any, ...]
    frequency_ns_: int
    provenance_: dict[str, Any]
    fitted_: bool

    def __init__(self, steps: list[tuple[str, Step]]) -> None:
        """Store ordered named steps.

        Parameters:
            steps (list[tuple[str, Step]]): Unique step names and component instances.

        Returns:
            None.
        """
        self.steps = steps

    def __getitem__(self, name: str) -> Step:
        """Find a configured step by name.

        Parameters:
            name (str): Exact configured name.

        Returns:
            Step: Matching component.
        """
        return dict(self.steps)[name]

    def _call(
        self,
        step: Step,
        operation: str,
        data: PipelineData,
        context: StepContext,
        validation: PipelineData | None = None,
    ) -> Any:
        """Execute an operation with lifecycle events and original exception context.

        Parameters:
            step (Step): Current component.
            operation (str): Fit, transform or predict.
            data (PipelineData): Operation input.
            context (StepContext): Scoped execution diagnostics.
            validation (PipelineData | None): Validation representation for fit.

        Returns:
            Any: Component result, preserving its declared operation contract.
        """
        started = perf_counter()
        context.log_event("started", {"input_shape": data.X.shape})
        try:
            method = getattr(step, operation, None)
            if not callable(method):
                raise TypeError(f"Step must provide {operation}")
            result = (
                method(data, validation, context=context)
                if operation == "fit"
                else (method(data, context=context))
            )
            if isinstance(result, PipelineData) and (
                len(result.X) != len(result.samples)
                or (result.y is not None and len(result.y) != len(result.X))
            ):
                raise ValueError("Step must keep X, y and sample metadata aligned")
            shape = (
                result.X.shape
                if isinstance(result, PipelineData)
                else (result.values.shape if isinstance(result, PredictionData) else None)
            )
            context.log_event(
                "completed", {"seconds": perf_counter() - started, "output_shape": shape}
            )
            return result
        except Exception as exc:
            context.log_event("failed", {"error": str(exc), "seconds": perf_counter() - started})
            exc.add_note(f"Pipeline step {context.path}, operation {operation}")
            raise

    def fit(self, data: PipelineData, *, context: StepContext | None = None) -> Self:
        """Fit each component on its current train representation.

        Parameters:
            data (PipelineData): Complete regular grid with temporal split labels.
            context (StepContext | None): Run-scoped tracking or no-op execution.

        Returns:
            Self: Fitted pipeline, including its original input schema and provenance.
        """
        context = context or StepContext()
        self.input_names_ = data.feature_names
        self.frequency_ns_ = data.temporal.frequency.value
        self.provenance_ = {
            "validation_start": str(data.temporal.boundaries.validation_start),
            "test_start": str(data.temporal.boundaries.test_start),
        }
        for index, (name, step) in enumerate(self.steps):
            train, validation = data.split("train"), data.split("validation")
            self._call(
                step,
                "fit",
                train,
                context.child(name, "fit"),
                validation if len(validation.X) else None,
            )
            if index < len(self.steps) - 1:
                data = self._call(step, "transform", data, context.child(name, "transform"))
        self.fitted_ = True
        return self

    def transform(
        self,
        data: PipelineData,
        *,
        context: StepContext | None = None,
        through: str | None = None,
    ) -> PipelineData:
        """Apply every step as a transformation, including the final encoder if present.

        Parameters:
            data (PipelineData): Input in the original fitted schema and sampling interval.
            context (StepContext | None): Execution-scoped tracking.
            through (str | None): Last included step name, or all steps.

        Returns:
            PipelineData: Final transformed representation.
        """
        self._check_input(data)
        context = context or StepContext(operation="transform")
        names = [name for name, _ in self.steps]
        stop = names.index(through) + 1 if through is not None else len(names)
        for name, step in self.steps[:stop]:
            data = self._call(step, "transform", data, context.child(name, "transform"))
        return data

    def _check_input(self, data: PipelineData) -> None:
        """Check fitted schema where a changed feature order could silently alter results.

        Parameters:
            data (PipelineData): New input grid.

        Returns:
            None.
        """
        if data.feature_names != self.input_names_:
            raise ValueError("Pipeline feature names and order must match fit")
        if data.temporal.frequency.value != self.frequency_ns_:
            raise ValueError("Pipeline sampling frequency must match fit")

    def predict(
        self,
        data: PipelineData,
        *,
        context: StepContext | None = None,
        splits: tuple[str, ...] | None = None,
    ) -> PredictionData:
        """Reuse the fitted transform prefix and restore model outputs to original units.

        Parameters:
            data (PipelineData): Raw input grid with optional external targets.
            context (StepContext | None): Prediction-scoped tracking.
            splits (tuple[str, ...] | None): Splits to predict after preparing full history.

        Returns:
            PredictionData: Restored predictions, references and temporal metadata.
        """
        self._check_input(data)
        # Lossy transforms cannot recover the original reference by inversion.
        reference = replace(data, X=data.X.copy(), samples=data.samples.copy())
        context = context or StepContext(operation="predict")
        for name, step in self.steps[:-1]:
            data = self._call(step, "transform", data, context.child(name, "transform"))
        if splits is not None:
            data = data.select(data.samples["split"].isin(splits).to_numpy())
        name, model = self.steps[-1]
        result = self._call(model, "predict", data, context.child(name, "predict"))
        for _, step in reversed(self.steps[:-1]):
            result = step.inverse_prediction(result)
        if result.kind == "features":
            result = self._restore_reference(reference, result)
        return result

    @staticmethod
    def _restore_reference(
        source: PipelineData,
        prediction: PredictionData,
    ) -> PredictionData:
        """Attach actual input features after restoring reconstructed predictions.

        Parameters:
            source (PipelineData): Original pipeline inputs before any transformations.
            prediction (PredictionData): Restored features with origin times and feature names.

        Returns:
            PredictionData: Predictions paired with original observations, including endpoints
                when the output represents the last point of each input window.
        """
        origins = pd.Index(source.samples["origin_time"])
        positions = origins.get_indexer(pd.Index(prediction.samples["origin_time"]))
        columns = pd.Index(source.feature_names).get_indexer(pd.Index(prediction.feature_names))
        if (positions < 0).any() or (columns < 0).any():
            raise ValueError("Reconstruction references must match original times and features")
        truth = np.asarray(source.X)[positions][..., columns]
        if truth.ndim == 3 and prediction.values.ndim == 2:
            truth = truth[:, -1, :]
        if truth.shape != prediction.values.shape:
            raise ValueError("Restored reconstruction shape must match original observations")
        return replace(prediction, truth=truth)

    def export_transform(
        self,
        path: str | Path,
        *,
        through: str,
        mode: str | None = None,
    ) -> Path:
        """Save a fitted prefix for frozen reuse, optionally changing its copied output mode.

        Parameters:
            path (str | Path): Trusted pickle destination; existing files are replaced.
            through (str): Last included step name, which must support transform.
            mode (str | None): Output mode override for the copied terminal estimator.

        Returns:
            Path: Absolute path to a self-contained transform artifact.
        """
        names = [name for name, _ in self.steps]
        stop = names.index(through) + 1
        exported = deepcopy(self)
        exported.steps = exported.steps[:stop]
        terminal = exported.steps[-1][1]
        estimator = getattr(terminal, "estimator", terminal)
        if mode is not None:
            estimator.set_params(mode=mode)
        if not callable(getattr(estimator, "transform", None)):
            raise TypeError("Exported terminal component must provide transform")
        artifact = TransformArtifact(
            exported, self.input_names_, self.frequency_ns_, dict(self.provenance_)
        )
        destination = Path(path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream:
            pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
        return destination
