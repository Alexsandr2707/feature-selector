"""Adapt sklearn transformers and numerical models to aligned pipeline data."""

from dataclasses import replace
from inspect import signature
from typing import Any, Self, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ..tracking import StepContext
from .base import Step
from .data import PipelineData, PredictionData


class TransformerStep(Step):
    """Apply an estimator to X, retaining the sample alignment and external targets."""

    def __init__(self, estimator: BaseEstimator, requires_y: bool = False) -> None:
        """Store a transformer and its training target requirement.

        Parameters:
            estimator (BaseEstimator): Estimator exposing fit and transform.
            requires_y (bool): Filter complete feature/target pairs before fit.

        Returns:
            None.
        """
        self.estimator = estimator
        self.requires_y = requires_y

    def fit(
        self,
        train: PipelineData,
        validation: PipelineData | None = None,
        *,
        context: StepContext | None = None,
    ) -> Self:
        """Fit the transformer on nonexcluded training examples.

        Parameters:
            train (PipelineData): Current training representation.
            validation (PipelineData | None): Unused validation representation.
            context (StepContext | None): Receives feature-selection metadata.

        Returns:
            Self: Fitted adapter.
        """
        mask = (
            train.valid(require_y=True)
            if self.requires_y
            else (~train.samples["excluded"].to_numpy(dtype=bool))
        )
        selected = train.select(mask)
        fit = getattr(self.estimator, "fit", None)
        if not callable(fit):
            raise TypeError("Transformer must provide fit")
        fit(selected.X, selected.y)
        self.input_names_ = train.feature_names
        indices = getattr(self.estimator, "get_selected_indices", None)
        if context and callable(indices):
            context.log_metadata({"selected_indices": indices()})
        return self

    def transform(
        self,
        data: PipelineData,
        *,
        context: StepContext | None = None,
    ) -> PipelineData:
        """Transform X and derive output feature identities.

        Parameters:
            data (PipelineData): Values with the fitted feature schema.
            context (StepContext | None): Execution diagnostics.

        Returns:
            PipelineData: Transformed values with unchanged examples and y.
        """
        if data.feature_names != self.input_names_:
            raise ValueError("Feature names and order must match fit")
        transform = getattr(self.estimator, "transform", None)
        if not callable(transform):
            raise TypeError("Transformer must provide transform")
        values = transform(data.X)
        values = cast(pd.DataFrame, values)

        names = data.feature_names
        get_names = getattr(self.estimator, "get_feature_names_out", None)
        if callable(get_names):
            names = tuple(get_names(list(names)))  # type: ignore
        elif values.shape[-1] != len(names):
            names = tuple(f"x{i}" for i in range(values.shape[-1]))
        if isinstance(data.X, pd.DataFrame) and not isinstance(values, pd.DataFrame):
            values = pd.DataFrame(values, index=data.X.index, columns=list(names))
        return replace(data, X=values, feature_names=names)

    def inverse_prediction(self, prediction: PredictionData) -> PredictionData:
        """Restore feature reconstructions through reversible preprocessing.

        Parameters:
            prediction (PredictionData): Target predictions or reconstructed features.

        Returns:
            PredictionData: Feature values in the preceding representation's units.
        """
        if prediction.kind != "features":
            return prediction
        inverse = getattr(self.estimator, "inverse_transform", None)
        if not callable(inverse):
            raise TypeError("Reconstruction reporting requires reversible feature preprocessing")
        shape = prediction.values.shape
        values = np.asarray(inverse(prediction.values.reshape(-1, shape[-1]))).reshape(shape)
        truth = (
            None
            if prediction.truth is None
            else np.asarray(inverse(prediction.truth.reshape(-1, shape[-1]))).reshape(shape)
        )
        return replace(prediction, values=values, truth=truth, feature_names=self.input_names_)


class TargetScale(Step):
    """Scale external targets and restore scalar model predictions."""

    def __init__(self, scaler: BaseEstimator) -> None:
        """Store a reversible target scaler.

        Parameters:
            scaler (BaseEstimator): Transformer accepting a single numeric column.

        Returns:
            None.
        """
        self.scaler = scaler

    def fit(
        self,
        train: PipelineData,
        validation: PipelineData | None = None,
        *,
        context: StepContext | None = None,
    ) -> Self:
        """Fit using targets of usable train examples only.

        Parameters:
            train (PipelineData): Current model inputs and external targets.
            validation (PipelineData | None): Unused validation examples.
            context (StepContext | None): Execution diagnostics.

        Returns:
            Self: Fitted target transformer.
        """
        selected = train.select(train.valid(require_y=True))
        fit = getattr(self.scaler, "fit", None)
        if not callable(fit):
            raise TypeError("Target scaler must provide fit")
        fit(np.asarray(selected.y).reshape(-1, 1))
        return self

    def transform(
        self,
        data: PipelineData,
        *,
        context: StepContext | None = None,
    ) -> PipelineData:
        """Scale observed y, preserving missing targets and all input examples.

        Parameters:
            data (PipelineData): Current data with optional external targets.
            context (StepContext | None): Execution diagnostics.

        Returns:
            PipelineData: Scaled targets or unchanged data when y is absent.
        """
        if data.y is None:
            return data
        values = np.array(data.y, dtype=float, copy=True)
        observed = np.isfinite(values)
        transform = getattr(self.scaler, "transform", None)
        if not callable(transform):
            raise TypeError("Target scaler must provide transform")
        if observed.any():
            values[observed] = transform(values[observed, None]).ravel()  # type:ignore
        return replace(data, y=values)

    def inverse_prediction(self, prediction: PredictionData) -> PredictionData:
        """Restore scalar predictions and references to original target units.

        Parameters:
            prediction (PredictionData): Scalar target predictions or unrelated output.

        Returns:
            PredictionData: Restored target values and reference.
        """
        if prediction.kind != "target":
            return prediction
        inverse = getattr(self.scaler, "inverse_transform", None)
        if not callable(inverse):
            raise TypeError("Target scaler must provide inverse_transform")
        values = inverse(prediction.values.reshape(-1, 1)).ravel()  # type:ignore

        truth = (
            None if prediction.truth is None else inverse(prediction.truth.reshape(-1, 1)).ravel()  # type:ignore
        )
        return replace(prediction, values=values, truth=truth)


class ModelStep(Step):
    """Expose a numerical estimator as a supervised or self-reconstructing model step."""

    def __init__(self, estimator: BaseEstimator, reconstructs: bool = False) -> None:
        """Store an estimator and the source of its training reference.

        Parameters:
            estimator (BaseEstimator): Model providing fit/predict and optionally transform.
            reconstructs (bool): Train against X rather than external y.

        Returns:
            None.
        """
        self.estimator = estimator
        self.reconstructs = reconstructs

    def fit(
        self,
        train: PipelineData,
        validation: PipelineData | None = None,
        *,
        context: StepContext | None = None,
    ) -> Self:
        """Select suitable examples and delegate fitting and telemetry to the estimator.

        Parameters:
            train (PipelineData): Current train inputs; missing y is allowed for reconstruction.
            validation (PipelineData | None): Current validation inputs for checkpoint selection.
            context (StepContext | None): Execution-scoped runner telemetry.

        Returns:
            Self: Fitted adapter retaining its input schema.
        """
        selected = train.select(train.valid(require_y=not self.reconstructs))
        if not len(selected.X):
            raise ValueError("No usable train examples for model")
        valid = None
        if validation is not None:
            valid = validation.select(validation.valid(require_y=not self.reconstructs))
            if not len(valid.X):
                raise ValueError("No usable validation examples for model")
        fit = getattr(self.estimator, "fit", None)
        if not callable(fit):
            raise TypeError("Model must provide fit")
        params: dict[str, Any] = {}
        if "validation_data" in signature(fit).parameters and valid is not None:
            params["validation_data"] = (
                np.asarray(valid.X)
                if self.reconstructs
                else (np.asarray(valid.X), np.asarray(valid.y))
            )
        if "context" in signature(fit).parameters:
            params["context"] = context
        fit(np.asarray(selected.X), None if self.reconstructs else np.asarray(selected.y), **params)
        self.input_names_ = train.feature_names
        if context:
            context.log_metadata(
                {
                    "train_samples": len(selected.X),
                    "validation_samples": len(valid.X) if valid is not None else 0,
                }
            )
        return self

    def transform(
        self,
        data: PipelineData,
        *,
        context: StepContext | None = None,
    ) -> PipelineData:
        """Apply the estimator's representation transform for downstream steps.

        Parameters:
            data (PipelineData): Current inputs matching training schema.
            context (StepContext | None): Execution diagnostics.

        Returns:
            PipelineData: New representation with unchanged targets and sample metadata.
        """
        if data.feature_names != self.input_names_:
            raise ValueError("Model feature schema must match fit")
        transform = getattr(self.estimator, "transform", None)
        if not callable(transform):
            raise TypeError("Intermediate model must provide transform")
        selected = data.select(data.valid())
        values = transform(np.asarray(selected.X))
        values = cast(np.ndarray, values)
        encoded = getattr(self.estimator, "mode", None) == "encode"
        names = tuple(f"z{i}" for i in range(values.shape[-1]))
        if not encoded and values.shape[-1] == len(self.input_names_):
            names = self.input_names_
        return replace(selected, X=values, feature_names=names)

    def predict(
        self,
        data: PipelineData,
        *,
        context: StepContext | None = None,
    ) -> PredictionData:
        """Predict all usable inputs, retaining missing references for unlabelled inference.

        Parameters:
            data (PipelineData): Current inputs and optional targets.
            context (StepContext | None): Execution diagnostics.

        Returns:
            PredictionData: Model output and the corresponding external or input reference.
        """
        if data.feature_names != self.input_names_:
            raise ValueError("Model feature schema must match fit")
        selected = data.select(data.valid())
        if not len(selected.X):
            raise ValueError("No usable prediction examples")
        predict = getattr(self.estimator, "predict", None)
        if not callable(predict):
            raise TypeError("Model must provide predict")
        values = np.asarray(predict(np.asarray(selected.X)))
        mode = getattr(self.estimator, "mode", "reconstruct")
        kind = (
            "latent"
            if self.reconstructs and mode == "encode"
            else ("features" if self.reconstructs else "target")
        )
        truth = (
            np.asarray(selected.X)
            if kind == "features"
            else (np.asarray(selected.y) if selected.y is not None and kind == "target" else None)
        )
        names = (
            selected.feature_names
            if kind != "latent"
            else tuple(f"z{i}" for i in range(values.shape[-1]))
        )
        return PredictionData(values, truth, selected.samples, names, kind)

    def inverse_prediction(self, prediction: PredictionData) -> PredictionData:
        """Decode a downstream reconstruction through an intermediate encoder.

        Parameters:
            prediction (PredictionData): Reconstructed latent inputs or scalar targets.

        Returns:
            PredictionData: Decoded feature reconstruction when relevant. The pipeline
                supplies original references after completing the inverse path.
        """
        if prediction.kind != "features" or getattr(self.estimator, "mode", None) == "reconstruct":
            return prediction
        inverse = getattr(self.estimator, "inverse_transform", None)
        if not callable(inverse):
            raise TypeError("Intermediate model cannot restore reconstructed features")
        # Decoding a latent reference would hide errors introduced by the encoder.
        return replace(
            prediction,
            values=inverse(prediction.values),
            truth=None,
            feature_names=self.input_names_,
        )
