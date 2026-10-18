"""Metrics in original target units, with explanations for undefined values."""

from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if TYPE_CHECKING:
    from .pipeline import PredictionData


class RegressionMetrics(TypedDict):
    """Regression scores, sample count and explanations for undefined scores."""

    mae: float
    rmse: float
    r2: float
    pearson: float
    n: int
    notes: dict[str, str]


class ReconstructionMetrics(TypedDict):
    """Scores over last-window points, pooled and separated by input feature."""

    overall: RegressionMetrics
    per_feature: dict[str, RegressionMetrics]


def reconstruction_metrics(report: pd.DataFrame) -> ReconstructionMetrics:
    """Evaluate last-point reconstruction reports in original feature units.

    Parameters:
        report (pd.DataFrame): Nonempty long report with feature, y_true and y_pred
            columns. Each row represents one feature at one accepted window endpoint.

    Returns:
        ReconstructionMetrics: Pooled scores and scores for each feature; overall n
            counts feature-point pairs. Pooled errors weight original feature scales.
    """
    if report["feature"].isna().any():
        raise ValueError("Reconstruction feature names must not be missing")
    overall = regression_metrics(report["y_true"], report["y_pred"])
    per_feature = {
        str(feature): regression_metrics(frame["y_true"], frame["y_pred"])
        for feature, frame in report.groupby("feature", sort=False)
    }
    return ReconstructionMetrics(overall=overall, per_feature=per_feature)


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> RegressionMetrics:
    """Compute regression scores, reporting undefined R² and Pearson values as NaN.

    Parameters:
        y_true (ArrayLike): Nonempty one-dimensional finite targets convertible to float.
        y_pred (ArrayLike): Finite predictions with the same shape, in the same target units.

    Returns:
        RegressionMetrics: MAE, RMSE, R², Pearson correlation, sample count and explanations
            in notes for scores undefined on constant or single-observation inputs.
    """
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)

    if truth.ndim != 1 or prediction.shape != truth.shape or not len(truth):
        raise ValueError("Metrics require nonempty, equally sized one-dimensional arrays")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Metrics require finite values")

    sample_count = len(truth)
    notes: dict[str, str] = {}
    r2 = float("nan")
    pearson = float("nan")
    has_variable_target = sample_count >= 2 and np.ptp(truth) != 0

    # Keep undefined scores visible instead of using sklearn's constant-target fallback.
    if not has_variable_target:
        notes["r2"] = "R² requires at least two observations and a nonconstant target"
    else:
        r2 = float(r2_score(truth, prediction))

    if not has_variable_target or np.ptp(prediction) == 0:
        notes["pearson"] = "Pearson requires at least two observations and two nonconstant arrays"
    else:
        pearson = cast(float, pearsonr(truth, prediction)[0])

    return RegressionMetrics(
        mae=float(mean_absolute_error(truth, prediction)),
        rmse=float(np.sqrt(mean_squared_error(truth, prediction))),
        r2=r2,
        pearson=pearson,
        n=sample_count,
        notes=notes,
    )


def prediction_report(prediction: "PredictionData", mask: "NDArray[np.bool_]") -> pd.DataFrame:
    """Flatten restored predictions into timestamped scalar or per-feature reports.

    Parameters:
        prediction (PredictionData): Restored model output and optional reference.
        mask (NDArray[np.bool_]): Selected examples, for example one temporal split.

    Returns:
        pd.DataFrame: Report with origin_time, target_time, y_true and y_pred;
            multifeature reports also include feature and latent sequences include timestep.
    """
    values = prediction.values[mask]
    truth = (
        prediction.truth[mask]
        if prediction.truth is not None
        else np.full_like(values, np.nan, dtype=float)
    )
    samples = prediction.samples.loc[mask]
    if values.ndim == 1:
        return pd.DataFrame(
            {
                "origin_time": samples["origin_time"].to_numpy(),
                "target_time": samples["target_time"].to_numpy(),
                "y_true": truth,
                "y_pred": values,
            }
        )
    if values.ndim == 3 and prediction.kind == "features":
        values, truth = values[:, -1, :], truth[:, -1, :]
    width = int(np.prod(values.shape[1:]))
    report = pd.DataFrame(
        {
            "origin_time": samples["origin_time"].repeat(width).to_numpy(),
            "target_time": samples["target_time"].repeat(width).to_numpy(),
            "feature": np.tile(
                prediction.feature_names, len(values) * (values.shape[1] if values.ndim == 3 else 1)
            ),
            "y_true": truth.ravel(),
            "y_pred": values.ravel(),
        }
    )
    if values.ndim == 3:
        report["timestep"] = np.tile(
            np.repeat(np.arange(values.shape[1]), values.shape[2]), len(values)
        )
    return report


def report_metrics(report: pd.DataFrame) -> dict[str, Any]:
    """Evaluate observed references, leaving unlabelled predictions available in reports.

    Parameters:
        report (pd.DataFrame): Scalar or feature report in original units.

    Returns:
        dict[str, Any]: Regression metrics, pooled/per-feature metrics, or no observed scores.
    """
    if not np.isfinite(report["y_pred"]).all():
        raise ValueError("Predictions must contain finite values")
    observed = report.loc[report["y_true"].notna()]
    if observed.empty:
        return {}
    if "feature" in observed:
        return dict(reconstruction_metrics(observed))
    return dict(regression_metrics(observed["y_true"], observed["y_pred"]))
