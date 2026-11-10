"""Metrics in original target units, with explanations for undefined values."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if TYPE_CHECKING:
    from .pipeline import PredictionData


MetricName = Literal["mae", "rmse", "mape", "pearson", "pearson_pvalue", "r2"]
METRIC_NAMES: tuple[MetricName, ...] = ("mae", "rmse", "mape", "pearson", "pearson_pvalue", "r2")


class RegressionMetrics(TypedDict):
    """Regression scores, sample count and explanations for undefined scores."""

    mae: float
    rmse: float
    mape: float
    r2: float
    pearson: float
    pearson_pvalue: float
    n: int
    notes: dict[str, str]


class MetricSummary(TypedDict):
    """Mean and sample deviation over finite run scores, with explicit coverage."""

    mean: float
    std: float
    n_valid: int
    n_total: int


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
    """Compute regression errors and Pearson correlation with its two-sided p-value.

    Parameters:
        y_true (ArrayLike): Nonempty one-dimensional finite targets convertible to float.
        y_pred (ArrayLike): Finite predictions with the same shape, in the same target units.

    Returns:
        RegressionMetrics: MAE, RMSE, MAPE in percent, R², Pearson correlation and p-value,
            sample count and notes. MAPE is NaN if any target is zero. Undefined R²,
            correlation and p-value are also NaN with explanations in notes.
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
    pearson_pvalue = float("nan")
    mape = float("nan")
    has_variable_target = sample_count >= 2 and np.ptp(truth) != 0

    # A percentage error at zero is undefined; an epsilon would hide that limitation.
    if np.any(truth == 0):
        notes["mape"] = "MAPE requires nonzero targets"
    else:
        mape = float(np.mean(np.abs((truth - prediction) / truth)) * 100)

    # Keep undefined scores visible instead of using sklearn's constant-target fallback.
    if not has_variable_target:
        notes["r2"] = "R² requires at least two observations and a nonconstant target"
    else:
        r2 = float(r2_score(truth, prediction))

    if not has_variable_target or np.ptp(prediction) == 0:
        notes["pearson"] = "Pearson requires at least two observations and two nonconstant arrays"
        notes["pearson_pvalue"] = notes["pearson"]
    else:
        correlation = pearsonr(truth, prediction)
        pearson = float(correlation[0])  # type: ignore
        pearson_pvalue = float(correlation[1])  # type: ignore

    return RegressionMetrics(
        mae=float(mean_absolute_error(truth, prediction)),
        rmse=float(np.sqrt(mean_squared_error(truth, prediction))),
        mape=mape,
        r2=r2,
        pearson=pearson,
        pearson_pvalue=pearson_pvalue,
        n=sample_count,
        notes=notes,
    )


def summarize_regression_metrics(
    metrics: Sequence[RegressionMetrics],
) -> dict[MetricName, MetricSummary]:
    """Summarize equally weighted run scores as means and sample standard deviations.

    Parameters:
        metrics (Sequence[RegressionMetrics]): Nonempty sequence of per-run scores.
            Runs should evaluate the same examples; n is not aggregated or used as a weight.

    Returns:
        dict[MetricName, MetricSummary]: Per-score mean, std with ddof=1, and finite/total
            run counts. Nonfinite scores are excluded. Mean is NaN without finite scores;
            std is NaN with fewer than two. P-values are summarized, not statistically combined.
    """
    if not metrics:
        raise ValueError("Metric summaries require at least one run")

    summaries: dict[MetricName, MetricSummary] = {}
    for name in METRIC_NAMES:
        values = np.asarray([scores[name] for scores in metrics], dtype=float)
        finite = values[np.isfinite(values)]
        summaries[name] = MetricSummary(
            mean=float(np.mean(finite)) if len(finite) else float("nan"),
            std=float(np.std(finite, ddof=1)) if len(finite) >= 2 else float("nan"),
            n_valid=len(finite),
            n_total=len(metrics),
        )
    return summaries


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
