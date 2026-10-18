"""Prediction timelines and epoch training history plots."""

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes


def plot_training_history(
    history: Sequence[Mapping[str, Any]],
    best_epoch: int | None = None,
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Plot epoch losses and an optional selected epoch in training target units.

    Parameters:
        history (Sequence[Mapping[str, Any]]): Epoch records with numeric epoch and
            train_loss; validation_loss may be missing or None. Empty history is allowed.
            Other fields, including batch records, are ignored.
        best_epoch (int | None): One-based selected epoch, or None to omit its marker.
        ax (Axes | None): Existing axes or None to create a figure.
        title (str | None): Optional plot title.

    Returns:
        Axes: Axes with loss curves or an empty-history message, ready for saving.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (training target units)")
    ax.grid(True)
    if title is not None:
        ax.set_title(title)

    # Noniterative estimators such as Ridge have no epoch records.
    if not history:
        ax.text(0.5, 0.5, "No training history", ha="center", va="center", transform=ax.transAxes)
        return ax

    frame = pd.DataFrame(history).set_index("epoch").sort_index()
    ax.plot(frame.index, frame["train_loss"], label="train_loss")
    if "validation_loss" in frame and frame["validation_loss"].notna().any():
        ax.plot(frame.index, frame["validation_loss"], label="validation_loss")
    if best_epoch is not None:
        ax.axvline(best_epoch, color="gray", linestyle="--", label="Best epoch")
    ax.legend()
    return ax


def _plot_predictions(
    predictions: pd.DataFrame,
    ax: Axes,
    label: str,
    color: str,
    truth_color: str | None = None,
) -> None:
    """Connect available observations, omitting missing values from each curve.

    Parameters:
        predictions (pd.DataFrame): Unique target_time values and numeric y_true/y_pred columns.
        ax (Axes): Axes to draw on.
        label (str): Legend prefix.
        color (str): Prediction color.
        truth_color (str | None): Target color; None uses the prediction color.

    Returns:
        None: Adds two curves to the axes.
    """
    frame = predictions.loc[:, ["target_time", "y_true", "y_pred"]].copy()
    frame["target_time"] = pd.to_datetime(frame["target_time"], errors="raise")
    if frame["target_time"].isna().any() or frame["target_time"].duplicated().any():
        raise ValueError("target_time must contain unique, nonmissing timestamps")
    frame = frame.set_index("target_time").sort_index()

    truth = frame["y_true"].dropna()
    prediction = frame["y_pred"].dropna()
    ax.plot(
        truth.index,
        truth,
        color=truth_color or color,
        linewidth=0.8,
        marker="." if len(truth) == 1 else None,
        markersize=1,
        label=f"{label}ground truth",
    )
    ax.plot(
        prediction.index,
        prediction,
        color=color,
        linewidth=0.7,
        marker="." if len(prediction) == 1 else None,
        markersize=1,
        label=f"{label}prediction",
    )


def _prepare_axes(ax: Axes | None, title: str | None) -> Axes:
    """Create or configure axes for a prediction plot.

    Parameters:
        ax (Axes | None): Existing axes or None to create a figure.
        title (str | None): Optional plot title.

    Returns:
        Axes: Axes with time and target labels and a grid.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 7), layout="constrained")
    ax.set_xlabel("Target time")
    ax.set_ylabel("Target")
    ax.grid(True)
    if title is not None:
        ax.set_title(title)
    return ax


def plot_predictions(
    predictions: pd.DataFrame,
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Plot ground truth and predictions against target time, including isolated points.

    Parameters:
        predictions (pd.DataFrame): Report with target_time, y_true and y_pred columns.
            Timestamps must be unique; missing values are omitted and observations connected.
        ax (Axes | None): Existing axes or None to create a figure.
        title (str | None): Optional plot title.

    Returns:
        Axes: Plot axes for further customization or saving.
    """
    ax = _prepare_axes(ax, title)
    _plot_predictions(predictions, ax, label="", color="tab:blue")
    ax.legend()
    return ax


def plot_split_predictions(
    predictions: Mapping[str, pd.DataFrame],
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Plot red targets and distinct prediction colors for train, validation and test.

    Parameters:
        predictions (Mapping[str, pd.DataFrame]): Nonempty mapping with train, validation
            or test keys and reports accepted by plot_predictions.
        ax (Axes | None): Existing axes or None to create a figure.
        title (str | None): Optional plot title.

    Returns:
        Axes: Plot axes with solid target and prediction lines for each split.
    """
    colors = {"train": "tab:blue", "validation": "tab:orange", "test": "tab:green"}
    if not predictions or predictions.keys() - colors.keys():
        raise ValueError("Provide reports keyed by train, validation or test")

    ax = _prepare_axes(ax, title)
    for split, color in colors.items():
        if split in predictions:
            _plot_predictions(
                predictions[split], ax, label=f"{split}: ", color=color, truth_color="tab:red"
            )
    ax.legend()
    return ax


def plot_reconstruction_predictions(
    reports: Mapping[str, pd.DataFrame],
    feature: str,
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Compare one original feature with its last-point window reconstructions.

    Parameters:
        reports (Mapping[str, pd.DataFrame]): Long reconstruction reports keyed by train,
            validation or test, with feature, target_time, y_true and y_pred columns.
        feature (str): Input feature to plot, present in every nonempty report.
        ax (Axes | None): Existing axes or None to create a figure.
        title (str | None): Optional title; None uses the feature name.

    Returns:
        Axes: Original-unit truth and reconstruction curves for the requested feature only.
    """
    selected: dict[str, pd.DataFrame] = {}
    for split, report in reports.items():
        if "feature" not in report:
            raise ValueError("Reconstruction reports require a feature column")
        frame = report.loc[report["feature"] == feature]
        if not report.empty and frame.empty:
            raise ValueError(f"Feature {feature!r} is absent from the {split} report")
        selected[split] = frame
    ax = plot_split_predictions(selected, ax=ax, title=feature if title is None else title)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{feature} (original units)")
    return ax
