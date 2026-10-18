"""Prediction plotting contracts using a headless backend."""

import unittest

import numpy as np
import pandas as pd


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import date2num

from feature_selector.visualization import (
    plot_predictions,
    plot_reconstruction_predictions,
    plot_split_predictions,
    plot_training_history,
)


class PredictionPlots(unittest.TestCase):
    def setUp(self) -> None:
        self.addCleanup(plt.close, "all")
        self.report = pd.DataFrame({
            "target_time": ["2024-01-01 03:00", "2024-01-01 00:00", "2024-01-01 01:00"],
            "y_true": [4.0, 1.0, 2.0],
            "y_pred": [3.5, 1.5, 2.5],
        })

    def test_timeline_and_observation_order(self) -> None:
        original = self.report.copy(deep=True)
        ax = plot_predictions(self.report, title="Validation")
        truth, prediction = ax.lines
        np.testing.assert_allclose(truth.get_ydata(), [1, 2, 4])
        np.testing.assert_allclose(prediction.get_ydata(), [1.5, 2.5, 3.5])
        expected_times = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00",
                                       "2024-01-01 03:00"])
        np.testing.assert_allclose(truth.get_xdata(orig=False),
                                   date2num(expected_times.to_pydatetime()))
        self.assertEqual(ax.get_title(), "Validation")
        pd.testing.assert_frame_equal(self.report, original)

    def test_splits_share_axes_but_not_colors(self) -> None:
        _, ax = plt.subplots()
        reports = {split: self.report for split in ("test", "train", "validation")}
        self.assertIs(plot_split_predictions(reports, ax=ax), ax)
        self.assertEqual(len(ax.lines), 6)
        self.assertEqual(len({line.get_color() for line in ax.lines[1::2]}), 3)
        for truth, prediction in zip(ax.lines[::2], ax.lines[1::2]):
            self.assertEqual(truth.get_color(), "tab:red")
            self.assertNotEqual(truth.get_color(), prediction.get_color())
            self.assertEqual(truth.get_linestyle(), "-")
            self.assertEqual(prediction.get_linestyle(), "-")
        self.assertEqual(len(ax.get_legend().get_texts()), 6)

    def test_available_splits_and_invalid_inputs(self) -> None:
        self.assertEqual(len(plot_split_predictions({"validation": self.report}).lines), 2)
        for reports in ({}, {"unknown": self.report}):
            with self.assertRaises(ValueError):
                plot_split_predictions(reports)
        duplicate = pd.concat([self.report, self.report.iloc[:1]])
        with self.assertRaisesRegex(ValueError, "unique"):
            plot_predictions(duplicate)

    def test_empty_report(self) -> None:
        ax = plot_predictions(self.report.iloc[:0])
        self.assertEqual(len(ax.lines), 2)
        self.assertEqual(len(ax.lines[0].get_ydata()), 0)

    def test_sparse_observations_remain_visible(self) -> None:
        report = self.report.copy()
        report["target_time"] = pd.date_range("2024-01-01", periods=3, freq="2D")
        ax = plot_predictions(report)
        for line in ax.lines:
            self.assertIn(line.get_marker(), (None, "None", "", " "))
            self.assertGreater(line.get_linewidth(), 0)
            self.assertEqual(np.isfinite(line.get_ydata()).sum(), 3)
            self.assertEqual(len(line.get_ydata()), 3)
        ax.figure.canvas.draw()

    def test_single_observation_uses_tiny_marker(self) -> None:
        ax = plot_predictions(self.report.iloc[:1])
        for line in ax.lines:
            self.assertEqual(line.get_marker(), ".")
            self.assertEqual(line.get_markersize(), 1)

    def test_missing_values_are_omitted_per_curve(self) -> None:
        report = self.report.copy()
        report.loc[1, "y_true"] = np.nan
        report.loc[2, "y_pred"] = np.nan
        ax = plot_predictions(report)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [2, 4])
        np.testing.assert_allclose(ax.lines[1].get_ydata(), [1.5, 3.5])

    def test_reconstruction_selects_one_feature(self) -> None:
        """Filter repeated endpoint timestamps by feature before drawing curves.

        Returns:
            None.
        """
        first = self.report.assign(feature="a")
        second = self.report.assign(feature="b", y_true=100.0, y_pred=200.0)
        report = pd.concat([first, second], ignore_index=True)
        original = report.copy()
        _, ax = plt.subplots()
        result = plot_reconstruction_predictions({"validation": report}, "a", ax=ax)
        self.assertIs(result, ax)
        self.assertEqual(len(ax.lines), 2)
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [1, 2, 4])
        self.assertEqual(ax.get_title(), "a")
        self.assertIn("original units", ax.get_ylabel())
        pd.testing.assert_frame_equal(report, original)
        with self.assertRaisesRegex(ValueError, "absent"):
            plot_reconstruction_predictions({"train": report}, "missing")
        with self.assertRaisesRegex(ValueError, "feature column"):
            plot_reconstruction_predictions({"train": self.report}, "a")
        with self.assertRaises(ValueError):
            plot_reconstruction_predictions({"unknown": report}, "a")
        with self.assertRaises(ValueError):
            plot_reconstruction_predictions({}, "a")


class TrainingHistoryPlots(unittest.TestCase):
    def test_losses_epoch_marker_and_existing_axes(self) -> None:
        """Plot ordered epoch losses on supplied axes with the selected checkpoint.

        Returns:
            None: Checks curve values, labels, marker position and rendering.
        """
        self.addCleanup(plt.close, "all")
        history = [
            {"epoch": 2, "train_loss": 0.2, "validation_loss": 0.4, "batches": []},
            {"epoch": 1, "train_loss": 0.8, "validation_loss": 0.9, "batches": []},
        ]
        _, ax = plt.subplots()
        self.assertIs(plot_training_history(history, best_epoch=2, ax=ax, title="Loss"), ax)
        self.assertEqual(len(ax.lines), 3)
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1, 2])
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.8, 0.2])
        np.testing.assert_allclose(ax.lines[1].get_ydata(), [0.9, 0.4])
        np.testing.assert_array_equal(ax.lines[2].get_xdata(), [2, 2])
        self.assertEqual(ax.get_title(), "Loss")
        self.assertEqual(ax.get_xlabel(), "Epoch")
        self.assertEqual([text.get_text() for text in ax.get_legend().get_texts()],
                         ["train_loss", "validation_loss", "Best epoch"])
        ax.figure.canvas.draw()

    def test_missing_validation_and_empty_history(self) -> None:
        """Handle disabled validation and estimators with no epoch history.

        Returns:
            None: Checks train-only curves and the empty-history message.
        """
        self.addCleanup(plt.close, "all")
        for record in (
            {"epoch": 1, "train_loss": 0.5, "validation_loss": None},
            {"epoch": 1, "train_loss": 0.5},
        ):
            ax = plot_training_history([record])
            self.assertEqual(len(ax.lines), 1)
            ax.figure.canvas.draw()
        ax = plot_training_history([])
        self.assertEqual(len(ax.lines), 0)
        self.assertEqual(ax.texts[0].get_text(), "No training history")
        ax.figure.canvas.draw()
