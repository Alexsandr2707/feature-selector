"""Metrics select observed targets and retain definedness explanations."""

import unittest

import numpy as np
import pandas as pd

from feature_selector.evaluation import (
    regression_metrics,
    report_metrics,
    summarize_regression_metrics,
)


class EvaluationTests(unittest.TestCase):
    def test_percentage_error_and_pearson_pvalue(self) -> None:
        """Check percentage units and a known two-sided Pearson test result.

        Returns:
            None.
        """
        scores = regression_metrics([1, 2, 3], [1, 3, 2])

        self.assertAlmostEqual(scores["mape"], 100 * (1 / 2 + 1 / 3) / 3)
        self.assertAlmostEqual(scores["pearson"], 0.5)
        self.assertAlmostEqual(scores["pearson_pvalue"], 2 / 3)
        self.assertEqual(scores["notes"], {})

        negative = regression_metrics([-2, -4], [-1, -2])
        self.assertEqual(negative["mape"], 50)
        self.assertEqual(negative["pearson_pvalue"], 1)

    def test_zero_target_does_not_discard_percentage_errors(self) -> None:
        """Mark MAPE undefined at zero while retaining every observation for other scores.

        Returns:
            None.
        """
        scores = regression_metrics([0, 2], [1, 2])

        self.assertTrue(np.isnan(scores["mape"]))
        self.assertIn("mape", scores["notes"])
        self.assertEqual(scores["n"], 2)
        self.assertEqual(scores["mae"], 0.5)

    def test_undefined_pearson_pvalue(self) -> None:
        """Explain unavailable p-values for insufficient or constant observations.

        Returns:
            None.
        """
        for truth, prediction in (([1], [2]), ([2, 2], [2, 3]), ([1, 2], [3, 3])):
            with self.subTest(truth=truth, prediction=prediction):
                scores = regression_metrics(truth, prediction)
                self.assertTrue(np.isnan(scores["pearson"]))
                self.assertTrue(np.isnan(scores["pearson_pvalue"]))
                self.assertIn("pearson_pvalue", scores["notes"])

    def test_summary_uses_sample_standard_deviation(self) -> None:
        """Give each run equal weight and distinguish deviation from half the variance.

        Returns:
            None.
        """
        truth = np.array([1, 2, 3])
        scores = [regression_metrics(truth, truth + offset) for offset in (1, 2, 3)]

        summary = summarize_regression_metrics(scores)

        self.assertEqual(summary["mae"], {"mean": 2, "std": 1, "n_valid": 3, "n_total": 3})
        self.assertEqual(summary["rmse"], summary["mae"])
        self.assertEqual(set(summary), {"mae", "rmse", "mape", "pearson", "pearson_pvalue", "r2"})

    def test_summary_reports_coverage_and_missing_deviation(self) -> None:
        """Expose missing scores and keep one-run deviations undefined.

        Returns:
            None.
        """
        scores = [
            regression_metrics([1, 2], [2, 3]),
            regression_metrics([0, 2], [1, 3]),
        ]

        summary = summarize_regression_metrics(scores)

        self.assertEqual(summary["mape"]["mean"], 75)
        self.assertTrue(np.isnan(summary["mape"]["std"]))
        self.assertEqual(summary["mape"]["n_valid"], 1)
        self.assertEqual(summary["mape"]["n_total"], 2)
        self.assertEqual(summary["mae"]["std"], 0)

        single = summarize_regression_metrics(scores[:1])
        self.assertEqual(single["mae"]["mean"], 1)
        self.assertTrue(np.isnan(single["mae"]["std"]))

    def test_summary_handles_all_undefined_scores_and_rejects_no_runs(self) -> None:
        """Summarize unavailable scores without warnings and reject an absent experiment.

        Returns:
            None.
        """
        summary = summarize_regression_metrics([regression_metrics([0], [1])])
        for name in ("mape", "r2", "pearson", "pearson_pvalue"):
            with self.subTest(metric=name):
                self.assertTrue(np.isnan(summary[name]["mean"]))
                self.assertTrue(np.isnan(summary[name]["std"]))
                self.assertEqual(summary[name]["n_valid"], 0)
                self.assertEqual(summary[name]["n_total"], 1)

        with self.assertRaises(ValueError):
            summarize_regression_metrics([])

    def test_missing_targets_and_per_feature_scores(self) -> None:
        """Score only observed reference values without removing predictions from reports.

        Returns:
            None.
        """
        report = pd.DataFrame({"y_true": [1, np.nan, 3], "y_pred": [2, 8, 4]})

        self.assertEqual(report_metrics(report)["n"], 2)
        self.assertEqual(report_metrics(report)["mae"], 1)

        report["feature"] = ["a", "b", "a"]
        self.assertEqual(report_metrics(report)["overall"]["n"], 2)
        self.assertEqual(set(report_metrics(report)["per_feature"]), {"a"})

        self.assertEqual(report_metrics(report.assign(y_true=np.nan)), {})

    def test_undefined_and_misaligned_metrics(self) -> None:
        """Explain undefined correlation and reject silently broadcastable shapes.

        Returns:
            None.
        """
        scores = regression_metrics([2, 2], [2, 3])

        self.assertTrue(np.isnan(scores["r2"]))
        self.assertTrue(np.isnan(scores["pearson"]))
        self.assertIn("r2", scores["notes"])

        with self.assertRaises(ValueError):
            regression_metrics([1, 2], [[1], [2]])

    def test_invalid_predictions_cannot_improve_scores(self) -> None:
        """Reject invalid predictions instead of silently reducing the sample count.

        Returns:
            None.
        """
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(prediction=value), self.assertRaises(ValueError):
                report_metrics(pd.DataFrame({"y_true": [0, 100], "y_pred": [0, value]}))
            with self.subTest(unlabelled_prediction=value), self.assertRaises(ValueError):
                report_metrics(pd.DataFrame({"y_true": [np.nan], "y_pred": [value]}))

        with self.assertRaises(ValueError):
            report_metrics(pd.DataFrame({"y_true": [1, np.inf], "y_pred": [1, 2]}))
