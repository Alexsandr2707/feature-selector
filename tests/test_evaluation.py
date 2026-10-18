"""Metrics select observed targets and retain definedness explanations."""

import unittest

import numpy as np
import pandas as pd

from feature_selector.evaluation import regression_metrics, report_metrics


class EvaluationTests(unittest.TestCase):
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
