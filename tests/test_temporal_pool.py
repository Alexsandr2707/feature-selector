"""Temporal pooling values, alignment and frozen-prefix forecast integration."""

from dataclasses import replace
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
from sklearn.base import clone
import torch

from feature_selector.factory import build_pipeline
from feature_selector.pipeline.temporal import AlignTarget, TemporalPool, Windows
from support import sample_data


class TemporalPoolTests(unittest.TestCase):
    """Check local averaging without changing the forecast examples."""

    def test_values_alignment_and_input_views(self) -> None:
        """Average adjacent points independently for each sample and feature.

        Returns:
            None.
        """
        data = Windows(8).transform(AlignTarget(1).transform(sample_data(sparse=True)))
        for values in (data.X, data.X[:, ::-1], data.X.astype(np.float32)):
            with self.subTest(dtype=values.dtype, strides=values.strides):
                original = values.copy()
                source = replace(data, X=values)

                result = TemporalPool(4).transform(source)

                expected = np.stack(
                    (values[:, :4].mean(axis=1), values[:, 4:].mean(axis=1)), axis=1
                )

                np.testing.assert_allclose(result.X, expected)
                np.testing.assert_array_equal(values, original)
                self.assertIs(result.y, source.y)
                self.assertIs(result.samples, source.samples)
                self.assertIs(result.temporal, source.temporal)
                self.assertEqual(result.feature_names, source.feature_names)

    def test_identity_empty_samples_and_missing_values(self) -> None:
        """Keep shape edge cases and propagate missing values within their groups.

        Returns:
            None.
        """
        data = Windows(8).transform(sample_data())
        np.testing.assert_array_equal(TemporalPool(1).transform(data).X, data.X)

        empty = replace(data, X=np.empty((0, 128, 2)), y=None, samples=data.samples.iloc[:0])
        self.assertEqual(TemporalPool(4).transform(empty).X.shape, (0, 32, 2))

        missing = np.ones((len(data.X), 8, 2))
        missing[0, 0, 0] = np.nan
        result = TemporalPool(4).transform(replace(data, X=missing, y=None))
        self.assertTrue(np.isnan(result.X[0, 0, 0]))
        self.assertEqual(result.X[0, 1, 0], 1)
        self.assertIsNone(result.y)

    def test_invalid_parameters_and_inputs_at_transform(self) -> None:
        """Reject invalid factors and incompatible windows at their consuming step.

        Returns:
            None.
        """
        data = Windows(8).transform(sample_data())

        for factor in (0, -1, True, 2.0, "4"):
            with self.subTest(factor=factor):
                step = TemporalPool(factor)
                self.assertIs(step.fit(data), step)
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    step.transform(data)

        for values in (sample_data().X, np.ones((3, 2)), np.empty((3, 0, 2))):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                TemporalPool(4).transform(replace(data, X=values))

        for factor in (3, 16):
            with self.subTest(factor=factor), self.assertRaisesRegex(ValueError, "divisible"):
                TemporalPool(factor).transform(data)

    def test_factory_clone_frozen_prefix_and_prediction_reload(self) -> None:
        """Pool a frozen prefix before forecasting and retain alignment after reload.

        Returns:
            None.
        """
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

        data = sample_data(sparse=True)
        prefix = build_pipeline(
            [
                {"name": "windows", "type": "windows", "params": {"lookback": 8}},
                {"name": "model", "type": "ridge"},
            ]
        ).fit(data)

        with tempfile.TemporaryDirectory() as directory:
            path = prefix.export_transform(Path(directory) / "prefix.pkl", through="windows")
            pipeline = build_pipeline(
                [
                    {"name": "align", "type": "align_target", "params": {"horizon": 1}},
                    {"name": "encoder", "type": "frozen_pipeline", "params": {"path": path}},
                    {"name": "pool", "type": "temporal_pool", "params": {"factor": 4}},
                    {
                        "name": "model",
                        "type": "gru_ensemble",
                        "params": {
                            "n_estimators": 2,
                            "gru_params": {
                                "batch_size": 1,
                                "hidden_size": 2,
                                "epochs": 1,
                                "scheduler": False,
                            },
                        },
                    },
                ]
            ).fit(data)

            self.assertEqual(clone(pipeline["pool"]).factor, 4)

            encoded = pipeline.transform(data, through="encoder")
            pooled = pipeline.transform(data, through="pool")
            self.assertEqual(pooled.X.shape, (len(encoded.X), 2, 2))
            np.testing.assert_array_equal(pooled.y, encoded.y)
            self.assertTrue(pooled.samples.equals(encoded.samples))

            prediction = pipeline.predict(data, splits=("test",))
            restored = pickle.loads(pickle.dumps(pipeline))

        reloaded = restored.predict(data, splits=("test",))
        np.testing.assert_array_equal(reloaded.values, prediction.values)
        self.assertTrue(reloaded.samples.equals(prediction.samples))


if __name__ == "__main__":
    unittest.main()
