"""Feature-only compression preserves time through training and frozen reuse."""

from copy import deepcopy
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone
import torch

from feature_selector import run_experiment
from feature_selector.artifacts import load_run
from feature_selector.data import load_data
from feature_selector.models.autoencoder import ConvAutoencoder
from support import write_config


class FeatureAutoencoderTests(unittest.TestCase):
    """Check independent timestep mappings and the existing artifact workflow."""

    def setUp(self) -> None:
        """Limit CPU overhead for small convolutional batches.

        Returns:
            None.
        """
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def test_shapes_independence_and_reload(self) -> None:
        """Preserve time and apply the same feature mapping at every position.

        Returns:
            None.
        """
        for length in (1, 4, 5):
            with self.subTest(length=length):
                X = np.random.default_rng(5).normal(size=(6, length, 7)).astype(np.float32)
                model = ConvAutoencoder(
                    compression="features", hidden_channels=4, latent_channels=2,
                    epochs=2, batch_size=4, prediction_batch_size=2,
                ).fit(X, validation_data=X[:2])
                Z = model.encode(X)
                self.assertEqual(Z.shape, (6, length, 2))
                self.assertEqual(model.latent_shape_, (length, 2))
                self.assertEqual(model.predict(X).shape, X.shape)
                np.testing.assert_allclose(model.decode(Z), model.predict(X), atol=1e-6)
                self.assertEqual(model.encode(X[:0]).shape, (0, length, 2))
                self.assertEqual(model.decode(Z[:0]).shape, (0, length, 7))
                self.assertEqual(model.predict(X[:0]).shape, (0, length, 7))

                changed = X.copy()
                changed[:, 0] += 10
                np.testing.assert_array_equal(model.encode(changed)[:, 1:], Z[:, 1:])
                np.testing.assert_array_equal(
                    model.predict(changed)[:, 1:], model.predict(X)[:, 1:]
                )
                changed_latent = Z.copy()
                changed_latent[:, 0] += 10
                np.testing.assert_array_equal(
                    model.decode(changed_latent)[:, 1:], model.decode(Z)[:, 1:]
                )
                np.testing.assert_allclose(model.encode(X[:, ::-1]), Z[:, ::-1], atol=1e-6)
                model.set_params(mode="encode")
                np.testing.assert_array_equal(model.transform(X), Z)
                restored = pickle.loads(pickle.dumps(model))
                np.testing.assert_array_equal(restored.predict(X), Z)
                fresh = clone(model)
                self.assertEqual(fresh.get_params(), model.get_params())
                self.assertFalse(hasattr(fresh, "runner_"))
                self.assertTrue(all(e["validation_loss"] is not None for e in model.history_))
                self.assertEqual(
                    [batch["samples"] for batch in model.history_[0]["batches"]], [4, 2]
                )

    def test_learning_and_compression_switch_on_refit(self) -> None:
        """Learn reconstruction and replace latent metadata when architecture changes.

        Returns:
            None.
        """
        X = np.ones((6, 5, 3), dtype=np.float32)
        model = ConvAutoencoder(
            compression="features", hidden_channels=4, latent_channels=2,
            epochs=20, learning_rate=0.03, scheduler=False,
        ).fit(X)
        self.assertLess(model.history_[-1]["train_loss"], model.history_[0]["train_loss"])
        self.assertLess(float(np.mean((model.predict(X) - X) ** 2)), 0.1)
        model.set_params(compression="features_time", epochs=1).fit(X)
        self.assertEqual(model.latent_shape_, (3, 2))
        model.set_params(compression="features").fit(X)
        self.assertEqual(model.latent_shape_, (5, 2))
        with self.assertRaisesRegex(ValueError, "compression"):
            model.set_params(compression="invalid").fit(X)

    def test_export_frozen_gru_and_self_contained_run(self) -> None:
        """Train on dense windows and reuse saved features without fitting the encoder.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forecast = write_config(root, sparse=True)
            reconstruction = deepcopy(forecast)
            reconstruction["data"] = {"features_path": forecast["data"]["features_path"]}
            reconstruction["pipeline"] = [
                forecast["pipeline"][0], forecast["pipeline"][2],
                {"name": "ae", "type": "conv_autoencoder", "params": {
                    "compression": "features", "epochs": 1,
                    "hidden_channels": 2, "latent_channels": 1, "batch_size": 8}},
            ]
            pretrained = run_experiment(reconstruction)
            self.assertEqual(pretrained.model.latent_shape_, (5, 1))
            self.assertEqual(pretrained.metrics["validation"]["overall"]["n"], 24)
            path = pretrained.pipeline.export_transform(
                root / "feature_encoder.pkl", through="ae", mode="encode"
            )
            forecast["pipeline"] = [
                forecast["pipeline"][1],
                {"name": "encoder", "type": "frozen_pipeline", "params": {"path": str(path)}},
                forecast["pipeline"][3],
                {"name": "gru", "type": "gru", "params": {
                    "epochs": 1, "hidden_size": 2, "batch_size": 8}},
            ]
            with patch.object(ConvAutoencoder, "fit", side_effect=AssertionError("refit")):
                run = run_experiment(forecast)
            self.assertEqual(run.model.window_shape_, (5, 1))
            self.assertEqual(
                sum(b["samples"] for b in pretrained.model.history_[0]["batches"]), 32
            )
            self.assertEqual(sum(b["samples"] for b in run.model.history_[0]["batches"]), 7)
            path.unlink()
            data = load_data(**forecast["data"])
            restored = load_run(run.artifact_dir)
            np.testing.assert_array_equal(
                restored.predict(data, "test").y_pred, run.predict(data, "test").y_pred
            )
