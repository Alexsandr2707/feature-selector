"""Recurrent feature compression and reuse through the existing pipeline interface."""

from copy import deepcopy
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone
import torch
import yaml

from feature_selector import run_experiment
from feature_selector.artifacts import load_run
from feature_selector.config import load_config
from feature_selector.data import load_data
from feature_selector.models.autoencoder import GRUConvAutoencoder
from support import write_config


class GRUAutoencoderTests(unittest.TestCase):
    """Check sequence semantics, training and frozen encoder reuse."""

    def setUp(self) -> None:
        """Limit thread overhead for small CPU networks.

        Returns:
            None.
        """
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def test_shapes_modes_and_batch_independence(self) -> None:
        """Preserve window length and reset recurrent states between windows and calls.

        Returns:
            None.
        """
        for length in (1, 4, 5):
            with self.subTest(length=length):
                X = np.random.default_rng(7).normal(size=(7, length, 6)).astype(np.float32)
                model = GRUConvAutoencoder(
                    hidden_channels=4, latent_channels=2, epochs=1,
                    batch_size=3, prediction_batch_size=2,
                ).fit(X)
                Z = model.encode(X)
                predicted = model.predict(X)
                self.assertEqual(Z.shape, (7, length, 2))
                self.assertEqual(model.latent_shape_, (length, 2))
                self.assertEqual(predicted.shape, X.shape)
                np.testing.assert_allclose(model.decode(Z), predicted, atol=1e-6)
                self.assertEqual(model.encode(X[:0]).shape, (0, length, 2))
                self.assertEqual(model.decode(Z[:0]).shape, (0, length, 6))
                self.assertEqual(model.predict(X[:0]).shape, (0, length, 6))
                model.set_params(prediction_batch_size=7)
                np.testing.assert_allclose(model.encode(X), Z, atol=1e-6)
                np.testing.assert_allclose(model.predict(X), predicted, atol=1e-6)
                np.testing.assert_allclose(model.predict(X[::-1]), predicted[::-1], atol=1e-6)
                model.set_params(mode="encode")
                np.testing.assert_allclose(model.transform(X), Z, atol=1e-6)
                np.testing.assert_allclose(model.inverse_transform(Z), predicted, atol=1e-6)

    def test_recurrent_context_is_causal_and_decoder_uses_only_latents(self) -> None:
        """Use past context while keeping future inputs and encoder state out of decoding.

        Returns:
            None.
        """
        X = np.random.default_rng(3).normal(size=(5, 5, 6)).astype(np.float32)
        model = GRUConvAutoencoder(hidden_channels=4, latent_channels=2, epochs=1).fit(X)
        Z = model.encode(X)
        changed = X.copy()
        changed[:, -1] += 10
        np.testing.assert_array_equal(model.encode(changed)[:, :-1], Z[:, :-1])
        changed = X.copy()
        changed[:, 0] += 10
        self.assertGreater(float(np.max(np.abs(model.encode(changed)[:, 1:] - Z[:, 1:]))), 0)
        decoded = model.decode(Z)
        model.encode(changed)
        np.testing.assert_array_equal(model.decode(Z), decoded)
        changed_latent = Z.copy()
        changed_latent[:, -1] += 10
        np.testing.assert_array_equal(model.decode(changed_latent)[:, :-1], decoded[:, :-1])
        changed_latent = Z.copy()
        changed_latent[:, 0] += 10
        self.assertGreater(
            float(np.max(np.abs(model.decode(changed_latent)[:, 1:] - decoded[:, 1:]))), 0
        )

    def test_learning_history_clone_and_reload(self) -> None:
        """Learn reconstruction, keep batch losses and serialize a cloneable estimator.

        Returns:
            None.
        """
        X = np.ones((7, 5, 3), dtype=np.float32)
        state = torch.get_rng_state().clone()
        model = GRUConvAutoencoder(
            hidden_channels=4, latent_channels=2, epochs=30, batch_size=4,
            scheduler=False, learning_rate=0.03, max_grad_norm=1.0,
        ).fit(X)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertLess(model.history_[-1]["train_loss"], model.history_[0]["train_loss"])
        self.assertLess(float(np.mean((model.predict(X) - X) ** 2)), 0.1)
        self.assertIsNone(model.best_epoch_)
        for epoch in model.history_:
            batches = epoch["batches"]
            self.assertEqual([batch["samples"] for batch in batches], [4, 3])
            expected = sum(batch["train_loss"] * batch["samples"] for batch in batches) / 7
            self.assertAlmostEqual(epoch["train_loss"], expected)
        fresh = clone(model)
        self.assertEqual(fresh.get_params(), model.get_params())
        self.assertNotIn("compression", fresh.get_params())
        self.assertFalse(hasattr(fresh, "runner_"))
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(restored.predict(X), model.predict(X))
        np.testing.assert_array_equal(restored.encode(X), model.encode(X))
        model.set_params(epochs=1).fit(np.ones((3, 2, 5), dtype=np.float32))
        self.assertEqual(model.window_shape_, (2, 5))
        self.assertEqual(model.latent_shape_, (2, 2))

    def test_validation_does_not_update_weights(self) -> None:
        """Log validation reconstruction loss and select a checkpoint without training on it.

        Returns:
            None.
        """
        X = np.random.default_rng(9).normal(size=(7, 5, 3)).astype(np.float32)
        params = dict(hidden_channels=4, latent_channels=2, epochs=1, batch_size=4)
        plain = GRUConvAutoencoder(**params).fit(X)
        validation = -2 * X[:3]
        model = GRUConvAutoencoder(**params).fit(X, validation_data=validation)
        np.testing.assert_array_equal(model.predict(X), plain.predict(X))
        self.assertEqual(model.best_epoch_, 1)
        expected = float(np.mean((model.predict(validation) - validation) ** 2))
        self.assertAlmostEqual(model.history_[0]["validation_loss"], expected, places=5)

    def test_export_and_frozen_gru_pipeline(self) -> None:
        """Train on dense windows and use the exported encoder with sparse GRU targets.

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
                {"name": "ae", "type": "gru_conv_autoencoder", "params": {
                    "hidden_channels": 3, "latent_channels": 1, "epochs": 1, "batch_size": 8}},
            ]
            config_path = root / "reconstruction.yaml"
            config_path.write_text(yaml.safe_dump(reconstruction), encoding="utf-8")
            pretrained = run_experiment(load_config(config_path))
            self.assertIsInstance(pretrained.model, GRUConvAutoencoder)
            self.assertEqual(pretrained.model.latent_shape_, (5, 1))
            self.assertEqual(
                sum(b["samples"] for b in pretrained.model.history_[0]["batches"]), 32
            )
            path = pretrained.pipeline.export_transform(
                root / "gru_encoder.pkl", through="ae", mode="encode"
            )
            forecast["pipeline"] = [
                forecast["pipeline"][1],
                {"name": "encoder", "type": "frozen_pipeline", "params": {"path": str(path)}},
                forecast["pipeline"][3],
                {"name": "gru", "type": "gru", "params": {
                    "epochs": 1, "hidden_size": 2, "batch_size": 8}},
            ]
            with patch.object(GRUConvAutoencoder, "fit", side_effect=AssertionError("refit")):
                run = run_experiment(forecast)
            self.assertEqual(run.model.window_shape_, (5, 1))
            self.assertEqual(sum(b["samples"] for b in run.model.history_[0]["batches"]), 7)
            path.unlink()
            restored = load_run(run.artifact_dir)
            data = load_data(**forecast["data"])
            np.testing.assert_array_equal(
                restored.predict(data, "test").y_pred, run.predict(data, "test").y_pred
            )
