"""Numerical contracts for convolutional reconstruction and encoding."""

import json
import pickle
import subprocess
import sys
import unittest

import numpy as np
from numpy.typing import NDArray



class ConvAutoencoderTests(unittest.TestCase):
    """Exercise reconstruction, transformer integration and estimator lifecycle."""

    original_threads: int

    @classmethod
    def setUpClass(cls) -> None:
        """Avoid thread pool overhead for tiny convolutional test batches.

        Returns:
            None.
        """
        import torch

        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        """Restore the caller's CPU threading configuration.

        Returns:
            None.
        """
        import torch

        torch.set_num_threads(cls.original_threads)

    @staticmethod
    def _windows(timesteps: int = 5) -> NDArray[np.float32]:
        """Create reproducible finite training windows.

        Parameters:
            timesteps (int): Positive number of rows per window.

        Returns:
            NDArray[np.float32]: Five windows with three features.
        """
        return np.random.default_rng(4).normal(size=(5, timesteps, 3)).astype(np.float32)

    def test_shapes_and_roundtrip(self) -> None:
        """Preserve complete windows and expose a shorter encoded sequence.

        Returns:
            None.
        """
        from feature_selector.models.autoencoder import ConvAutoencoder

        for length in (1, 4, 5):
            with self.subTest(length=length):
                X = self._windows(length)
                before = X.copy()
                model = ConvAutoencoder(
                    hidden_channels=4,
                    latent_channels=2,
                    epochs=1,
                    batch_size=3,
                    prediction_batch_size=2,
                )
                self.assertIs(model.fit(X), model)
                encoded = model.encode(X)
                self.assertEqual(encoded.shape, (5, (length + 1) // 2, 2))
                prediction = model.predict(X)
                self.assertEqual(prediction.shape, X.shape)
                self.assertEqual(prediction.dtype, np.float32)
                np.testing.assert_allclose(model.decode(encoded), prediction, atol=1e-6)
                np.testing.assert_array_equal(model.transform(X), prediction)
                model.set_params(mode="encode")
                np.testing.assert_array_equal(model.transform(X), encoded)
                np.testing.assert_array_equal(model.predict(X), encoded)
                np.testing.assert_array_equal(
                    model.inverse_transform(encoded), model.decode(encoded)
                )
                np.testing.assert_array_equal(X, before)
                self.assertEqual(model.n_features_in_, 3)

    def test_learning_history_and_rng(self) -> None:
        """Train reconstruction, retain final weights and isolate the random state.

        Returns:
            None.
        """
        import torch

        from feature_selector.models.autoencoder import ConvAutoencoder

        X = np.ones((5, 5, 3), dtype=np.float32)
        state = torch.get_rng_state().clone()
        model = ConvAutoencoder(
            hidden_channels=4,
            latent_channels=2,
            epochs=20,
            batch_size=3,
            scheduler=False,
            learning_rate=0.03,
        ).fit(X)

        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertIsNone(model.best_epoch_)
        self.assertIs(model.history_, model.runner_.history_)
        self.assertEqual(len(model.history_), 20)
        self.assertTrue(all(epoch["validation_loss"] is None for epoch in model.history_))
        self.assertLess(model.history_[-1]["train_loss"], model.history_[0]["train_loss"])
        self.assertLess(float(np.mean((model.predict(X) - X) ** 2)), 0.1)
        for epoch in model.history_:
            self.assertEqual([batch["samples"] for batch in epoch["batches"]], [3, 2])
            expected = sum(b["train_loss"] * b["samples"] for b in epoch["batches"]) / 5
            self.assertAlmostEqual(epoch["train_loss"], expected)
        json.dumps(model.history_, allow_nan=False)

    def test_validation_selects_checkpoint_without_updating_weights(self) -> None:
        """Use validation to select weights without taking gradient steps on it.

        Returns:
            None.
        """
        from feature_selector.models.autoencoder import ConvAutoencoder

        X = self._windows()
        validation = -2 * X
        params = dict(hidden_channels=4, latent_channels=2, epochs=1, batch_size=3)
        plain = ConvAutoencoder(**params).fit(X)
        model = ConvAutoencoder(**params).fit(X, validation_data=validation)
        self.assertEqual(model.best_epoch_, 1)
        np.testing.assert_array_equal(model.predict(X), plain.predict(X))
        expected = float(np.mean((model.predict(validation) - validation) ** 2))
        self.assertAlmostEqual(model.history_[0]["validation_loss"], expected, places=5)

        stopped = ConvAutoencoder(
            hidden_channels=4,
            latent_channels=2,
            epochs=8,
            batch_size=3,
            learning_rate=1e-30,
            min_lr=0,
            patience=1,
        ).fit(X, validation_data=validation)
        self.assertEqual(len(stopped.history_), 2)
        self.assertEqual(stopped.best_epoch_, 1)

    def test_pipeline_clone_and_reload(self) -> None:
        """Support sklearn window pipelines, cloning and serialized predictions.

        Returns:
            None.
        """
        from sklearn.base import clone
        from sklearn.exceptions import NotFittedError
        from sklearn.pipeline import Pipeline

        from feature_selector.models.autoencoder import ConvAutoencoder
        from feature_selector.models.ridge import WindowRidge

        X = self._windows()
        y = X[:, -1, 0]
        encoder = ConvAutoencoder(hidden_channels=4, latent_channels=2, epochs=1)
        pipeline = Pipeline([("encoder", encoder), ("model", WindowRidge())])
        pipeline.fit(X, y)
        self.assertEqual(pipeline.predict(X).shape, (5,))
        restored = pickle.loads(pickle.dumps(pipeline))
        np.testing.assert_array_equal(restored.predict(X), pipeline.predict(X))
        np.testing.assert_array_equal(restored["encoder"].predict(X), encoder.predict(X))
        fresh = clone(encoder)
        self.assertEqual(fresh.get_params(), encoder.get_params())
        self.assertFalse(hasattr(fresh, "runner_"))
        for method in (fresh.predict, fresh.encode, fresh.decode):
            with self.assertRaises(NotFittedError):
                method(X)

    def test_batch_limits_empty_inputs_and_order(self) -> None:
        """Bound inference batches and preserve order for encoding and decoding.

        Returns:
            None.
        """
        import torch

        from feature_selector.models.autoencoder import ConvAutoencoder

        X = self._windows()
        model = ConvAutoencoder(
            hidden_channels=4, latent_channels=2, epochs=1, prediction_batch_size=2
        ).fit(X)
        for method, module, values in (
            (model.predict, model.network_.encoder, X),
            (model.encode, model.network_.encoder, X),
            (model.decode, model.network_.decoder, model.encode(X)),
        ):
            batches: list[int] = []
            hook = module.register_forward_pre_hook(
                lambda module, inputs: batches.append(len(inputs[0]))
            )
            try:
                result = method(values)
            finally:
                hook.remove()
            self.assertEqual(batches, [2, 2, 1])
            model.set_params(prediction_batch_size=5)
            np.testing.assert_allclose(method(values), result, atol=1e-6)
            model.set_params(prediction_batch_size=2)
        self.assertEqual(model.encode(X[:0]).shape, (0, 3, 2))
        self.assertEqual(model.predict(X[:0]).shape, (0, 5, 3))
        self.assertEqual(model.decode(np.empty((0, 3, 2))).shape, (0, 5, 3))
        self.assertFalse(model.network_.training)
        self.assertTrue(
            all(
                parameter.grad is None or not parameter.grad.requires_grad
                for parameter in model.network_.parameters()
            )
        )
        with torch.no_grad():
            expected = model.network_(torch.as_tensor(X)).numpy()
        np.testing.assert_allclose(model.predict(X), expected, atol=1e-6)

    def test_parameter_errors_at_consumption(self) -> None:
        """Delegate optimizer errors and reject empty training loops at fit time.

        Returns:
            None.
        """
        from feature_selector.models.autoencoder import ConvAutoencoder

        model = ConvAutoencoder(learning_rate=-1, epochs=1)
        self.assertEqual(model.learning_rate, -1)
        with self.assertRaises(ValueError):
            model.fit(self._windows())
        with self.assertRaises(ValueError):
            ConvAutoencoder(epochs=0).fit(self._windows())

    def test_invalid_arrays_and_refit(self) -> None:
        """Validate input and latent shapes and replace metadata on repeated fit.

        Returns:
            None.
        """
        from feature_selector.models.autoencoder import ConvAutoencoder

        X = self._windows()
        model = ConvAutoencoder(hidden_channels=4, latent_channels=2, epochs=1)
        for values in (
            np.array(1),
            X[0],
            X[:0],
            X[:, :0],
            X[:, :, :0],
            np.full_like(X, np.nan),
            np.full_like(X, np.inf),
            np.full(X.shape, 1e100),
            X.astype(complex),
            X.astype(str),
            X.tolist(),
        ):
            with self.subTest(shape=np.shape(values)), self.assertRaises(ValueError):
                model.fit(values)
        with self.assertRaises(ValueError):
            model.fit(X, validation_data=X[:, :2])
        self.assertFalse(hasattr(model, "runner_"))
        model.fit(X)
        for values in (X[:, :2], X[:, :, :2], np.full_like(X, np.nan)):
            with self.assertRaises(ValueError):
                model.predict(values)
        with self.assertRaises(ValueError):
            model.decode(X)
        model.fit(np.ones((4, 6, 1)))
        self.assertEqual(model.window_shape_, (6, 1))
        self.assertEqual(model.latent_shape_, (3, 2))
        self.assertEqual(model.n_features_in_, 1)
        with self.assertRaises(ValueError):
            model.encode(X)

    def test_readonly_reversed_views_and_required_dependencies(self) -> None:
        """Accept reversed read-only windows and import all runtime dependencies.

        Returns:
            None.
        """
        from feature_selector.models.autoencoder import ConvAutoencoder

        X = self._windows()[:, ::-1]
        X.flags.writeable = False
        model = ConvAutoencoder(hidden_channels=4, latent_channels=2, epochs=1).fit(X)
        self.assertEqual(model.predict(X).shape, X.shape)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import feature_selector; import feature_selector.factory; "
                "import torch, matplotlib, tensorboard",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
