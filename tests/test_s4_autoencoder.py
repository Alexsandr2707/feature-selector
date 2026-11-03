"""Full S4 reconstruction, feature compression and frozen encoder reuse."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
import torch

from feature_selector.factory import build_pipeline
from feature_selector.models.autoencoder import S4Autoencoder
from feature_selector.pipeline.adapters import ModelStep, TransformerStep
from feature_selector.pipeline.core import ConfiguredPipeline
from feature_selector.pipeline.frozen import FrozenPipeline
from feature_selector.pipeline.temporal import Windows
from support import sample_data


class S4AutoencoderTests(unittest.TestCase):
    """Check temporal semantics, optimizer settings and estimator lifecycle on CPU."""

    original_threads: int

    @classmethod
    def setUpClass(cls) -> None:
        """Limit thread overhead for small FFT-based networks.

        Returns:
            None.
        """
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        """Restore the caller's CPU threading configuration.

        Returns:
            None.
        """
        torch.set_num_threads(cls.original_threads)

    @staticmethod
    def _windows(length: int = 5) -> NDArray[np.float32]:
        """Create complete windows with independent features.

        Parameters:
            length (int): Positive number of timesteps per window.

        Returns:
            NDArray[np.float32]: Seven windows with six input features.
        """
        return np.random.default_rng(7).normal(size=(7, length, 6)).astype(np.float32)

    def test_shapes_modes_empty_inputs_and_batch_independence(self) -> None:
        """Preserve temporal length across modes, empty inputs and inference batch sizes.

        Returns:
            None.
        """
        for length in (1, 4, 5, 48):
            with self.subTest(length=length):
                X = self._windows(length)
                original = X.copy()
                model = S4Autoencoder(
                    latent_channels=2, state_size=8, epochs=1,
                    batch_size=3, prediction_batch_size=2,
                ).fit(X)
                Z = model.encode(X)
                prediction = model.predict(X)
                self.assertEqual(Z.shape, (7, length, 2))
                self.assertEqual(model.latent_shape_, (length, 2))
                self.assertEqual(prediction.shape, X.shape)
                self.assertEqual(prediction.dtype, np.float32)
                np.testing.assert_array_equal(X, original)
                np.testing.assert_allclose(model.decode(Z), prediction, atol=1e-6)
                self.assertEqual(model.encode(X[:0]).shape, (0, length, 2))
                self.assertEqual(model.decode(Z[:0]).shape, (0, length, 6))
                self.assertEqual(model.predict(X[:0]).shape, (0, length, 6))
                model.set_params(prediction_batch_size=7)
                np.testing.assert_allclose(model.encode(X), Z, atol=1e-6)
                np.testing.assert_allclose(model.predict(X[::-1]), prediction[::-1], atol=1e-6)
                model.set_params(mode="encode")
                np.testing.assert_allclose(model.transform(X), Z, atol=1e-6)
                np.testing.assert_allclose(model.predict(X), Z, atol=1e-6)
                np.testing.assert_allclose(model.inverse_transform(Z), prediction, atol=1e-6)

    def test_causal_context_and_latent_only_decoding(self) -> None:
        """Use past context without passing encoder information around the latent sequence.

        Returns:
            None.
        """
        X = self._windows()
        model = S4Autoencoder(
            batch_size=1,
            latent_channels=2, state_size=8, epochs=1,
        ).fit(X)
        Z = model.encode(X)
        changed = X.copy()
        changed[:, 0, 0] += 10
        difference = model.encode(changed)[:, 1:] - Z[:, 1:]
        self.assertGreater(float(np.max(np.abs(difference))), 1e-6)
        changed = X.copy()
        changed[:, -1, 0] += 10
        np.testing.assert_allclose(model.encode(changed)[:, :-1], Z[:, :-1], atol=1e-6)

        decoded = model.decode(Z)
        model.encode(X * 2)
        np.testing.assert_array_equal(model.decode(Z), decoded)
        changed_latent = Z.copy()
        changed_latent[:, -1, 0] += 10
        np.testing.assert_allclose(model.decode(changed_latent)[:, :-1], decoded[:, :-1], atol=1e-6)
        changed_latent = Z.copy()
        changed_latent[:, 0, 0] += 10
        difference = model.decode(changed_latent)[:, 1:] - decoded[:, 1:]
        self.assertGreater(float(np.max(np.abs(difference))), 1e-6)

    def test_learning_history_finite_gradients_and_rng_isolation(self) -> None:
        """Learn reconstruction with finite gradients, weighted losses and isolated RNG.

        Returns:
            None.
        """
        X = np.ones((7, 5, 3), dtype=np.float32)
        state = torch.get_rng_state().clone()
        model = S4Autoencoder(
            latent_channels=2, state_size=8, epochs=25, batch_size=4,
            scheduler=False, learning_rate=0.03, ssm_learning_rate=0.003, max_grad_norm=1.0,
        ).fit(X, y=np.full(7, np.nan))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertLess(model.history_[-1]["train_loss"], model.history_[0]["train_loss"])
        self.assertLess(float(np.mean((model.predict(X) - X) ** 2)), 0.1)
        self.assertIsNone(model.best_epoch_)
        for epoch in model.history_:
            self.assertIsNone(epoch["validation_loss"])
            batches = epoch["batches"]
            self.assertEqual([batch["samples"] for batch in batches], [4])
            expected = sum(batch["train_loss"] * batch["samples"] for batch in batches) / 4
            self.assertAlmostEqual(epoch["train_loss"], expected)

        model.network_.zero_grad()
        values = torch.tensor(X, requires_grad=True)
        model.network_(values).square().mean().backward()
        input_gradient = values.grad
        if input_gradient is None:
            self.fail("Reconstruction must propagate gradients to its input")
        self.assertTrue(bool(torch.isfinite(input_gradient).all()))
        for parameter in model.network_.parameters():
            gradient = parameter.grad
            if gradient is None:
                self.fail("Every trainable parameter must receive a gradient")
            self.assertTrue(bool(torch.isfinite(gradient).all()))

    def test_optimizer_groups_and_common_scheduler_minimum(self) -> None:
        """Apply S4 rates and decay overrides while retaining ordinary-parameter telemetry.

        Returns:
            None.
        """
        model = S4Autoencoder(
            batch_size=1,
            latent_channels=2, state_size=8, epochs=1,
            learning_rate=0.02, ssm_learning_rate=0.002, weight_decay=0.05, scheduler=False,
        ).fit(self._windows())
        groups = model.runner_.optimizer.param_groups
        self.assertEqual(groups[0]["lr"], 0.02)
        grouped = [parameter for group in groups for parameter in group["params"]]
        self.assertEqual(len(grouped), len({id(parameter) for parameter in grouped}))
        self.assertEqual({id(p) for p in grouped}, {id(p) for p in model.network_.parameters()})
        by_parameter = {id(p): group for group in groups for p in group["params"]}
        for name, parameter in model.network_.named_parameters():
            group = by_parameter[id(parameter)]
            if ".s4.layer.kernel." in name:
                self.assertEqual(group["lr"], 0.002)
                self.assertEqual(group["weight_decay"], 0.05 if name.endswith(".C") else 0.0)
            else:
                self.assertEqual((group["lr"], group["weight_decay"]), (0.02, 0.05))
        self.assertEqual(model.history_[0]["batches"][0]["learning_rate"], 0.02)

        model.set_params(scheduler=True, epochs=2, min_lr=0.0001).fit(self._windows())
        for group in model.runner_.optimizer.param_groups:
            self.assertAlmostEqual(group["lr"], 0.0001)

    def test_validation_checkpoint_and_deterministic_training(self) -> None:
        """Keep validation out of weight updates and restore the selected checkpoint.

        Returns:
            None.
        """
        X = self._windows()
        params = dict(batch_size=1, latent_channels=2, state_size=8, epochs=1)
        plain = S4Autoencoder(**params).fit(X)
        validated = S4Autoencoder(**params).fit(X, validation_data=X * 2)
        self.assertEqual(validated.best_epoch_, 1)
        self.assertIsNotNone(validated.history_[0]["validation_loss"])
        for name, value in plain.network_.state_dict().items():
            torch.testing.assert_close(value, validated.network_.state_dict()[name], rtol=0, atol=0)

        stopped = S4Autoencoder(
            batch_size=1,
            latent_channels=2, state_size=8, epochs=5,
            patience=1, scheduler=False, learning_rate=1e-30, ssm_learning_rate=1e-30,
        ).fit(X, validation_data=X)
        self.assertEqual(len(stopped.history_), 2)
        self.assertEqual(stopped.best_epoch_, 1)

    def test_clone_pickle_and_refit(self) -> None:
        """Serialize full S4 state, clone constructor parameters and replace fitted shapes.

        Returns:
            None.
        """
        X = self._windows()
        model = S4Autoencoder(
            batch_size=1,
            latent_channels=2, epochs=1,
        ).fit(X)
        fresh = clone(model)
        self.assertEqual(fresh.get_params(), model.get_params())
        self.assertNotIn("compression", fresh.get_params())
        self.assertNotIn("hidden_channels", fresh.get_params())
        self.assertNotIn("hidden_size", fresh.get_params())
        for operation, values in ((fresh.predict, X), (fresh.encode, X), (fresh.decode, X)):
            with self.assertRaises(NotFittedError):
                operation(values)
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(restored.predict(X), model.predict(X))
        np.testing.assert_array_equal(restored.encode(X), model.encode(X))
        model.fit(np.ones((3, 2, 5), dtype=np.float32))
        self.assertEqual(model.window_shape_, (2, 5))
        self.assertEqual(model.latent_shape_, (2, 2))
        self.assertEqual(model.n_features_in_, 5)

    def test_frozen_export_preserves_metadata_and_predicts_without_source_file(self) -> None:
        """Reuse S4 latents before GRU without refitting or retaining the export file.

        Returns:
            None.
        """
        data = sample_data(sparse=True)
        prefix = ConfiguredPipeline([
            ("scale", TransformerStep(StandardScaler())),
            ("windows", Windows(5)),
        ])
        pretrained = build_pipeline([
            *prefix.steps,
            {"name": "ae", "type": "s4_autoencoder", "params": {
                "latent_channels": 1, "state_size": 8,
                "epochs": 1, "batch_size": 8,
            }},
        ]).fit(replace(data, y=None))
        encoder_step = pretrained["ae"]
        if not isinstance(encoder_step, ModelStep):
            self.fail("The factory must adapt S4 as a model step")
        estimator = encoder_step.estimator
        if not isinstance(estimator, S4Autoencoder):
            self.fail("The factory must construct an S4 autoencoder")
        weights = deepcopy(estimator.network_.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = pretrained.export_transform(Path(directory) / "encoder.pkl",
                                                through="ae", mode="encode")
            pipeline = build_pipeline([
                {"name": "encoder", "type": "frozen_pipeline", "params": {"path": str(path)}},
                {"name": "gru", "type": "gru", "params": {
                    "hidden_size": 2, "epochs": 1, "batch_size": 2,
                }},
            ])
            with patch.object(S4Autoencoder, "fit", side_effect=AssertionError("refit")):
                pipeline.fit(data)
            frozen = pipeline["encoder"]
            if not isinstance(frozen, FrozenPipeline):
                self.fail("The pipeline must contain the frozen encoder adapter")
            encoded = frozen.transform(data)
            windows = pretrained.transform(data, through="windows")
            selected = windows.select(windows.valid())
            pd.testing.assert_frame_equal(encoded.samples, selected.samples)
            np.testing.assert_array_equal(encoded.y, selected.y)
            self.assertEqual(encoded.temporal, selected.temporal)
            self.assertEqual(encoded.X.shape, (len(selected.X), 5, 1))
            self.assertEqual(encoded.feature_names, ("z0",))
            self.assertEqual(estimator.mode, "reconstruct")
            nested_step = frozen.artifact_.pipeline["ae"]
            if not isinstance(nested_step, ModelStep):
                self.fail("The exported prefix must contain the model adapter")
            nested = nested_step.estimator
            if not isinstance(nested, S4Autoencoder):
                self.fail("The exported model must retain its S4 estimator")
            for name, value in weights.items():
                torch.testing.assert_close(
                    value, nested.network_.state_dict()[name], rtol=0, atol=0,
                )
            prediction = pipeline.predict(data, splits=("test",))
            restored = pickle.loads(pickle.dumps(pipeline))
            path.unlink()
            np.testing.assert_array_equal(
                restored.predict(data, splits=("test",)).values, prediction.values,
            )

    def test_tied_dropout_training_and_inference(self) -> None:
        """Share training masks across time and disable dropout for inference.

        Returns:
            None.
        """
        model = S4Autoencoder(
            state_size=8, latent_channels=2, epochs=1, batch_size=3, dropout=0.5,
        )
        self.assertNotIn("tie_dropout", model.get_params())
        self.assertEqual(clone(model).dropout, 0.5)
        X = self._windows()
        model.fit(X)
        for block in (model.network_.encoder, model.network_.decoder):
            for dropout in (block.s4.drop, block.s4.layer.drop):
                with self.subTest(dropout=dropout):
                    values = torch.ones((8, 6, 5) if dropout.transposed else (8, 5, 6))
                    dropout.train()
                    dropped = dropout(values)
                    temporal_axis = -1 if dropout.transposed else 1
                    first = dropped.select(temporal_axis, 0).unsqueeze(temporal_axis)
                    torch.testing.assert_close(dropped, first.expand_as(dropped))
                    self.assertTrue(bool((dropped == 0).any()))
                    self.assertTrue(bool((dropped == 2).any()))
                    dropout.eval()
                    torch.testing.assert_close(dropout(values), values)
        prediction = model.predict(X)
        np.testing.assert_array_equal(model.predict(X), prediction)
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(restored.predict(X), prediction)

    def test_s4_parameter_constraints_are_checked_at_fit(self) -> None:
        """Reject state sizes and rates unsupported by the S4 integration when used.

        Returns:
            None.
        """
        for size in (0, -2, 3, True, 4.0):
            with self.subTest(state_size=size):
                model = S4Autoencoder(batch_size=1, state_size=size, epochs=1)
                with self.assertRaisesRegex(ValueError, "state_size"):
                    model.fit(self._windows())
        for rate in (0, -0.1, np.inf, np.nan, True):
            with self.subTest(ssm_learning_rate=rate):
                model = S4Autoencoder(batch_size=1, ssm_learning_rate=rate, epochs=1)
                with self.assertRaisesRegex(ValueError, "ssm_learning_rate"):
                    model.fit(self._windows())
        for probability in (-0.1, 1, np.inf, np.nan, True):
            with self.subTest(dropout=probability):
                model = S4Autoencoder(batch_size=1, dropout=probability, epochs=1)
                with self.assertRaisesRegex(ValueError, "dropout"):
                    model.fit(self._windows())


if __name__ == "__main__":
    unittest.main()
