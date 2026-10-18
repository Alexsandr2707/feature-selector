"""Optional clipping bounds gradients before optimizer updates for all torch models."""

import unittest
from typing import Literal
from unittest.mock import patch

import numpy as np
from sklearn.base import clone
import torch

from feature_selector.models.autoencoder import ConvAutoencoder
from feature_selector.models.base import TorchModelRunner
from feature_selector.models.gru import GRURegressor


class GradientClippingTests(unittest.TestCase):
    """Verify actual updates, disabled clipping and model parameter forwarding."""

    def setUp(self) -> None:
        """Keep small CPU training runs inexpensive.

        Returns:
            None.
        """
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def test_joint_gradient_norm_before_optimizer_step(self) -> None:
        """Clip weight and bias gradients together before the optimizer consumes them.

        Returns:
            None.
        """
        for limit in (None, 1.0):
            with self.subTest(max_grad_norm=limit):
                network = torch.nn.Linear(1, 1)
                with torch.no_grad():
                    network.weight.zero_()
                    network.bias.zero_()
                optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
                runner = TorchModelRunner(network, optimizer)
                norms: list[float] = []
                original_step = optimizer.step

                def observed_step() -> None:
                    """Record the joint norm immediately before the real weight update.

                    Returns:
                        None.
                    """
                    gradients = [p.grad.flatten() for p in network.parameters()
                                 if p.grad is not None]
                    norms.append(float(torch.linalg.vector_norm(torch.cat(gradients))))
                    original_step()

                X = np.ones((3, 1), dtype=np.float32)
                y = np.full(3, 10, dtype=np.float32)
                with patch.object(optimizer, "step", observed_step):
                    runner.fit(
                        X, y, epochs=1, batch_size=3, patience=1, random_state=0,
                        max_grad_norm=limit,
                    )
                expected_norm = np.sqrt(800) if limit is None else limit
                self.assertAlmostEqual(norms[0], expected_norm, places=5)
                expected_prediction = 4 if limit is None else np.sqrt(2) * 0.1
                np.testing.assert_allclose(runner.predict(X), expected_prediction, rtol=1e-6)
                self.assertEqual(runner.history_[0]["train_loss"], 100)

    def test_gru_and_autoencoder_forward_optional_limit(self) -> None:
        """Expose the limit through sklearn parameters and both autoencoder architectures.

        Returns:
            None.
        """
        X = np.ones((4, 3, 2), dtype=np.float32)
        y = np.ones(4, dtype=np.float32)
        compressions: tuple[Literal["features_time", "features"], ...] = (
            "features_time", "features"
        )
        models = [
            GRURegressor(hidden_size=2, epochs=1, batch_size=2, max_grad_norm=0.5),
            *[ConvAutoencoder(
                compression=compression, hidden_channels=2, latent_channels=1,
                epochs=1, batch_size=2, max_grad_norm=0.5,
            ) for compression in compressions],
        ]
        original_clip = torch.nn.utils.clip_grad_norm_
        for model in models:
            with self.subTest(model=type(model).__name__, params=model.get_params()):
                self.assertEqual(clone(model).max_grad_norm, 0.5)
                with patch("torch.nn.utils.clip_grad_norm_", wraps=original_clip) as clip:
                    model.fit(X, y)
                self.assertEqual(clip.call_count, 2)
                self.assertTrue(all(call.args[1] == 0.5 for call in clip.call_args_list))
                model.set_params(max_grad_norm=None)
                with patch("torch.nn.utils.clip_grad_norm_", wraps=original_clip) as clip:
                    model.fit(X, y)
                clip.assert_not_called()

    def test_invalid_limits_fail_before_updates(self) -> None:
        """Reject limits that could silently disable clipping or reverse gradients.

        Returns:
            None.
        """
        for limit in (0, -1, float("inf"), float("nan")):
            with self.subTest(max_grad_norm=limit):
                network = torch.nn.Linear(1, 1)
                optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
                runner = TorchModelRunner(network, optimizer)
                X = np.ones((2, 1), dtype=np.float32)
                with patch.object(optimizer, "step") as step:
                    with self.assertRaisesRegex(ValueError, "max_grad_norm"):
                        runner.fit(
                            X, X[:, 0], epochs=1, batch_size=2, patience=1,
                            random_state=0, max_grad_norm=limit,
                        )
                step.assert_not_called()
