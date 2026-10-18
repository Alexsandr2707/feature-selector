"""Contracts for training and running reconstruction networks."""

import json
import pickle
import unittest

import numpy as np



class ReconstructionRunnerTests(unittest.TestCase):
    def test_reconstruction_history_and_prediction(self) -> None:
        """Preserve windows, weighted losses, batch limits and restored checkpoints.

        Returns:
            None.
        """
        import torch

        from feature_selector.models.base import TorchModelRunner

        network = torch.nn.Linear(2, 2)
        with torch.no_grad():
            network.weight.zero_()
            network.bias.zero_()
        runner = TorchModelRunner(network, torch.optim.SGD(network.parameters(), lr=0))
        X = np.arange(30, dtype=np.float32).reshape(5, 3, 2) / 30
        runner.fit(X, X, (X, X), epochs=8, batch_size=2, patience=2, random_state=3)

        self.assertEqual(runner.best_epoch_, 1)
        self.assertEqual(len(runner.history_), 3)
        expected_loss = float(np.mean(X ** 2))
        for epoch in runner.history_:
            self.assertEqual([batch["samples"] for batch in epoch["batches"]], [2, 2, 1])
            weighted_loss = sum(b["train_loss"] * b["samples"] for b in epoch["batches"]) / 5
            self.assertAlmostEqual(epoch["train_loss"], weighted_loss)
            self.assertAlmostEqual(epoch["train_loss"], expected_loss)
            self.assertAlmostEqual(epoch["validation_loss"], expected_loss)
        json.dumps(runner.history_, allow_nan=False)

        seen_batches: list[int] = []
        hook = network.register_forward_pre_hook(
            lambda module, inputs: seen_batches.append(len(inputs[0]))
        )
        try:
            prediction = runner.predict(X, batch_size=2)
        finally:
            hook.remove()
        self.assertEqual(seen_batches, [2, 2, 1])
        np.testing.assert_array_equal(prediction, np.zeros_like(X))
        restored = pickle.loads(pickle.dumps(runner))
        np.testing.assert_array_equal(restored.predict(X), prediction)

    def test_reconstruction_training_without_validation(self) -> None:
        """Learn window targets and retain the final weights without validation.

        Returns:
            None.
        """
        import torch

        from feature_selector.models.base import TorchModelRunner

        network = torch.nn.Linear(1, 1)
        with torch.no_grad():
            network.weight.zero_()
            network.bias.zero_()
        runner = TorchModelRunner(network, torch.optim.SGD(network.parameters(), lr=0.25))
        X = np.zeros((5, 3, 1), dtype=np.float32)
        y = np.ones_like(X)
        runner.fit(X, y, epochs=3, batch_size=5, patience=1, random_state=0)

        self.assertEqual(len(runner.history_), 3)
        self.assertIsNone(runner.best_epoch_)
        self.assertTrue(all(epoch["validation_loss"] is None for epoch in runner.history_))
        np.testing.assert_allclose(runner.predict(X), np.full_like(X, 0.875))

    def test_invalid_target_shapes(self) -> None:
        """Reject empty, misaligned and incompatible reconstruction targets.

        Returns:
            None.
        """
        import torch

        from feature_selector.models.base import TorchModelRunner

        X = np.ones((5, 3, 2), dtype=np.float32)
        for targets in (np.array(1), X[:4], X[:, :, :0], X.transpose(0, 2, 1)):
            with self.subTest(shape=targets.shape), self.assertRaises(ValueError):
                network = torch.nn.Linear(2, 2)
                runner = TorchModelRunner(network, torch.optim.SGD(network.parameters(), lr=0))
                runner.fit(X, targets, epochs=1, batch_size=2, patience=1, random_state=0)

        network = torch.nn.Linear(2, 2)
        runner = TorchModelRunner(network, torch.optim.SGD(network.parameters(), lr=0))
        with self.assertRaisesRegex(ValueError, "Validation target shape"):
            runner.fit(
                X, X, (X, X[:, :1]), epochs=1, batch_size=2, patience=1, random_state=0
            )

    def test_reconstruction_restores_best_weights(self) -> None:
        """Restore weights from the best reconstruction validation epoch.

        Returns:
            None.
        """
        import torch

        from feature_selector.models.base import TorchModelRunner

        network = torch.nn.Linear(1, 1)
        with torch.no_grad():
            network.weight.zero_()
            network.bias.zero_()
        runner = TorchModelRunner(network, torch.optim.SGD(network.parameters(), lr=0.25))
        X = np.zeros((5, 3, 1), dtype=np.float32)
        validation_targets = np.full_like(X, 0.5)
        runner.fit(
            X, np.ones_like(X), (X, validation_targets),
            epochs=8, batch_size=5, patience=2, random_state=0,
        )

        self.assertEqual(runner.best_epoch_, 1)
        self.assertEqual(len(runner.history_), 3)
        np.testing.assert_allclose(runner.predict(X), validation_targets, atol=1e-7)
