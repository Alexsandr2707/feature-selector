"""Contracts for PyTorch training and batch history."""

import json
import pickle
import unittest

import numpy as np



class TorchRunnerTests(unittest.TestCase):
    def test_checkpoint_start_ignores_early_minimum(self) -> None:
        """Select weights and count patience only from the first eligible epoch.

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
        X = np.zeros((5, 1), dtype=np.float32)
        train = np.ones(5, dtype=np.float32)
        validation = np.full(5, 0.5, dtype=np.float32)
        runner.fit(
            X, train, (X, validation), epochs=10, batch_size=5, patience=1,
            random_state=0, checkpoint_start_epoch=3,
        )
        self.assertEqual(runner.best_epoch_, 3)
        self.assertEqual(len(runner.history_), 4)
        self.assertEqual(runner.history_[0]["validation_loss"], 0)
        np.testing.assert_allclose(runner.predict(X), 0.875)
        self.assertTrue(all(epoch["batches"] for epoch in runner.history_))

    def test_checkpoint_start_beyond_training_keeps_final_weights(self) -> None:
        """Keep final weights if validation has no eligible checkpoint epoch.

        Returns:
            None.
        """
        import torch
        from feature_selector.models.base import TorchModelRunner

        for use_validation in (True, False):
            with self.subTest(validation=use_validation):
                network = torch.nn.Linear(1, 1)
                with torch.no_grad():
                    network.weight.zero_()
                    network.bias.zero_()
                runner = TorchModelRunner(
                    network, torch.optim.SGD(network.parameters(), lr=0.25)
                )
                X = np.zeros((5, 1), dtype=np.float32)
                y = np.ones(5, dtype=np.float32)
                validation = (X, np.full(5, 0.5, dtype=np.float32)) if use_validation else None
                runner.fit(
                    X, y, validation, epochs=2, batch_size=5, patience=1,
                    random_state=0, checkpoint_start_epoch=3,
                )
                self.assertIsNone(runner.best_epoch_)
                self.assertEqual(len(runner.history_), 2)
                np.testing.assert_allclose(runner.predict(X), 0.75)

    def test_gru_checkpoint_start_delegation(self) -> None:
        """Expose the checkpoint boundary through sklearn parameters and GRU fitting.

        Returns:
            None.
        """
        from sklearn.base import clone
        from unittest.mock import patch
        from feature_selector.models.base import TorchModelRunner
        from feature_selector.models.gru import GRURegressor

        X = np.ones((4, 2, 1), dtype=np.float32)
        y = np.ones(4, dtype=np.float32)
        model = GRURegressor(
            hidden_size=2, epochs=6, patience=1, checkpoint_start_epoch=3
        )
        # The first minimum must not prevent the first eligible checkpoint from being selected.
        with patch.object(TorchModelRunner, "_validation_loss", side_effect=[0, 1, 2, 3]):
            model.fit(X, y, (X, y))
        self.assertEqual(model.best_epoch_, 3)
        self.assertEqual(len(model.history_), 4)
        self.assertEqual(clone(model).checkpoint_start_epoch, 3)
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(restored.predict(X), model.predict(X))

    def test_training_without_validation_keeps_final_weights(self) -> None:
        """Run all epochs without early stopping or selecting a checkpoint."""
        import torch
        from feature_selector.models.base import TorchModelRunner

        network = torch.nn.Linear(1, 1)
        with torch.no_grad():
            network.weight.zero_()
            network.bias.zero_()
        runner = TorchModelRunner(network, torch.optim.SGD(network.parameters(), lr=0.25))
        X = np.zeros((5, 1), dtype=np.float32)
        y = np.ones(5, dtype=np.float32)
        runner.fit(X, y, epochs=3, batch_size=5, patience=1, random_state=0)

        self.assertEqual(len(runner.history_), 3)
        self.assertIsNone(runner.best_epoch_)
        self.assertTrue(all(epoch["validation_loss"] is None for epoch in runner.history_))
        np.testing.assert_allclose(runner.predict(X), 0.875)
        json.dumps(runner.history_, allow_nan=False)

    def test_column_predictions_use_samplewise_loss(self) -> None:
        """Column outputs must preserve samplewise MSE and one-dimensional predictions."""
        import torch
        from feature_selector.models.base import TorchModelRunner

        network = torch.nn.Linear(1, 1)
        with torch.no_grad():
            network.weight.fill_(1)
            network.bias.zero_()
        runner = TorchModelRunner(network, torch.optim.SGD(network.parameters(), lr=0))
        X = np.arange(5, dtype=np.float32).reshape(-1, 1)
        y = 2 * X[:, 0]
        runner.fit(X, y, (X, y), epochs=1, batch_size=2, patience=1, random_state=3)

        expected_loss = float(np.mean((X[:, 0] - y) ** 2))
        self.assertAlmostEqual(runner.history_[0]["train_loss"], expected_loss)
        self.assertAlmostEqual(runner.history_[0]["validation_loss"], expected_loss)
        prediction = runner.predict(X, batch_size=2)
        self.assertEqual(prediction.shape, (len(X),))
        np.testing.assert_array_equal(prediction, X[:, 0])

    def test_batch_history_early_stopping_and_reload(self) -> None:
        """Keep unequal batch sizes, epoch averaging and restored weights consistent."""
        import torch
        from feature_selector.models.base import TorchModelRunner

        network = torch.nn.Linear(2, 1)
        model = torch.nn.Sequential(network, torch.nn.Flatten(start_dim=0))
        with torch.no_grad():
            network.weight.zero_()
            network.bias.zero_()
        optimizer = torch.optim.SGD(model.parameters(), lr=0)
        runner = TorchModelRunner(model, optimizer)
        X = np.zeros((5, 2), dtype=np.float32)
        y = np.arange(5, dtype=np.float32)
        runner.fit(X, y, (X, y), epochs=8, batch_size=2, patience=2, random_state=3)

        self.assertEqual(runner.best_epoch_, 1)
        self.assertEqual(len(runner.history_), 3)
        for epoch in runner.history_:
            batches = epoch["batches"]
            self.assertEqual([batch["batch"] for batch in batches], [1, 2, 3])
            self.assertEqual([batch["samples"] for batch in batches], [2, 2, 1])
            weighted_loss = sum(b["train_loss"] * b["samples"] for b in batches) / 5
            self.assertAlmostEqual(epoch["train_loss"], weighted_loss)
            self.assertAlmostEqual(weighted_loss, float(np.mean(y ** 2)))
        json.dumps(runner.history_, allow_nan=False)
        restored = pickle.loads(pickle.dumps(runner))
        np.testing.assert_array_equal(restored.predict(X), runner.predict(X))

    def test_restores_best_validation_weights(self) -> None:
        """Restore the best checkpoint after later optimizer steps worsen validation loss."""
        import torch
        from feature_selector.models.base import TorchModelRunner

        network = torch.nn.Linear(1, 1)
        with torch.no_grad():
            network.weight.zero_()
            network.bias.zero_()
        runner = TorchModelRunner(network, torch.optim.SGD(network.parameters(), lr=0.25))
        X = np.zeros((5, 1), dtype=np.float32)
        y_train = np.ones(5, dtype=np.float32)
        y_validation = np.full(5, 0.5, dtype=np.float32)

        # Bias moves through 0.5, 0.75 and 0.875; validation is best at the first step.
        runner.fit(
            X, y_train, (X, y_validation),
            epochs=8, batch_size=5, patience=2, random_state=3,
        )

        self.assertEqual(runner.best_epoch_, 1)
        self.assertEqual(len(runner.history_), 3)
        self.assertGreater(
            runner.history_[-1]["validation_loss"], runner.history_[0]["validation_loss"]
        )
        prediction = runner.predict(X)
        np.testing.assert_allclose(prediction, y_validation, atol=1e-7)
        restored_loss = float(np.mean((prediction - y_validation) ** 2))
        self.assertAlmostEqual(
            restored_loss, min(epoch["validation_loss"] for epoch in runner.history_)
        )

    def test_gru_delegation_and_rng(self) -> None:
        """Expose runner history while retaining sklearn cloning and RNG isolation."""
        import torch
        from sklearn.base import clone
        from feature_selector.models.gru import GRURegressor

        X = np.random.default_rng(4).normal(size=(5, 2, 1)).astype(np.float32)
        y = X[:, -1, 0].copy()
        state = torch.get_rng_state().clone()
        model = GRURegressor(
            hidden_size=2, epochs=1, batch_size=3, prediction_batch_size=2
        )
        model.fit(X, y, (X, y))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertIs(model.history_, model.runner_.history_)
        self.assertEqual(len(model.history_[0]["batches"]), 2)
        np.testing.assert_array_equal(
            model.predict(X), model.runner_.predict(X, batch_size=2)
        )
        self.assertEqual(clone(model).prediction_batch_size, 2)
        self.assertFalse(hasattr(clone(model), "runner_"))

    def test_prediction_batch_size(self) -> None:
        """Respect batch limits and preserve sample order for incomplete batches."""
        import torch
        from feature_selector.models.base import TorchModelRunner

        model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Flatten(start_dim=0))
        runner = TorchModelRunner(model, torch.optim.SGD(model.parameters(), lr=0))
        X = np.arange(14, dtype=np.float32).reshape(7, 2)
        seen_batches: list[int] = []
        hook = model.register_forward_pre_hook(
            lambda module, inputs: seen_batches.append(len(inputs[0]))
        )
        try:
            prediction = runner.predict(X, batch_size=3)
        finally:
            hook.remove()
        self.assertEqual(sum(seen_batches), len(X))
        self.assertLessEqual(max(seen_batches), 3)
        np.testing.assert_allclose(prediction, runner.predict(X, batch_size=7), atol=1e-6)
        for invalid in (0, -1, True, 1.5):
            with self.subTest(batch_size=invalid), self.assertRaises(ValueError):
                runner.predict(X, batch_size=invalid)
