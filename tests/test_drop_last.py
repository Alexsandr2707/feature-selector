"""Incomplete training batch handling across the shared runner and estimators."""

import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone
import torch
from torch import nn

from feature_selector.models.autoencoder import ConvAutoencoder, GRUAutoencoder, S4Autoencoder
from feature_selector.models.base import TorchModelRunner
from feature_selector.models.ensemble import GRUEnsembleRegressor
from feature_selector.models.gru import GRURegressor
from feature_selector.tracking import StepContext


class DropLastTests(unittest.TestCase):
    """Drop only training remainders and keep inference and validation complete."""

    def setUp(self) -> None:
        """Limit CPU threading overhead for small neural networks.

        Returns:
            None.
        """
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def test_runner_batches_loss_telemetry_and_complete_evaluation(self) -> None:
        """Report processed samples and retain all validation and inference windows.

        Returns:
            None.
        """
        X = np.arange(14, dtype=np.float32).reshape(7, 2, 1) / 14
        sizes = [3, 3]
        network = nn.Linear(1, 1)
        with torch.no_grad():
            network.weight.zero_()
            network.bias.zero_()
        runner = TorchModelRunner(network, torch.optim.SGD(network.parameters(), lr=0))
        with patch.object(StepContext, "log_event") as log_event:
            runner.fit(
                X, X, (X, X), epochs=1, batch_size=3, patience=1,
                random_state=0,
            )
        epoch = runner.history_[0]
        self.assertEqual([batch["samples"] for batch in epoch["batches"]], sizes)
        expected = sum(b["samples"] * b["train_loss"] for b in epoch["batches"])
        self.assertAlmostEqual(epoch["train_loss"], expected / sum(sizes))
        self.assertAlmostEqual(epoch["validation_loss"], float(np.mean(X ** 2)))
        train_events = [
            call.args[1] for call in log_event.call_args_list
            if call.args[0] == "metrics" and call.args[1].get("split") == "train"
        ]
        self.assertEqual(train_events[-1]["samples"], sum(sizes))
        self.assertEqual(runner.predict(X, batch_size=3).shape, X.shape)

    def test_runner_rejects_empty_epoch_but_accepts_exact_batch(self) -> None:
        """Fail clearly when dropping the only batch would prevent all updates.

        Returns:
            None.
        """
        network = nn.Linear(1, 1)
        runner = TorchModelRunner(network, torch.optim.SGD(network.parameters(), lr=0))
        X = np.ones((2, 4, 1), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "drop_last=True.*batch_size"):
            runner.fit(
                X, X, epochs=1, batch_size=3, patience=1, random_state=0,
            )
        runner.fit(X, X, epochs=1, batch_size=2, patience=1, random_state=0)
        self.assertEqual([b["samples"] for b in runner.history_[0]["batches"]], [2])

    def test_estimators_clone_training_and_prediction(self) -> None:
        """Use full batches in each public estimator without truncating predictions.

        Returns:
            None.
        """
        X = np.random.default_rng(4).normal(size=(7, 4, 2)).astype(np.float32)
        y = np.arange(7, dtype=np.float32)
        for constructor, extra in (
            (ConvAutoencoder, {"hidden_channels": 2, "latent_channels": 1}),
            (GRUAutoencoder, {"hidden_channels": 2, "latent_channels": 1}),
            (S4Autoencoder, {"latent_channels": 1, "state_size": 4}),
            (GRURegressor, {"hidden_size": 2}),
        ):
            with self.subTest(model=constructor.__name__):
                model = constructor(**extra, epochs=1, batch_size=3)
                self.assertNotIn("drop_last", clone(model).get_params())
                model.fit(X, y)
                self.assertEqual([b["samples"] for b in model.history_[0]["batches"]], [3, 3])
                self.assertEqual(len(model.predict(X)), len(X))

    def test_ensemble_batches_and_prediction(self) -> None:
        """Use full batches in every member and retain all prediction samples.

        Returns:
            None.
        """
        X = np.random.default_rng(2).normal(size=(7, 4, 2)).astype(np.float32)
        y = np.arange(7, dtype=np.float32)
        model = GRUEnsembleRegressor(n_estimators=2, gru_params={
            "hidden_size": 2, "epochs": 1, "batch_size": 3,
        }).fit(X, y, validation_data=(X[:2], y[:2]))
        for member in model.estimators_:
            self.assertNotIn("drop_last", member.get_params())
            self.assertEqual([b["samples"] for b in member.history_[0]["batches"]], [3, 3])
        self.assertEqual(len(model.predict(X)), len(X))

if __name__ == "__main__":
    unittest.main()
