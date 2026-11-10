"""Contracts for validation-trained GRU stacking and scoped training telemetry."""

import json
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from feature_selector import run_experiment
from feature_selector.artifacts import load_run
from feature_selector.models.base import TorchModelRunner
from feature_selector.models.ensemble import GRUEnsembleRegressor
from feature_selector.models.gru import GRURegressor
from feature_selector.pipeline.adapters import ModelStep
from feature_selector.tracking import RunTracker, StepContext
from support import write_config


def sample_windows() -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Create distinct train and validation examples for stacking checks.

    Returns:
        tuple[NDArray[np.float32], NDArray[np.float32]]: Eight windows and scalar targets.
    """
    X = np.random.default_rng(4).normal(size=(8, 2, 1)).astype(np.float32)
    y = (2 * X[:, -1, 0] + 3).astype(np.float32)
    return X, y


def small_ensemble(patience: int = 4) -> GRUEnsembleRegressor:
    """Configure inexpensive GRUs with distinct initialization seeds.

    Parameters:
        patience (int): Epochs without improvement before stopping a GRU.

    Returns:
        GRUEnsembleRegressor: Unfitted two-member CPU ensemble.
    """
    return GRUEnsembleRegressor(
        n_estimators=2,
        gru_params={"hidden_size": 2, "epochs": 4, "batch_size": 3, "patience": patience},
        random_state=7,
        alpha=0.2,
    )


class EnsembleTests(unittest.TestCase):
    def test_ridge_uses_all_validation_and_grus_use_train(self) -> None:
        """Train Ridge on member validation predictions and preserve train-only GRU updates.

        Returns:
            None.
        """
        X, y = sample_windows()
        validation = (X[5:], y[5:])
        model = small_ensemble()
        original_fit = GRURegressor.fit

        with patch.object(GRURegressor, "fit", autospec=True, side_effect=original_fit) as fit:
            self.assertIs(model.fit(X[:5], y[:5], validation), model)

        self.assertEqual(fit.call_count, 2)
        for call in fit.call_args_list:
            np.testing.assert_array_equal(call.args[1], X[:5])
            np.testing.assert_array_equal(call.args[2], y[:5])
            self.assertIs(call.kwargs["validation_data"], validation)

        meta_X = np.column_stack([member.predict(X[5:]) for member in model.estimators_])
        expected = Ridge(alpha=0.2).fit(meta_X, y[5:])
        np.testing.assert_allclose(model.meta_model_.coef_, expected.coef_)
        np.testing.assert_allclose(model.meta_model_.intercept_, expected.intercept_)

        self.assertEqual(model.predict(X).shape, (8,))
        np.testing.assert_allclose(
            model.predict(X),
            expected.predict(np.column_stack([m.predict(X) for m in model.estimators_])),
        )

    def test_large_patience_runs_all_epochs_and_keeps_best_weights(self) -> None:
        """Run every epoch while retaining validation loss and the best checkpoint.

        Returns:
            None.
        """
        X, y = sample_windows()

        with patch.object(TorchModelRunner, "_validation_loss", side_effect=[1, 2, 3, 4] * 2):
            model = small_ensemble().fit(X[:5], y[:5], (X[5:], y[5:]))

        for member in model.estimators_:
            self.assertEqual(len(member.history_), 4)
            self.assertEqual(member.best_epoch_, 1)
            self.assertEqual(
                [record["validation_loss"] for record in member.history_], [1, 2, 3, 4]
            )

            reference = clone(member).set_params(epochs=1).fit(X[:5], y[:5])
            np.testing.assert_array_equal(member.predict(X), reference.predict(X))

        self.assertEqual([member.random_state for member in model.estimators_], [7, 8])
        self.assertFalse(
            np.array_equal(model.estimators_[0].predict(X), model.estimators_[1].predict(X))
        )

        json.dumps(model.history_, allow_nan=False)

    def test_enabled_stopping_selects_checkpoints_on_same_validation(self) -> None:
        """Stop each GRU independently and pass the entire validation pair to both.

        Returns:
            None.
        """
        X, y = sample_windows()
        validation = (X[5:], y[5:])
        model = small_ensemble(patience=1)
        original_fit = GRURegressor.fit

        with (
            patch.object(TorchModelRunner, "_validation_loss", side_effect=[1, 2, 3, 1, 2, 3]),
            patch.object(GRURegressor, "fit", autospec=True, side_effect=original_fit) as fit,
        ):
            model.fit(X[:5], y[:5], validation)

        for call in fit.call_args_list:
            self.assertIs(call.kwargs["validation_data"], validation)
        self.assertEqual([len(m.history_) for m in model.estimators_], [2, 3])
        self.assertEqual([m.best_epoch_ for m in model.estimators_], [1, 2])

        meta_X = np.column_stack([member.predict(X[5:]) for member in model.estimators_])
        expected = Ridge(alpha=0.2).fit(meta_X, y[5:])
        np.testing.assert_array_equal(model.meta_model_.coef_, expected.coef_)

    def test_clone_reload_reproducibility_and_refit(self) -> None:
        """Keep configuration clonable, saved predictions stable and refits independent.

        Returns:
            None.
        """
        X, y = sample_windows()
        params = {"batch_size": 1, "hidden_size": 2, "epochs": 1}
        model = GRUEnsembleRegressor(n_estimators=2, gru_params=params)

        with self.assertRaises(NotFittedError):
            model.predict(X)

        model.fit(X[:5], y[:5], (X[5:], y[5:]))

        unfitted = clone(model)
        self.assertFalse(hasattr(unfitted, "meta_model_"))
        unfitted.fit(X[:5], y[:5], (X[5:], y[5:]))
        np.testing.assert_array_equal(unfitted.predict(X), model.predict(X))

        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(restored.predict(X), model.predict(X))

        old_member = model.estimators_[0]
        model.set_params(n_estimators=1)
        model.fit(X[:5], y[:5], (X[5:], y[5:]))
        self.assertEqual(len(model.estimators_), 1)
        self.assertEqual(len(model.history_), 1)
        self.assertEqual(model.meta_model_.n_features_in_, 1)
        self.assertIsNot(model.estimators_[0], old_member)
        self.assertEqual(params, {"batch_size": 1, "hidden_size": 2, "epochs": 1})

    def test_missing_or_invalid_validation_rejected_before_training(self) -> None:
        """Require validation for Ridge and delegate finite-value checks to the runner.

        Returns:
            None.
        """
        X, y = sample_windows()

        for validation in (
            None,
            (X[:0], y[:0]),
            (X, y[:, None]),
            (X[:, :, :0], y),
            (X[:, :1], y),
            (X, np.full(8, np.nan)),
            (X, y[:2]),
        ):
            model = small_ensemble()
            with self.subTest(validation=validation), self.assertRaises(ValueError):
                model.fit(X, y, validation)
            self.assertFalse(hasattr(model, "meta_model_"))

    def test_parameter_and_prediction_validation(self) -> None:
        """Reject invalid ensemble settings and incompatible prediction windows.

        Returns:
            None.
        """
        X, y = sample_windows()

        for params in (
            {"n_estimators": 0},
            {"n_estimators": True},
            {"n_estimators": 1.5},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                GRUEnsembleRegressor(**params).fit(X, y, (X, y))

        for params in ({"random_state": 5}, {"unknown": 1}):
            with self.subTest(gru_params=params), self.assertRaises(TypeError):
                GRUEnsembleRegressor(gru_params=params).fit(X, y, (X, y))

        model = small_ensemble().fit(X[:5], y[:5], (X[5:], y[5:]))
        for invalid in (X[:, :1], X[:0], np.full_like(X, np.nan), X[:, :, 0]):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                model.predict(invalid)

    def test_saved_history_jsonl_tensorboard_and_metadata_agree(self) -> None:
        """Keep member telemetry separate and persist validation curves for large patience.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            spec = write_config(Path(directory))
            spec["pipeline"][-1] = {
                "name": "model",
                "type": "gru_ensemble",
                "params": small_ensemble().get_params(deep=False),
            }
            config_path = Path(directory) / "ensemble.yaml"
            config_path.write_text(yaml.safe_dump(spec), encoding="utf-8")

            run = run_experiment(config_path)

            cloned = clone(run.pipeline)
            model_step = cloned["model"]
            assert isinstance(model_step, ModelStep)
            self.assertIsInstance(model_step.estimator, GRUEnsembleRegressor)
            self.assertFalse(hasattr(model_step.estimator, "meta_model_"))
            self.assertEqual(
                model_step.estimator.get_params(deep=False), run.model.get_params(deep=False)
            )

            root = run.artifact_dir
            records = [
                json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()
            ]
            metadata = json.loads((root / "metadata.json").read_text())["steps"]
            histories = json.loads((root / "history.json").read_text())["model"]["history"]
            tensorboard = EventAccumulator(str(root / "tensorboard")).Reload()
            self.assertFalse(any(path.is_dir() for path in (root / "tensorboard").iterdir()))

            for index, member in enumerate(run.model.estimators_):
                path = f"pipeline/model/gru_{index}"
                self.assertEqual(histories[index]["history"], member.history_)
                self.assertEqual(histories[index]["best_epoch"], member.best_epoch_)
                self.assertEqual(metadata[path]["random_state"], 7 + index)
                self.assertEqual(metadata[path]["best_epoch"], member.best_epoch_)
                self.assertEqual(metadata[path]["stopping_reason"], "epochs_completed")
                self.assertTrue(metadata[path]["checkpoint_restored"])
                self.assertIn(path, (root / "run.log").read_text())

                metrics = [
                    record["data"]
                    for record in records
                    if record["path"] == path and record["event"] == "metrics"
                ]
                expected_batches = [
                    batch for epoch in member.history_ for batch in epoch["batches"]
                ]
                batches = [record for record in metrics if record["batch"] is not None]
                self.assertEqual(len(batches), len(expected_batches))
                self.assertTrue(all(record["operation"] == "fit" for record in batches))
                self.assertEqual(
                    [record["step"] for record in batches],
                    [batch["step"] for batch in expected_batches],
                )

                for record, batch in zip(batches, expected_batches, strict=True):
                    self.assertEqual(record["samples"], batch["samples"])
                    self.assertEqual(record["values"]["loss"], batch["train_loss"])
                    self.assertEqual(record["values"]["learning_rate"], batch["learning_rate"])

                for split, key in (("train", "train_loss"), ("validation", "validation_loss")):
                    epochs = [
                        record
                        for record in metrics
                        if record["split"] == split and record["batch"] is None
                    ]
                    self.assertEqual(
                        [record["values"]["loss"] for record in epochs],
                        [epoch[key] for epoch in member.history_],
                    )
                    if split == "train":
                        self.assertNotIn(f"{path}/train/epoch/loss", tensorboard.Tags()["scalars"])
                        self.assertNotIn(f"{path} train epoch=", (root / "run.log").read_text())
                        continue
                    scalars = tensorboard.Scalars(f"{path}/{split}/epoch/loss")
                    self.assertEqual([event.step for event in scalars], [1, 2, 3, 4])
                    np.testing.assert_allclose(
                        [event.value for event in scalars],
                        [epoch[key] for epoch in member.history_],
                        rtol=1e-6,
                    )

                for metric, key in (("loss", "train_loss"), ("learning_rate", "learning_rate")):
                    scalars = tensorboard.Scalars(f"{path}/train/batch/{metric}")
                    self.assertEqual(
                        [event.step for event in scalars],
                        [batch["step"] for batch in expected_batches],
                    )
                    np.testing.assert_allclose(
                        [event.value for event in scalars],
                        [batch[key] for batch in expected_batches],
                        rtol=1e-6,
                    )

            ridge = metadata["pipeline/model/ridge"]
            self.assertEqual(
                ridge["train_samples"], metadata["pipeline/model"]["validation_samples"]
            )
            np.testing.assert_array_equal(ridge["coefficients"], run.model.meta_model_.coef_)
            self.assertEqual(ridge["intercept"], run.model.meta_model_.intercept_)
            self.assertNotIn("random_state", metadata["pipeline/model"])
            self.assertIn("results/validation/mae", tensorboard.Tags()["scalars"])
            self.assertFalse(any("/test/" in tag for tag in tensorboard.Tags()["scalars"]))
            self.assertFalse(run.metadata["test_evaluated"])

            loaded = load_run(root)
            for index, member in enumerate(loaded.model.estimators_):
                self.assertEqual(member.history_, histories[index]["history"])

    def test_interrupted_fit_keeps_member_telemetry(self) -> None:
        """Retain completed batch records and member identity if validation fails later.

        Returns:
            None.
        """
        X, y = sample_windows()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracker = RunTracker(root)
            model = small_ensemble()

            try:
                with (
                    patch.object(
                        TorchModelRunner,
                        "_validation_loss",
                        side_effect=[1, RuntimeError("interrupted")],
                    ),
                    self.assertRaisesRegex(RuntimeError, "interrupted"),
                ):
                    model.fit(
                        X[:5], y[:5], (X[5:], y[5:]), context=StepContext(tracker).child("model")
                    )
            finally:
                tracker.close()

            records = [
                json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()
            ]
            batches = [
                record
                for record in records
                if record["event"] == "metrics" and record["data"]["batch"] is not None
            ]

            self.assertEqual(len(batches), 2)
            self.assertTrue(all(record["path"] == "pipeline/model/gru_0" for record in batches))
            self.assertEqual(tracker.metadata["pipeline/model/gru_0"]["random_state"], 7)
            self.assertNotIn("pipeline/model/gru_1", tracker.metadata)
            self.assertFalse(hasattr(model, "meta_model_"))
