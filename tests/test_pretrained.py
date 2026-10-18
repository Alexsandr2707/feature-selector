"""Frozen nested pipelines preserve fitted preprocessing, weights and provenance."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from feature_selector import run_experiment
from feature_selector.artifacts import load_run
from feature_selector.data import load_data
from feature_selector.factory import build_pipeline
from feature_selector.models.autoencoder import ConvAutoencoder
from support import sample_data, write_config


class FrozenPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        """Use one CPU thread for tiny networks.

        Returns:
            None.
        """
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def test_dense_pretraining_frozen_gru_and_self_contained_reload(self) -> None:
        """Reuse an exported prefix without fitting any nested model or scaler.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forecast = write_config(root, sparse=True)
            ae = deepcopy(forecast)
            ae["data"] = {"features_path": forecast["data"]["features_path"]}
            ae["pipeline"] = [ae["pipeline"][0], ae["pipeline"][2],
                              {"name": "ae", "type": "conv_autoencoder", "params": {
                                  "epochs": 1, "batch_size": 8, "hidden_channels": 2,
                                  "latent_channels": 1}}]
            pretrained = run_experiment(ae)
            destination = pretrained.pipeline.export_transform(root / "encoder.pkl",
                                                                through="ae", mode="encode")
            self.assertEqual(pretrained.model.mode, "reconstruct")
            weights = deepcopy(pretrained.model.network_.state_dict())
            forecast["pipeline"] = [forecast["pipeline"][1],
                                    {"name": "encoder", "type": "frozen_pipeline",
                                     "params": {"path": str(destination)}},
                                    forecast["pipeline"][3],
                                    {"name": "gru", "type": "gru", "params": {
                                        "epochs": 1, "hidden_size": 2, "batch_size": 8}}]
            with patch.object(ConvAutoencoder, "fit", side_effect=AssertionError("refit")):
                run = run_experiment(forecast)
            nested = run.pipeline["encoder"].artifact_.pipeline
            for name, value in nested["ae"].estimator.network_.state_dict().items():
                torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
            pre_count = sum(b["samples"] for b in pretrained.model.history_[0]["batches"])
            gru_count = sum(b["samples"] for b in run.model.history_[0]["batches"])
            self.assertEqual(pre_count, 32)
            self.assertEqual(gru_count, 7)
            events = [json.loads(line) for line in (run.artifact_dir / "events.jsonl").read_text(
            ).splitlines()]
            self.assertFalse(any(
                e["event"] == "metrics" and "encoder" in e["path"] for e in events
            ))
            destination.unlink()
            source = load_data(**forecast["data"])
            np.testing.assert_array_equal(run.predict(source, "test").y_pred,
                                          load_run(run.artifact_dir).predict(source, "test").y_pred)

    def test_inline_encoder_uses_unobserved_windows(self) -> None:
        """Train intermediate encoders before the supervised model filters missing y.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            spec = write_config(Path(directory), sparse=True)
            spec["pipeline"].insert(3, {"name": "encoder", "type": "conv_autoencoder", "params": {
                "mode": "encode", "epochs": 1, "hidden_channels": 2, "latent_channels": 1}})
            run = run_experiment(spec)
            encoder = run.pipeline["encoder"].estimator
            self.assertEqual(sum(b["samples"] for b in encoder.history_[0]["batches"]), 31)
            self.assertEqual(run.metrics["train"]["n"], 7)

    def test_reconstruction_original_units_and_mode(self) -> None:
        """Restore window endpoints and preserve the reference through feature scaling.

        Returns:
            None.
        """
        with tempfile.TemporaryDirectory() as directory:
            spec = write_config(Path(directory))
            spec["data"].pop("target_path")
            spec["data"].pop("target_column")
            spec["pipeline"] = [spec["pipeline"][0], spec["pipeline"][2],
                                {"name": "ae", "type": "conv_autoencoder", "params": {
                                    "epochs": 1, "hidden_channels": 2, "latent_channels": 1}}]
            run = run_experiment(spec)
            a = run.predictions["validation"].query("feature == 'a'")
            np.testing.assert_allclose(a.y_true, np.arange(36, 48), atol=1e-6)
            self.assertEqual(run.metrics["validation"]["overall"]["n"], 24)
            self.assertEqual(set(run.metrics["validation"]["per_feature"]), {"a", "b"})

    def test_stacked_autoencoders_keep_original_reference(self) -> None:
        """Score the complete lossy chain against actual inputs, including frozen prefixes.

        Returns:
            None.
        """
        data = sample_data(30)
        encoder_spec = {"name": "encoder", "type": "conv_autoencoder", "params": {
            "epochs": 1, "hidden_channels": 2, "latent_channels": 1, "mode": "encode"}}
        decoder_spec = {"name": "decoder", "type": "conv_autoencoder", "params": {
            "epochs": 1, "hidden_channels": 2, "latent_channels": 1}}
        prefix = [{"name": "scale", "type": "standard"},
                  {"name": "windows", "type": "windows", "params": {"lookback": 4}},
                  encoder_spec]
        direct = build_pipeline([*prefix, decoder_spec]).fit(data)
        with tempfile.TemporaryDirectory() as directory:
            path = direct.export_transform(Path(directory) / "encoder.pkl", through="encoder")
            frozen = build_pipeline([
                {"name": "prefix", "type": "frozen_pipeline", "params": {"path": str(path)}},
                decoder_spec,
            ]).fit(data)
            for pipeline in (direct, frozen):
                with self.subTest(frozen=pipeline is frozen):
                    prediction = pipeline.predict(data, splits=("validation",))
                    times = pd.DatetimeIndex(prediction.samples["origin_time"])
                    expected = data.X.loc[times].to_numpy()
                    np.testing.assert_array_equal(prediction.truth, expected)
                    self.assertEqual(prediction.feature_names, data.feature_names)
                    self.assertGreater(np.max(np.abs(prediction.values - expected)), 1)

    def test_encoder_transform_rejects_permuted_schema(self) -> None:
        """Apply the same fitted feature-order check to encoding and prediction.

        Returns:
            None.
        """
        from dataclasses import replace

        from feature_selector.pipeline.temporal import Windows

        data = sample_data(30)
        pipeline = build_pipeline([
            {"name": "windows", "type": "windows", "params": {"lookback": 4}},
            {"name": "encoder", "type": "conv_autoencoder", "params": {
                "epochs": 1, "hidden_channels": 2, "latent_channels": 1, "mode": "encode"}},
        ]).fit(data)
        windows = Windows(4).transform(data)
        changed = replace(windows, X=windows.X[:, :, ::-1],
                          feature_names=tuple(reversed(windows.feature_names)))
        with self.assertRaisesRegex(ValueError, "schema"):
            pipeline["encoder"].transform(changed)
        with self.assertRaisesRegex(ValueError, "schema"):
            pipeline["encoder"].predict(changed)
