"""Configuration structure and paths, independent of model parameter validity."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import yaml

from feature_selector import ExperimentConfig, load_config
from feature_selector.factory import build_pipeline
from feature_selector.pipeline.frozen import FrozenPipeline


class ConfigTests(unittest.TestCase):
    def test_python_step_paths_use_the_configuration_base(self) -> None:
        """Resolve copied Python components before their saved config changes directories.

        Returns:
            None.
        """
        step = FrozenPipeline("encoder.pkl")
        settings = {"data": {"features_path": "x.csv"}, "pipeline": [
            ("encoder", step), {"name": "model", "type": "ridge"},
        ]}
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig.from_dict(settings, base_dir=directory)
            self.assertEqual(config.pipeline[0][1].path, str(Path(directory) / "encoder.pkl"))
            self.assertEqual(step.path, "encoder.pkl")

    def test_paths_defaults_and_input_ownership(self) -> None:
        """Resolve copied settings without mutating caller-owned data.

        Returns:
            None.
        """
        settings = {"data": {"features_path": "x.csv"}, "pipeline": [
            {"name": "encoder", "type": "frozen_pipeline", "params": {"path": "ae.pkl"}},
            {"name": "model", "type": "gru", "params": {"learning_rate": -1}},
        ]}
        before = deepcopy(settings)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(settings))
            config = load_config(path)
            self.assertEqual(config.data["features_path"], str(Path(directory) / "x.csv"))
            self.assertEqual(config.pipeline[0]["params"]["path"], str(Path(directory) / "ae.pkl"))
            self.assertEqual(config.run["seed"], 0)
            self.assertEqual(config.run["experiment_name"], "default")
            # Even factory construction does not prevalidate optimizer arguments.
            self.assertEqual(build_pipeline(config.pipeline)["model"].estimator.learning_rate, -1)
        self.assertEqual(settings, before)

    def test_experiment_name(self) -> None:
        """Keep explicit names and reject invalid directory components.

        Returns:
            None: Checks name validation without creating output directories.
        """
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "runs"
            settings = {
                "data": {"features_path": "x.csv"},
                "pipeline": [{"name": "model", "type": "ridge"}],
                "run": {"output_dir": str(output), "experiment_name": "uzk_gru"},
            }
            before = deepcopy(settings)
            config = ExperimentConfig.from_dict(settings)
            self.assertEqual(config.run["experiment_name"], "uzk_gru")
            self.assertEqual(settings, before)
            for name in (None, 7, True, [], "", "  ", ".", "..", "a/b", "a\\b", "a\0b"):
                with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, "run.experiment_name"
                ):
                    ExperimentConfig.from_dict({**settings, "run": {
                        **settings["run"], "experiment_name": name,
                    }})
            self.assertFalse(output.exists())

    def test_structure_and_removed_schema(self) -> None:
        """Reject ambiguous structure while leaving algorithm checks to consumers.

        Returns:
            None.
        """
        for settings in (
            {"data": {}, "pipeline": []},
            {"data": {"features_path": "x"}, "pipeline": []},
            {"data": {"features_path": "x", "target_path": "y"},
             "pipeline": [{"name": "m", "type": "ridge"}]},
            {"data": {"features_path": "x"}, "pipeline": [
                {"name": "x", "type": "standard"}, {"name": "x", "type": "ridge"}]},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                ExperimentConfig.from_dict(settings)
        with self.assertRaises(TypeError):
            ExperimentConfig.from_dict({"data": {"features_path": "x"},
                                        "pipeline": [], "task": "forecast"})
