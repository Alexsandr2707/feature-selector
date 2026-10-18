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
            # Even factory construction does not prevalidate optimizer arguments.
            self.assertEqual(build_pipeline(config.pipeline)["model"].estimator.learning_rate, -1)
        self.assertEqual(settings, before)

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

    def test_repository_configs_construct(self) -> None:
        """Keep every shipped config in the new schema without opening model files.

        Returns:
            None.
        """
        for path in Path("configs").glob("*.yaml"):
            with self.subTest(path=path):
                config = load_config(path)
                build_pipeline(config.pipeline)
                self.assertNotIn("exclude_intervals", config.data)
                self.assertFalse(hasattr(config, "task"))
