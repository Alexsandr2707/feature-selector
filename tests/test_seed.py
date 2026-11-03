"""Configured run seeds initialize global generators before data loading."""

from pathlib import Path
import random
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

import numpy as np
import torch

from feature_selector import run_experiment
from feature_selector.data import TemporalData, load_data
from feature_selector.experiment import _set_global_seed
from support import write_config


class RunSeedTests(unittest.TestCase):
    def setUp(self) -> None:
        """Restore CPU generator states after each test.

        Returns:
            None.
        """
        self.addCleanup(random.setstate, random.getstate())
        self.addCleanup(np.random.set_state, np.random.get_state())
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())

    def test_global_random_sequences_repeat_for_the_same_seed(self) -> None:
        """Repeat Python, NumPy and CPU torch draws and change them with a different seed.

        Returns:
            None.
        """
        _set_global_seed(42)
        python_value = random.random()
        numpy_values = np.random.random(4)
        torch_values = torch.rand(4)
        _set_global_seed(42)
        self.assertEqual(random.random(), python_value)
        np.testing.assert_array_equal(np.random.random(4), numpy_values)
        torch.testing.assert_close(torch.rand(4), torch_values, rtol=0, atol=0)
        _set_global_seed(43)
        self.assertNotEqual(random.random(), python_value)
        self.assertFalse(np.array_equal(np.random.random(4), numpy_values))
        self.assertFalse(torch.equal(torch.rand(4), torch_values))

    def test_run_seed_is_applied_before_loading_and_recorded(self) -> None:
        """Seed global draws and request both CUDA seed operations from the configured run seed.

        Returns:
            None.
        """
        observed: list[tuple[float, float, float]] = []

        def observe_load(**params: Any) -> TemporalData:
            """Capture the first global draws before delegating data loading.

            Parameters:
                **params (Any): Arguments forwarded to the original loader.

            Returns:
                TemporalData: Loaded inputs for the experiment.
            """
            observed.append((random.random(), float(np.random.random()), torch.rand(1).item()))
            return load_data(**params)

        with tempfile.TemporaryDirectory() as directory:
            settings = write_config(Path(directory))
            for seed in (0, 17, 42):
                with self.subTest(seed=seed):
                    if seed:
                        settings["run"]["seed"] = seed
                    with (
                        patch("feature_selector.experiment.load_data", side_effect=observe_load),
                        patch("feature_selector.experiment.torch.cuda.manual_seed") as cuda_seed,
                        patch("feature_selector.experiment.torch.cuda.manual_seed_all") as cuda_all,
                    ):
                        run = run_experiment(settings)
                    expected = (
                        random.Random(seed).random(),
                        float(np.random.RandomState(seed).random()),
                        torch.rand(1, generator=torch.Generator().manual_seed(seed)).item(),
                    )
                    self.assertEqual(observed[-1], expected)
                    cuda_seed.assert_called_with(seed)
                    cuda_all.assert_any_call(seed)
                    self.assertEqual(run.metadata["seed"], seed)
                    self.assertEqual(run.pipeline.provenance_["seed"], seed)
