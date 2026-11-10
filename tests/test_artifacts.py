"""Run directory naming and collision handling."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from feature_selector.artifacts import create_run_dir


class RunDirectoryTests(unittest.TestCase):
    def test_local_time_and_existing_destinations(self) -> None:
        """Use local seconds and preserve existing files and directories.

        Returns:
            None: Checks sequential collisions with a fixed local clock.
        """
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("feature_selector.artifacts.datetime") as clock,
        ):
            clock.now.return_value = datetime(2026, 11, 3, 14, 5, 9, 123456)
            parent = Path(temporary) / "nested" / "runs"

            first = create_run_dir(str(parent))
            self.assertEqual(first, parent / "2026-11-03_14-05-09")

            marker = first / "run.log"
            marker.write_text("existing log", encoding="utf-8")
            blocker = parent / f"{first.name}_2"
            blocker.write_text("existing file", encoding="utf-8")

            third = create_run_dir(parent)
            self.assertEqual(third.name, f"{first.name}_3")
            self.assertTrue(third.is_dir())
            self.assertEqual(marker.read_text(encoding="utf-8"), "existing log")
            self.assertEqual(blocker.read_text(encoding="utf-8"), "existing file")

            self.assertEqual(create_run_dir(parent).name, f"{first.name}_4")
            clock.now.assert_called_with()

    def test_concurrent_launches(self) -> None:
        """Reserve distinct directories for concurrent launches in the same second.

        Returns:
            None: Checks creation with a shared missing parent and a fixed clock.
        """
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("feature_selector.artifacts.datetime") as clock,
        ):
            clock.now.return_value = datetime(2026, 11, 3, 14, 5, 9)
            parent = Path(temporary) / "runs"

            with ThreadPoolExecutor(max_workers=8) as executor:
                directories = list(executor.map(create_run_dir, [parent] * 16))

            stamp = "2026-11-03_14-05-09"
            expected = {stamp} | {f"{stamp}_{number}" for number in range(2, 17)}
            self.assertEqual({directory.name for directory in directories}, expected)
            self.assertTrue(all(directory.is_dir() for directory in directories))

    def test_creation_errors_propagate(self) -> None:
        """Propagate access errors instead of retrying them as collisions.

        Returns:
            None: Checks errors from both parent and run directory creation.
        """
        with patch.object(Path, "mkdir", side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                create_run_dir("runs")

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(Path, "mkdir", side_effect=[None, PermissionError("denied")]),
        ):
            with self.assertRaises(PermissionError):
                create_run_dir(temporary)
