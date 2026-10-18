"""Unique run directories, metadata and reloadable trained objects."""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from .pipeline.core import ConfiguredPipeline, TransformArtifact

if TYPE_CHECKING:
    from .config import ExperimentConfig
    from .results import ExperimentRun


def serializable(value: Any) -> Any:
    """Convert supported objects into values suitable for artifact JSON and YAML.

    Parameters:
        value (Any): Nested dataclass instances, mappings, sequences, NumPy arrays or
            scalars, paths, dates, timestamps and estimators. Unsupported objects pass through.

    Returns:
        Any: Converted value, with string mapping keys and nonfinite floats replaced by None.
    """
    if is_dataclass(value):
        if isinstance(value, type):
            raise TypeError("asdict() should be called on dataclass instances")
        return serializable(asdict(value))
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [serializable(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)

    get_params = getattr(value, "get_params", None)
    if callable(get_params):
        return {
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "params": serializable(get_params(deep=False)),
        }

    return value


def write_json(path: str | Path, value: Any) -> None:
    """Write artifact data as UTF-8 JSON, replacing an existing file.

    Parameters:
        path (str | Path): Destination file in an existing directory.
        value (Any): Data supported by serializable and the JSON encoder.

    Returns:
        None: Writes the JSON file.
    """
    content = json.dumps(serializable(value), indent=2, ensure_ascii=False, allow_nan=False)
    Path(path).write_text(content, encoding="utf-8")


def create_run_dir(output_dir: str | Path) -> Path:
    """Create a run directory named with a UTC timestamp and random suffix.

    Parameters:
        output_dir (str | Path): Parent directory, created if missing.

    Returns:
        Path: Newly created directory; an existing destination is never reused.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path(output_dir) / f"{stamp}-{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def input_metadata(config: ExperimentConfig) -> dict[str, Any]:
    """Collect input file hashes, seed and runtime versions for a run.

    Parameters:
        config (ExperimentConfig): Settings containing input paths and the run seed.

    Returns:
        dict[str, Any]: Input paths and SHA-256 hashes, seed, Python and package versions.
    """
    files: dict[str, dict[str, str]] = {}
    keys = [key for key in ("features_path", "target_path") if key in config.data]
    for key in keys:
        path = Path(config.data[key])
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        files[key] = {"path": str(path), "sha256": checksum}
    packages = (
        "feature-selector",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "PyYAML",
        "torch",
        "matplotlib",
        "tensorboard",
    )
    return {
        "inputs": files,
        "seed": config.run["seed"],
        "python": platform.python_version(),
        "versions": {name: version(name) for name in packages},
    }


def pipeline_histories(pipeline: ConfiguredPipeline, prefix: str = "") -> dict[str, Any]:
    """Collect histories by step path, including frozen provenance without replaying events.

    Parameters:
        pipeline (ConfiguredPipeline): Fitted pipeline or exported nested prefix.
        prefix (str): Parent namespace for nested history entries.

    Returns:
        dict[str, Any]: Epoch histories and selected epochs indexed by component paths.
    """
    histories: dict[str, Any] = {}
    for name, step in pipeline.steps:
        path = f"{prefix}/{name}" if prefix else name
        estimator = getattr(step, "estimator", step)
        histories[path] = {
            "history": getattr(estimator, "history_", []),
            "best_epoch": getattr(estimator, "best_epoch_", None),
        }
        artifact = getattr(step, "artifact_", None)
        if isinstance(artifact, TransformArtifact):
            histories[path]["provenance"] = artifact.provenance
            histories.update(pipeline_histories(artifact.pipeline, path))
    return histories


def save_run(run: ExperimentRun) -> None:
    """Save run reports, predictions and reloadable state in its artifact directory.

    Parameters:
        run (ExperimentRun): Completed run with an existing
            artifact directory and model history.

    Returns:
        None: Writes YAML, JSON, CSV and pickle artifacts, replacing matching files.
    """
    directory = run.artifact_dir
    settings = asdict(run.config)
    # YAML has no tuple/estimator objects; retain their importable constructor descriptions.
    settings["pipeline"] = [
        {"name": spec[0], **serializable(spec[1])} if isinstance(spec, tuple) else spec
        for spec in run.config.pipeline
    ]
    config_yaml = yaml.safe_dump(serializable(settings), sort_keys=False, allow_unicode=True)
    (directory / "config.yaml").write_text(config_yaml, encoding="utf-8")
    write_json(directory / "metadata.json", run.metadata)
    write_json(directory / "metrics.json", run.metrics)
    write_json(directory / "features.json", run.feature_names)
    write_json(directory / "history.json", pipeline_histories(run.pipeline))

    for split, frame in run.predictions.items():
        frame.to_csv(directory / f"predictions_{split}.csv", index=False)

    with (directory / "run.pkl").open("wb") as stream:
        pickle.dump(run, stream, protocol=pickle.HIGHEST_PROTOCOL)


def load_run(path: str | Path) -> ExperimentRun:
    """Load a saved run from a trusted pickle artifact.

    Parameters:
        path (str | Path): Run directory or pickle file. Load only trusted files:
            unpickling can execute code.

    Returns:
        ExperimentRun: Saved run with its trained components and reports.
    """
    source = Path(path)
    artifact = source / "run.pkl" if source.is_dir() else source
    with artifact.open("rb") as stream:
        return pickle.load(stream)
