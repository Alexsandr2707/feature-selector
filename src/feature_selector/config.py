"""Project settings and YAML-relative paths, without algorithm parameter prevalidation."""

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self, cast

import yaml


@dataclass
class ExperimentConfig:
    """Input sources, temporal split, ordered pipeline and run output settings."""

    data: dict[str, Any]
    pipeline: list[Any]
    split: dict[str, Any] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any], base_dir: str | Path | None = None) -> Self:
        """Copy settings and resolve file paths without constructing model components.

        Parameters:
            value (dict[str, Any]): Data sources and a nonempty ordered pipeline list.
            base_dir (str | Path | None): Relative-path base, or the working directory.

        Returns:
            Self: Settings with resolved paths and project defaults.
        """
        config = cls(**deepcopy(value))

        if not isinstance(config.data, dict) or "features_path" not in config.data:
            raise ValueError("data requires features_path")
        if not isinstance(config.pipeline, list) or not config.pipeline:
            raise ValueError("pipeline must be a nonempty list")

        base = Path(base_dir or Path.cwd())

        for key in ("features_path", "target_path"):
            if key in config.data:
                config.data[key] = str((base / config.data[key]).resolve())

        if ("target_path" in config.data) != ("target_column" in config.data):
            raise ValueError("Provide target_path and target_column together")

        config.data.setdefault("freq", "1h")
        names: set[str] = set()

        for spec in config.pipeline:
            if isinstance(spec, tuple):
                name = spec[0]
                get_params = getattr(spec[1], "get_params", None)
                params = get_params(deep=False) if callable(get_params) else {}
            else:
                if set(spec) - {"name", "type", "class", "params"} or (
                    ("type" in spec) == ("class" in spec)
                ):
                    raise ValueError(
                        "Pipeline steps require name, type or class, and optional params"
                    )

                name = spec["name"]
                if not isinstance(spec.get("params", {}), dict):
                    raise TypeError("Step params must be a mapping")
                params = spec.get("params", {})

            params = cast(dict, params)

            # Resolve Python and YAML component paths before saving a rerunnable config.
            paths = {
                key: str((base / item).resolve())
                for key, item in params.items()
                if key == "path" or key.endswith("_path")
            }

            if isinstance(spec, tuple):
                set_params = getattr(spec[1], "set_params", None)
                if paths and callable(set_params):
                    set_params(**paths)
            else:
                params.update(paths)

            if not isinstance(name, str) or not name or "/" in name or name in names:
                raise ValueError(f"Invalid or duplicate step name: {name!r}")

            names.add(name)

        config.run.setdefault("seed", 0)
        config.run["output_dir"] = str((base / config.run.get("output_dir", "runs")).resolve())

        return config


def load_config(path: str | Path) -> ExperimentConfig:
    """Load YAML using its directory as the path base.

    Parameters:
        path (str | Path): YAML configuration file in the current schema.

    Returns:
        ExperimentConfig: Copied settings with absolute file paths.
    """
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        return ExperimentConfig.from_dict(yaml.safe_load(stream), path.parent)
