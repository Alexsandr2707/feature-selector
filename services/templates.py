"""Restricted presentation parameters over unchanged experiment configurations."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PARAMETERS = {
    "run_seed": {"type": "int", "min": 0, "max": 2**32 - 1},
    "horizon": {"type": "int", "min": 1, "max": 168},
    "lookback": {"type": "int", "min": 1, "max": 336},
    "alpha": {"type": "float", "min": 0.000001, "max": 1000000},
    "model_seed": {"type": "int", "min": 0, "max": 2**32 - 1},
    "epochs": {"type": "int", "min": 1, "max": 1000},
    "batch_size": {"type": "int", "min": 1, "max": 1024},
    "hidden_size": {"type": "int", "min": 1, "max": 256},
    "learning_rate": {"type": "float", "min": 0.000001, "max": 1.0},
}


def configured_templates(settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Describe available templates and editable fields.

    Parameters:
        settings (dict[str, Any]): Service configuration containing templates.

    Returns:
        list[dict[str, Any]]: GUI descriptors without server filesystem paths.
    """
    return [
        {
            "id": key,
            "name": spec["name"],
            "source_id": spec["source_id"],
            "parameters": {
                name: {**PARAMETERS[name], "default": value}
                for name, value in spec["parameters"].items()
            },
        }
        for key, spec in settings["templates"].items()
    ]


def resolve_template(
    settings: dict[str, Any],
    template_id: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Resolve allowlisted parameters into a copied experiment YAML mapping.

    Parameters:
        settings (dict[str, Any]): Template configuration.
        template_id (str): Configured template ID.
        parameters (dict[str, Any]): Restricted GUI overrides.

    Returns:
        dict[str, Any]: Experiment configuration; data paths are replaced by the worker.
    """
    import math

    if template_id not in settings["templates"]:
        raise ValueError("Unknown template")
    spec = settings["templates"][template_id]
    if parameters.keys() - spec["parameters"].keys():
        raise ValueError("Unsupported template parameter")

    values = {**spec["parameters"], **parameters}
    for name, value in values.items():
        field = PARAMETERS[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if field["type"] == "int" and not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if not math.isfinite(value) or not field["min"] <= value <= field["max"]:
            raise ValueError(f"{name} is outside supported bounds")

    config = deepcopy(yaml.safe_load(Path(spec["config_path"]).read_text("utf-8")))
    config.setdefault("run", {})["seed"] = values.pop("run_seed")
    for step in config["pipeline"]:
        params = step.setdefault("params", {})
        if step["type"] == "windows":
            params["lookback"] = values["lookback"]
        elif step["type"] == "align_target":
            params["horizon"] = values["horizon"]
        elif step["name"] == "model":
            for name, value in values.items():
                if name not in {"lookback", "horizon"}:
                    params["random_state" if name == "model_seed" else name] = value
            if step["type"] == "gru":
                params.update(device="cpu", checkpoint_start_epoch=1, patience=values["epochs"])
    return config
