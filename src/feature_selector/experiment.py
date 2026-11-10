"""Run orchestration independent of concrete model and transformation classes."""

import json
import random
import traceback
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone

from .artifacts import create_run_dir, input_metadata, save_run, write_json
from .config import ExperimentConfig, load_config
from .data import load_data, load_features
from .evaluation import (
    METRIC_NAMES,
    RegressionMetrics,
    prediction_report,
    report_metrics,
    summarize_regression_metrics,
)
from .factory import build_pipeline
from .pipeline import Step
from .pipeline.data import prepare_data
from .results import ExperimentRun
from .tracking import RunTracker, StepContext


@dataclass
class SeedExperimentResult:
    """Saved series directory and test scores per seed and across completed runs."""

    artifact_dir: Path
    metrics_per_seed: pd.DataFrame
    metrics_summary: pd.DataFrame


def _set_global_seed(seed: int) -> None:
    """Initialize the process-wide Python, NumPy and PyTorch random generators.

    Parameters:
        seed (int): Seed in [0, 2**32 - 1], shared by Python, NumPy, CPU and CUDA generators.

    Returns:
        None: Resets global RNG states for subsequent random operations.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _log_final_metrics(tracker: RunTracker, metrics: dict[str, Any]) -> None:
    """Publish final scores in original units as scalars and a compact results table.

    Parameters:
        tracker (RunTracker): Open tracker receiving TensorBoard summaries.
        metrics (dict[str, Any]): Split reports containing regression scores or
            reconstruction overall/per_feature scores. Undefined scores are shown
            as unavailable in the table and omitted from scalar summaries.

    Returns:
        None: Writes aggregate results without additional runs or event directories.
    """
    names = ("mae", "rmse", "r2", "pearson", "n")
    rows = ["| Split | MAE | RMSE | R² | Pearson | n |", "| --- | --- | --- | --- | --- | --- |"]
    for split, report in metrics.items():
        scores = report.get("overall", report)
        if not scores:
            continue
        cells = []
        for name in names:
            value = scores.get(name)
            if value is None or not isfinite(value):
                cells.append("—")
                continue
            tracker.writer.add_scalar(f"results/{split}/{name}", value, global_step=0)
            cells.append(str(value) if name == "n" else f"{value:.6g}")
        rows.append(f"| {split} | {' | '.join(cells)} |")

    if len(rows) > 2:
        tracker.writer.add_text("results/summary", "\n".join(rows), global_step=0)


def run_experiment(
    config: ExperimentConfig | dict[str, Any] | str | Path,
) -> ExperimentRun:
    """Fit an ordered pipeline and evaluate train and enabled validation only.

    Parameters:
        config (ExperimentConfig | dict[str, Any] | str | Path): Settings or YAML path.
            run.seed initializes global Python, NumPy and PyTorch RNG states before loading
            data and constructing the pipeline; estimator random_state parameters remain local.
            run.experiment_name groups artifacts below run.output_dir.

    Returns:
        ExperimentRun: Fitted pipeline and saved reports; test requires explicit prediction.
    """
    if isinstance(config, (str, Path)):
        config = load_config(config)
    else:
        config = ExperimentConfig.from_dict(
            asdict(config) if isinstance(config, ExperimentConfig) else config
        )

    directory = create_run_dir(Path(config.run["output_dir"]) / config.run["experiment_name"])
    tracker = RunTracker(directory)
    context = StepContext(tracker)

    started = perf_counter()
    try:
        context.log_event("run_started", {"seed": config.run["seed"]})
        _set_global_seed(config.run["seed"])

        source = (
            load_data(**config.data)
            if "target_path" in config.data
            else load_features(**config.data)
        )
        data = prepare_data(source, config.split, config.data["freq"])
        metadata = input_metadata(config)

        pipeline = build_pipeline(config.pipeline)
        pipeline.fit(data, context=context)
        pipeline.provenance_.update(
            source_run=str(directory),
            inputs=metadata["inputs"],
            seed=config.run["seed"],
            versions=metadata["versions"],
        )

        prediction = pipeline.predict(
            data, context=StepContext(tracker, operation="predict"), splits=("train", "validation")
        )

        reports, metrics = {}, {}
        for split in ("train", "validation"):
            mask = prediction.samples["split"].to_numpy() == split
            if mask.any():
                report = prediction_report(prediction, mask)
                reports[split], metrics[split] = report, report_metrics(report)

        metadata.update(
            status="completed",
            duration_seconds=perf_counter() - started,
            boundaries=asdict(data.temporal.boundaries),
            steps=tracker.metadata,
            test_evaluated=False,
        )

        run = ExperimentRun(
            config, pipeline, data.temporal.boundaries, reports, metrics, metadata, directory
        )
        save_run(run)
        _log_final_metrics(tracker, metrics)
        context.log_event("run_completed", {"seconds": perf_counter() - started})

        return run

    except Exception as exc:
        tracker.logger.error("Experiment failed\n%s", traceback.format_exc())
        context.log_event("run_failed", {"error": str(exc)})
        try:
            write_json(
                directory / "failure.json",
                {"error": str(exc), "type": type(exc).__name__, "config": config},
            )
        except Exception:
            tracker.logger.exception("Could not save failure report")
        raise
    finally:
        tracker.close()


def _seeded_step(template: Step, seed: int) -> Step:
    """Clone a final step and set its numerical estimator's initialization seed.

    Parameters:
        template (Step): Final step or estimator adapter exposing a random_state parameter.
        seed (int): Initialization and, when supported, batch shuffling seed.

    Returns:
        Step: Unfitted step configured for this seed.
    """
    step = clone(template)
    estimator = getattr(step, "estimator", step)
    get_params = getattr(estimator, "get_params", None)
    set_params = getattr(estimator, "set_params", None)

    if not callable(get_params) or not callable(set_params):
        raise TypeError("The final estimator must provide get_params and set_params")
    if "random_state" not in get_params(deep=False):  # type: ignore
        raise ValueError("The final estimator must expose a random_state parameter")
    if getattr(step, "reconstructs", False):
        raise ValueError("Seed experiments require a scalar target regression model")

    set_params(random_state=seed)
    return step


def _save_seed_tables(
    result: SeedExperimentResult,
    rows: list[dict[str, Any]],
    scores: Sequence[RegressionMetrics],
) -> None:
    """Update in-memory score tables and replace their CSV artifacts.

    Parameters:
        result (SeedExperimentResult): Series result whose tables are updated in place.
        rows (list[dict[str, Any]]): Completed seeds, scores, JSON notes and artifact paths.
        scores (Sequence[RegressionMetrics]): Matching scalar test scores for aggregation.

    Returns:
        None: Writes per-seed and summary CSV files, including headers for an empty series.
    """
    result.metrics_per_seed = pd.DataFrame(
        rows, columns=["seed", *METRIC_NAMES, "n", "notes", "artifact_dir"]
    )
    summaries = summarize_regression_metrics(scores) if scores else {}
    result.metrics_summary = pd.DataFrame(
        [{"metric": name, **summary} for name, summary in summaries.items()],
        columns=["metric", "mean", "std", "n_valid", "n_total"],
    )

    result.metrics_per_seed.to_csv(result.artifact_dir / "metrics_per_seed.csv", index=False)
    result.metrics_summary.to_csv(result.artifact_dir / "metrics_summary.csv", index=False)


def run_seed_experiments(
    config: ExperimentConfig | dict[str, Any] | str | Path,
    seeds: Sequence[int] = tuple(range(10)),
) -> SeedExperimentResult:
    """Train scalar regressors with fixed preprocessing and explicitly evaluate test.

    Parameters:
        config (ExperimentConfig | dict[str, Any] | str | Path): Settings or YAML path with
            observed test targets and a final estimator exposing random_state. The first
            run fits preprocessing; later runs reuse its exported prefix without fitting.
            run.seed stays fixed. Each run saves its model, reports and test predictions.
            run.experiment_name groups the series below run.output_dir; individual
            runs are nested under runs/seed_<seed> within the series directory.
        seeds (Sequence[int]): Nonempty sequence of distinct initialization seeds in
            [0, 2**32 - 1], applied to the final estimator in the supplied order.

    Returns:
        SeedExperimentResult: Series directory and per-seed/summary test score tables.
            CSV tables and series.json are updated after each completed run. Failures
            stop the series and preserve completed artifacts plus failure.json. Only one
            trained run is retained in memory; load saved runs to inspect their models.
    """
    seed_values = tuple(seeds)
    if not seed_values or any(
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or not 0 <= seed < 2**32
        for seed in seed_values
    ):
        raise ValueError("seeds must be a nonempty sequence of integers in [0, 2**32 - 1]")
    seed_values = tuple(int(seed) for seed in seed_values)
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be distinct")

    if isinstance(config, (str, Path)):
        config = load_config(config)
    else:
        config = ExperimentConfig.from_dict(
            asdict(config) if isinstance(config, ExperimentConfig) else config
        )
    if "target_path" not in config.data:
        raise ValueError("Seed experiments require target data")
    name, template = build_pipeline([config.pipeline[-1]]).steps[-1]
    template = _seeded_step(template, seed_values[0])

    _set_global_seed(config.run["seed"])
    source = load_data(**config.data)
    data = prepare_data(source, config.split, config.data["freq"])
    if not (data.samples["split"] == "test").any():
        raise ValueError("Seed experiments require a nonempty test split")

    directory = create_run_dir(Path(config.run["output_dir"]) / config.run["experiment_name"])
    prefix_path = directory / "preprocessing.pkl" if len(config.pipeline) > 1 else None
    result = SeedExperimentResult(directory, pd.DataFrame(), pd.DataFrame())
    rows: list[dict[str, Any]] = []
    scores: list[RegressionMetrics] = []
    reference: pd.DataFrame | None = None
    inputs: dict[str, Any] | None = None
    manifest: dict[str, Any] = {
        "status": "running",
        "seeds": seed_values,
        "completed_seeds": [],
        "run_seed": config.run["seed"],
        "std_ddof": 1,
        "preprocessing_path": prefix_path,
    }
    seed = seed_values[0]

    try:
        _save_seed_tables(result, rows, scores)
        write_json(directory / "series.json", manifest)

        for index, seed in enumerate(seed_values):
            settings = deepcopy(config)
            model = (name, _seeded_step(template, seed))
            if index and prefix_path is not None:
                settings.pipeline = [
                    {
                        "name": f"{name}_preprocessing",
                        "type": "frozen_pipeline",
                        "params": {"path": str(prefix_path)},
                    },
                    model,
                ]
            else:
                settings.pipeline[-1] = model
            settings.run["output_dir"] = str(directory / "runs")
            settings.run["experiment_name"] = f"seed_{seed}"

            run = run_experiment(settings)
            if inputs is None:
                inputs = run.metadata["inputs"]
            elif run.metadata["inputs"] != inputs:
                raise ValueError("Input files changed between seed runs")

            report = run.predict(source, split="test")
            if "feature" in report:
                raise ValueError("Seed experiments require scalar target predictions")

            computed = report_metrics(report)
            if not computed:
                raise ValueError("No observed usable test targets for seed experiments")
            # A nonempty scalar report is evaluated as RegressionMetrics.
            test_scores = cast(RegressionMetrics, computed)

            observed = report[["origin_time", "target_time", "y_true"]]
            if reference is None:
                reference = observed
            elif not reference.equals(observed):
                raise ValueError("Test examples and targets must match across seed runs")

            run.predictions["test"] = report
            run.metrics["test"] = test_scores
            run.metadata.update(test_evaluated=True, model_seed=seed, seed_series=str(directory))
            run.pipeline.provenance_.update(model_seed=seed, seed_series=str(directory))
            save_run(run)

            rows.append(
                {
                    "seed": seed,
                    **{metric: test_scores[metric] for metric in METRIC_NAMES},
                    "n": test_scores["n"],
                    "notes": json.dumps(test_scores["notes"], ensure_ascii=False),
                    "artifact_dir": str(run.artifact_dir),
                }
            )
            scores.append(test_scores)
            _save_seed_tables(result, rows, scores)
            manifest["completed_seeds"].append(seed)
            write_json(directory / "series.json", manifest)

            if index == 0 and prefix_path is not None:
                # Freeze every fitted step before the final model, including target scaling.
                run.pipeline.export_transform(prefix_path, through=run.pipeline.steps[-2][0])
            del run

        manifest["status"] = "completed"
        write_json(directory / "series.json", manifest)
        return result
    except Exception as exc:
        manifest.update(status="failed", failed_seed=seed, error=str(exc))
        try:
            write_json(directory / "series.json", manifest)
            write_json(
                directory / "failure.json",
                {"seed": seed, "error": str(exc), "type": type(exc).__name__},
            )
        except Exception as artifact_error:  # noqa: BLE001
            exc.add_note(f"Could not save series failure report: {artifact_error}")
        exc.add_note(f"Seed experiment series: {directory}, seed: {seed}")
        raise
