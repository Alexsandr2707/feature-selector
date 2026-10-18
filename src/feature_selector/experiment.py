"""Run orchestration independent of concrete model and transformation classes."""

import traceback
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any

from .artifacts import create_run_dir, input_metadata, save_run, write_json
from .config import ExperimentConfig, load_config
from .data import load_data, load_features
from .evaluation import prediction_report, report_metrics
from .factory import build_pipeline
from .pipeline.data import prepare_data
from .results import ExperimentRun
from .tracking import RunTracker, StepContext


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

    Returns:
        ExperimentRun: Fitted pipeline and saved reports; test requires explicit prediction.
    """
    if isinstance(config, (str, Path)):
        config = load_config(config)
    else:
        config = ExperimentConfig.from_dict(
            asdict(config) if isinstance(config, ExperimentConfig) else config
        )

    directory = create_run_dir(config.run["output_dir"])
    tracker = RunTracker(directory)
    context = StepContext(tracker)

    started = perf_counter()
    try:
        context.log_event("run_started", {"seed": config.run["seed"]})
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
