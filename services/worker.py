"""Isolated collection, experiment and prediction tasks, without import-time ML work."""

import csv
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any

from services.catalog import Catalog
from services.common import Remote, atomic_json, read_json

logger = logging.getLogger(__name__)


def source_fingerprint(root: Path) -> str:
    """Identify Python sources independently of package version labels.

    Parameters:
        root (Path): Package directory.

    Returns:
        str: SHA-256 of sorted relative names and source bytes.
    """
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def progress(directory: Path, stage: str, **values: Any) -> None:
    """Publish the latest stage without claiming an overall completion percentage.

    Parameters:
        directory (Path): Job directory.
        stage (str): Current execution stage.
        **values (Any): Optional row counts and diagnostics.

    Returns:
        None: Replaces progress.json.
    """
    atomic_json(directory / "progress.json", {"stage": stage, **values})


def collect(request: dict[str, Any], catalog: Catalog, directory: Path) -> None:
    """Collect a whole source and publish compatible feature and target CSVs.

    Parameters:
        request (dict[str, Any]): Job settings and selected source.
        catalog (Catalog): Storage client.
        directory (Path): Job working directory.

    Returns:
        None: Publishes the dataset under the job ID.
    """
    settings, payload = request["settings"], request["payload"]
    collector = Remote(settings["collector_url"], settings["timeout"])
    source_id = payload["source_id"]
    source = collector.json("GET", f"/sources/{source_id}")

    paths = {"features.csv": directory / "features.csv", "target.csv": directory / "target.csv"}
    counts = {"features": 0, "targets": 0}
    offset = 0

    with (
        paths["features.csv"].open("w", newline="") as features,
        paths["target.csv"].open("w", newline="") as targets,
    ):
        writers = {"features": csv.writer(features), "targets": csv.writer(targets)}
        while True:
            progress(directory, "collecting", rows=offset, total_rows=source["rows"])
            batch = collector.json(
                "GET",
                f"/sources/{source_id}/batches",
                params={
                    "offset": offset,
                    "limit": settings["batch_size"],
                },
            )

            for key, writer in writers.items():
                if offset == 0:
                    writer.writerow(batch[key]["columns"])
                writer.writerows(batch[key]["rows"])
                counts[key] += len(batch[key]["rows"])

            following = batch["next_offset"]
            if batch["done"]:
                break
            if following <= offset:
                raise ValueError("Collector did not advance its cursor")
            offset = following

    if counts["features"] != source["rows"]:
        raise ValueError("Collected feature count differs from source")

    progress(directory, "publishing", rows=counts["features"])
    catalog.publish(
        "datasets",
        request["job_id"],
        payload["name"],
        {
            **source,
            "source_id": source_id,
            "collected_rows": counts,
        },
        {},
        paths,
        reservation_token=request.get("reservation_token"),
    )


def download_dataset(catalog: Catalog, key: str, directory: Path) -> dict[str, Any]:
    """Materialize raw CSV inputs locally for the existing file-based loader.

    Parameters:
        catalog (Catalog): Storage client.
        key (str): Dataset ID.
        directory (Path): Local working directory.

    Returns:
        dict[str, Any]: Dataset manifest.
    """
    manifest = catalog.get("datasets", key)
    for filename in ("features.csv", "target.csv"):
        catalog.download("datasets", key, filename, directory / filename)
    return manifest


def train(request: dict[str, Any], catalog: Catalog, directory: Path) -> None:
    """Run an existing experiment and publish its complete reloadable state.

    Parameters:
        request (dict[str, Any]): Selected model specification and configuration.
        catalog (Catalog): Storage client.
        directory (Path): Local working directory.

    Returns:
        None: Publishes a Run without evaluating test.
    """
    import feature_selector
    from feature_selector.artifacts import serializable
    from feature_selector.experiment import run_experiment

    model = request["payload"]["model"]
    dataset_id = model["references"]["dataset_id"]
    progress(directory, "downloading")
    dataset = download_dataset(catalog, dataset_id, directory)

    config = model["metadata"]["config"]
    config["data"].update(
        features_path=str(directory / "features.csv"),
        target_path=str(directory / "target.csv"),
    )
    config["run"].update(output_dir=str(directory / "artifacts"), experiment_name="service")

    progress(directory, "training")
    run = run_experiment(config)

    files = {path.name: path for path in run.artifact_dir.iterdir() if path.is_file()}
    # Preserve TensorBoard files as flat downloadable artifacts without changing core output.
    for index, path in enumerate(sorted((run.artifact_dir / "tensorboard").glob("*"))):
        if path.is_file():
            files[f"tensorboard_{index}_{path.name}"] = path

    alignment = next(s for s in config["pipeline"] if s["type"] == "align_target")
    validation = run.predictions.get("validation")
    stamps = (
        []
        if validation is None
        else validation.loc[validation["y_true"].notna(), "target_time"].astype(str).tolist()
    )

    progress(directory, "publishing")
    catalog.publish(
        "runs",
        request["job_id"],
        model["name"],
        serializable(
            {
                "metrics": run.metrics,
                "template_id": model["metadata"]["template_id"],
                "parameters": model["metadata"]["parameters"],
                "comparison": {
                    "dataset_id": dataset_id,
                    "target": config["data"]["target_column"],
                    "freq": config["data"]["freq"],
                    "split": config["split"],
                    "horizon": alignment["params"]["horizon"],
                },
                "input_checksums": dataset["files"],
                "runtime": run.metadata,
                "validation_times_sha256": hashlib.sha256("\n".join(stamps).encode()).hexdigest(),
                "source_sha256": {
                    "core": source_fingerprint(Path(feature_selector.__file__).parent),
                    "services": source_fingerprint(Path(__file__).parent),
                },
            }
        ),
        {"dataset_id": dataset_id, "model_id": model["id"]},
        files,
        reservation_token=request.get("reservation_token"),
    )


def predict(request: dict[str, Any], catalog: Catalog, directory: Path) -> None:
    """Predict one saved split and publish a separate report in original units.

    Parameters:
        request (dict[str, Any]): Run ID and requested split.
        catalog (Catalog): Storage client.
        directory (Path): Local working directory.

    Returns:
        None: Publishes Prediction without modifying Run.
    """
    from feature_selector.artifacts import load_run, write_json
    from feature_selector.data import load_data
    from feature_selector.evaluation import report_metrics

    payload = request["payload"]
    manifest = catalog.get("runs", payload["run_id"])
    dataset_id = manifest["references"]["dataset_id"]

    progress(directory, "downloading")
    download_dataset(catalog, dataset_id, directory)
    catalog.download("runs", manifest["id"], "run.pkl", directory / "run.pkl")

    run = load_run(directory / "run.pkl")
    source = load_data(
        directory / "features.csv",
        directory / "target.csv",
        run.config.data["target_column"],
        run.config.data["freq"],
    )

    progress(directory, "predicting")
    report = run.predict(source, split=payload["split"])
    if report.empty:
        raise ValueError("No usable examples in selected split")

    metrics = report_metrics(report)
    report.to_csv(directory / "predictions.csv", index=False)
    write_json(directory / "metrics.json", metrics)

    progress(directory, "publishing")
    suffix = f" / {payload['split']}"
    catalog.publish(
        "predictions",
        request["job_id"],
        f"{manifest['name'][: 200 - len(suffix)]}{suffix}",
        {
            "split": payload["split"],
            "run_name": manifest["name"],
            "metrics": read_json(directory / "metrics.json"),
            "rows": len(report),
        },
        {"dataset_id": dataset_id, "run_id": manifest["id"]},
        {
            "predictions.csv": directory / "predictions.csv",
            "metrics.json": directory / "metrics.json",
        },
        reservation_token=request.get("reservation_token"),
    )


def main(path: str) -> int:
    """Execute one job and persist its outcome for the parent process.

    Parameters:
        path (str): JSON request path created by Jobs.

    Returns:
        int: Zero after publication, one on execution failure.
    """
    request = read_json(Path(path))
    directory = Path(request["directory"])
    catalog = Catalog(request["settings"]["storage_url"], request["settings"]["timeout"])

    try:
        {"collect": collect, "train": train, "predict": predict}[request["action"]](
            request, catalog, directory
        )

        atomic_json(directory / "outcome.json", {"result_id": request["job_id"]})
        progress(directory, "completed")
        return 0
    except Exception as exc:
        logger.exception("Job %s failed", request.get("job_id"))
        atomic_json(
            directory / "outcome.json",
            {
                "error": str(exc),
                "type": type(exc).__name__,
            },
        )
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
