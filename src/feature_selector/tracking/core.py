"""Local structured events and TensorBoard metrics scoped to pipeline steps."""

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from torch.utils.tensorboard import SummaryWriter


def json_value(value: Any) -> Any:
    """Convert diagnostic values to JSON-compatible objects.

    Parameters:
        value (Any): Scalars, collections, paths or objects with tolist/item methods.

    Returns:
        Any: JSON-compatible diagnostic value.
    """
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        from math import isfinite

        return value if isfinite(value) else None
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return json_value(tolist())
    return str(value)


class RunTracker:
    """Own event files and TensorBoard writers for one execution."""

    def __init__(self, directory: Path) -> None:
        """Open outputs in an existing run directory.

        Parameters:
            directory (Path): Destination directory for append-only events and logs.

        Returns:
            None.
        """
        self.events: TextIO = (directory / "events.jsonl").open("a", encoding="utf-8")
        self.writer = SummaryWriter(str(directory / "tensorboard"))
        self.metadata: dict[str, dict[str, Any]] = {}

        self.logger = logging.getLogger(f"feature_selector.run.{directory.name}")
        self.logger.setLevel(logging.INFO)

        self.handler = logging.FileHandler(directory / "run.log", encoding="utf-8")
        self.handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(self.handler)

    def record(self, path: str, event: str, values: dict[str, Any]) -> None:
        """Persist an event immediately so interrupted fits retain completed batches.

        Parameters:
            path (str): Slash-separated pipeline step path.
            event (str): Event name, such as metrics or completed.
            values (dict[str, Any]): Diagnostic fields belonging to this event.

        Returns:
            None: Appends and flushes one JSONL record.
        """
        record = {
            "time": datetime.now(UTC).isoformat(),
            "path": path,
            "event": event,
            "data": json_value(values),
        }
        self.events.write(json.dumps(record, allow_nan=False) + "\n")
        self.events.flush()
        lifecycle = {"started", "completed", "failed", "run_started", "run_completed", "run_failed"}
        if event in lifecycle:
            self.logger.info("%s %s %s", path, event, json_value(values))

    def close(self) -> None:
        """Flush and close all outputs, including after execution errors.

        Returns:
            None.
        """
        try:
            self.writer.close()
        finally:
            self.events.close()
            self.handler.close()
            self.logger.removeHandler(self.handler)


@dataclass(frozen=True)
class StepContext:
    """Bind optional tracking to a step without storing it inside estimators."""

    tracker: RunTracker | None = field(default=None, repr=False)
    path: str = "pipeline"
    operation: str = "fit"

    def child(self, name: str, operation: str | None = None) -> "StepContext":
        """Create a nested scope for a step or composite pipeline.

        Parameters:
            name (str): Child step name.
            operation (str | None): Operation override, or the current operation.

        Returns:
            StepContext: Bound child context sharing the same tracker.
        """
        return StepContext(self.tracker, f"{self.path}/{name}", operation or self.operation)

    def log_event(self, name: str, values: dict[str, Any]) -> None:
        """Write step-specific diagnostic information.

        Parameters:
            name (str): Event name.
            values (dict[str, Any]): Diagnostic fields in this step's namespace.

        Returns:
            None.
        """
        if self.tracker is not None:
            self.tracker.record(self.path, name, {"operation": self.operation, **values})

    def log_metadata(self, values: dict[str, Any]) -> None:
        """Update only this step's summary metadata and record the update.

        Parameters:
            values (dict[str, Any]): Small diagnostic values; large arrays belong in artifacts.

        Returns:
            None.
        """
        if self.tracker is not None:
            self.tracker.metadata.setdefault(self.path, {}).update(json_value(values))
        self.log_event("metadata", values)

    def log_metrics(
        self,
        values: dict[str, float | None],
        *,
        split: str | None = None,
        epoch: int | None = None,
        batch: int | None = None,
        step: int | None = None,
        samples: int | None = None,
    ) -> None:
        """Record metrics and publish scalars using batch or epoch coordinates.

        Parameters:
            values (dict[str, float | None]): Scalar metric names and values.
            split (str | None): Train or validation label.
            epoch (int | None): One-based epoch.
            batch (int | None): One-based batch within the epoch.
            step (int | None): Monotonic optimizer step for batch metrics.
            samples (int | None): Actual number of examples represented by the metric.

        Returns:
            None.
        """
        self.log_event(
            "metrics",
            {
                "values": values,
                "split": split,
                "epoch": epoch,
                "batch": batch,
                "step": step,
                "samples": samples,
            },
        )

        if self.tracker is not None:
            coordinate = step if step is not None else epoch or 0
            level = "batch" if batch is not None else "epoch" if epoch is not None else "summary"
            scope = f"{self.path}/{split or 'summary'}/{level}"
            if batch is None:
                self.tracker.logger.info("%s %s epoch=%s %s", self.path, split, epoch, values)
            for name, value in values.items():
                if value is not None:
                    self.tracker.writer.add_scalar(f"{scope}/{name}", value, coordinate)

    def log_message(self, message: str, *, level: str = "info") -> None:
        """Write a readable message in the run log and event stream.

        Parameters:
            message (str): Human-readable diagnostic message.
            level (str): Standard logging level name.

        Returns:
            None.
        """
        if self.tracker is not None:
            self.tracker.logger.log(getattr(logging, level.upper()), "%s: %s", self.path, message)
        self.log_event("message", {"level": level, "message": message})
