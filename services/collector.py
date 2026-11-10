"""Read-only chronological batches from configured CSV sources."""

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query

from services.common import API, config_for, make_app


def read_frame(path: Path) -> pd.DataFrame:
    """Read a numeric CSV indexed by unique timestamps.

    Parameters:
        path (Path): Source CSV with timestamps in its first column.

    Returns:
        pd.DataFrame: Sorted numeric observations with no infinite values.
    """
    import csv

    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if len(header) < 2 or len(set(header)) != len(header):
        raise ValueError("CSV requires unique column names")

    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="raise"))
    if frame.empty or frame.index.hasnans or frame.index.has_duplicates:
        raise ValueError("CSV requires nonempty, unique timestamps")

    frame = frame.sort_index().apply(pd.to_numeric, errors="raise")
    if np.isinf(frame.to_numpy(dtype=float)).any():
        raise ValueError("Infinite observations are unsupported")
    return frame


def table(frame: pd.DataFrame) -> dict[str, Any]:
    """Encode a frame with explicit timestamp and null missing values.

    Parameters:
        frame (pd.DataFrame): Chronological observations.

    Returns:
        dict[str, Any]: CSV-compatible columns and rows.
    """
    rows = frame.astype(object).where(frame.notna(), None).to_numpy().tolist()

    timestamp_column = "timestamp"
    while timestamp_column in frame.columns:
        timestamp_column += "_"

    return {
        "columns": [timestamp_column, *frame.columns.tolist()],
        "rows": [[str(stamp), *row] for stamp, row in zip(frame.index, rows)],
    }


def create_app(settings: dict[str, Any] | None = None) -> FastAPI:
    """Create a collector whose cached inputs must remain unchanged until restart.

    Parameters:
        settings (dict[str, Any] | None): Configuration or default YAML selection.

    Returns:
        FastAPI: Source discovery and batch API.
    """
    settings = settings or config_for("collector")
    app = make_app("collector", settings)
    cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    lock = threading.Lock()

    def source(source_id: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
        """Load and retain a consistent source snapshot.

        Parameters:
            source_id (str): Configured source name.

        Returns:
            tuple: Source configuration, features and targets.
        """
        if source_id not in settings["sources"]:
            raise FileNotFoundError("Unknown source")

        spec = settings["sources"][source_id]
        with lock:
            if source_id not in cache:
                features = read_frame(Path(spec["features_path"]))
                targets = read_frame(Path(spec["target_path"]))
                if pd.DatetimeIndex(features.index).tz != pd.DatetimeIndex(targets.index).tz:
                    raise ValueError("Feature and target timezones must match")
                if spec["target_column"] not in targets:
                    raise ValueError("Unknown target column")

                cache[source_id] = features, targets
            return spec, *cache[source_id]

    @app.get(f"{API}/sources")
    def sources() -> list[dict[str, Any]]:
        """List sources including unavailable inputs.

        Returns:
            list[dict[str, Any]]: Source descriptions and availability errors.
        """
        result = []
        for key in settings["sources"]:
            try:
                result.append(describe(key))
            except (OSError, ValueError, KeyError) as exc:
                result.append({"id": key, "available": False, "error": str(exc)})
        return result

    @app.get(f"{API}/sources/{{source_id}}")
    def describe(source_id: str) -> dict[str, Any]:
        """Describe a loaded source.

        Parameters:
            source_id (str): Configured source name.

        Returns:
            dict[str, Any]: Schema and time range.
        """
        spec, features, targets = source(source_id)
        return {
            "id": source_id,
            "available": True,
            "rows": len(features),
            "feature_names": features.columns.tolist(),
            "target_names": targets.columns.tolist(),
            "start": str(features.index[0]),
            "end": str(features.index[-1]),
            "freq": spec["freq"],
            "target_column": spec["target_column"],
        }

    @app.get(f"{API}/sources/{{source_id}}/batches")
    def batch(
        source_id: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(1000, ge=1),
    ) -> dict[str, Any]:
        """Return disjoint feature-row batches and targets in their temporal interval.

        Parameters:
            source_id (str): Configured source name.
            offset (int): Feature row position, at most the source length.
            limit (int): Requested row count, bounded by service configuration.

        Returns:
            dict[str, Any]: Tables, next cursor and end marker.
        """
        _, features, targets = source(source_id)
        if offset > len(features) or limit > settings["max_batch_size"]:
            raise ValueError("Cursor or batch size exceeds source limits")

        stop = min(offset + limit, len(features))
        selected = features.iloc[offset:stop]

        chosen = targets.iloc[:0]
        if len(selected):
            mask = targets.index >= selected.index[0]
            if stop < len(features):
                mask &= targets.index < features.index[stop]
            else:
                mask &= targets.index <= features.index[-1]
            chosen = targets.loc[mask]

        return {
            "features": table(selected),
            "targets": table(chosen),
            "next_offset": stop,
            "done": stop == len(features),
        }

    return app
