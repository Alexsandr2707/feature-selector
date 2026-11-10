"""Scenario gateway; only this API is called by the visualization service."""

import csv
import io
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import httpx
from fastapi import FastAPI, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from services.catalog import Catalog
from services.common import API, Remote, config_for, identifier, make_app
from services.contracts import ModelInput, PredictionInput, RenameInput
from services.jobs import Jobs


class CollectionInput(BaseModel):
    """Whole-source collection request."""

    source_id: str
    name: str = Field(min_length=1, max_length=200)


def create_app(settings: dict[str, Any] | None = None) -> FastAPI:
    """Create gateway routes and the collection executor.

    Parameters:
        settings (dict[str, Any] | None): Configuration or default YAML selection.

    Returns:
        FastAPI: Scenario API without numerical imports.
    """
    settings = settings or config_for("master")
    app = make_app("master", settings)

    catalog = Catalog(settings["storage_url"], settings["timeout"])
    collector = Remote(settings["collector_url"], settings["timeout"])
    ml = Remote(settings["ml_url"], settings["timeout"])

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """Own collection executor lifetime.

        Parameters:
            application (FastAPI): Application receiving executor state.

        Returns:
            AsyncIterator[None]: Lifespan scope.
        """
        application.state.jobs = Jobs(settings)
        try:
            yield
        finally:
            application.state.jobs.close()

    app.router.lifespan_context = lifespan

    @app.get(f"{API}/status")
    def status() -> dict[str, Any]:
        """Probe downstream services independently.

        Returns:
            dict[str, Any]: Availability and errors per service.
        """
        result = {}
        for name, remote in {"collector": collector, "storage": catalog, "ml": ml}.items():
            try:
                result[name] = Remote(remote.url, 3).json("GET", "/health")
            except (httpx.HTTPError, ValueError) as exc:
                result[name] = {"status": "unavailable", "error": str(exc)}
        return result

    @app.get(f"{API}/sources")
    def sources() -> Any:
        """List source availability.

        Returns:
            Any: Collector source descriptors.
        """
        return collector.json("GET", "/sources")

    @app.post(f"{API}/collections", status_code=202)
    def collect(body: CollectionInput) -> dict[str, Any]:
        """Start collection independently of the browser session.

        Parameters:
            body (CollectionInput): Whole-source request.

        Returns:
            dict[str, Any]: Running collection job.
        """
        identifier(body.source_id)
        collector.json("GET", f"/sources/{body.source_id}")
        return app.state.jobs.start("collect", body.model_dump(), "datasets")

    @app.get(f"{API}/collections")
    def collections() -> list[dict[str, Any]]:
        """List collection attempts.

        Returns:
            list[dict[str, Any]]: Persisted collection jobs.
        """
        return app.state.jobs.list()

    @app.get(f"{API}/collections/{{key}}")
    def collection(key: str) -> dict[str, Any]:
        """Read collection progress.

        Parameters:
            key (str): Collection job ID.

        Returns:
            dict[str, Any]: Job state.
        """
        return app.state.jobs.get(key)

    @app.get(f"{API}/templates")
    def templates() -> Any:
        """Read ML template descriptors.

        Returns:
            Any: Editable template fields.
        """
        return ml.json("GET", "/templates")

    @app.post(f"{API}/models", status_code=201)
    def model(body: ModelInput) -> Any:
        """Create a model specification through ML Service.

        Parameters:
            body (ModelInput): Specification input.

        Returns:
            Any: Published specification.
        """
        return ml.json("POST", "/models", json=body.model_dump())

    @app.get(f"{API}/models/{{key}}")
    def summary(key: str) -> Any:
        """Read model summary.

        Parameters:
            key (str): Model ID.

        Returns:
            Any: Model and related runs.
        """
        return ml.json("GET", f"/models/{identifier(key)}")

    @app.post(f"{API}/models/{{key}}/train", status_code=202)
    def train(key: str) -> Any:
        """Submit model training.

        Parameters:
            key (str): Model ID.

        Returns:
            Any: Compute job.
        """
        return ml.json("POST", f"/models/{identifier(key)}/train")

    @app.post(f"{API}/runs/{{key}}/predict", status_code=202)
    def predict(key: str, body: PredictionInput) -> Any:
        """Submit explicit split prediction.

        Parameters:
            key (str): Run ID.
            body (PredictionInput): Requested split.

        Returns:
            Any: Compute job.
        """
        return ml.json("POST", f"/runs/{identifier(key)}/predict", json=body.model_dump())

    @app.get(f"{API}/jobs")
    def jobs() -> Any:
        """List ML attempts.

        Returns:
            Any: ML job states.
        """
        return ml.json("GET", "/jobs")

    @app.get(f"{API}/jobs/{{key}}")
    def job(key: str) -> Any:
        """Read an ML job.

        Parameters:
            key (str): ML job ID.

        Returns:
            Any: State and progress.
        """
        return ml.json("GET", f"/jobs/{identifier(key)}")

    @app.get(f"{API}/jobs/{{key}}/events")
    def events(key: str) -> Any:
        """Read recent ML events.

        Parameters:
            key (str): ML job ID.

        Returns:
            Any: Bounded log tail.
        """
        return ml.json("GET", f"/jobs/{identifier(key)}/events")

    @app.get(f"{API}/comparison")
    def comparison(run_ids: Annotated[list[str], Query()]) -> dict[str, Any]:
        """Compare only matching data, target, horizon and split protocols.

        Parameters:
            run_ids (list[str]): Runs selected for comparison.

        Returns:
            dict[str, Any]: Validation scores, parameters and sample-count warning.
        """
        runs = [catalog.get("runs", key) for key in run_ids]
        if not runs:
            raise ValueError("Select at least one run")
        groups = {json.dumps(run["metadata"]["comparison"], sort_keys=True) for run in runs}
        if len(groups) != 1:
            raise ValueError("Runs differ in dataset, target, horizon, frequency or split")

        rows = [
            {
                "id": run["id"],
                "name": run["name"],
                "parameters": run["metadata"]["parameters"],
                **run["metadata"]["metrics"].get("validation", {}),
            }
            for run in runs
        ]

        return {
            "rows": rows,
            "sample_count_warning": len({row.get("n") for row in rows}) > 1,
            "sample_set_warning": len(
                {run["metadata"].get("validation_times_sha256") for run in runs}
            )
            > 1,
            "note": "Validation metrics use observed targets in original units",
        }

    @app.get(f"{API}/objects/{{kind}}")
    def listing(kind: str) -> Any:
        """List published catalog objects.

        Parameters:
            kind (str): Collection name.

        Returns:
            Any: Published manifests.
        """
        return catalog.json("GET", f"/objects/{identifier(kind)}")

    @app.get(f"{API}/drafts/{{kind}}")
    def drafts(kind: str) -> Any:
        """List drafts and public ownership for diagnosing abandoned uploads.

        Parameters:
            kind (str): Collection name.

        Returns:
            Any: Draft manifests without private reservation capabilities.
        """
        return catalog.json("GET", f"/drafts/{identifier(kind)}")

    @app.get(f"{API}/drafts/{{kind}}/{{key}}")
    def draft(kind: str, key: str) -> Any:
        """Describe one unpublished draft.

        Parameters:
            kind (str): Collection name.
            key (str): Draft ID.

        Returns:
            Any: Public draft manifest.
        """
        return catalog.json("GET", f"/drafts/{identifier(kind)}/{identifier(key)}")

    @app.delete(f"{API}/drafts/{{kind}}/{{key}}", status_code=204)
    def discard(kind: str, key: str) -> None:
        """Delete an ordinary abandoned draft, preserving active job reservations.

        Parameters:
            kind (str): Collection name.
            key (str): Draft ID.

        Returns:
            None: Removes the draft unless Storage reports protected ownership.
        """
        catalog.request("DELETE", f"/drafts/{identifier(kind)}/{identifier(key)}")

    @app.get(f"{API}/objects/{{kind}}/{{key}}")
    def describe(kind: str, key: str) -> Any:
        """Describe a published object.

        Parameters:
            kind (str): Collection name.
            key (str): Object ID.

        Returns:
            Any: Manifest.
        """
        return catalog.get(kind, key)

    @app.patch(f"{API}/objects/{{kind}}/{{key}}")
    def rename(kind: str, key: str, body: RenameInput) -> Any:
        """Rename a published object.

        Parameters:
            kind (str): Collection name.
            key (str): Object ID.
            body (RenameInput): Display name.

        Returns:
            Any: Updated manifest.
        """
        return catalog.json(
            "PATCH", f"/objects/{identifier(kind)}/{identifier(key)}", json=body.model_dump()
        )

    @app.delete(f"{API}/objects/{{kind}}/{{key}}", status_code=204)
    def delete(kind: str, key: str) -> None:
        """Delete an unreferenced object through Storage.

        Parameters:
            kind (str): Collection name.
            key (str): Object ID.

        Returns:
            None: Removes the object if permitted.
        """
        catalog.request("DELETE", f"/objects/{identifier(kind)}/{identifier(key)}")

    @app.get(f"{API}/objects/{{kind}}/{{key}}/files/{{filename}}")
    def download(kind: str, key: str, filename: str) -> Response:
        """Download an artifact through the gateway.

        Parameters:
            kind (str): Collection name.
            key (str): Object ID.
            filename (str): Flat artifact name.

        Returns:
            Response: Downloadable bytes.
        """
        identifier(filename)
        response = catalog.request(
            "GET", f"/objects/{identifier(kind)}/{identifier(key)}/files/{filename}"
        )
        return Response(
            response.content,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )

    @app.get(f"{API}/datasets/{{key}}/preview")
    def preview(key: str) -> dict[str, Any]:
        """Return a bounded row preview of a saved feature file.

        Parameters:
            key (str): Dataset ID.

        Returns:
            dict[str, Any]: First twenty rows.
        """
        content = catalog.request("GET", f"/objects/datasets/{identifier(key)}/files/features.csv")
        reader = csv.DictReader(io.StringIO(content.text))
        from itertools import islice

        return {"rows": list(islice(reader, 20))}

    return app
