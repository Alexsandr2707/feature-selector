"""Model specifications and isolated compute job API."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI

from services.catalog import Catalog
from services.common import API, config_for, make_app
from services.contracts import ModelInput, PredictionInput
from services.jobs import Jobs
from services.templates import configured_templates, resolve_template


def create_app(settings: dict[str, Any] | None = None) -> FastAPI:
    """Create the ML control plane without importing numerical algorithms.

    Parameters:
        settings (dict[str, Any] | None): Configuration or default YAML selection.

    Returns:
        FastAPI: Template, specification and compute endpoints.
    """
    settings = settings or config_for("ml")
    app = make_app("ml", settings)
    catalog = Catalog(settings["storage_url"], settings["timeout"])

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """Own executor recovery and shutdown.

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

    @app.get(f"{API}/templates")
    def templates() -> list[dict[str, Any]]:
        """List restricted template descriptors.

        Returns:
            list[dict[str, Any]]: Available model templates.
        """
        return configured_templates(settings)

    @app.post(f"{API}/models", status_code=201)
    def create_model(body: ModelInput) -> dict[str, Any]:
        """Resolve and freeze a model specification, without fitting an estimator.

        Parameters:
            body (ModelInput): Dataset, template and editable parameters.

        Returns:
            dict[str, Any]: Published model specification.
        """
        config = resolve_template(settings, body.template_id, body.parameters)
        dataset = catalog.get("datasets", body.dataset_id)
        spec = settings["templates"][body.template_id]

        if dataset["metadata"]["source_id"] != spec["source_id"]:
            raise ValueError("Template is incompatible with dataset source")
        if config["data"]["freq"] != dataset["metadata"]["freq"]:
            raise ValueError("Dataset frequency differs from template")

        parameters = {**spec["parameters"], **body.parameters}
        return catalog.publish(
            "models",
            uuid4().hex,
            body.name,
            {
                "template_id": body.template_id,
                "parameters": parameters,
                "config": config,
            },
            {"dataset_id": body.dataset_id},
            {},
        )

    @app.get(f"{API}/models")
    def models() -> list[dict[str, Any]]:
        """List model specifications.

        Returns:
            list[dict[str, Any]]: Published specifications.
        """
        return catalog.json("GET", "/objects/models")

    @app.get(f"{API}/models/{{key}}")
    def model(key: str) -> dict[str, Any]:
        """Describe a model and its completed runs.

        Parameters:
            key (str): Model ID.

        Returns:
            dict[str, Any]: Specification with run summaries.
        """
        value = catalog.get("models", key)
        value["runs"] = [
            run
            for run in catalog.json("GET", "/objects/runs")
            if run["references"]["model_id"] == key
        ]
        return value

    @app.post(f"{API}/models/{{key}}/train", status_code=202)
    def train_model(key: str) -> dict[str, Any]:
        """Submit training if the executor is idle.

        Parameters:
            key (str): Published model ID.

        Returns:
            dict[str, Any]: Running job.
        """
        return app.state.jobs.start("train", {"model": catalog.get("models", key)}, "runs")

    @app.get(f"{API}/runs/{{key}}")
    def run(key: str) -> dict[str, Any]:
        """Describe a completed run.

        Parameters:
            key (str): Run ID.

        Returns:
            dict[str, Any]: Run manifest and summary metrics.
        """
        return catalog.get("runs", key)

    @app.post(f"{API}/runs/{{key}}/predict", status_code=202)
    def predict_run(key: str, body: PredictionInput) -> dict[str, Any]:
        """Submit explicit split evaluation against the original dataset.

        Parameters:
            key (str): Run ID.
            body (PredictionInput): Requested temporal split.

        Returns:
            dict[str, Any]: Running prediction job.
        """
        catalog.get("runs", key)
        return app.state.jobs.start("predict", {"run_id": key, "split": body.split}, "predictions")

    @app.get(f"{API}/jobs")
    def jobs() -> list[dict[str, Any]]:
        """List persisted jobs including failed and interrupted attempts.

        Returns:
            list[dict[str, Any]]: Job states.
        """
        return app.state.jobs.list()

    @app.get(f"{API}/jobs/{{key}}")
    def job(key: str) -> dict[str, Any]:
        """Read one job and its current stage.

        Parameters:
            key (str): Job ID.

        Returns:
            dict[str, Any]: Persisted job state.
        """
        return app.state.jobs.get(key)

    @app.get(f"{API}/jobs/{{key}}/events")
    def events(key: str) -> dict[str, Any]:
        """Read recent diagnostics.

        Parameters:
            key (str): Job ID.

        Returns:
            dict[str, Any]: Bounded event and log tails.
        """
        return app.state.jobs.events(key)

    return app
