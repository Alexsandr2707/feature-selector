"""Single-process file catalog with draft publication and dependency protection."""

import hashlib
import secrets
import shutil
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from services.common import API, atomic_json, config_for, identifier, make_app, now, read_json
from services.contracts import DraftUpdateInput, ObjectInput, RenameInput

KINDS = {"datasets", "models", "runs", "predictions"}
REQUIRED = {
    "datasets": {"features.csv", "target.csv"},
    "models": set(),
    "runs": {"run.pkl", "config.yaml", "metrics.json", "metadata.json", "history.json"},
    "predictions": {"predictions.csv", "metrics.json"},
}
REFERENCES = {"dataset_id": "datasets", "model_id": "models", "run_id": "runs"}


def create_app(settings: dict[str, Any] | None = None) -> FastAPI:
    """Create the object catalog; run with exactly one API worker.

    Parameters:
        settings (dict[str, Any] | None): Configuration or default YAML selection.

    Returns:
        FastAPI: Draft, publication, download and CRUD API.
    """
    settings = settings or config_for("storage")
    app = make_app("storage", settings)

    root = Path(settings["state_dir"]) / "objects"
    root.mkdir(parents=True, exist_ok=True)
    lock = threading.RLock()

    def directory(kind: str, object_id: str) -> Path:
        """Resolve a catalog location without accepting paths.

        Parameters:
            kind (str): Known collection.
            object_id (str): Safe identifier.

        Returns:
            Path: Object directory.
        """
        if kind not in KINDS:
            raise FileNotFoundError("Unknown collection")
        return root / kind / identifier(object_id)

    def get(kind: str, object_id: str, published: bool = True) -> dict[str, Any]:
        """Read a manifest and optionally require publication.

        Parameters:
            kind (str): Collection name.
            object_id (str): Object ID.
            published (bool): Whether drafts must remain hidden.

        Returns:
            dict[str, Any]: Manifest.
        """
        value = read_json(directory(kind, object_id) / "manifest.json")
        if published and value["status"] != "published":
            raise FileNotFoundError("Object is not published")
        return value

    def authorize(kind: str, object_id: str, value: dict[str, Any], request: Request) -> None:
        """Require the controller capability when mutating an active job draft.

        Parameters:
            kind (str): Collection name.
            object_id (str): Draft ID.
            value (dict[str, Any]): Current manifest.
            request (Request): Request carrying the private reservation header.

        Returns:
            None: Rejects changes by ordinary CRUD clients while a job owns the draft.
        """
        if value.get("reserved"):
            expected = (directory(kind, object_id) / ".reservation").read_text("utf-8")
            token = request.headers.get("X-Reservation-Token", "")
            actual = hashlib.sha256(token.encode()).hexdigest()
            if not secrets.compare_digest(expected, actual):
                raise HTTPException(409, "Draft is reserved by an active or unresolved job")

    @app.get(f"{API}/drafts/{{kind}}")
    def drafts(kind: str) -> list[dict[str, Any]]:
        """List unpublished drafts, including their public job ownership.

        Parameters:
            kind (str): Collection name.

        Returns:
            list[dict[str, Any]]: Draft manifests without reservation capabilities.
        """
        directory(kind, "check")
        with lock:
            values = [read_json(path) for path in (root / kind).glob("*/manifest.json")]
            return sorted(
                [value for value in values if value["status"] == "draft"],
                key=lambda value: value["created_at"],
            )

    @app.get(f"{API}/drafts/{{kind}}/{{object_id}}")
    def draft(kind: str, object_id: str) -> dict[str, Any]:
        """Describe an unpublished draft for diagnostics and cleanup.

        Parameters:
            kind (str): Collection name.
            object_id (str): Draft ID.

        Returns:
            dict[str, Any]: Public draft manifest.
        """
        with lock:
            value = get(kind, object_id, published=False)
            if value["status"] != "draft":
                raise HTTPException(409, "Object is already published")
            return value

    @app.patch(f"{API}/drafts/{{kind}}/{{object_id}}")
    def update_draft(
        kind: str, object_id: str, body: DraftUpdateInput, request: Request
    ) -> dict[str, Any]:
        """Set final draft metadata without changing its protected dependencies.

        Parameters:
            kind (str): Collection name.
            object_id (str): Draft ID.
            body (DraftUpdateInput): Final result name and metadata.
            request (Request): Owner capability for a reserved draft.

        Returns:
            dict[str, Any]: Updated draft manifest.
        """
        with lock:
            value = get(kind, object_id, published=False)
            if value["status"] != "draft":
                raise HTTPException(409, "Published content is immutable")
            authorize(kind, object_id, value, request)

            value.update(name=body.name, metadata=body.metadata)
            atomic_json(directory(kind, object_id) / "manifest.json", value)
            return value

    @app.delete(f"{API}/drafts/{{kind}}/{{object_id}}", status_code=204)
    def discard(kind: str, object_id: str, request: Request) -> None:
        """Remove an abandoned draft, requiring ownership for an active reservation.

        Parameters:
            kind (str): Collection name.
            object_id (str): Draft ID.
            request (Request): Optional controller capability.

        Returns:
            None: Atomically releases dependencies by deleting the draft directory.
        """
        with lock:
            value = get(kind, object_id, published=False)
            if value["status"] != "draft":
                raise HTTPException(409, "Published objects require ordinary object deletion")
            authorize(kind, object_id, value, request)
            shutil.rmtree(directory(kind, object_id))

    @app.get(f"{API}/objects/{{kind}}")
    def listing(kind: str) -> list[dict[str, Any]]:
        """List published objects.

        Parameters:
            kind (str): Collection name.

        Returns:
            list[dict[str, Any]]: Manifests in creation order.
        """
        directory(kind, "check")
        with lock:
            values = [read_json(path) for path in (root / kind).glob("*/manifest.json")]
            return sorted(
                [value for value in values if value["status"] == "published"],
                key=lambda value: value["created_at"],
            )

    @app.post(f"{API}/objects/{{kind}}", status_code=201)
    def create(kind: str, body: ObjectInput) -> dict[str, Any]:
        """Create a hidden draft with validated references.

        Parameters:
            kind (str): Collection name.
            body (ObjectInput): Draft metadata.

        Returns:
            dict[str, Any]: Draft manifest.
        """
        key = body.id or uuid4().hex
        destination = directory(kind, key)
        with lock:
            if (body.job_id is None) != (body.reservation_token is None):
                raise ValueError("Job ID and reservation token must be provided together")
            if body.job_id is not None and body.job_id != key:
                raise ValueError("Job reservation ID must match its result ID")

            for reference, target in body.references.items():
                if reference not in REFERENCES:
                    raise ValueError("Unknown reference type")
                get(REFERENCES[reference], target)
            required_refs = {
                "datasets": set(),
                "models": {"dataset_id"},
                "runs": {"dataset_id", "model_id"},
                "predictions": {"dataset_id", "run_id"},
            }
            if not required_refs[kind] <= body.references.keys():
                raise ValueError("Missing required object references")
            if destination.exists():
                raise HTTPException(409, "Object ID already exists")

            destination.mkdir(parents=True)
            if body.reservation_token is not None:
                (destination / ".reservation").write_text(
                    hashlib.sha256(body.reservation_token.encode()).hexdigest(), "utf-8"
                )

            value = {
                "id": key,
                "kind": kind,
                "name": body.name,
                "created_at": now(),
                "status": "draft",
                "metadata": body.metadata,
                "references": body.references,
                "files": {},
                "job_id": body.job_id,
                "reserved": body.reservation_token is not None,
            }

            atomic_json(destination / "manifest.json", value)
            return value

    @app.get(f"{API}/objects/{{kind}}/{{object_id}}")
    def describe(kind: str, object_id: str) -> dict[str, Any]:
        """Read a published manifest.

        Parameters:
            kind (str): Collection name.
            object_id (str): Object ID.

        Returns:
            dict[str, Any]: Published manifest.
        """
        with lock:
            return get(kind, object_id)

    @app.put(f"{API}/objects/{{kind}}/{{object_id}}/files/{{filename}}")
    async def upload(kind: str, object_id: str, filename: str, request: Request) -> dict[str, Any]:
        """Upload one bounded file to a draft using an isolated temporary file.

        Parameters:
            kind (str): Collection name.
            object_id (str): Draft ID.
            filename (str): Flat artifact name.
            request (Request): Raw file bytes.

        Returns:
            dict[str, Any]: Saved length and checksum.
        """
        identifier(filename)
        if filename == "manifest.json":
            raise ValueError("Reserved filename")

        # Bound the body before entering the catalog lock.
        content = bytearray()
        async for chunk in request.stream():
            content.extend(chunk)
            if len(content) > settings["max_file_bytes"]:
                raise HTTPException(413, "Artifact exceeds configured upload limit")

        with lock:
            value = get(kind, object_id, published=False)
            if value["status"] != "draft":
                raise HTTPException(409, "Published content is immutable")
            authorize(kind, object_id, value, request)

            path = directory(kind, object_id) / filename
            temporary = path.with_name(f".{uuid4().hex}.upload")
            try:
                temporary.write_bytes(content)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)

            result = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            value["files"][filename] = result
            atomic_json(path.parent / "manifest.json", value)
            return result

    @app.post(f"{API}/objects/{{kind}}/{{object_id}}/publish")
    def publish(kind: str, object_id: str, request: Request) -> dict[str, Any]:
        """Make a complete draft visible atomically.

        Parameters:
            kind (str): Collection name.
            object_id (str): Draft ID.
            request (Request): Owner capability for a reserved draft.

        Returns:
            dict[str, Any]: Published manifest.
        """
        with lock:
            value = get(kind, object_id, published=False)
            if value["status"] == "published":
                return value
            authorize(kind, object_id, value, request)
            if not REQUIRED[kind] <= value["files"].keys():
                raise ValueError("Missing required artifact files")

            for filename, expected in value["files"].items():
                path = directory(kind, object_id) / filename
                with path.open("rb") as stream:
                    checksum = hashlib.file_digest(stream, "sha256").hexdigest()
                if path.stat().st_size != expected["bytes"] or checksum != expected["sha256"]:
                    raise ValueError("Draft artifact checksum mismatch")

            value["status"] = "published"
            value["reserved"] = False
            atomic_json(directory(kind, object_id) / "manifest.json", value)
            return value

    @app.get(f"{API}/objects/{{kind}}/{{object_id}}/files/{{filename}}")
    def download(kind: str, object_id: str, filename: str) -> FileResponse:
        """Download an advertised published artifact.

        Parameters:
            kind (str): Collection name.
            object_id (str): Published ID.
            filename (str): Flat artifact name.

        Returns:
            FileResponse: Artifact content.
        """
        with lock:
            value = get(kind, object_id)
            if identifier(filename) not in value["files"]:
                raise FileNotFoundError("Unknown artifact")
            return FileResponse(directory(kind, object_id) / filename, filename=filename)

    @app.patch(f"{API}/objects/{{kind}}/{{object_id}}")
    def rename(kind: str, object_id: str, body: RenameInput) -> dict[str, Any]:
        """Change only the display name.

        Parameters:
            kind (str): Collection name.
            object_id (str): Published ID.
            body (RenameInput): New name.

        Returns:
            dict[str, Any]: Updated manifest.
        """
        with lock:
            value = get(kind, object_id)
            value["name"] = body.name
            atomic_json(directory(kind, object_id) / "manifest.json", value)
            return value

    @app.delete(f"{API}/objects/{{kind}}/{{object_id}}", status_code=204)
    def delete(kind: str, object_id: str) -> None:
        """Delete an unreferenced object, including abandoned drafts.

        Parameters:
            kind (str): Collection name.
            object_id (str): Object ID.

        Returns:
            None: Removes its catalog directory.
        """
        with lock:
            value = get(kind, object_id, published=False)
            if value.get("reserved"):
                raise HTTPException(409, "Object is reserved by an active or unresolved job")

            dependents = []
            for path in root.glob("*/*/manifest.json"):
                value = read_json(path)
                for name, target in value["references"].items():
                    if REFERENCES[name] == kind and target == object_id:
                        dependents.append(f"{value['kind']}/{value['id']}")
            if dependents:
                raise HTTPException(409, {"dependents": dependents})

            shutil.rmtree(directory(kind, object_id))

    return app
