"""Configuration, atomic state and HTTP utilities shared by service applications."""

import json
import logging
import os
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

API = "/api/v1"


def now() -> str:
    """Return the current UTC timestamp.

    Returns:
        str: ISO timestamp.
    """
    return datetime.now(UTC).isoformat()


def identifier(value: str) -> str:
    """Validate a single filesystem-safe resource component.

    Parameters:
        value (str): Identifier or artifact filename.

    Returns:
        str: Validated component without path separators.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", value):
        raise ValueError("Invalid identifier or filename")
    return value


def atomic_json(path: Path, value: Any) -> None:
    """Replace a JSON document after writing a complete temporary file.

    Parameters:
        path (Path): Destination whose parents may not exist.
        value (Any): JSON-compatible content without nonfinite numbers.

    Returns:
        None: Publishes the completed document atomically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")

    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), "utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON document.

    Parameters:
        path (Path): Existing JSON file.

    Returns:
        Any: Decoded content.
    """
    return json.loads(path.read_text("utf-8"))


def config_for(name: str) -> dict[str, Any]:
    """Load service settings from the selected YAML file.

    Parameters:
        name (str): Service name used for the default config path.

    Returns:
        dict[str, Any]: Configuration mapping.
    """
    path = Path(os.environ.get("SERVICE_CONFIG", f"deploy/{name}.yaml"))
    return yaml.safe_load(path.read_text("utf-8"))


def make_app(name: str, settings: dict[str, Any]) -> FastAPI:
    """Create an application with health reporting and consistent handled errors.

    Parameters:
        name (str): Service name.
        settings (dict[str, Any]): Includes state_dir for persistent logs.

    Returns:
        FastAPI: Application with health and error handlers.
    """
    state = Path(settings["state_dir"])
    state.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"services.{name}.{state.resolve()}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(state / "service.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)

    app = FastAPI(title=f"Feature selector: {name}")

    @app.middleware("http")
    async def log_request(request: Request, call_next: Any) -> Any:
        """Record request outcomes.

        Parameters:
            request (Request): Incoming request.
            call_next (Any): ASGI continuation.

        Returns:
            Any: Downstream response.
        """
        try:
            response = await call_next(request)
            logger.info("%s %s %s", request.method, request.url.path, response.status_code)
            return response
        except Exception:
            logger.exception("%s %s", request.method, request.url.path)
            raise

    @app.get(f"{API}/health")
    def health() -> dict[str, str]:
        """Report process health.

        Returns:
            dict[str, str]: Service identity and status.
        """
        return {"service": name, "status": "ok"}

    @app.exception_handler(Exception)
    async def handle_error(request: Request, exc: Exception) -> JSONResponse:
        """Translate operational exceptions to structured HTTP errors.

        Parameters:
            request (Request): Failed request.
            exc (Exception): Original exception.

        Returns:
            JSONResponse: Error without a public traceback.
        """
        logger.warning("%s: %s", request.url.path, exc)
        if isinstance(exc, FileNotFoundError):
            status, code = 404, "not_found"
        elif isinstance(exc, (ValueError, KeyError, TypeError)):
            status, code = 422, "invalid_input"
        elif isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            try:
                content = exc.response.json()
            except ValueError:
                content = {"detail": exc.response.text[:2000]}
            return JSONResponse(status_code=status, content=content)
        elif isinstance(exc, httpx.RequestError):
            status, code = 503, "dependency_unavailable"
        else:
            status, code = 500, "internal_error"

        detail = str(exc) if status != 500 else "See the service log for details"
        return JSONResponse(status_code=status, content={"code": code, "detail": detail})

    for handled in (
        FileNotFoundError,
        ValueError,
        KeyError,
        TypeError,
        httpx.HTTPStatusError,
        httpx.RequestError,
    ):
        app.add_exception_handler(handled, handle_error)

    return app


class Remote:
    """Bounded synchronous client for one service; requests are never retried implicitly."""

    def __init__(self, url: str, timeout: float = 60) -> None:
        """Store endpoint and timeout.

        Parameters:
            url (str): Base URL including the API prefix.
            timeout (float): Request timeout in seconds.

        Returns:
            None.
        """
        self.url = url.rstrip("/")
        self.timeout = timeout

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Send one request and propagate upstream HTTP errors.

        Parameters:
            method (str): HTTP method.
            path (str): Relative API path.
            **kwargs (Any): HTTPX request arguments.

        Returns:
            httpx.Response: Successful response.
        """
        response = httpx.request(method, self.url + path, timeout=self.timeout, **kwargs)
        response.raise_for_status()
        return response

    def json(self, method: str, path: str, **kwargs: Any) -> Any:
        """Send one request expecting JSON.

        Parameters:
            method (str): HTTP method.
            path (str): Relative API path.
            **kwargs (Any): HTTPX request arguments.

        Returns:
            Any: Decoded response.
        """
        return self.request(method, path, **kwargs).json()


class StateFiles:
    """Small serialized JSON catalog for a single service process."""

    def __init__(self, root: Path) -> None:
        """Create the catalog directory and lock.

        Parameters:
            root (Path): Catalog directory.

        Returns:
            None.
        """
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def get(self, key: str) -> dict[str, Any]:
        """Read a catalog entry.

        Parameters:
            key (str): Safe entry ID.

        Returns:
            dict[str, Any]: Entry content.
        """
        with self.lock:
            return read_json(self.root / f"{identifier(key)}.json")

    def put(self, value: dict[str, Any]) -> None:
        """Replace an entry by its ID.

        Parameters:
            value (dict[str, Any]): Entry containing id.

        Returns:
            None: Saves the entry.
        """
        with self.lock:
            atomic_json(self.root / f"{identifier(value['id'])}.json", value)

    def list(self) -> list[dict[str, Any]]:
        """Read catalog entries in creation order.

        Returns:
            list[dict[str, Any]]: Saved entries.
        """
        with self.lock:
            return sorted(
                [read_json(path) for path in self.root.glob("*.json")],
                key=lambda value: value["created_at"],
            )
