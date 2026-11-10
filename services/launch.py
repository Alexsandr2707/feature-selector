"""Configured process entrypoints and supervised Streamlit lifecycle."""

import os
import signal
import subprocess
import sys
import time
from typing import Any

import uvicorn

from services.common import API, config_for, make_app


def visualization_api() -> Any:
    """Expose visualization health independently of Streamlit internals.

    Returns:
        Any: Service API application.
    """
    settings = config_for("visualization")
    app = make_app("visualization", settings)

    @app.get(f"{API}/summary")
    def summary() -> dict[str, Any]:
        """Describe GUI capabilities.

        Returns:
            dict[str, Any]: Read-only visualization description.
        """
        return {"interface": "streamlit", "pages": ["data", "training", "prediction", "comparison"]}

    return app


def supervise(settings: dict[str, Any]) -> int:
    """Run Streamlit and its API together, stopping both on shutdown or child failure.

    Parameters:
        settings (dict[str, Any]): Visualization ports and host.

    Returns:
        int: Child failure or clean shutdown code.
    """
    processes: list[subprocess.Popen[bytes]] = []
    stopping = False

    def stop(signum: int, frame: Any) -> None:
        """Request orderly child shutdown.

        Parameters:
            signum (int): Received termination signal.
            frame (Any): Interrupted frame.

        Returns:
            None: Marks the supervisor for shutdown.
        """
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)

    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([os.getcwd(), os.environ.get("PYTHONPATH", "")]),
    }

    try:
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "services.launch:visualization_api",
                    "--factory",
                    "--host",
                    settings["host"],
                    "--port",
                    str(settings["api_port"]),
                ],
                env=environment,
            )
        )

        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    "services/visualization.py",
                    "--server.address",
                    settings["host"],
                    "--server.port",
                    str(settings["port"]),
                    "--server.headless",
                    "true",
                    "--browser.gatherUsageStats",
                    "false",
                ],
                env=environment,
            )
        )

        while not stopping:
            if any(process.poll() is not None for process in processes):
                return 1
            time.sleep(0.25)
        return 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()

        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main() -> int:
    """Start the service selected on the command line.

    Returns:
        int: Process exit status.
    """
    name = sys.argv[1]
    if name not in {"collector", "storage", "ml", "master", "visualization"}:
        raise ValueError("Unknown service")

    settings = config_for(name)
    if name == "visualization":
        return supervise(settings)

    uvicorn.run(
        f"services.{name}:create_app",
        factory=True,
        host=settings["host"],
        port=settings["port"],
        workers=1,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
