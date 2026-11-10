"""Executable four-scenario demonstration against the public gateway."""

import argparse
import json
import time
from typing import Any

from services.common import Remote


def wait(remote: Remote, route: str, job: dict[str, Any], timeout: float) -> str:
    """Poll running or reconciling jobs until a terminal outcome or a bounded deadline.

    Parameters:
        remote (Remote): Gateway client.
        route (str): Job collection route.
        job (dict[str, Any]): Submission response.
        timeout (float): Maximum wait in seconds.

    Returns:
        str: Published result ID; failures include diagnostic state.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = remote.json("GET", f"{route}/{job['id']}")
        if state["status"] == "completed":
            return state["result_id"]
        if state["status"] not in {"running", "reconciling"}:
            raise RuntimeError(json.dumps(state, ensure_ascii=False))
        time.sleep(1)
    raise TimeoutError(f"Job {job['id']} is not resolved; inspect it through the API")


def main() -> None:
    """Collect UZK, train Ridge and GRU, explicitly evaluate test and compare validation.

    Returns:
        None: Prints IDs and the validation comparison as JSON.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/v1")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()

    remote = Remote(args.url, 120)

    dataset = wait(
        remote,
        "/collections",
        remote.json(
            "POST",
            "/collections",
            json={
                "source_id": "uzk",
                "name": "UZK / demonstration",
            },
        ),
        args.timeout,
    )

    runs, predictions = [], []
    for template in ("uzk_ridge", "uzk_gru"):
        model = remote.json(
            "POST",
            "/models",
            json={
                "dataset_id": dataset,
                "template_id": template,
                "name": template,
            },
        )
        run = wait(
            remote, "/jobs", remote.json("POST", f"/models/{model['id']}/train"), args.timeout
        )
        runs.append(run)

        predictions.append(
            wait(
                remote,
                "/jobs",
                remote.json("POST", f"/runs/{run}/predict", json={"split": "test"}),
                args.timeout,
            )
        )

    comparison = remote.json("GET", "/comparison", params=[("run_ids", key) for key in runs])
    print(
        json.dumps(
            {
                "dataset_id": dataset,
                "run_ids": runs,
                "prediction_ids": predictions,
                "comparison": comparison,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
