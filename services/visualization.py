"""Streamlit scenario client; all data operations go through Web Master."""

import io
import json
import logging
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

from services.common import Remote, config_for

logger = logging.getLogger(__name__)

DEFAULT_UTC_OFFSET = 180
SETTINGS = config_for("visualization")
MASTER = Remote(SETTINGS["master_url"], SETTINGS["timeout"])


def objects(kind: str) -> list[dict[str, Any]]:
    """List published objects through the gateway.

    Parameters:
        kind (str): Catalog collection.

    Returns:
        list[dict[str, Any]]: Published objects.
    """
    return MASTER.json("GET", f"/objects/{kind}")


def timezone_label(offset: int) -> str:
    """Format a fixed UTC offset for the time zone selector.

    Parameters:
        offset (int): Signed offset in minutes.

    Returns:
        str: Explicit UTC offset, such as UTC+03:00.
    """
    hours, minutes = divmod(abs(offset), 60)
    sign = "+" if offset >= 0 else "-"
    return f"UTC{sign}{hours:02d}:{minutes:02d}"


def display_name(value: dict[str, Any]) -> str:
    """Build an object or job label using the session's selected UTC offset.

    Parameters:
        value (dict[str, Any]): Catalog object or job, with optional resolved name.

    Returns:
        str: Readable label; an ID is used only when no descriptive fields exist.
    """
    action = {"collect": "Collection", "train": "Training", "predict": "Prediction"}.get(
        value.get("action", "")
    )
    parts = [part for part in (action, value.get("name")) if part]
    if value.get("created_at"):
        created = datetime.fromisoformat(value["created_at"])
        # Interpret legacy naive timestamps as UTC rather than the server's local time.
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        offset = st.session_state.get("display-utc-offset", DEFAULT_UTC_OFFSET)
        created = created.astimezone(timezone(timedelta(minutes=offset)))
        parts.append(f"{created:%Y-%m-%d %H:%M:%S} {timezone_label(offset)}")
    if value.get("status"):
        parts.append(value["status"])
    return " · ".join(parts) or value["id"]


def choose(label: str, values: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    """Select the newest object initially and when a newer object appears.

    Parameters:
        label (str): Widget label.
        values (list[dict[str, Any]]): Objects with id, optional name and created_at.
            Undated catalogs retain their first item as the initial selection.
        key (str): Unique widget key, also used to track the newest object in session state.

    Returns:
        dict[str, Any] | None: Selected object, or None for an empty catalog. Manual
            selection persists until the newest object changes or the selection is deleted.
    """
    latest_key = f"latest-object:{key}"
    if not values:
        st.session_state[latest_key] = None
        st.session_state.pop(key, None)
        st.info("No objects available yet. Lists refresh automatically.")
        return None

    mapping = {value["id"]: value for value in values}
    latest = max(values, key=lambda value: value.get("created_at", ""))["id"]
    if st.session_state.get(latest_key) != latest or st.session_state.get(key) not in mapping:
        st.session_state[key] = latest
    st.session_state[latest_key] = latest

    names = {item: display_name(value) for item, value in mapping.items()}
    selected = st.selectbox(
        label,
        list(mapping),
        key=key,
        format_func=names.__getitem__,
    )
    return mapping[selected]


def monitor(collection: bool = False) -> None:
    """Refresh persisted job status using GET requests only.

    Parameters:
        collection (bool): Whether to inspect Master collection jobs instead of ML jobs.

    Returns:
        None: Displays state and recent numerical events.
    """
    try:
        route = "/collections" if collection else "/jobs"
        attempts = MASTER.json("GET", route)
        if not attempts:
            return

        names = {}
        for kind in {attempt["result_kind"] for attempt in attempts}:
            drafts = MASTER.json("GET", f"/drafts/{kind}")
            names.update({value["id"]: value["name"] for value in drafts + objects(kind)})
        attempts = [dict(attempt, name=names.get(attempt["id"])) for attempt in attempts]
        selected = choose("Job", list(reversed(attempts)), f"job-{collection}")
        if selected:
            state = MASTER.json("GET", f"{route}/{selected['id']}")
            st.write(display_name(dict(state, name=selected.get("name"))))
            if state.get("error"):
                st.warning(f"Job error: {state['error']}")
            with st.expander("Job details / technical ID"):
                st.json(state)
            if not collection:
                lines = MASTER.json("GET", f"{route}/{selected['id']}/events")["lines"]
                with st.expander(f"Recent events / log · {display_name(selected)}"):
                    st.code("\n".join(lines), language="json")
    except httpx.HTTPStatusError as exc:
        st.error(exc.response.text)
    except Exception as exc:
        logger.exception("Could not refresh job status")
        st.error(str(exc))


@st.fragment(run_every=SETTINGS["refresh_seconds"])
def data_page() -> None:
    """Display source availability, collect requests and saved data previews.

    Returns:
        None: Renders the data page.
    """
    sources = MASTER.json("GET", "/sources")
    st.json(sources)
    available = [source for source in sources if source.get("available")]

    with st.form("collect"):
        source = choose("Source", available, "source")
        name = st.text_input("Dataset name", "UZK")
        submitted = st.form_submit_button("Collect whole source", disabled=source is None)
    if submitted and source:
        job = MASTER.json("POST", "/collections", json={"source_id": source["id"], "name": name})
        st.success(f"Collection started: {display_name(dict(job, name=name))}")

    monitor(collection=True)
    dataset = choose("Saved dataset", objects("datasets"), "dataset-preview")
    if dataset:
        st.json(dataset)
        st.dataframe(MASTER.json("GET", f"/datasets/{dataset['id']}/preview")["rows"])


@st.fragment(run_every=SETTINGS["refresh_seconds"])
def training_page() -> None:
    """Render template fields and create/train actions without side effects on reruns.

    Returns:
        None: Renders the training page.
    """
    dataset = choose("Dataset", objects("datasets"), "train-dataset")
    templates = MASTER.json("GET", "/templates")
    template = choose("Template", templates, "template")

    if dataset and template:
        with st.form(f"train-{template['id']}"):
            name = st.text_input("Model name", template["name"])
            parameters = {}
            for field, spec in template["parameters"].items():
                integer = spec["type"] == "int"
                convert = int if integer else float
                parameters[field] = st.number_input(
                    field,
                    min_value=convert(spec["min"]),
                    max_value=convert(spec["max"]),
                    value=convert(spec["default"]),
                    step=1 if integer else 0.000001,
                    format="%d" if integer else "%.6f",
                )
            submitted = st.form_submit_button("Create model and train")

        if submitted:
            model = MASTER.json(
                "POST",
                "/models",
                json={
                    "name": name,
                    "dataset_id": dataset["id"],
                    "template_id": template["id"],
                    "parameters": parameters,
                },
            )
            job = MASTER.json("POST", f"/models/{model['id']}/train")
            st.success(f"Training started: {display_name(dict(job, name=name))}")

    model = choose("Existing specification", objects("models"), "existing-model")
    if model:
        with st.expander("Summary"):
            st.json(MASTER.json("GET", f"/models/{model['id']}"))
        if st.button("Train existing specification again"):
            job = MASTER.json("POST", f"/models/{model['id']}/train")
            st.success(f"Training started: {display_name(dict(job, name=model['name']))}")

    monitor()


def artifact(kind: str, key: str, filename: str) -> bytes:
    """Download a published file through Master.

    Parameters:
        kind (str): Collection name.
        key (str): Object ID.
        filename (str): Artifact name.

    Returns:
        bytes: File content.
    """
    return MASTER.request("GET", f"/objects/{kind}/{key}/files/{filename}").content


@st.fragment(run_every=SETTINGS["refresh_seconds"])
def prediction_page() -> None:
    """Apply a saved run explicitly and display original-unit predictions.

    Returns:
        None: Renders the prediction page.
    """
    run = choose("Trained run", objects("runs"), "predict-run")
    with st.form("predict"):
        split = st.selectbox("Split", ["test", "validation", "train"])
        st.caption("Test is evaluated only by this action, never during training.")
        submitted = st.form_submit_button("Get predictions and metrics", disabled=run is None)
    if submitted and run:
        job = MASTER.json("POST", f"/runs/{run['id']}/predict", json={"split": split})
        st.success(f"Prediction started: {display_name(dict(job, name=run['name']))}")

    monitor()
    prediction = choose("Saved prediction", objects("predictions"), "prediction")
    if prediction:
        content = artifact("predictions", prediction["id"], "predictions.csv")
        frame = pd.read_csv(io.BytesIO(content))
        st.json(prediction["metadata"])

        series = frame.melt(id_vars=["target_time"], value_vars=["y_true", "y_pred"])
        series = series.dropna(subset=["value"]).sort_values("target_time")
        st.plotly_chart(
            px.line(
                series,
                x="target_time",
                y="value",
                color="variable",
                labels={"value": "Original units"},
            ),
            width="stretch",
        )
        st.download_button("Download CSV", content, "predictions.csv", "text/csv")


@st.fragment(run_every=SETTINGS["refresh_seconds"])
def comparison_page() -> None:
    """Compare validation scores and epoch histories for matching evaluation protocols.

    Returns:
        None: Renders the comparison page.
    """
    runs = {run["id"]: run for run in objects("runs")}
    names = {key: display_name(run) for key, run in runs.items()}
    selected = st.multiselect("Runs", list(runs), format_func=names.__getitem__)
    if not selected:
        return

    comparison = MASTER.json("GET", "/comparison", params=[("run_ids", key) for key in selected])
    st.dataframe(comparison["rows"])
    st.caption(comparison["note"])
    if comparison["sample_set_warning"]:
        st.warning(
            "Validation metrics use different timestamps; "
            "take this into account when comparing runs."
        )

    curves = []
    for key in selected:
        histories = json.loads(artifact("runs", key, "history.json"))
        for step, history in histories.items():
            for epoch in history["history"]:
                for metric in ("train_loss", "validation_loss"):
                    if epoch.get(metric) is not None:
                        curves.append(
                            {
                                "epoch": epoch["epoch"],
                                "loss": epoch[metric],
                                "curve": f"{display_name(runs[key])} / {step} / {metric}",
                            }
                        )

    if curves:
        st.plotly_chart(
            px.line(
                pd.DataFrame(curves),
                x="epoch",
                y="loss",
                color="curve",
                labels={"loss": "MSE in training units"},
            ),
            width="stretch",
        )
    else:
        st.info("The selected models have no epoch history.")


def main() -> None:
    """Render a scenario page with visible operational errors.

    Returns:
        None: Builds the Streamlit interface.
    """
    st.set_page_config(page_title="Feature selector", layout="wide")
    st.title("Temporal feature experiments")
    st.sidebar.button("Refresh lists")
    offsets = list(range(-12 * 60, 14 * 60 + 1, 15))
    st.sidebar.selectbox(
        "Time zone",
        offsets,
        index=offsets.index(DEFAULT_UTC_OFFSET),
        format_func=timezone_label,
        key="display-utc-offset",
        help="Fixed UTC offset for run and job labels. Moscow: UTC+03:00. "
        "Daylight saving time is not applied automatically.",
    )

    pages = {
        "Data": data_page,
        "Training": training_page,
        "Prediction": prediction_page,
        "Comparison": comparison_page,
    }
    page = st.sidebar.radio("Scenario", list(pages))

    try:
        with st.sidebar.expander("Service status"):
            st.json(MASTER.json("GET", "/status"))
        pages[page]()
    except httpx.HTTPStatusError as exc:
        st.error(exc.response.text)
    except Exception as exc:
        logger.exception("Could not render scenario page")
        st.error(str(exc))


if __name__ == "__main__":
    main()
