# Development and checks

Python style rules are in [code_style.md](code_style.md); contracts are in
[architecture.md](architecture.md) and [services.md](services.md#service-architecture).
Notebook and Git filter instructions are in [notebooks.md](notebooks.md).
Run commands from the repository root after installation as described in [README](../README.md).

## Library and build checks

```bash
.venv/bin/python -m unittest discover -s tests -v
uv lock --check
uv build
```

Use `-p` to select a test module, for example
`.venv/bin/python -m unittest discover -s tests -p test_gru.py -v`.
Build verification includes inspecting the wheel and importing its CLI from an
extracted wheel in an isolated directory, beyond checking the builder's exit
status. Do not commit `dist/` artifacts. Runtime tests do not establish GPU,
Pylance or full notebook execution results; report those checks separately.

Create temporary Python experiments and service environments with Python 3.12
under `/mnt/data/tmp/`, using the existing `UV_CACHE_DIR`. Small, short-lived
environments may use `/tmp`, but large packages, models and data must remain
on `/mnt/data`.

## Service checks

Service dependencies are separate from the library. For local checks, create
a separate Python 3.12 environment on `/mnt/data`, install the core from the
lockfile, then install `requirements.txt`. Do not override the UV cache.

The current `pyproject.toml` is the version source: web libraries come from the
`dev` group, while pandas and PyYAML come from the main dependencies. Container
requirements pin the corresponding exact versions from `uv.lock`. Update these
requirements together when project dependencies change.

```bash
uv venv --python 3.12 /mnt/data/tmp/feature-selector-services/.venv
UV_PROJECT_ENVIRONMENT=/mnt/data/tmp/feature-selector-services/.venv uv sync --locked --no-dev
uv pip install --python /mnt/data/tmp/feature-selector-services/.venv/bin/python -r requirements.txt
/mnt/data/tmp/feature-selector-services/.venv/bin/python -m unittest discover -s services/tests -v
docker compose config --quiet
```

Tests start local HTTP servers and child processes, so loopback access is
required. They use synthetic CSVs, do not depend on UZK and do not replace
verification of `docker compose up --build`.

## Add a step

Subclass `Step`, implement `transform` and optionally `fit`, then register the
constructor with `register_step("name", Constructor)` before execution. The Python
API also accepts `(name, step)` pairs in the pipeline list; instances are cloned.
When a step changes examples, it must update X, y and metadata consistently.
In saved `config.yaml`, a Python-defined step uses `name`, `class` and `params`;
nested estimators use `class` and `params`. Reloading imports classes by their
fully qualified names, so define them at the top level of an importable Python
module. Regular YAML configurations continue to use `type`.

`context` is optional and is not stored in the model. It provides `log_metrics`,
`log_metadata`, `log_event` and `log_message`, scoped to the step's namespace.
Components do not open log files or create TensorBoard writers themselves.

## Local DataSet1 helper

`scripts/dataset1.py` exists in the current working copy but is ignored by Git
and is not included in a repository clone. The following commands apply only
when this file has been supplied separately; they are not the standard entrypoint:

```bash
uv run python scripts/dataset1.py prepare
uv run python scripts/dataset1.py run configs/dataset1/gru.yaml
```

DataSet1 preparation removes laboratory/quality signals from features and preserves
numeric values. The script explicitly evaluates test after training and saves a
comparison with the mean of observed train targets. The DataSet1 notebook also
requires this helper.

## Acceptance checks

- Known windows/horizons; gaps, intervals and split-boundary history.
- Missing y does not shrink AE training; supervised models select observed targets.
- Scaler-before-alignment fits on original train rows, excluding validation/test from scaler.fit.
- One event per completed train batch and validation metric per applicable epoch.
- Batch and validation losses agree across history, JSONL and TensorBoard;
  weighted train epoch means agree across history and JSONL and are absent from TensorBoard.
- Interrupted fit retains readable events; scoped metadata cannot overwrite siblings.
- Frozen preprocessing/weights do not update; reload works without source export.
- Original-unit predictions and reconstruction references survive save/load.
- User-defined step works without changes to executor/config logic.
- Validation-disabled training keeps final weights and no selected epoch.
- Full unittest, short CPU Ridge/GRU/AE runs, AE/GRU notebook execution, lockfile
  consistency, wheel contents and imports from the built package.

## Report results

State the checks performed, their results and limitations in the change description.
Rerun appropriate checks after behavior changes.
