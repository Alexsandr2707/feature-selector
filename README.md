# feature_selector

Experiments with temporal features: Ridge, GRU, GRU ensembles and autoencoders.
A single ordered list of YAML steps defines processing: interval exclusions,
target filtering and alignment, scaling, feature selection, windows and the final model.

Use the project as a local Python library with a CLI and notebooks, or as a
five-service system with a GUI. The numerical core works independently of the
services. Their interface, application messages and logs are in English;
user-provided names and source data are preserved without translation.

Run all commands below from the repository root.

## Run the microservices

Docker Engine, Docker Compose and access to the Docker daemon are required.
Source data is not included in the repository: prepare `datasets/uzk/uzk.csv`
and `datasets/uzk/uzk_lab.csv` using the [format description](docs/services.md#data-preparation).
Then run:

```bash
docker compose up --build -d
docker compose ps
```

After startup, open:

- [GUI](http://127.0.0.1:8501).
- [Master API and interactive documentation](http://127.0.0.1:8000/docs).
- [Dependency status](http://127.0.0.1:8000/api/v1/status).

The interface supports four scenarios: collect a Dataset, train Ridge/GRU,
predict with a saved Run and compare validation results. Test evaluation requires
an explicit action. See the [scenario walkthrough](docs/services.md#four-scenario-walkthrough)
for the steps and GUI buttons. For a quick API demonstration:

```bash
docker compose exec master python -m services.demo --url http://127.0.0.1:8000/api/v1
```

Data paths, environment settings, container management, saved objects and v1
limitations are described in the [service guide](docs/services.md).

## Local library and CLI

Python >= 3.12 and uv are required. PyTorch, matplotlib and TensorBoard are
installed with the project:

```bash
uv sync
```

CSV files contain timestamps in the first column and numeric values. Features
and targets may be stored in separate files. YAML data paths are relative to
the configuration file. After preparing data for the selected configuration,
run, for example:

```bash
uv run feature-selector run configs/uzk/gru.yaml
```

Configurations live in `configs/`; notebooks use a matching layout under
`notebooks/`. Select the `.venv` kernel for notebooks. Git filter setup is covered
in the [notebook guide](docs/notebooks.md). Python API examples, the detailed
data format, model parameters and artifacts are in the [experiment guide](docs/experiments.md).

## Documentation

| Document | Contents |
| --- | --- |
| [Experiments](docs/experiments.md) | Data, YAML, CLI/Python API, models, seed series, artifacts and TensorBoard |
| [Microservices](docs/services.md) | CSV preparation, deployment, GUI scenarios, API, object and job lifecycle |
| [Architecture](docs/architecture.md) | Module boundaries, data and step contracts, temporal processing and evaluation |
| [Development](docs/development.md) | Extension, environments, tests, builds and acceptance checks |
| [Code style](docs/code_style.md) | Python, type annotation and docstring rules |
| [Notebooks](docs/notebooks.md) | Git filter setup that preserves local outputs |

## Checks

```bash
.venv/bin/python -m unittest discover -s tests -v
uv lock --check
uv build
```

The separate environment and service test commands are described in the
[development guide](docs/development.md#service-checks).
