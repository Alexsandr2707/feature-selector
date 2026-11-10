# Experiments and configuration

Guide to the local library, CLI and notebooks. The service interface is described
in [services.md](services.md), internal contracts in [architecture.md](architecture.md),
and installation in [README](../README.md). Run all commands from the repository root.

## Configurations and notebooks

Files are grouped by dataset and experiment family. Configurations live under
`configs/`, with the same directory layout under `notebooks/`. `pretrain` denotes
autoencoder training and export; other names identify prediction scenarios.
The general configuration example is `configs/example.yaml`.

Change experiment parameters in YAML files and notebooks. The examples below
illustrate the API and individual steps; they do not prescribe the contents of
files in `configs/`. Choose an encoder export path when saving it and pass that
path to `frozen_pipeline`.

Run notebooks sequentially with the `.venv` kernel after `uv sync`. The initial
cell finds the project root in the current directory or its parents and makes
it the working directory, keeping shared `datasets/`, `runs/` and `tmp/` paths.
For the Feature autoencoder, use the shared notebooks under `uzk/autoencoder/`
and select the appropriate configuration and export path.

Before committing notebooks, configure the [Git filter](notebooks.md): local
plots and outputs are preserved, while Git receives a stripped copy.

## Data

CSV files have a header, timestamps in the first column and numeric values.
Features and targets may be in separate files; the target file is optional.
The loader sorts timestamps, checks uniqueness and restores a regular grid.
Missing values remain NaN: rows are not compacted or automatically interpolated.

```yaml
data:
  features_path: ../datasets/uzk/uzk.csv
  target_path: ../datasets/uzk/uzk_lab.csv
  target_column: UZK.Q.81AY00108.FINALPOINT
  freq: 1h
```

Paths are relative to the YAML file. Autoencoder training only requires
`features_path` and `freq`. Define exclusions as a separate step, not a `data` field.

## Pipeline configuration

```yaml
data:
  features_path: ../data/features.csv
  target_path: ../data/target.csv
  target_column: target
  freq: 1h

split:
  train_fraction: 0.6
  validation_fraction: 0.2

pipeline:
  - name: exclusions
    type: exclude_intervals
    params:
      intervals:
        - ['2022-10-01', '2022-10-16 23:00:00']
  - name: scaler
    type: standard
  - name: future_target
    type: align_target
    params:
      horizon: 1
  - name: selector
    type: static
    params:
      columns: [sensor_a, sensor_b]
  - name: windows
    type: windows
    params:
      lookback: 48
  - name: target_scaler
    type: target_scale
    params:
      scaler:
        type: standard
  - name: model
    type: gru
    params:
      hidden_size: 16
      epochs: 200
      batch_size: 128
      learning_rate: 0.01
      device: cpu

run:
  output_dir: ../runs
  experiment_name: uzk_gru
```

Built-in types: `exclude_intervals`, `filter`, `align_target`, `windows`, `temporal_pool`,
`standard`, `minmax`, `robust`, `static`, `elasticnet`, `pls`, `target_scale`, `ridge`,
`gru`, `gru_ensemble`, `conv_autoencoder`, `gru_autoencoder`, `s4_autoencoder`,
`frozen_pipeline`, `passthrough`.

Each step receives `PipelineData`: X, optional y, feature names, and metadata for
time, splits and exclusions. Steps run sequentially through the same
`fit(train, validation, context=...)` and `transform(data, context=...)` methods.
The final step also provides `predict`.

- `exclude_intervals` masks inclusive intervals without compacting the grid;
  after window creation, it accounts for overlap with window history.
- `align_target` shifts y to the input point and updates target time and split.
- `windows` creates `(samples, time, features)`. Missing y does not remove a window;
  the algorithm consuming targets determines their usability.
- A scaler before `align_target` fits on the original train rows. The step does
  not know the parameters of later steps.
- Neural models fit on train; validation monitors training and selects weights.
  In `gru_ensemble`, Ridge fits on the base GRUs' validation predictions.
  Test is not predicted or evaluated automatically.
- After alignment, split membership follows target time. Window history may
  reach into the preceding split; no additional gap is applied.

Instead of fractions, specify `validation_start` and `test_start`. Validation
and test fractions may be zero. Without validation, neural models run all epochs
and retain final weights; `validation_loss` and `best_epoch_` are None.

For GRU, `checkpoint_start_epoch` sets the first epoch eligible for best-weight
selection and `patience` counting, using one-based numbering. Earlier epochs
train and log normally, but their validation loss does not select checkpoints.
Selection starts at the first epoch by default. If training ends before the
specified epoch, final weights are retained and `best_epoch_` is None.
Autoencoder checkpoint selection starts at the first epoch.

GRU and autoencoders support optional `max_grad_norm` in `params`. For example,
`max_grad_norm: 1.0` limits the combined L2 gradient norm across all parameters
after `backward()` and before each weight update. The value must be finite and
positive; omission or `null` disables clipping.

The pipeline author controls causality and processing order. Configuration
validation checks structure; algorithms check necessary conditions when used.
The factory does not perform full model-parameter validation in advance.
The target filter uses future values: its statistics may depend on validation/test,
even though those examples are not passed to the steps' `fit` methods.

## Target filtering

The `filter` step applies only to `y`:

```yaml
- name: filter
  type: filter
  params:
    filter_freq: 1W
    max_diff: 30
```

It computes period means and the absolute differences between consecutive means,
resamples the differences to the original frequency using `nearest`, and replaces
targets with NaN when the difference exceeds `max_diff`. Frequency comes from the
temporal context. It uses future values and recomputes statistics over the entire
supplied time grid, including different splits.

The threshold uses the target units at the step's input. X and rows are retained.
NaN targets are excluded from training and metrics, but predictions for those
rows remain available. Unknown y makes the step a no-op. Newly masked counts by
split are written to `metadata.json` and JSONL as `masked_targets` under
`pipeline/filter`.

## Reproducibility

Before loading data and building the pipeline, `run.seed` initializes the global
Python `random`, NumPy and PyTorch CPU/CUDA generators. It defaults to 0 and
accepts integers from 0 through `2**32 - 1`.

```yaml
run:
  seed: 42
  output_dir: ../runs
  experiment_name: uzk_gru
```

Model `random_state` parameters are separate: they control weight initialization
and batch shuffling. Each ensemble GRU receives `random_state + index`.
`run.seed` does not replace these parameters; it is stored in run metadata.

## Experiment directories

`run.experiment_name` names the experiment. All run artifacts, including
`run.log`, `events.jsonl`, TensorBoard, metrics and the model, are stored together:

```text
<run.output_dir>/<run.experiment_name>/2026-11-03_14-05-09/
```

Directory names use the machine's local time to the second. Collisions receive
suffixes `_2`, `_3` and so on; existing results are not overwritten. An experiment
name must be a nonempty string without path separators or a null character;
`.` and `..` are forbidden. The default name is `default`.
Relative `output_dir` paths are resolved against the YAML file.

## Training and prediction

```bash
uv run feature-selector run configs/uzk/gru.yaml
```

```python
from feature_selector import load_config, run_experiment
from feature_selector.artifacts import load_run
from feature_selector.data import load_data
from feature_selector.evaluation import report_metrics

config = load_config("configs/uzk/gru.yaml")
run = run_experiment(config)
print(run.metrics)

saved = load_run(run.artifact_dir)
data = load_data(**config.data)
test_report = saved.predict(data, split="test")
print(report_metrics(test_report))
```

Prediction accepts unknown targets and returns `y_true=NaN` for them. Metrics
use only observed targets. For entirely unlabeled prediction, pass the result
of `load_features(...)`. NaN/inf predictions are errors, including at points
with unknown targets.

Metrics include `mae`, `rmse`, `mape` (percent), `r2`, `pearson` and
`pearson_pvalue` (two-sided p-value). If any observed target is zero, MAPE is NaN
with an explanation in `notes`; other metrics use all observations. Undefined
R², correlation and p-value also include explanations.

`feature_selector.evaluation.summarize_regression_metrics(scores)` accepts a
nonempty list of `regression_metrics` results from runs on the same examples.
For each metric, it returns `mean`, sample standard deviation `std` (`ddof=1`),
finite-value count `n_valid` and total run count `n_total`. Undefined values are
excluded from the summary. With no available values, the mean is NaN; with fewer
than two values, standard deviation is NaN. The p-value summary describes variation
between runs, not a combined significance test.

### Seed series

```python
from feature_selector.experiment import run_seed_experiments

series = run_seed_experiments("configs/uzk/gru.yaml")  # Seeds 0, ..., 9.
print(series.metrics_per_seed)
print(series.metrics_summary)
print(series.artifact_dir)
```

Pass `seeds=[7, 13, 42]` for a custom set. Seeds must be unique integers within
`[0, 2**32 - 1]`; execution order is preserved. The function supports single-target
regression with a final algorithm exposing `random_state`, including GRU and GRU
ensembles. For an ensemble, the seed starts the members' seed sequence.
Deterministic `WindowRidge` does not expose this parameter. A test split with
observed test targets is required.

The first run fits the entire pipeline, then saves its prefix before the final
model to `preprocessing.pkl`. Later runs use that prefix through `frozen_pipeline`
without refitting, including the target scaler and any intermediate autoencoders.
Only the final algorithm's `random_state` changes; `run.seed`, splits and fitted
preprocessing stay fixed. The variation therefore describes the final model under
one fitted data representation. Changing input files or the set of test examples
between runs is an error.

The series is saved under `<run.output_dir>/<run.experiment_name>/<series-time>/`.
It contains:

- `metrics_per_seed.csv`: seed, all test metrics, observation count `n`, JSON
  `notes` and the individual run's `artifact_dir` path.
- `metrics_summary.csv`: metric, `mean`, `std`, `n_valid`, `n_total`.
- `series.json`: requested and completed seeds, series status, shared `run_seed`
  and the configured `std_ddof=1`.
- `runs/seed_<seed>/<run-time>/`: regular artifacts for each run, with test metrics
  added to `metrics.json` and `run.pkl`, plus `predictions_test.csv`.

The series explicitly evaluates test after each algorithm is trained; test is
not used to select a seed or weights. Individual run metadata stores `model_seed`
separately from the shared `seed`. Saved `run.pkl` embeds the fitted prefix and
works without the original `preprocessing.pkl`.

Tables update after each completed seed. On failure, the series stops, retains
completed results and writes `failure.json`; `series.json` status becomes `failed`.
For a partial series, `n_total` refers to completed runs, while `series.json`
retains the full requested list. The function result contains tables and the
artifact path; load models individually with `load_run`.

Truth and prediction plots omit missing points separately for each curve and
connect remaining points across gaps. Report data is not filled. The truth curve
is drawn over predictions.

`run.pipeline["name"]` returns a step; its algorithm is available as `.estimator`
on an adapter. `run.model` is the final algorithm. For an intermediate representation:

```python
from feature_selector.pipeline.data import prepare_data

prepared = prepare_data(data, run.boundaries, config.data["freq"])
windows = run.pipeline.transform(prepared, through="windows")
```

## GRU ensemble

`gru_ensemble` trains several GRUs with the same parameters and different seeds.
Each GRU fits on train and selects weights using validation. Ridge then fits on
their predictions over that same entire validation split; each input column to
Ridge represents one GRU. Test does not participate in training.

Example final pipeline step:

```yaml
- name: model
  type: gru_ensemble
  params:
    n_estimators: 10
    alpha: 0.2
    fit_intercept: true
    random_state: 0
    gru_params:
      hidden_size: 16
      epochs: 200
      patience: 200
      batch_size: 128
      device: cpu
```

Set GRU parameters in `gru_params`, except `random_state`: the ensemble increments
it for each subsequent model. With `patience >= epochs`, all epochs run,
validation loss continues to be logged and the best weights are restored.
Validation is required to fit Ridge. Validation metrics describe data used for
GRU weight selection and Ridge fitting; independently evaluate the final ensemble
on test with an explicit call.

```bash
uv run feature-selector run configs/uzk/gru_ensemble.yaml
```

Fitted models are available in `run.model.estimators_`, and Ridge in
`run.model.meta_model_`. Ensemble `run.model.history_` contains a separate record
for each GRU: `estimator`, `random_state`, `best_epoch`, `history`. To plot one
model, use its `history_` and `best_epoch_`. TensorBoard losses use separate paths
`pipeline/model/gru_0`, `pipeline/model/gru_1`, and so on; Ridge coefficients are
saved in metadata under `pipeline/model/ridge`.

`notebooks/uzk/gru_ensemble.ipynb` shows a summary of all GRUs, the selected model's
history, Ridge coefficients and explicit test evaluation after reloading the
saved pipeline.

To train on autoencoder features, pass the exported prefix to `frozen_pipeline`.
Scaling, windows and the encoder load from the export without refitting. The
target scaler and final model fit in the new run.

## Autoencoders and saved encoders

Autoencoders accept `(samples, time, features)` windows and learn to reconstruct
X regardless of whether y is present. `mode: reconstruct` returns reconstructed
windows; `mode: encode` returns latent sequences.

`conv_autoencoder` supports two compression modes. `compression: features_time`
compresses features and time: temporal convolutions use `kernel_size=3`, and the
decoder restores window length by interpolation. `compression: features` only
compresses features: fully connected layers process each timestamp independently
with shared weights. Hidden layers use GELU; outputs have no activation.

`gru_autoencoder` uses `GRU → Linear` in the encoder and
`Linear → GELU → GRU → Linear` in the decoder. Window length is preserved.
The decoder receives only the latent sequence, without the encoder's hidden
state. `hidden_channels` sets both GRUs' width, while `latent_channels` is the
number of compressed features. This model has no `compression` parameter.

`s4_autoencoder` uses unidirectional S4 (`mode="diag"`) without external residual
connections or LayerNorm. Encoder: `S4(F) → Linear(F, C)`; decoder:
`Linear(C, F) → GELU → S4(F) → Linear(F, F)`. Both blocks operate at the input
feature width `F`. `latent_channels` sets width `C`; `state_size` is the positive,
even S4 state size. Window length is preserved.

S4 `dropout` sets the channel dropout probability: a finite number in `[0, 1)`.
Masks are shared across time within a window (`tie_dropout=True`). Dropout is
disabled during prediction.

The optional `temporal_pool` step averages adjacent points within each window.
`factor` is a positive integer that must divide window length without remainder;
`factor: 1` preserves length. Examples, targets, features and temporal history
boundaries are retained. Pooling after a frozen encoder changes temporal
resolution without refitting the encoder; the downstream model must be fitted again.

During PyTorch model training, the last incomplete batch is dropped after
shuffling windows in each epoch. At least one full train batch is required.
Validation and prediction use all examples; mean train loss accounts only for
windows actually processed.

Example fitted-prefix export:

```python
run.pipeline.export_transform(
    "runs/encoder.pkl", through="autoencoder", mode="encode"
)
```

`through` must name a step in the fitted pipeline. The export includes all steps
up to and including it; a mode change applies to a copy of the model.
To use the export, add:

```yaml
- name: encoder
  type: frozen_pipeline
  params:
    path: ../runs/encoder.pkl
```

The frozen step does not fit anything. A complete saved run embeds its state and
works after deleting the original export. Feature names/order and frequency must
match prefix training. Pretraining boundaries are retained as provenance; the
experiment author determines whether those boundaries are appropriate.
Load pickle files only from trusted sources.

Autoencoder loss uses full windows in training units. Reconstruction reports use
window endpoints in original units: `overall` and `per_feature`. In a chain of
autoencoders, references come from the original observations, so metrics reflect
losses across the entire chain.

## History, events and TensorBoard

```bash
uv run tensorboard --logdir runs
```

View training plots under **Scalars**, filtered by `loss|learning_rate`.
All models' final metrics are saved to `metrics.json`, with parameters in
`config.yaml`. TensorBoard final train/validation metrics appear under **Scalars**
with the `results/` filter, and the results table under **Text → results/summary**.
Autoencoders publish `overall` metrics in original units; per-feature results
remain in `metrics.json`. Undefined R² and Pearson values appear as `—` in the
table and are not logged as scalars. Test is not computed or logged automatically.
Final results appear after the run completes.

Each completed train batch is recorded immediately: loss before the weight
update, actual batch size, epoch, batch, global step and learning rate.
TensorBoard logs train loss and learning rate per batch, and validation loss per
epoch when validation is enabled. Mean train epoch loss remains in `history_`
and `events.jsonl`, and is absent from TensorBoard and `run.log`. It is a weighted
mean of batch losses, computed without an extra train pass. Weights are selected
by validation loss. History is available as `run.model.history_`, including after
reload. For ensemble epoch or batch plots, select an individual member's history:
`run.model.estimators_[member_index].history_`.

```python
import pandas as pd

model = run.model  # For an ensemble, select run.model.estimators_[member_index].
batches = pd.DataFrame([
    {"epoch": epoch["epoch"], **batch}
    for epoch in model.history_ for batch in epoch["batches"]
])
batches.plot(x="step", y="train_loss")
```

Run outputs: `config.yaml`, `metadata.json`, `events.jsonl`, `run.log`,
`tensorboard/`, `history.json`, `features.json`, `metrics.json`, `predictions_*.csv`,
`run.pkl`. On failure, `failure.json` is written and completed batch events are retained.

JSONL records contain `time`, `path`, `event` and `data`. Paths identify steps and
nested pipelines. The executor records start/end, sizes and duration; the runner
records training; steps record their own diagnostics. Run metadata includes input
paths/hashes, versions, boundaries and a summary of each step. Large arrays are
not written to logs.
