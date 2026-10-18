# feature_selector

Эксперименты с временными признаками: Ridge, GRU и свёрточный автоэнкодер.
Порядок обработки задаётся единым списком шагов в YAML. Окна, исключения интервалов,
сдвиг цели и масштабирование входят в этот список.

Карта архитектуры, контракты и согласованный план миграции:
[docs/architecture.md](docs/architecture.md).

## Установка

Python >= 3.12, управление окружением через uv:

```bash
uv sync
```

PyTorch, matplotlib и TensorBoard устанавливаются вместе с проектом.

## Данные

CSV содержит время в первом столбце, заголовок и числовые значения. Признаки и цели
могут находиться в разных файлах; целевой файл необязателен. Загрузчик сортирует
временные метки, проверяет их уникальность и восстанавливает регулярную сетку.
Пропуски остаются NaN: строки не уплотняются и автоматически не интерполируются.

```yaml
data:
  features_path: ../datasets/uzk/uzk.csv
  target_path: ../datasets/uzk/uzk_lab.csv
  target_column: UZK.Q.81AY00108.FINALPOINT
  freq: 1h
```

Пути относительны YAML-файлу. Для обучения автоэнкодера достаточно `features_path`
и `freq`. Исключения задаются отдельным шагом, а не полем `data`.

## Конфигурация пайплайна

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
```

Встроенные типы: `exclude_intervals`, `align_target`, `windows`, `standard`,
`minmax`, `robust`, `static`, `elasticnet`, `pls`, `target_scale`, `ridge`, `gru`,
`conv_autoencoder`, `frozen_pipeline`, `passthrough`.

Каждый шаг получает `PipelineData`: X, необязательный y, имена признаков и метаданные
времени, выборки и исключений. Шаги вызываются последовательно через одинаковые
`fit(train, validation, context=...)` и `transform(data, context=...)`. Финальный шаг
также предоставляет `predict`.

- `exclude_intervals` маскирует включительные интервалы без сжатия сетки; после
  создания окон учитывает пересечение интервала с историей окна.
- `align_target` сдвигает y к входной точке, обновляет время цели и выборку.
- `windows` формирует `(samples, time, features)`. Отсутствие y не удаляет окно;
  пригодность целей определяет потребляющий их алгоритм.
- Scaler до `align_target` обучается на исходных строках train. Шаг не знает
  параметры последующих шагов.
- Обучение использует train, validation служит для контроля и выбора весов.
  Test автоматически не предсказывается и не оценивается.
- Выборка после сдвига определяется временем цели. История окна может заходить
  в предыдущую выборку; дополнительного зазора нет.

Вместо долей можно задать `validation_start` и `test_start`. Доли validation/test
могут быть нулевыми. Без validation нейронные модели выполняют все эпохи и сохраняют
финальные веса; `validation_loss` и `best_epoch_` равны None.

У GRU параметр `checkpoint_start_epoch` задаёт первую эпоху, участвующую в выборе
лучших весов и отсчёте `patience` (нумерация от 1). Ранние эпохи обучаются и
логируются полностью, но их validation loss не влияет на сохранение checkpoint.
По умолчанию выбор начинается с первой эпохи, в UZK-конфигах GRU — с десятой.
Если обучение завершилось раньше указанной эпохи, сохраняются финальные веса,
`best_epoch_` равен None. Автоэнкодер сохраняет прежнее поведение.

GRU и оба варианта автоэнкодера поддерживают опциональный `max_grad_norm` в `params`:
например, `max_grad_norm: 1.0` ограничивает общую L2-норму градиентов всех параметров
после `backward()` перед каждым обновлением весов. Значение должно быть конечным
и положительным; отсутствие параметра или `null` отключает клиппинг.

Причинность и порядок обработки определяет автор пайплайна. Конфигурация проверяет
структуру, алгоритмы — необходимые условия в момент использования. Фабрика не
вызывает предварительную полную проверку параметров моделей.

Старые поля `task`, `features`, `window`, `window_features`, `model` и
`target_scaler` вне списка pipeline удалены. Старые pickle-файлы нужно заменить
новым обучением и экспортом.

## Запуск и предсказание

```bash
uv run feature-selector run configs/uzk_gru.yaml
```

```python
from feature_selector import load_config, run_experiment
from feature_selector.artifacts import load_run
from feature_selector.data import load_data
from feature_selector.evaluation import report_metrics

config = load_config("configs/uzk_gru.yaml")
run = run_experiment(config)
print(run.metrics)

saved = load_run(run.artifact_dir)
data = load_data(**config.data)
test_report = saved.predict(data, split="test")
print(report_metrics(test_report))
```

Прогнозирование допускает неизвестные цели и возвращает для них `y_true=NaN`.
Метрики рассчитываются только по наблюдавшимся целям. Для полностью неразмеченного
прогноза можно передать результат `load_features(...)`.
Прогнозы NaN/inf считаются ошибкой, включая точки с неизвестной целью.

`run.pipeline["name"]` возвращает шаг; у адаптера алгоритм доступен как `.estimator`.
`run.model` — конечный алгоритм. Для промежуточного представления:

```python
from feature_selector.pipeline.data import prepare_data

prepared = prepare_data(data, run.boundaries, config.data["freq"])
windows = run.pipeline.transform(prepared, through="windows")
```

## Автоэнкодер и сохранённый энкодер перед GRU

`configs/uzk_autoencoder.yaml` задаёт:

```text
exclude_intervals → standard → windows → conv_autoencoder(mode=reconstruct)
```

По умолчанию `compression: features_time`: энкодер сжимает признаки и временную ось.
Декодер восстанавливает исходную длину интерполяцией и свёртками.
`compression: features` сжимает только признаки: обе свёртки энкодера и декодера
имеют `kernel_size=1` и обрабатывают каждый момент времени независимо, с общими
весами. Между свёртками используется GELU, выходы энкодера и декодера без активации.
Форма `(samples, time, features)` преобразуется в `(samples, time, latent_channels)`;
интерполяция не используется, временные зависимости может изучать последующая GRU.
Оба варианта остаются одним типом шага `conv_autoencoder`.

`mode: reconstruct` возвращает восстановленные окна,
`mode: encode` — латентные последовательности. Оба режима обучаются восстанавливать
X, включая окна без внешнего таргета.

```python
run = run_experiment("configs/uzk_autoencoder.yaml")
run.pipeline.export_transform(
    "runs/uzk_encoder.pkl", through="autoencoder", mode="encode"
)
```

Для варианта без сжатия времени используются отдельные конфиги и файл экспорта:

```python
run = run_experiment("configs/uzk_feature_autoencoder.yaml")
run.pipeline.export_transform(
    "runs/uzk_feature_encoder.pkl", through="autoencoder", mode="encode"
)
gru_run = run_experiment("configs/uzk_feature_autoencoder_gru.yaml")
```

В `Test_UZK_Autoencoder.ipynb` можно выбрать этот конфиг и путь экспорта,
а в `Test_UZK_Autoencoder_GRU.ipynb` — соответствующий конфиг GRU.

Экспорт содержит обученные scaler, окна, исключения и энкодер. Исходная модель
продолжает работать в своём режиме. `configs/uzk_autoencoder_gru.yaml` использует:

```text
align_target → frozen_pipeline(path=../runs/uzk_encoder.pkl) → target_scale → gru
```

Замороженный шаг ничего не обучает. Полный сохранённый запуск включает его состояние
и работает после удаления исходного экспорта. Имена/порядок признаков и частота
должны совпадать; границы обучения сохранены для проверки автором эксперимента.
Загружайте pickle только из доверенных источников.

Автоэнкодер считает loss по полному окну в единицах обучения. Отчёты реконструкции
и метрики используют последние точки окон в исходных единицах: `overall` и
`per_feature`. Пул общих ошибок зависит от масштаба исходных признаков.
В цепочке нескольких автоэнкодеров эталон берётся из исходных наблюдений,
а не из декодированного латентного представления: учитываются потери всей цепочки.
В AE-блокноте визуализируются только восемь ранее выбранных признаков, обучение
и метрики используют все признаки.

Блокноты:

- `Test_UZK_Autoencoder.ipynb` — обучение, реконструкция и экспорт.
- `Test_UZK_Autoencoder_GRU.ipynb` — готовый энкодер и обучение GRU.
- `Test_UZK.ipynb` — самостоятельная прогнозная модель.
- `Test_DataSet1.ipynb` — подготовка DataSet1 и эксперимент.

## История, события и TensorBoard

```bash
uv run tensorboard --logdir runs
```

Графики обучения смотрите в **Scalars** с фильтром `loss|learning_rate`.
Итоговые метрики всех моделей сохраняются в `metrics.json`, параметры — в
`config.yaml`. TensorBoard содержит только ход обучения.

Каждый выполненный train-батч записывается сразу: loss до обновления весов, фактический
размер батча, epoch, batch, сквозной step и learning rate. На каждой эпохе записываются
взвешенный train loss и validation loss при наличии validation. Веса выбираются по
validation loss. История доступна как `run.model.history_` и после загрузки.

```python
import pandas as pd

batches = pd.DataFrame([
    {"epoch": epoch["epoch"], **batch}
    for epoch in run.model.history_ for batch in epoch["batches"]
])
batches.plot(x="step", y="train_loss")
```

Выходы запуска: `config.yaml`, `metadata.json`, `events.jsonl`, `run.log`,
`tensorboard/`, `history.json`, `features.json`, `metrics.json`, `predictions_*.csv`,
`run.pkl`. При ошибке создаётся `failure.json`, завершённые batch-события сохраняются.

JSONL содержит `time`, `path`, `event`, `data`. Путь идентифицирует шаг и вложенный
подпайплайн. Исполнитель записывает начало/конец, размеры и длительность; runner —
обучение; шаги — собственную диагностику. Метаданные запуска содержат пути/хеши
входов, версии, границы и сводку каждого шага. Большие массивы не записываются в лог.

## Добавление шага

Наследуйте `Step`, реализуйте `transform` и при необходимости `fit`, зарегистрируйте
конструктор через `register_step("name", Constructor)` перед запуском. Для Python API
также допустимы пары `(name, step)` в списке pipeline; экземпляры клонируются.
Шаг обязан согласованно обновлять X, y и метаданные при изменении примеров.
В сохранённом `config.yaml` Python-шаг представлен полями `name`, `class`, `params`;
вложенные estimators сохраняются через `class` и `params`. При повторной загрузке
классы импортируются по полному имени, поэтому они должны быть определены на верхнем
уровне доступного Python-модуля. Обычные YAML-конфиги продолжают использовать `type`.

`context` необязателен и не хранится в модели. Он предоставляет `log_metrics`,
`log_metadata`, `log_event`, `log_message`; данные привязаны к пространству имён шага.
Компонент не открывает файлы и не создаёт TensorBoard writer самостоятельно.

## DataSet1 и проверки

```bash
uv run python scripts/dataset1.py prepare
uv run python scripts/dataset1.py run configs/dataset1_gru.yaml
.venv/bin/python -m unittest discover -s tests -v
uv lock --check
uv build
```

Подготовка DataSet1 удаляет лабораторные/качественные сигналы из признаков и сохраняет
числовые значения. Скрипт явно оценивает test после обучения и сохраняет сравнение
со средним наблюдавшихся train-целей.
