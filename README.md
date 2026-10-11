# epic-competiton
Epic Autosome competition - Large Milkweed Bug leaderboard

## Ноутбуки

| ноутбук | что внутри |
|---|---|
| `epic_eda_explained(лучший_вариант).ipynb` | EDA датасета Nematostella: геном, реплики, whitelist, пучки позиций, мотивы, PWM-бейзлайн |
| `epic_pipeline_annotated(лучший_вариант).ipynb` | первый пайплайн: dilated CNN, обучение и сабмит |
| `epic_v2_model.ipynb` | **новая модель v2** поверх бейзлайна InitiationNet: ансамбль, EMA, attention, головы на реплики, TTA, подбор квантования на hold-out; все параметры и архитектура визуализированы |

Какие модели обучены, с какими тегами и что вошло в каждый сабмит — в [MODELS.md](MODELS.md).

[Открыть v2 в Colab](https://colab.research.google.com/github/Mitochondrion3000/epic-competiton/blob/main/epic_v2_model.ipynb)

## Простая интерпретируемая модель (папка `simple_model/`)

Пошаговая серия: понятные биологические признаки из последовательности → таблица → простая модель.

| шаг | ноутбук | что внутри |
|---|---|---|
| 1 | `simple_model/step1_data_table.ipynb` | чтение данных, «позиция-цепь» и ориентация, валидационные контиги, таблица строк (все старты + случайные отрицательные), первый взгляд на сигнал (динуклеотид на старте, GC) |

[Открыть шаг 1 в Colab](https://colab.research.google.com/github/Mitochondrion3000/epic-competiton/blob/claude/gracious-brown-4ybsmj/simple_model/step1_data_table.ipynb) (CPU достаточно)

## Как запустить v2

1. Colab → Runtime → Change runtime type → GPU (лучше L4 или A100), для `Ofas_2.0` включить High-RAM.
2. В ячейке 1.2 выбрать вид: `SPECIES = "jaNemVect1.1"` (учебный, есть ответы теста) или `"Ofas_2.0"` (наш лидерборд).
3. Выполнить все ячейки. Сабмит появится в `OUT_DIR/<вид>.predictions.txt.gz`.

Сначала стоит прогнать Nematostella: только у этого вида есть ответы теста, поэтому ноутбук сравнивает v2 с бейзлайном
на официальном тесте, а раздел 9 (абляция) показывает вклад каждого изменения.

**Прервалась сессия или кончился GPU?** Просто запустите ноутбук заново: каждые `CKPT_EVERY` шагов сохраняется полное
состояние обучения, и оно продолжится с последнего чекпоинта. В Colab чекпоинты лежат на Google Диске (`MyDrive/epic_runs/<вид>/`).
Перенос в Kaggle: загрузить эти `*.pt` как Kaggle Dataset, подключить его как Input (Internet on, GPU T4), запустить —
ноутбук сам найдёт чекпоинты в `/kaggle/input`. Подробности в разделе 6 ноутбука.

Для проверки кода без GPU: `EPIC_FAST_DEV=1` ставит крошечные настройки (несколько минут на CPU).
