# epic-competiton
Epic Autosome competition - Large Milkweed Bug leaderboard

## Ноутбуки

| ноутбук | что внутри |
|---|---|
| `epic_eda_explained(лучший_вариант).ipynb` | EDA датасета Nematostella: геном, реплики, whitelist, пучки позиций, мотивы, PWM-бейзлайн |
| `epic_pipeline_annotated(лучший_вариант).ipynb` | первый пайплайн: dilated CNN, обучение и сабмит |
| `epic_v2_model.ipynb` | **новая модель v2** поверх бейзлайна InitiationNet: ансамбль, EMA, attention, головы на реплики, TTA, подбор квантования на hold-out; все параметры и архитектура визуализированы |

[Открыть v2 в Colab](https://colab.research.google.com/github/Mitochondrion3000/epic-competiton/blob/main/epic_v2_model.ipynb)

## Простая интерпретируемая модель (папка `simple_model/`)

Пошаговая серия: понятные биологические признаки из последовательности → таблица → простая модель.

| шаг | ноутбук | что внутри |
|---|---|---|
| 1 | `simple_model/step1_data_table.ipynb` | чтение данных, «позиция-цепь» и ориентация, валидационные контиги, таблица строк (все старты + случайные отрицательные), первый взгляд на сигнал (динуклеотид на старте, GC) |
| 2 | `simple_model/step2_features.ipynb` + `simple_model/features.py` | 27 признаков из последовательности (Initiator, TATA, CCAAT, E-бокс, CRE, боксы тРНК, донор сплайсинга, поли-A/T, GC, CpG, повторы, найденные 6-меры); профили мотивов вокруг стартов; полезность каждого признака по отдельности |
| 3 | `simple_model/step3_motif_db_and_model.ipynb` + `simple_model/motifs.py` | скрининг 2805 мотивов JASPAR 2024 / HOCOMOCO v12 (YY1, TAF1, NF-Y, CREB/ATF, AP-1…), подписи k-меров, широкий контекст ±1 т.н., три модели (логистическая регрессия, аддитивная, деревья глубины 3), кривые влияния признаков, разбор ошибок. Валидация: AUPRC 0.020 (×27 случайного), Spearman 0.32 |
| 3б | `simple_model/step3b_positional.ipynb` + `simple_model/scoring.py` | плотная валидация (все позиции подряд, официальные метрики), окна пиков для каждого фактора (YY1 −2..+15, NF-Y −112..−43…), локальный контраст Initiator, подбор квантования сабмита. AUPRC на плотной валидации 0.017 → 0.025 |
| 4 | `simple_model/step4_test_and_submission.ipynb` | предсказание 72.6 млн позиций теста, официальные метрики, участки генома глазами, файл сабмита. **Тест Nematostella: AUPRC 0.020, Spearman 0.196** (бейзлайн организаторов 0.0012 / 0.036, лучшая CNN 0.132 / 0.391) |

Открыть в Colab: [шаг 1](https://colab.research.google.com/github/Mitochondrion3000/epic-competiton/blob/claude/gracious-brown-4ybsmj/simple_model/step1_data_table.ipynb), [шаг 2](https://colab.research.google.com/github/Mitochondrion3000/epic-competiton/blob/claude/gracious-brown-4ybsmj/simple_model/step2_features.ipynb), [шаг 3](https://colab.research.google.com/github/Mitochondrion3000/epic-competiton/blob/claude/gracious-brown-4ybsmj/simple_model/step3_motif_db_and_model.ipynb), [шаг 3б](https://colab.research.google.com/github/Mitochondrion3000/epic-competiton/blob/claude/gracious-brown-4ybsmj/simple_model/step3b_positional.ipynb), [шаг 4](https://colab.research.google.com/github/Mitochondrion3000/epic-competiton/blob/claude/gracious-brown-4ybsmj/simple_model/step4_test_and_submission.ipynb) (CPU достаточно; каждый шаг читает то, что предыдущий сохранил на Google Диск)

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
