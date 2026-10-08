# epic-competiton
Epic Autosome competition - Large Milkweed Bug leaderboard

## Ноутбуки

| ноутбук | что внутри |
|---|---|
| `epic_eda_explained(лучший_вариант).ipynb` | EDA датасета Nematostella: геном, реплики, whitelist, пучки позиций, мотивы, PWM-бейзлайн |
| `epic_pipeline_annotated(лучший_вариант).ipynb` | первый пайплайн: dilated CNN, обучение и сабмит |
| `epic_v2_model.ipynb` | **новая модель v2** поверх бейзлайна InitiationNet: ансамбль, EMA, attention, головы на реплики, TTA, подбор квантования на hold-out; все параметры и архитектура визуализированы |

[Открыть v2 в Colab](https://colab.research.google.com/github/Mitochondrion3000/epic-competiton/blob/main/epic_v2_model.ipynb)

## v3: мультивидовой BPNet + ансамбль с v2 (`epic_v3_bpnet.ipynb`)

* Архитектура BPNet/ProCapNet: dilated CNN и две головы — «профиль» (где в окне старт, softmax + кросс-энтропия)
  и «счёт» (сколько прочтений в окне, MSE по log1p). Предсказание в позиции = счёт × доля профиля.
* Тело общее для всех видов, головы свои у каждого вида. По умолчанию учимся на клопе `Ofas_2.0`
  плюс моль, устрица и Nematostella (переменные `EPIC_SPECIES`, `EPIC_AUX`).
* Если рядом лежат предсказания v2 (`holdout_*_lograte.npy`, `test_lograte.npy`, `holdout_groups.json`: их теперь сохраняет v2),
  v3 подбирает смесь на группе tune, проверяет на report и делает сабмит из смеси.
* Ноутбук генерируется скриптом: правки вносить в `tools/build_v3_notebook.py`, затем `python3 tools/build_v3_notebook.py`.

Запуск на Kaggle: GPU T4 x2, Internet On, Run All. Для проверки кода: `EPIC_FAST_DEV=1 EPIC_SPECIES=jaNemVect1.1 EPIC_AUX=ilPloInte3.2`.

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
