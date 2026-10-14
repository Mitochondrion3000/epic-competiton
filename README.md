# epic-competiton

**Подробный отчёт о проекте: [REPORT.md](REPORT.md).**
Epic Autosome competition - Large Milkweed Bug leaderboard

**Текущее состояние, планы и как продолжить работу — в [STATUS.md](STATUS.md).**

## Ноутбуки

| ноутбук | что внутри |
|---|---|
| `epic_eda_explained(лучший_вариант).ipynb` | EDA датасета Nematostella: геном, реплики, whitelist, пучки позиций, мотивы, PWM-бейзлайн |
| `epic_pipeline_annotated(лучший_вариант).ipynb` | первый пайплайн: dilated CNN, обучение и сабмит |
| `epic_v2_model.ipynb` | **новая модель v2** поверх бейзлайна InitiationNet: ансамбль, EMA, attention, головы на реплики, TTA, подбор квантования на hold-out; все параметры и архитектура визуализированы |

Какие модели обучены, с какими тегами и что вошло в каждый сабмит — в [MODELS.md](MODELS.md).

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
| 2 | `simple_model/step2_features.ipynb` + `simple_model/features.py` | 27 признаков из последовательности (Initiator, TATA, CCAAT, E-бокс, CRE, боксы тРНК, донор сплайсинга, поли-A/T, GC, CpG, повторы, найденные 6-меры); профили мотивов вокруг стартов; полезность каждого признака по отдельности |
| 3 | `simple_model/step3_motif_db_and_model.ipynb` + `simple_model/motifs.py` | скрининг 2805 мотивов JASPAR 2024 / HOCOMOCO v12 (YY1, TAF1, NF-Y, CREB/ATF, AP-1…), подписи k-меров, широкий контекст ±1 т.н., три модели (логистическая регрессия, аддитивная, деревья глубины 3), кривые влияния признаков, разбор ошибок. Валидация: AUPRC 0.020 (×27 случайного), Spearman 0.32 |
| 3б | `simple_model/step3b_positional.ipynb` + `simple_model/scoring.py` | плотная валидация (все позиции подряд, официальные метрики), окна пиков для каждого фактора (YY1 −2..+15, NF-Y −112..−43…), локальный контраст Initiator, подбор квантования сабмита. AUPRC на плотной валидации 0.017 → 0.025 |
| 4 | `simple_model/step4_test_and_submission.ipynb` | предсказание 72.6 млн позиций теста, официальные метрики, участки генома глазами, файл сабмита. **Тест Nematostella: AUPRC 0.020, Spearman 0.196** (бейзлайн организаторов 0.0012 / 0.036, лучшая CNN 0.132 / 0.391) |
| 5 | `simple_model/step5_logo.ipynb` | позиционный «логотип» (буква на каждом смещении −50..+30, 324 признака) + 61 признак, деревья глубины 6, картинка важности позиций, тест и сабмит. **Тест Nematostella: AUPRC 0.029, Spearman 0.274** (+45% / +40% к шагу 4) |
| 6 | `simple_model/step6_final.ipynb` | финальная версия: + пары букв (динуклеотиды −5..+30), все строки и все train-контиги, 3 модели; проверка пайплайна официальными `validate`/`scorer.py` по `template.test.txt.gz`, `template.train` + `csRNA-train.txt.gz` и тесту. **Тест Nematostella: AUPRC 0.033, Spearman 0.291** |

Открыть в Colab: [шаг 1](https://colab.research.google.com/github/Mitochondrion3000/epic-competiton/blob/claude/gracious-brown-4ybsmj/simple_model/step1_data_table.ipynb), [шаг 2](https://colab.research.google.com/github/Mitochondrion3000/epic-competiton/blob/claude/gracious-brown-4ybsmj/simple_model/step2_features.ipynb), [шаг 3](https://colab.research.google.com/github/Mitochondrion3000/epic-competiton/blob/claude/gracious-brown-4ybsmj/simple_model/step3_motif_db_and_model.ipynb), [шаг 3б](https://colab.research.google.com/github/Mitochondrion3000/epic-competiton/blob/claude/gracious-brown-4ybsmj/simple_model/step3b_positional.ipynb), [шаг 4](https://colab.research.google.com/github/Mitochondrion3000/epic-competiton/blob/claude/gracious-brown-4ybsmj/simple_model/step4_test_and_submission.ipynb), [шаг 5](https://colab.research.google.com/github/Mitochondrion3000/epic-competiton/blob/claude/gracious-brown-4ybsmj/simple_model/step5_logo.ipynb), [шаг 6](https://colab.research.google.com/github/Mitochondrion3000/epic-competiton/blob/claude/gracious-brown-4ybsmj/simple_model/step6_final.ipynb) (CPU достаточно; каждый шаг читает то, что предыдущий сохранил на Google Диск)

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

### Запуск на Kaggle (CPU)

`kaggle/ofas_a` (шаги 1–3) → `kaggle/ofas_b` (шаг 3б) → `kaggle/ofas_c` (шаг 5: логотип, тест, сабмит). Каждый скрипт клонирует
эту ветку, берёт результаты предыдущего ноутбука как вход и пишет свои в `/kaggle/working`. Отправка: `kaggle kernels push -p kaggle/ofas_a`.
