# Обученные модели простой интерпретируемой модели

| папка | вид | версия | признаки | моделей | alpha | качество |
|---|---|---|---|---|---|---|
| `jaNemVect1.1_step6/` | *Nematostella* (учебный) | шаг 6 (финальная) | 61 понятный + логотип −50..+30 + пары букв −5..+30 = 961 | 3 сида | 0.6 | тест: AUPRC 0.033, Spearman 0.291 |
| `Ofas_2.0_step5/` | клоп (лидерборд) | шаг 5 | 57 понятных + логотип −50..+30 = 381 | 1 | 0.6 | плотная валидация: AUPRC 0.029, Spearman 0.370 |

Финальная модель для `Ofas_2.0` (шаг 6, 3 сида) сейчас обучается на Kaggle (`epic-simple-ofas-final-train`) и будет
добавлена в `Ofas_2.0_step6/`.

В каждой папке:
* `*_cls*.txt` — LightGBM-модель «шанс старта» (бинарная классификация);
* `*_reg*.txt` — LightGBM-модель «сила старта» (`log(1 + счёт)` среди стартов);
* `*.step3b_params.json` — параметры понятных признаков (PWM старта, k-меры, мотивы JASPAR/HOCOMOCO с порогами и окнами);
* `*_summary.json` — alpha квантования сабмита и метрики.

## Как использовать

Код признаков — в `simple_model/` (`features.py`, `motifs.py`, `scoring.py`, `final.py`). Счёт позиции:
`log P(старт) + E[log(1 + счёт) | старт]`, при нескольких моделях — среднее; для сабмита — `scoring.rank_power(счёт, alpha)`.

```python
import sys, numpy as np, lightgbm as lgb
sys.path.insert(0, "simple_model")
import features as F, final as FN, scoring as S

M = "models/jaNemVect1.1_step6"
params, BASE, ALL = FN.load_params(f"{M}/jaNemVect1.1.step3b_params.json")     # для шага 5 логотип без пар букв, см. ниже
models = [(lgb.Booster(model_file=f"{M}/jaNemVect1.1.step6_cls_seed{s}.txt"),
           lgb.Booster(model_file=f"{M}/jaNemVect1.1.step6_reg_seed{s}.txt")) for s in range(3)]

SEQ, LOW = FN.read_genome("jaNemVect1.1/genome/genome.fa.gz", {"NC_064037.1"})   # геном из архива Zenodo
pos = np.arange(1_000_000, 1_001_000)                                           # позиции генома (0-based)
score = FN.predict_positions(models, SEQ["NC_064037.1"], LOW["NC_064037.1"], pos, "+", params, BASE)
```

Для `Ofas_2.0_step5` признаки — 57 понятных + только логотип (`F.logo_names()`), без пар букв; модель одна:
счёт = `log(cls.predict(X)) + reg.predict(X)`, где `X = [feature_matrix(...), logo_onehot(...)]` (см. `simple_model/step5_logo.ipynb`).

Готовые сабмиты не хранятся в репозитории (большие файлы): они в Output ноутбуков Kaggle
`epic-simple-ofas-c` (клоп, шаг 5) и `epic-simple-nem-step6` (*Nematostella*, шаг 6).
