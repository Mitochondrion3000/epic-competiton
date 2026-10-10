# Журнал сабмитов EPIC

Каждый сабмит: файл, откуда взят, что за модель, оценка на отложенных контигах (report) и результат лидерборда после отправки.
Метрики report считаются с рангами, растянутыми до размера теста (как в настоящем файле), поэтому сопоставимы с лидербордом.

---

## 2026-10-10 · sSquSuc5.hap2 (акула) · v3 на своих данных, alpha 0.5

**Файл:** `sSquSuc5.hap2.predictions.txt.gz` (в датасете `ivanmitochondrion/epic-predictions`: `submissions/sSquSuc5.hap2.predictions.v3.alpha0.50.txt.gz.bin`).
Kaggle `k13polinaa/epic-v3-shark-pred` → Output → `epic_v3/sSquSuc5.hap2/` (веса из `k13polinaa/epic-v3-shark`).
Ветка `claude/youthful-hopper-hjme9f`, коммит `bd1214e`, ноутбук `epic_v3_bpnet.ipynb` с `EPIC_SPECIES=sSquSuc5.hap2`, `EPIC_AUX=''`, `EPIC_TAG=v3b2`.

**Модель:** v3 (BPNet / ProCapNet: dilated CNN, головы «профиль» и «счёт»), обучена с нуля только на train-данных акулы, 30k шагов, LR 5e-4.
1 модель (seed0); seed1 разошёлся (5 откатов) и в сабмит не вошёл. alpha 0.5 выбран правилом по tune. Внешних данных и предобученных весов нет.

**Оценка на report (честная: tune/report отложены; выборка 30 Мб из 371 Мб группы, ранги растянуты до размера теста):**

| | AUPRC | Spearman |
|---|---|---|
| перенос v2 с насекомых (`epic-v2-transfer-dogfish`, report) | 0.006 | 0.195 |
| бейзлайн CNN из write-up организаторов (их отложенные данные, сравнение примерное) | 0.110 | 0.358 |
| **этот сабмит** | **0.057** | **0.246** |

Сетка alpha на tune: 0.5 → 0.052 / 0.274; 0.70 → 0.050 / 0.286 (файл `…alpha0.70.txt.gz` в том же Output).

**Проверки файла:** 344 993 974 строки = позиций шаблона; значения в [0, 1]; N-позиции = 0; 4 068 уровней; 289.9 МБ.

**Лидерборд:** —

**Теги:** `CNN`, `Raw` + свой `BPNet`.

**Текст для формы сабмита (EN):**
> BPNet/ProCapNet-style dilated CNN (sequence only) with profile (multinomial) and count (MSE) heads,
> trained from scratch on sSquSuc5.hap2 training data only (1 model), rank-power quantised (α = 0.5).
> No external data or pretrained weights. Hold-out contigs (sSquSuc5.hap2): AUPRC 0.057, Spearman 0.246.

---

## 2026-10-10 · ilPloInte3.2 (моль) · смесь v2moth + v3b2, alpha 0.70

**Файл:** `ilPloInte3.2.predictions.txt.gz` (в датасете `ivanmitochondrion/epic-predictions`: `submissions/ilPloInte3.2.predictions.blend_v2v3.alpha0.70.txt.gz.bin`).
Kaggle `k13polinaa/epic-v3-moth-blend` (версия 3) → Output → `epic_v3/ilPloInte3.2/`. Ветка `claude/youthful-hopper-hjme9f`, коммит `547c632`.
Источники: `k13polinaa/epic-v2-moth-pred` (v2, веса `ivanmitochondrion/epic-v2-moth-ckpt`) и `k13polinaa/epic-v3b2-moth` (v3).

**Модель:** 0.6 × v2 + 0.4 × v3 (стандартизованные log-rate), rank-power квантование с alpha 0.70.
- **v2 (`bug+moth`):** InitiationNet + attention, общая сеть на клопе и моли со своей головой у каждого вида, 2 сида, 96k шагов
  (`Ofas_2.0_v2moth_seed0/1.pt`, ветка `claude/dreamy-albattani-3tmvp6`). Предсказание головой моли.
- **v3 (`v3b2`):** BPNet / ProCapNet, обучена с нуля на клопе + моли + устрице + Nematostella (с тестом Nematostella), 30k шагов, LR 5e-4, 2 сида;
  tune/report моли отложены. Предсказание головами моли.
- **Вес и alpha заданы заранее** (`EPIC_BLEND=z:0.4`, `EPIC_ALPHA=0.7`, как у смеси клопа), а не подобраны: v2 училась на **всех** контигах моли,
  включая tune/report, поэтому сетка на tune смещена в её пользу (она выбрала бы w_v3 = 0.1).
- Внешних данных и предобученных весов нет.

**Оценка на report (ранги растянуты до размера теста):**

| | AUPRC | Spearman | честная? |
|---|---|---|---|
| v2 | 0.386 | 0.780 | нет: v2 видела эти контиги |
| v3 | 0.171 | 0.339 | да |
| **этот сабмит** | 0.359 | 0.785 | нет |

Честного прогноза у смеси нет, сравнивать надо по лидерборду с нынешним сабмитом моли (v3 отдельно, 0.177 / 0.378).
Запасной файл в том же Output: `…alpha0.80.txt.gz` (report 0.346 / 0.808, тоже нечестный).

**Проверки файла:** 113 971 908 строк = позиций шаблона; значения в [0, 1]; N-позиции = 0; 1 720 уровней; 31.4 МБ.

**Лидерборд:** —

**Теги:** `CNN`, `Ensemble`, `Raw` + свои `multi-species`, `BPNet`.

**Текст для формы сабмита (EN):**
> Blend of two sequence-only CNNs (0.6 × v2 + 0.4 × v3, z-scored log-rates), rank-power quantised (α = 0.70).
> v2: InitiationNet-style multi-scale CNN with attention, trained jointly on Ofas_2.0 + ilPloInte3.2 with per-species heads (2 seeds); ilPloInte3.2 head.
> v3: BPNet/ProCapNet-style dilated CNN with profile (multinomial) and count (MSE) heads, shared body and per-species heads,
> trained from scratch on Ofas_2.0 + Plodia + Magallana + Nematostella training data; ilPloInte3.2 heads.
> No external data or pretrained weights. Blend weight and α fixed in advance (v2 saw the hold-out contigs of this species).

---

## 2026-10-09 · ilPloInte3.2 (моль) · v3 отдельно, alpha 0.55

**Файл:** `ilPloInte3.2.predictions.txt.gz`. Kaggle `k13polinaa/epic-v3-moth` → Output → `epic_v3/ilPloInte3.2/`.
Ветка `claude/youthful-hopper-hjme9f`, коммит `f25558a`, ноутбук `epic_v3_bpnet.ipynb` с `EPIC_SPECIES=ilPloInte3.2`, `EPIC_CKPT_FROM=Ofas_2.0`.

**Модель:** v3 (BPNet / ProCapNet, общее тело + головы на каждый вид) из прогона клопа `k13polinaa/epic-v3-ofas`:
обучена с нуля на клопе + моли + устрице + Nematostella, 30k шагов; предсказание головами вида ilPloInte3.2 (индекс 1), без дообучения.
В ансамбле только seed1 (seed0 разошёлся при обучении). Смеси нет: v2 для этого вида не обучали. alpha 0.55 выбран правилом по tune.
Внешних данных и предобученных весов нет.

**Оценка на report (5% train-контигов вида, ранги растянуты до размера теста):** AUPRC **0.278**, Spearman **0.617**.
Сетка alpha на tune: 0.55 → 0.255 / 0.601; 0.70 → 0.250 / 0.655 (файл `…alpha0.70.txt.gz` в том же Output). Spearman уже выше лидера, поэтому берём 0.55 ради AUPRC.

**Проверки файла:** 113 971 908 строк = позиций шаблона; значения в [0, 1]; N-позиции = 0; 81.9 МБ.

**Лидерборд (2026-10-09 12:33, 1-я половина теста):** AUPRC **0.177**, Spearman **0.378** → **21-е место** по моли (было 31-е, без сабмита). Лидер NemoAgents 0.282 / 0.576.

**⚠ Оценка report выше недействительна (утечка):** в прогоне клопа моль была вспомогательным видом и училась на всех своих train-контигах,
включая контиги report. Честная оценка — только лидерборд. Исправлено в коммите после `74b48a4`: группы tune/report теперь откладываются у каждого вида.

**Теги:** `CNN`, `Raw` + свои `multi-species`, `BPNet`.

**Текст для формы сабмита (EN):**
> BPNet/ProCapNet-style dilated CNN (sequence only) with profile (multinomial) and count (MSE) heads; shared body, per-species heads,
> trained from scratch on Ofas_2.0 + Plodia + Magallana + Nematostella training data; this file uses the ilPloInte3.2 heads, 1 model, rank-power α = 0.55.
> No external data or pretrained weights. Hold-out contigs (ilPloInte3.2): AUPRC 0.278, Spearman 0.617.

---

## 2026-10-09 · cgigas_uk_roslin_v1 (устрица) · v3 отдельно, alpha 0.5

**Файл:** `cgigas_uk_roslin_v1.predictions.txt.gz`. Kaggle `k13polinaa/epic-v3-oyster` → Output → `epic_v3/cgigas_uk_roslin_v1/`.
Ветка `claude/youthful-hopper-hjme9f`, коммит `f25558a`, ноутбук `epic_v3_bpnet.ipynb` с `EPIC_SPECIES=cgigas_uk_roslin_v1`, `EPIC_CKPT_FROM=Ofas_2.0`.

**Модель:** v3 (BPNet / ProCapNet, общее тело + головы на каждый вид) из прогона клопа `k13polinaa/epic-v3-ofas`:
обучена с нуля на клопе + моли + устрице + Nematostella, 30k шагов; предсказание головами вида cgigas_uk_roslin_v1 (индекс 2), без дообучения.
В ансамбле только seed1 (seed0 разошёлся при обучении). Смеси нет: v2 для этого вида не обучали. alpha 0.5 выбран правилом по tune.
Внешних данных и предобученных весов нет.

**Оценка на report (5% train-контигов вида, ранги растянуты до размера теста):** AUPRC **0.127**, Spearman **0.433**.
Сетка alpha на tune: 0.50 → 0.150 / 0.468; 0.70 → 0.141 / 0.526 (файл `…alpha0.70.txt.gz`). Spearman уже выше лидера, поэтому берём 0.5 ради AUPRC.

**Проверки файла:** 198 420 888 строк = позиций шаблона; значения в [0, 1]; N-позиции = 0; 172.9 МБ.

**Лидерборд (2026-10-09 13:08, 1-я половина теста):** AUPRC **0.085**, Spearman **0.225** → **16-е место** по устрице (было 23-е, без сабмита). Лидеры VibeGenomics 0.186 / 0.360, NemoAgents 0.186 / 0.412.

**⚠ Оценка report выше недействительна (утечка):** та же причина, что у моли — устрица училась на своих контигах report.

**Теги:** `CNN`, `Raw` + свои `multi-species`, `BPNet`.

**Текст для формы сабмита (EN):**
> BPNet/ProCapNet-style dilated CNN (sequence only) with profile (multinomial) and count (MSE) heads; shared body, per-species heads,
> trained from scratch on Ofas_2.0 + Plodia + Magallana + Nematostella training data; this file uses the cgigas_uk_roslin_v1 heads, 1 model, rank-power α = 0.5.
> No external data or pretrained weights. Hold-out contigs (cgigas_uk_roslin_v1): AUPRC 0.127, Spearman 0.433.

---

## 2026-10-09 · Ofas_2.0 (клоп) · смесь v2 + v3, alpha 0.70

**Файл:** `Ofas_2.0.predictions.alpha0.70.txt.gz`. Kaggle `k13polinaa/epic-v3-ofas-requant` → Output → `epic_v3/Ofas_2.0/`.
Ветка `claude/youthful-hopper-hjme9f`, коммит `6958807`, ноутбук `epic_v3_bpnet.ipynb`.

**Модель:** 0.6 × v2 + 0.4 × v3 (стандартизованные log-rate), затем rank-power квантование с alpha 0.70.
- **v2:** InitiationNet + attention, обучена только на клопе, 2 сида (seed0, seed1), 96k шагов.
- **v3:** BPNet / ProCapNet (общее тело + головы «профиль» и «счёт» на каждый вид), обучена с нуля на клопе + моли + устрице + Nematostella, 30k шагов.
  В ансамбле только seed1: seed0 разошёлся при обучении.
- Вес смеси выбран на группе tune, alpha 0.70 — компромисс по сетке tune и report.
- Внешних данных и предобученных весов нет.

**Оценка на report (5% train-контигов клопа):**

| | AUPRC | Spearman |
|---|---|---|
| v2 | 0.201 | 0.475 |
| v3 | 0.156 | 0.359 |
| **этот сабмит** | **0.217** | **0.506** |

**Проверки файла:** 328 090 834 строки = позиций шаблона; значения в [0, 1]; N-позиции = 0.

**Лидерборд (2026-10-09 09:33, 1-я половина теста):** AUPRC **0.2158**, Spearman **0.5015** → **2-е место** на клопе (было 0.203 / 0.478, 7-е). Прогноз по report был 0.217 / 0.506.

**Теги:** `CNN`, `Ensemble`, `Raw` + свои `multi-species`, `BPNet`.
Stage не ставим: Baseline/Finetune/Debug не подходят (модели обучены с нуля, предобученных весов нет).

**Текст для формы сабмита (EN):**
> Blend of two sequence-only CNNs (0.6 × v2 + 0.4 × v3, z-scored log-rates), rank-power quantised (α = 0.70).
> v2: InitiationNet-style multi-scale CNN with attention, trained on Ofas_2.0 only (2 seeds).
> v3: BPNet/ProCapNet-style dilated CNN with profile (multinomial) and count (MSE) heads, shared body and per-species heads,
> trained from scratch on Ofas_2.0 + Plodia + Magallana + Nematostella training data. No external data or pretrained weights.
> Hold-out contigs (Ofas_2.0): AUPRC 0.217, Spearman 0.506.

**Запасной вариант:** тот же прогон, `Ofas_2.0.predictions.txt.gz`, alpha 0.55 → report 0.222 / 0.472.
