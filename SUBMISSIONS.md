# Журнал сабмитов EPIC

Каждый сабмит: файл, откуда взят, что за модель, оценка на отложенных контигах (report) и результат лидерборда после отправки.
Метрики report считаются с рангами, растянутыми до размера теста (как в настоящем файле), поэтому сопоставимы с лидербордом.

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

**Лидерборд:** _отправлен 2026-10-09, результат ещё не появился в публичной таблице (до этого у команды было 0.203 / 0.478, 7-е место)._

**Теги:** `CNN`, `Ensemble`, `Raw` + свои `multi-species`, `BPNet`.
Stage не ставим: Baseline/Finetune/Debug не подходят (модели обучены с нуля, предобученных весов нет).

**Текст для формы сабмита (EN):**
> Blend of two sequence-only CNNs (0.6 × v2 + 0.4 × v3, z-scored log-rates), rank-power quantised (α = 0.70).
> v2: InitiationNet-style multi-scale CNN with attention, trained on Ofas_2.0 only (2 seeds).
> v3: BPNet/ProCapNet-style dilated CNN with profile (multinomial) and count (MSE) heads, shared body and per-species heads,
> trained from scratch on Ofas_2.0 + Plodia + Magallana + Nematostella training data. No external data or pretrained weights.
> Hold-out contigs (Ofas_2.0): AUPRC 0.217, Spearman 0.506.

**Запасной вариант:** тот же прогон, `Ofas_2.0.predictions.txt.gz`, alpha 0.55 → report 0.222 / 0.472.
