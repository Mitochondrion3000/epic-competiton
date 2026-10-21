# Журнал сабмитов EPIC

Каждый сабмит: файл, откуда взят, что за модель, оценка на отложенных контигах (report) и результат лидерборда после отправки.
Метрики report считаются с рангами, растянутыми до размера теста (как в настоящем файле), поэтому сопоставимы с лидербордом.

---

## 2026-10-21 · cgigas_uk_roslin_v1 (устрица) · смесь m3ft (веса 0.5 / 0.3) + прежний сабмит, alpha 0.65

**Файлы:** `cgigas_uk_roslin_v1.predictions.blend_m3ft_w0.5_old.alpha0.65.txt.gz` и `…_w0.3_…` (по ≈77.6 МБ). Kaggle `k13polinaa/epic-blend-m3ft-oyster` (CPU) → Output → `m3ft_blend/`. Скрипт `kaggle/blend_m3ft_oyster/blend.py`, ветка `claude/week0-code`, коммит `c542c62`.
**Модель:** w × z(m3ft) + (1 − w) × z(прежний). m3ft = `k13polinaa/epic-v2-m3ft-oyster`: v2 InitiationNet, претрейн на шести видах (`epic-v2-m3-pre2`, головы видов), дообучение на cgigas_uk_roslin_v1, 48 000 шагов, 2 сида, 2 T4 (≈2,8 ч). Прежний = v2 ft2 (дообучение от моли, 2 сида) (`epic-v2-ft2-oyster`). Spearman между входами на 2 млн позиций: 0.883.
**Оценка:** честного report нет (претрейн видел контиги report), по сидам m3ft AUPRC 0.115 / 0.118 (сырое, alpha tune 0.55). Прежний ft2 на честных контигах: AUPRC 0.120, Spearman 0.316. Веса заданы заранее, смесь на report не считали.
**Проверки файла (оба):** 198 420 888 строк = позиций шаблона; значения в [0, 1]; N-позиции = 0; 2 096 уровней.
**Лидерборд:** —
**Теги:** `CNN`, `Ensemble`, `Finetune`, `Raw` + свои `multi-species`, `InitiationNet`.
**Текст для формы (EN, 441 знаков; для веса 0.3 заменить «0.5/0.5» на «0.3/0.7», 441 знаков):**
> Blend of sequence-only CNNs (z-scored, m3ft weight 0.5/0.5, rank-power alpha 0.65). A: multi-scale CNN with attention, pretrained jointly on six EPIC species with per-species heads and fine-tuned on this species training data (2 seeds). B: earlier model: v2 CNN fine-tuned from a model pretrained on other EPIC species (2 seeds). No external data or third-party pretrained weights. Weights fixed in advance; not validated on honest hold-out.

## 2026-10-21 · sSquSuc5.hap2 (акула) · смесь m3ft (веса 0.5 / 0.3) + прежний сабмит, alpha 0.65

**Файлы:** `sSquSuc5.hap2.predictions.blend_m3ft_w0.5_old.alpha0.65.txt.gz` и `…_w0.3_…` (по ≈124.9 МБ). Kaggle `k13polinaa/epic-blend-m3ft-shark` (CPU) → Output → `m3ft_blend/`. Скрипт `kaggle/blend_m3ft_shark/blend.py`, ветка `claude/week0-code`, коммит `c542c62`.
**Модель:** w × z(m3ft) + (1 − w) × z(прежний). m3ft = `k13polinaa/epic-v2-m3ft-shark`: v2 InitiationNet, претрейн на шести видах (`epic-v2-m3-pre2`, головы видов), дообучение на sSquSuc5.hap2, 48 000 шагов, 2 сида, 2 T4 (≈2,8 ч). Прежний = смесь v2 ft2 + v3 (вес v3 0.3, alpha 0.55; на сайте 0.1206 / 0.4625) (`epic-blend-shark`). Spearman между входами на 2 млн позиций: 0.889.
**Оценка:** честного report нет (претрейн видел контиги report), по сидам m3ft AUPRC 0.122 / 0.094 (сырое, alpha tune 0.65). Прежний ft2 на честных контигах: AUPRC 0.102, Spearman 0.335. Веса заданы заранее, смесь на report не считали.
**Проверки файла (оба):** 344 993 974 строк = позиций шаблона; значения в [0, 1]; N-позиции = 0; 2 096 уровней.
**Лидерборд:** —
**Теги:** `CNN`, `Ensemble`, `Finetune`, `Raw` + свои `multi-species`, `InitiationNet`.
**Текст для формы (EN, 437 знаков; для веса 0.3 заменить «0.5/0.5» на «0.3/0.7», 437 знаков):**
> Blend of sequence-only CNNs (z-scored, m3ft weight 0.5/0.5, rank-power alpha 0.65). A: multi-scale CNN with attention, pretrained jointly on six EPIC species with per-species heads and fine-tuned on this species training data (2 seeds). B: earlier model: blend of that v2 CNN (2 seeds) and a species-specific BPNet-style CNN. No external data or third-party pretrained weights. Weights fixed in advance; not validated on honest hold-out.

## 2026-10-20 · ilPloInte3.2 (моль) · смесь v3 (веса 0.3 / 0.5) + лучший файл, alpha 0.70

**Файлы:** `ilPloInte3.2.predictions.blend_v3_w0.3_best.alpha0.70.txt.gz` и `..._w0.5_...` (по 31.9 МБ). Kaggle `k13polinaa/epic-moth-blend-v3` (CPU, ≈2,5 мин) → Output → `moth_blend_v3/`. Скрипт `kaggle/moth_blend_v3/blend.py`, ветка `claude/week0-code`.
**Модель:** w × z(v3) + (1 − w) × z(лучший), где «лучший» = 0.5 z(m3ft) + 0.5 z(прежняя смесь) (это файл `blend_m3ft_w0.5_old.alpha0.70`, лидерборд 0.23205 / 0.49419 для клопа; для моли 0.2577 / 0.566). v3 = ансамбль контроль + FP3 (`ivanmitochondrion/epic-v3-m6-pred-moth`, `test_final_score.npy`). Spearman между лучшим и v3 на 2 млн позиций: 0.640.
**Оценка:** смесь на report не считали (у m3ft и прежней смеси hold-out моли виден в обучении, честного report-сравнения нет); веса заданы заранее. v3 отдельно на честном report: AUPRC 0.1913, Spearman 0.3431.
**Проверки файла (оба):** 113 971 908 строк = позиций шаблона; значения в [0, 1]; N-позиции = 0; 1 720 уровней.
**Лидерборд:** 2026-10-20: вес v3 0.5 — `calm-bison-13b1` (10:46) 0.25051 / 0.51749, место сайта 159; вес v3 0.3 — `lunar-seal-d86f` (11:12) 0.25900 / 0.54742, место 61. Лучший наш (m3ft 0.5 + прежний, без v3) 0.2577 / 0.566: смесь с v3 при 0.3 даёт AUPRC +0,5%, Spearman −0.019, произведение мест хуже (11 против 9 среди команд по текущей таблице), при 0.5 хуже сильнее (14). Вывод: v3 в нынешнем виде для моли не помогает.
**Теги:** `CNN`, `Ensemble`, `Raw` + свои `multi-species`, `InitiationNet`, `BPNet`.
**Текст для формы (EN, 434 знаков, вес v3 0.3; для 0.5 заменить «0.3/0.7» на «0.5/0.5», 434 знаков):**
> Blend of sequence-only CNNs (z-scored, weights 0.3/0.7, rank-power alpha 0.70). A: BPNet/ProCapNet-style dilated CNN, 2-model ensemble, pretrained on six species with per-species heads and fine-tuned on this species. B: earlier blend of InitiationNet-style CNNs (pretrained on six species, fine-tuned) and a species-specific CNN. No external data or third-party pretrained weights. Weights fixed in advance; not validated on hold-out.


## 2026-10-20 · ilPloInte3.2 (моль) · v3 BPNet, ансамбль контроль + FP_WEIGHT=3, alpha 0.55

**Файл:** `ilPloInte3.2.predictions.txt.gz` (81.1 МБ). Kaggle `ivanmitochondrion/epic-v3-m6-pred-moth` (GPU T4×2, ≈17 мин) → Output → `epic_v3/ilPloInte3.2/` (там же `..alpha0.70..` и `..alpha0.80..`: хуже по tune, не нужны). Ветка `claude/week0-code`, `epic_v3_bpnet.ipynb`.
**Модель:** BPNetMS (1,93 млн параметров, рецептивное поле ±1069 нт): претрейн на 6 видах (`epic-v3-m6-pre`, 30 000 шагов, 2 сида, 4,15 ч), ft на моли 16 000 шагов (`epic-v3-m6-ft-moth-ab`, 2,01 ч). Две модели в ансамбле: контроль и FP_WEIGHT=3 (вес ×4 для окон, где предсказанный счёт выше цели на 0,5).
**Оценка на report (честная, hold-out не видел претрейн и ft):** ансамбль AUPRC 0.1913, Spearman 0.3431 (alpha 0.55 по tune, с поправкой n_ref); по одиночке 0.1797/0.3405 и 0.1823/0.3330. Лучшая одиночная v2 на моли для сравнения в этом же формате не считалась.
**Проверки файла:** 113 971 908 строк = позиций шаблона; значения в [0, 1]; N-позиции = 0; 3 220 уровней.
**Лидерборд:** 2026-10-20 09:11 `950 jade-falcon-891d` (файл по времени и цифрам — v3 отдельно, alpha 0.55; Иван не называл файл): AUPRC 0.18066 / Spearman 0.36737, место сайта 283 (лучший наш 0.2577 / 0.566, место 24). Честный report дал 0.191 / 0.343, то есть оценка на report совпала с лидербордом.
**Теги:** `CNN`, `Ensemble`, `Raw` + свои `multi-species`, `BPNet`.
**Текст для формы (EN, 404 знаков):**
> Sequence-only BPNet/ProCapNet-style dilated CNN (profile and count heads), pretrained jointly on six species with per-species heads, then fine-tuned on ilPloInte3.2. Ensemble of two models (one trained with extra weight on over-predicted false-positive windows), rank-power alpha 0.55. No external data or third-party pretrained weights. Held-out contigs, unseen in training: AUPRC 0.191, Spearman 0.343.


## 2026-10-19 · Ofas_2.0 (клоп) · смесь m3ft (веса 0.5 / 0.3 / 0.7) + прежний сабмит, alpha 0.70

**Файлы:** `Ofas_2.0.predictions.blend_m3ft_w0.5_old.alpha0.70.txt.gz` (51.5 МБ), `..._w0.3_...` (51.2 МБ), `..._w0.7_...` (51.7 МБ). Kaggle `k13polinaa/epic-bug-blend-m3ft` (CPU) → Output → `bug_blend/`.
Скрипт `kaggle/v2_ab/epic-bug-blend-m3ft/blend.py` (как `tools/blend_moth_m3ft.py`, ветка `claude/week0-code`). Источники: `k13polinaa/epic-v2-m3ft-bug` (`test_lograte.npy`) и `k13polinaa/epic-v3-ofas-requant` (`test_final_score.npy`, это сабмит 0.6 v2 + 0.4 v3, лидерборд 0.2158 / 0.5015).

**Модель:** w × z(m3ft) + (1 − w) × z(прежняя смесь), rank-power alpha 0.70 (как у прежнего сабмита). Корреляция Spearman между входами 0.80 (2 млн позиций).
m3ft клопа: претрейн на 6 видах (`epic-v2-m3-pre2`, 80k шагов), затем ft на клопе 48k шагов, LR 6e-4, 2 сида (`epic-v2-m3ft-bug`, ≈2,5 ч на двух T4). Контиги report клопа в претрейн не входили.
**Оценка на report (честная, 60 млн позиций):** m3ft отдельно, alpha 0.65: AUPRC 0.2217, Spearman 0.5204 (сиды 0.188 / 0.192); базовая одиночная модель 0.1685 / 0.4543; прежняя смесь на report 0.217 / 0.506. Смесь на report не считали: веса заданы заранее по опыту моли (0.3–0.5 лучше).
**Проверки файла (все три):** 328 090 834 строки = позиций шаблона; значения в [0, 1]; N-позиции = 0; 1 720 уровней.
**Лидерборд (API сайта, 2026-10-19 17:40 UTC, 1-я половина теста; соответствие файлам уточняет Иван):** `944 velvet-raven-9da5` = файл `blend_m3ft_w0.5_old.alpha0.70` (подтвердил Иван) (08:39) **0.23205 / 0.49419** → места 14×6 = 84, место ~3 (прежний сабмит 0.2158 / 0.5015: 36×6 = 216). `943 misty-jay-cb3a` (07:23) = файл `Ofas_2.0.predictions.blend_m3ft_w0.3_old.alpha0.70.txt.gz` (подтвердил Иван; w0.3), `946 vivid-fox-7622` (14:13), `948 keen-gecko-efbe` (17:13) дали одинаковые 0.22896 / 0.47759 (15×13 = 195) — 946 и 948 — дубли w0.3 (Иван, 2026-10-19).
**Теги:** `CNN`, `Ensemble`, `Raw` + свои `multi-species`, `InitiationNet`, `BPNet`.
**Текст для формы (EN, 431 знаков, вес 0.5; для 0.3 и 0.7 заменить «0.5/0.5» на «0.3/0.7» и «0.7/0.3»):**
> Blend of sequence-only CNNs (z-scored, weights 0.5/0.5, rank-power alpha 0.70). A: InitiationNet-style CNN with attention, 2 seeds, pretrained on six species with per-species heads, fine-tuned on this species. B: earlier blend of an Ofas_2.0-only CNN (0.6) and a BPNet/ProCapNet-style CNN (0.4). No external data or third-party pretrained weights. Weights fixed in advance; A alone on held-out contigs: AUPRC 0.222, Spearman 0.520.

---

## 2026-10-19 · Ofas_2.0 (клоп) · m3ft отдельно, alpha 0.65

**Файл:** `Ofas_2.0.predictions.txt.gz`. Kaggle `k13polinaa/epic-v2-m3ft-bug` → Output → `epic_runs/Ofas_2.0/` (alpha 0.65 выбрана ядром по правилу KEEP_FRAC на tune).
**Модель:** v2 (InitiationNet + attention), претрейн на 6 видах + ft на клопе, 2 сида (`Ofas_2.0_m3ftbug_seed0/1.pt` там же).
**Оценка на report (честная):** AUPRC 0.2217, Spearman 0.5204 (одиночная база 0.1685 / 0.4543; прежняя смесь 0.217 / 0.506).
**Проверки файла:** ⚠ построчно не перепроверялся (файл записан тем же кодом, что прежние сабмиты, проверка на 3,2 млн позиций прошла). Если загружать этот файл, сначала запустить проверку на 328 090 834 строки.
**Лидерборд:** (заполнить после отправки).
**Теги:** `CNN`, `Ensemble`, `Raw` + свои `multi-species`, `InitiationNet`.
**Текст для формы (EN, 324 знаков):**
> Sequence-only InitiationNet-style CNN with attention (2-seed ensemble), pretrained jointly on six species with per-species heads (80k steps), then fine-tuned on Ofas_2.0 (48k steps); rank-power alpha 0.65. No external data or third-party pretrained weights. Held-out contigs, unseen in training: AUPRC 0.222, Spearman 0.520.

---

## 2026-10-18 · ilPloInte3.2 (моль) · смесь m3ft (вес 0.7) + прежний сабмит, alpha 0.70

**Файл:** `ilPloInte3.2.predictions.blend_m3ft_w0.7_old.alpha0.70.txt.gz` (32.9 МБ). Kaggle `k13polinaa/epic-moth-blend-w07` (CPU) → Output → `moth_blend/`. Скрипт как `tools/blend_moth_m3ft.py` с весом 0.7. Источники те же.
**Почему 0.7:** на лидерборде вес m3ft 0.5 (emerald-wolf, Иван подтвердил) дал 0.2577 / 0.56609, вес 0.3 (misty-gecko) 0.25797 / 0.55844; больший вес поднимал Spearman при том же AUPRC.
**Проверки файла:** 113 971 908 строк = позиций шаблона; значения в [0, 1]; N-позиции = 0; 1 720 уровней.
**Лидерборд:** 2026-10-18 18:13, AUPRC 0.25209 / Spearman 0.56709 (вес m3ft 0.7), место сайта 25. Вес 0.5 остаётся лучшим по произведению мест.
**Теги:** `CNN`, `Ensemble`, `Raw` + свои `multi-species`, `InitiationNet`, `BPNet`.
**Текст для формы сабмита (EN, ≤512 знаков, 401):**
> Blend of sequence-only CNNs (z-scored, weights 0.7/0.3, rank-power alpha 0.70). A: InitiationNet-style CNN with attention, 2 seeds, pretrained on six species with per-species heads, fine-tuned on this species. B: earlier blend of that CNN (0.6) and a BPNet/ProCapNet-style CNN (0.4). No external data or third-party pretrained weights. Weights fixed in advance; hold-out contigs were seen in training.

---

## 2026-10-18 · ilPloInte3.2 (моль) · смесь m3ft (вес 0.5) + прежний сабмит, alpha 0.70

**Файл:** `ilPloInte3.2.predictions.blend_m3ft_w0.5_old.alpha0.70.txt.gz`. Kaggle `k13polinaa/epic-moth-blend-m3ft` (версия 1, CPU) → Output → `moth_blend/`.
Скрипт `tools/blend_moth_m3ft.py` (ветка `claude/week0-code`). Источники: `k13polinaa/epic-v2-m3ft-moth` (`test_lograte.npy`) и `k13polinaa/epic-v3-moth-blend` (`test_final_score.npy`, это смесь 0.6 v2 + 0.4 v3 прежнего сабмита 0.2400 / 0.5414).

**Модель:** 0.5 × z(m3ft) + 0.5 × z(прежняя смесь), rank-power quantisation alpha 0.70 (как у прежнего сабмита). Корреляция Spearman между входами 0.80 (2 млн позиций).
Вес и alpha заданы заранее, не подобраны: обе модели видели все контиги моли, честной оценки нет. Оценка только по лидерборду, база 0.23752 / 0.55 (m3ft) и 0.2400 / 0.5414.

**Проверки файла:** 113 971 908 строк = позиций шаблона; значения в [0, 1]; N-позиции = 0; 1 720 уровней.

**Лидерборд:** загружен 2026-10-18 17:17–17:19, (по ответу Ивана) вес 0.5 = `emerald-wolf-9e3d` 0.2577 / 0.56609; вес 0.3 = `misty-gecko-ff20` 0.25797 / 0.55844; места 115×16=1840 и 117×8=936 (прежний сабмит 200×36=7200), место в зачёте ~9.

**Теги:** `CNN`, `Ensemble`, `Raw` + свои `multi-species`, `InitiationNet`, `BPNet`.

**Текст для формы сабмита (EN, ≤512 знаков, 401):**
> Blend of sequence-only CNNs (z-scored, weights 0.5/0.5, rank-power alpha 0.70). A: InitiationNet-style CNN with attention, 2 seeds, pretrained on six species with per-species heads, fine-tuned on this species. B: earlier blend of that CNN (0.6) and a BPNet/ProCapNet-style CNN (0.4). No external data or third-party pretrained weights. Weights fixed in advance; hold-out contigs were seen in training.

---

## 2026-10-18 · ilPloInte3.2 (моль) · смесь m3ft (вес 0.3) + прежний сабмит, alpha 0.70

**Файл:** `ilPloInte3.2.predictions.blend_m3ft_w0.3_old.alpha0.70.txt.gz`. Kaggle `k13polinaa/epic-moth-blend-m3ft` (версия 1, CPU) → Output → `moth_blend/`.
Скрипт `tools/blend_moth_m3ft.py` (ветка `claude/week0-code`). Источники: `k13polinaa/epic-v2-m3ft-moth` (`test_lograte.npy`) и `k13polinaa/epic-v3-moth-blend` (`test_final_score.npy`, это смесь 0.6 v2 + 0.4 v3 прежнего сабмита 0.2400 / 0.5414).

**Модель:** 0.3 × z(m3ft) + 0.7 × z(прежняя смесь), rank-power quantisation alpha 0.70 (как у прежнего сабмита). Корреляция Spearman между входами 0.80 (2 млн позиций).
Вес и alpha заданы заранее, не подобраны: обе модели видели все контиги моли, честной оценки нет. Оценка только по лидерборду, база 0.23752 / 0.55 (m3ft) и 0.2400 / 0.5414.

**Проверки файла:** 113 971 908 строк = позиций шаблона; значения в [0, 1]; N-позиции = 0; 1 720 уровней.

**Лидерборд:** (тот же результат, см. пару выше) загружен 2026-10-18 17:17–17:19, (по ответу Ивана) вес 0.5 = `emerald-wolf-9e3d` 0.2577 / 0.56609; вес 0.3 = `misty-gecko-ff20` 0.25797 / 0.55844; места 115×16=1840 и 117×8=936 (прежний сабмит 200×36=7200), место в зачёте ~9.

**Теги:** `CNN`, `Ensemble`, `Raw` + свои `multi-species`, `InitiationNet`, `BPNet`.

**Текст для формы сабмита (EN, ≤512 знаков, 401):**
> Blend of sequence-only CNNs (z-scored, weights 0.3/0.7, rank-power alpha 0.70). A: InitiationNet-style CNN with attention, 2 seeds, pretrained on six species with per-species heads, fine-tuned on this species. B: earlier blend of that CNN (0.6) and a BPNet/ProCapNet-style CNN (0.4). No external data or third-party pretrained weights. Weights fixed in advance; hold-out contigs were seen in training.

---

## 2026-10-18 · ilPloInte3.2 (моль) · v2 M-3 (претрейн на 6 видах + ft моли), alpha 0.65

**Файл:** `ilPloInte3.2.predictions.txt.gz` (34.3 МБ). Kaggle `k13polinaa/epic-v2-m3ft-moth` (версия 1) → Output → `epic_runs/ilPloInte3.2/`.
В датасет `epic-predictions` пока не выложен. Ветка `claude/week0-code`, коммит — см. запись в `git log` (ноутбук `epic_v2_model.ipynb`).

**Модель:** v2 InitiationNet (CNN + attention), 2 сида, среднее log-rate, rank-power quantisation с alpha 0.65.
Путь: претрейн на 6 видах (клоп + моль + устрица + акула + осьминог + Nematostella, `epic-v2-m3-pre2`, 80k шагов, ≈8,0 ч), затем ft
на моли (голова 1, 48k шагов, LR 6e-4, `epic-v2-m3ft-moth`, ≈2,6 ч на обе карты). Сид 0 и сид 1 обучались независимо, по одному на карту.
Предобученных внешних моделей и внешних данных нет.

**Оценка на report (ранги растянуты до размера теста):**

| | AUPRC | Spearman | честная? |
|---|---|---|---|
| сид 0 | 0.268 | 0.693 | нет: претрейн видел все контиги моли |
| сид 1 | 0.267 | 0.612 | нет |
| **ансамбль, alpha 0.65** | 0.307 | 0.671 | нет |
| старая v2 (смесь 10.10) | 0.386 | 0.780 | нет |

Оценка нечестная (претрейн на всех контигах моли), а у старой v2 утечка была сильнее, так что сравнивать эти числа нельзя.
Ориентир только лидерборд: нынешняя моль 0.240 / 0.541 (id 882).
tune (EMA, шаг 48000): AUPRC 0.244 / 0.247, Spearman 0.338 / 0.321.

**Проверки файла:** 113 971 908 строк = позиций шаблона; значения в [0, 1], NaN нет; официальный валидатор не запускали.

**Лидерборд:** 2026-10-18 11:45, `silver-crane-1ee1`, AUPRC 0.23752 / Spearman 0.55, место сайта 92; места 193 × 25 = 4825 (текущий 0.2400/0.5414: 6650), место в зачёте ~11.

**Теги:** `CNN`, `Ensemble`, `Raw` + свои `multi-species`, `InitiationNet`. `Finetune` не ставить: дообучали собственную модель, не предобученную чужую.

**Текст для формы сабмита (EN):**
> Ensemble of two sequence-only CNNs with attention (InitiationNet-style multi-scale CNN, 2 seeds, averaged log-rates), rank-power quantised (alpha = 0.65).
> Pretrained jointly on six species (Ofas_2.0, ilPloInte3.2, Magallana, shark, octopus, Nematostella training data) with per-species heads,
> then fine-tuned on ilPloInte3.2 only. No external data or pretrained third-party weights.
> Hold-out contigs of this species were seen during multi-species pretraining, so our own hold-out numbers are optimistic (AUPRC 0.307, Spearman 0.671).

---

## Правила ведения и сверка с сайтом (O-2, 2026-10-16)

`SUBMISSIONS.md` — **единственный источник правды по сабмитам**. `STATUS.md`, `MODELS.md`, `PLAN.md` ссылаются сюда и не дублируют. В каждую запись добавлять поля **«id сабмита на сайте»** (`submissionId` из API) и **«место на момент проверки»** (оба — по `GET https://epic.autosome.org/api/v1/leaderboard/species/<id>?limit=500&show=all`, id вида: `Ofas_2.0`, `ilPloInte3.2`, `cgigas_uk_roslin_v1-Heart`, `sSquSuc5.hap2`, `ASM119413v2-Kidney`).

Сверка 2026-10-16 (команда Mitochondrion, места AUPRC / Spearman среди всех сабмитов вида; модель сайт не показывает, состав — по записям ниже):

| вид | id | имя на сайте | дата (UTC) | AUPRC / Spearman | места AUPRC / Spearman | состав (по записям) |
|---|---|---|---|---|---|---|
| клоп | 743 | `ivory-fox-3812` | 2026-10-09 09:33 | 0.2158 / 0.5015 | 33 / 6 | смесь 0.6 v2 + 0.4 v3, α 0.70 (в зачёте) |
| клоп | 717 | `velvet-crane-a26d` | 2026-10-08 17:57 | 0.2025 / 0.4784 | 46 / 14 | `bug` 0–3, α 0.75 |
| клоп | 739 | `tidal-badger-a021` | 2026-10-09 08:54 | 0.2124 / 0.4430 | 38 / 59 | вероятно `bug+moth`, α неизвестна |
| клоп | 708 | `emerald-heron-c654` | 2026-10-08 04:56 | 0.1921 / 0.4476 | 58 / 55 | `bug` 0, 1, α 0.65 |
| моль | 882 | `velvet-hawk-3cb7` | 2026-10-15 10:47 | 0.2400 / 0.5414 | 181 / 34 | смесь v2moth+v3b2 / ft (в зачёте) |
| моль | 858 | `golden-otter-7711` | 2026-10-14 21:15 | 0.2475 / 0.5330 | 161 / 63 | см. записи моли |
| моль | 745 | `quiet-fox-8ca7` | 2026-10-09 12:33 | 0.1770 / 0.3777 | 252 / 259 | v3 |
| устрица | 863 | `amber-bison-094d` | 2026-10-14 22:00 | 0.1185 / 0.3483 | 31 / 11 | v2 ft (в зачёте) |
| устрица | 746 | `calm-eagle-874b` | 2026-10-09 13:08 | 0.0854 / 0.2245 | 76 / 83 | v3 |
| акула | 862 | `swift-raven-dcfb` | 2026-10-14 21:57 | 0.1027 / 0.4909 | 47 / 25 | v2 ft (в зачёте) |
| акула | 889 | `quiet-lemur-8f4f` | 2026-10-15 20:35 | 0.1206 / 0.4625 | 37 / 35 | смесь v2 ft2 + v3 |
| акула | 631 | `rapid-heron-1ad7` | 2026-10-04 21:39 | 0.0003 / 0.0170 | 96 / 95 | тест, не в счёт |
| осьминог | 871 | `keen-falcon-b0df` | 2026-10-15 09:26 | 0.0699 / 0.3858 | 26 / 14 | v3 (в зачёте) |
| осьминог | 883 | `velvet-eagle-1519` | 2026-10-15 10:54 | 0.0702 / 0.3826 | 24 / 16 | см. записи осьминога |
| осьминог | 906 | `silent-panda-feab` | 2026-10-16 21:57 | 0.0791 / 0.3645 | 21 / 37 | загружен 2026-10-16, состав уточнить у Ивана |
| осьминог | 864 | `misty-ibis-00df` | 2026-10-14 22:09 | 0.0538 / 0.2774 | 53 / 68 | см. записи осьминога |

Состав сабмитов с пометкой «см. записи» и «уточнить» агент по сайту установить не может: Иван при следующей отправке указывает имя сабмита в сообщении.

---

## 2026-10-15 · sSquSuc5.hap2 (акула) · смесь v2 ft2 + v3, alpha 0.55

**Файл:** `sSquSuc5.hap2.predictions.txt.gz` (216.6 МБ). Kaggle `k13polinaa/epic-blend-shark` → Output (на CPU по готовым предсказаниям). Запасные: `…alpha0.70.txt.gz`, `…alpha0.80.txt.gz` там же. В датасет `epic-predictions` не загружен.
Ветка `main`, коммит см. `git log`; ядро собрано из `epic_v3_bpnet.ipynb` в режиме `EPIC_REUSE_V3=1`.

**Модель:** смесь `z`, вес v3 = 0.3 (подбор на tune: AUPRC 0.1001 при 0.3, максимум 0.1011 при 0.4; выбран по сумме относительных AUPRC и Spearman); v2 ft2 (`k13polinaa/epic-v2-ft2-dogfish`, 2 сида, дообучена на акуле),
v3 (BPNet, `k13polinaa/epic-v3-shark-pred`, 1 сид v3b2, головы акулы). alpha 0.55 по tune.

**Оценка на report (честные контиги):**

| | AUPRC | Spearman |
|---|---|---|
| v2 ft2 | 0.102 | 0.337 |
| v3 | 0.057 | 0.246 |
| **смесь, alpha 0.55** | **0.110** | **0.337** |
| смесь, alpha 0.70 | 0.108 | 0.343 |

**Проверки файла:** строк = позиций шаблона (344 993 974); значения в [0, 1]; N-позиции = 0; 3222 уровня.
**Лидерборд:** заполнить после отправки (сейчас у акулы 0.103 / 0.491).
**Теги:** `CNN`, `Ensemble`, `Finetune`, `Raw` + свои `multi-species`, `BPNet`.

**Текст для формы сабмита (EN):**
> Ensemble of two models: v2 (multi-scale CNN with attention, fine-tuned on the training data of sSquSuc5.hap2 from a model pretrained on other EPIC species, 2 seeds) and v3 (BPNet/ProCapNet trained on EPIC training data, 1 seed), averaged as z-scores with weight 0.3 for v3. Rank-power alpha = 0.55.


## 2026-10-10 · ASM119413v2 (осьминог) · v3 на своих данных + устрица, alpha 0.5

**Файл:** `ASM119413v2.predictions.txt.gz` (в датасете `ivanmitochondrion/epic-predictions`: `submissions/ASM119413v2.predictions.v3.alpha0.50.txt.gz.bin`).
Kaggle `k13polinaa/epic-v3-octopus-pred` → Output → `epic_v3/ASM119413v2/` (веса из `k13polinaa/epic-v3-octopus`).
Ветка `claude/youthful-hopper-hjme9f`, коммит `bd1214e`, ноутбук `epic_v3_bpnet.ipynb` с `EPIC_SPECIES=ASM119413v2`, `EPIC_TAG=v3b2`
(обучение: `EPIC_AUX=cgigas_uk_roslin_v1`).

**Модель:** v3 (BPNet / ProCapNet: dilated CNN, общее тело, головы «профиль» и «счёт» на каждый вид), обучена с нуля на train-данных
осьминога + устрицы (тоже моллюск), 30k шагов, LR 5e-4, 2 сида (оба в ансамбле); tune/report отложены у обоих видов. Предсказание головами осьминога.
alpha 0.5 выбран правилом по tune. Внешних данных и предобученных весов нет.

**Оценка на report (честная; выборка 30 Мб из 172 Мб группы, ранги растянуты до размера теста):**

| | AUPRC | Spearman |
|---|---|---|
| перенос v2 с насекомых (`epic-v2-transfer-octopus`, report) | 0.007 | 0.107 |
| бейзлайн CNN из write-up организаторов (их отложенные данные, сравнение примерное) | 0.068 | 0.291 |
| v3 seed 0 / seed 1 | 0.060 / 0.051 | 0.194 / 0.193 |
| **этот сабмит (2 сида)** | **0.063** | **0.198** |

Сетка alpha на tune: 0.5 → 0.050 / 0.306; 0.70 → 0.047 / 0.336 (файл `…alpha0.70.txt.gz` в том же Output).

**Проверки файла:** 416 885 800 строк = позиций шаблона; значения в [0, 1]; N-позиции = 0 (67 011 820 N); 4 068 уровней; 264.1 МБ.

**Лидерборд:** —

**Теги:** `CNN`, `Ensemble`, `Raw` + свои `multi-species`, `BPNet`.

**Текст для формы сабмита (EN):**
> BPNet/ProCapNet-style dilated CNN (sequence only) with profile (multinomial) and count (MSE) heads; shared body, per-species heads,
> trained from scratch on ASM119413v2 + Magallana (cgigas_uk_roslin_v1) training data; 2 seeds averaged, rank-power quantised (α = 0.5).
> No external data or pretrained weights. Hold-out contigs (ASM119413v2): AUPRC 0.063, Spearman 0.198.

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
