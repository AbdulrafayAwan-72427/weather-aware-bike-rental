# Weather-Aware Bike Rental Demand Prediction — Technical Report

**Course:** Artificial Intelligence (Semester Project) — BSAI, Fall 2026,
Faculty of Computing, Riphah International University, Islamabad.
**Instructor:** Anees Tariq. **Section:** AI 3-1. **Due:** 21 December 2026.
**Student:** Abdul Rafay Awan (SAP ID 72427).
**Status:** Working first version, built and verified 15 November 2026,
then re-verified the same day in a fresh environment (see §10).
All results below are from actual runs on this machine; nothing is invented.
Results in §8 are from the regenerated run (§10, Run 3); Runs 1–2 produced
identical MAE/RMSE values, as recorded in the run log.

---

## 1. Problem statement

Seoul's public bike-sharing system needs to know, hour by hour, how many
bikes will be rented across the whole system, so bikes can be available at
the right time and place. Demand changes sharply with the hour of day, the
season, the weather, and holidays, so a fixed schedule is not enough.

**Task.** Predict the total number of observed bike rentals across the
Seoul system during one hour, given the hour, calendar information,
weather measurements, holiday status, and whether the service is operating.

**What this is not.** It is a regression problem, not a classification
problem. It does not predict availability at any individual station, and
it does not measure unmet demand: the target is *observed* rentals, which
can be limited by how many bikes were available. A prediction of 500 does
not mean only 500 people wanted a bike.

## 2. Objectives

1. Build the complete AI lifecycle: requirements → dataset → preprocessing
   and EDA → model design → chronological evaluation → documented,
   runnable local application.
2. Compare an hourly-mean baseline, Linear Regression, and a Random Forest
   Regressor under identical, leakage-free conditions and select the model
   on validation evidence only.
3. Deliver a working local Flask interface that loads the saved model and
   estimates rentals for a weather scenario, with input validation.
4. Report results honestly, including where the model is weak, so the
   evaluation — not the headline score — is the basis for judgement.

## 3. Functional requirements

| ID | Requirement | Where it is met |
|----|-------------|-----------------|
| FR1 | Predict total hourly rentals (integer ≥ 0) from hour, date, weather, holiday, operating status | `bike/model.py`, `bike/app.py` |
| FR2 | Reproducible dataset acquisition with schema and integrity checks | `bike/data.py` |
| FR3 | Exploratory analysis with charts (hour, season, weather, distribution) | `bike/eda.py` → `reports/eda_*.png` |
| FR4 | Chronological train/validation/test split; no random split of the time series | `bike/train.py` |
| FR5 | Preprocessing fitted on training data only; same pipeline used for training and for the app | `bike/model.py` (`ColumnTransformer` inside each `Pipeline`) |
| FR6 | Compare baseline, Linear Regression, Random Forest; select on validation MAE | `bike/train.py`, `reports/metrics.json` |
| FR7 | Report MAE and RMSE, actual-vs-predicted charts, peak-hour and weather/season error analysis with sample sizes | §7, `reports/` |
| FR8 | Local web interface: labelled inputs with units, defaults, validation, saved-model loading (no retraining per prediction) | `bike/app.py`, `templates/index.html` |
| FR9 | Non-operating hours predict exactly 0; negative predictions clipped to 0 | `bike/model.py: predict()` |
| FR10 | Tests covering chronology, column validation, validation, closure, invalid inputs, saved-model loading, end-to-end prediction | `tests/test_project.py` (21 tests) |
| FR11 | Documentation and GitHub-ready structure (README, `.gitignore`) | `README.md`, this report |

## 4. Assumptions

- The hourly weather values in the dataset are treated as the conditions
  for that hour. In the app, a user types in a *scenario* (for example,
  "18 °C, 55% humidity at 18:00"); this is not a forecast. A genuine
  forecast would need weather estimates that are available *before* the
  hour, which this dataset does not provide.
- Season, weekday, and month are derived deterministically from the date
  (season by calendar month: Dec–Feb Winter, Mar–May Spring, Jun–Aug
  Summer, Sep–Nov Autumn). This matches the dataset's own season labels.
- Rentals during non-operating hours are structurally zero, not missing
  data, so they are handled by a rule (FR9) and models are trained on
  operating hours only.
- One year of data is enough to demonstrate the lifecycle and to compare
  models, but not enough to prove seasonal generalisation (§8.4).

## 5. Constraints

- Solo student project; basic Python and scikit-learn Linear Regression
  background; target working version in 2–3 focused days.
- Student laptop: modest tuning only (one Random Forest configuration,
  250 trees), no GPU, no paid APIs, no cloud deployment, no authentication.
- Single historical year (8,760 hours) from one city (Seoul).
- No live weather feed; the app is a scenario estimator, not a forecasting
  service.

## 6. Dataset

### 6.1 Source and justification

Seoul Bike Sharing Demand, UCI Machine Learning Repository, dataset 560:
https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand

- **Citation:** Seoul Bike Sharing Demand [Dataset]. (2020). UCI Machine
  Learning Repository. https://doi.org/10.24432/C5F62R
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Why this dataset:** it is the standard public benchmark for exactly
  this task (hourly system rentals + weather + holiday), it is labelled a
  regression task by UCI, it is a complete single year with no missing
  values, and its size (8,760 rows) fits a student laptop. No alternative
  dataset was compared in this project, so no claim is made about how
  other datasets would perform.

### 6.2 Schema (verified on download)

| Column (renamed) | Original column | Type | Notes |
|---|---|---|---|
| date | Date | date (dd/mm/yyyy in file) | 1 Dec 2017 – 30 Nov 2018 |
| rentals | Rented Bike Count | int | **target**, never an input |
| hour | Hour | int 0–23 | |
| temperature | Temperature (°C) | float | −17.8 to 39.4 |
| humidity | Humidity (%) | int | 0–98 |
| wind_speed | Wind speed (m/s) | float | 0–7.4 |
| visibility | Visibility (10 m) | int | 27–2000 |
| dew_point | Dew point temperature (°C) | float | −30.6 to 27.2 |
| solar | Solar Radiation (MJ/m²) | float | 0–3.52 |
| rainfall | Rainfall (mm) | float | 0–35 |
| snowfall | Snowfall (cm) | float | 0–8.8 |
| season | Seasons | Winter/Spring/Summer/Autumn | |
| holiday | Holiday | Holiday / No Holiday | 432 holiday hours |
| operating | Functioning Day | Yes / No | 295 "No" hours |

### 6.3 Validation checks (all passed)

`bike/data.py: load_data()` downloads the official ZIP, then *refuses to
proceed* unless every check passes:

- The **original column names and their order** are checked against the
  14 UCI headers (`validate_and_rename()` in `bike/data.py`) before any
  renaming happens. A missing, unexpected, or misordered column raises a
  `ValueError` naming the problem; columns are never renamed by position
  on an unchecked header. (The file is read as cp1252, not UTF-8: the
  degree sign in `Temperature(°C)` is byte 0xB0.)
- 8,760 rows; 0 missing cells.
- Timestamps form a complete hourly sequence with no duplicates and no
  gaps (1 Dec 2017 00:00 → 30 Nov 2018 23:00).
- Rentals are never negative; operating is only Yes/No; **every**
  non-operating hour has exactly 0 rentals (295 hours).
- SHA-256 of the downloaded CSV:
  `373339b71a8935d69e9af0abf26a70744632119862eeb3919efb389a7b749c60`

No missing values were invented, imputed for demonstration, or dropped:
there were none. The only "cleaning" is renaming columns and parsing the
date — the data is used as supplied.

### 6.4 Exploratory analysis (key findings)

Full charts: `reports/eda_hourly.png`, `eda_season.png`,
`eda_temperature.png`, `eda_distribution.png`,
`eda_weather_holiday.png`, `eda_monthly.png`; numbers:
`reports/eda_summary.json`.

- **Target distribution is right-skewed:** mean 704.6, median 504.5,
  max 3,556 rentals/hour. A few very busy hours pull the mean up; MAE is
  therefore the primary selection metric (RMSE is reported alongside it
  because it penalises the large peak-hour misses).
- **Strong daily pattern:** operating-hour means peak at 18:00 (1,554),
  with a morning peak at 08:00 (1,050); the quietest hour is 04:00 (137).
  Commuting hours dominate demand.
- **Strong seasonal pattern:** operating-hour means — Summer 1,034,
  Autumn 924, Spring 746, Winter 226. Winter demand is roughly a quarter
  of summer demand.
- **Temperature matters most:** correlation with rentals +0.54; mean
  rentals rise steadily with temperature band. Dew point +0.38 and solar
  radiation +0.26 are also positive; humidity −0.20, rainfall −0.12 and
  snowfall −0.14 are negative.
- **Wet hours are much quieter:** wet operating hours (rain or snow,
  n = 931) average 178 rentals vs 797 for dry hours — but wet hours are
  concentrated in colder months, so this is partly a seasonal effect, not
  a pure rain effect.
- **Holidays are quieter:** 529 vs 739 rentals/hour on non-holidays.
- **Operating hours never record 0 rentals** (minimum operating-hour
  count is above 0), while all 295 closed hours are exactly 0 — this is
  what justifies the closure rule in FR9 rather than letting the model
  learn it.

## 7. Method

### 7.1 Splits (chronological, no shuffling)

| Split | Dates | Hours | Closed hours |
|---|---|---|---|
| Train | 1 Dec 2017 00:00 – 13 Aug 2018 11:00 | 6,132 (70%) | 48 |
| Validation | 13 Aug 2018 12:00 – 6 Oct 2018 23:00 | 1,308 (15%) | 151 |
| Test | 7 Oct 2018 00:00 – 30 Nov 2018 23:00 | 1,320 (15%) | 96 |

A random split would leak the future into training (adjacent hours are
nearly identical) and would flatter every model. The chronological split
is the honest design — but it has a real consequence, discussed in §8.4:
**the training period contains no autumn at all** (Winter 2,160 h,
Spring 2,208 h, Summer 1,764 h), validation is Summer + Autumn, and the
test period is 100% Autumn.

### 7.2 Preprocessing and feature engineering

One shared `ColumnTransformer`, fitted on training data only and stored
inside each model's `Pipeline`, so training and the Flask app use the
identical transformation:

- **Numeric (8 weather + 2 hour-cycle features):** median imputation
  (a safety net for the app — the training data has no missing values)
  followed by standard scaling.
- **Categorical (hour, weekday, month, season, holiday):** one-hot
  encoding with `handle_unknown="ignore"`, so an unseen category (for
  example Autumn, which never occurs in training) becomes all zeros
  instead of crashing.
- **Hour as sin/cos:** `hour_sin`, `hour_cos` encode the circular nature
  of the hour (23:00 is adjacent to 00:00). The raw hour is also kept as
  a category so the model can learn the sharp 08:00 and 18:00 peaks.
- **Derived, never from the target:** weekday, month, season, and the
  sin/cos features come only from the date/hour. No feature uses the
  rental count, a lag, or a rolling mean, so there is no target leakage.
- **Operating status is not a model input.** It is applied afterwards as
  the closure rule: non-operating → 0, otherwise `max(prediction, 0)`.

### 7.3 Models compared

1. **Hourly-mean baseline** — for each hour of day, the mean rental count
   in training data (operating hours). No weather is used. This is the
   "what if we knew nothing about the weather" reference; any real model
   must beat it.
2. **Linear Regression** — the interpretable model: one coefficient per
   (scaled/encoded) feature. Fast and transparent. Because hour is
   one-hot encoded, it *can* represent the hourly pattern: each hour gets
   its own coefficient, so the sharp 08:00 and 18:00 peaks are within its
   reach. Its real limitations are different: it cannot represent
   *interactions* (for example, the effect of temperature differing by
   hour or by season) or *non-linear* weather effects (for example,
   rentals rising with temperature and then falling in extreme heat)
   without hand-made features, and it extrapolates linearly outside the
   range seen in training — which is what hurt it on the unseen autumn
   test period (§8.4).
3. **Random Forest Regressor** — 250 trees, `min_samples_leaf=3`,
   `max_features=0.8`, `random_state=42`, `n_jobs=-1`. Chosen as the
   non-linear candidate because it handles interactions and thresholds
   (for example, "cold *and* wet") without manual feature design, at a
   modest size for a laptop. Only this single configuration was tried:
   tuning on the validation set with a one-year dataset would overfit the
   validation period.

### 7.4 Selection rule

The model with the lowest **validation MAE** is selected, and its
training fit is frozen — it is not refitted on validation or test data.
Test results are reported afterwards, once, for the record. This is
recorded in `reports/metrics.json` (`"selection": "lowest validation
MAE; frozen training fit, no refit"`).

## 8. Evaluation

### 8.1 Headline results (MAE / RMSE in rentals per hour)

| Model | Validation MAE | Validation RMSE | Test MAE | Test RMSE | Fit time |
|---|---:|---:|---:|---:|---:|
| Hourly-mean baseline | 420.3 | 566.3 | 278.6 | 397.5 | <0.01 s |
| Linear Regression | 302.2 | 422.9 | 286.4 | 387.2 | 0.16 s |
| **Random Forest (selected)** | **162.2** | **250.9** | **243.3** | **346.4** | 6.4 s |

Fit times are the per-model `fit_seconds` recorded in
`reports/metrics.json` for this run. They vary between runs: the
fresh-environment reproduction (§10, Run 3) recorded 0.10 s for Linear
Regression and 6.9 s for Random Forest with identical MAE/RMSE, so
timings should be read as approximate, not as exact benchmarks.

Operating-hours-only test scores: baseline 300.5 / 412.8, Linear
308.8 / 402.1, Random Forest **262.4 / 359.7**. The full test set
(including the 96 closed hours, which every model predicts exactly as 0)
is the headline figure; the operating-only figure is the fairer measure
of the learned model.

**Selection:** Random Forest, by validation MAE (162.2 vs 302.2 for
Linear and 420.3 for the baseline — roughly half the error of Linear
Regression on validation). It also wins on the final test, but by a
smaller margin (243.3 vs 286.4), which is itself informative (§8.4).

### 8.2 Error analysis (selected model, final test, operating hours only)

| Segment | n | MAE | RMSE | Reading |
|---|---:|---:|---:|---|
| Peak hours (07–09, 17–19) | 306 | 482.3 | 584.9 | Hardest: counts are highest and most variable |
| Off-peak hours | 918 | 189.1 | 241.9 | Much easier; most of the day is predictable |
| Wet hours (rain or snow) | 76 | 163.0 | 225.0 | Low counts, so low absolute error — do not over-read |
| Dry hours | 1,148 | 269.0 | 366.9 | |
| Autumn | 1,224 | 262.4 | 359.7 | The entire test set |
| Winter / Spring / Summer | 0 | — | — | **Cannot be evaluated on the test set** |

Charts: `reports/test_actual_vs_predicted.png` (points sit below the
diagonal above ~1,500 rentals — systematic under-prediction of busy
hours), `reports/test_week.png` (first test week, observed vs predicted),
`reports/test_error_by_hour.png`. Per-hour predictions for every test
hour: `reports/test_predictions.csv`.

### 8.3 What the model learned

Random Forest feature importances (top): temperature 0.374, hour_sin
0.199, humidity 0.077, hour_cos 0.070, solar radiation 0.059,
season_Winter 0.045, rainfall 0.034, dew point 0.028. This agrees with
the EDA: temperature and time of day dominate; weather is a modifier,
not the whole story. (Importance is descriptive, not causal — temperature
also carries the seasonal signal.)

### 8.4 The honest limitation: validation and test disagree

Validation MAE (162.2) is much better than test MAE (243.3), and on the
test period the model **under-predicts operating hours by 233 rentals/hour
on average** (predicted mean 614.5 vs observed 847.7). Two structural
reasons, both visible before any modelling:

1. **No autumn in training.** The model never saw an autumn hour while
   learning. Autumn in Seoul is warm enough to ride but has different
   daylight and commuting patterns from summer; the one-hot season
   feature for Autumn is all zeros at prediction time, so the model must
   infer autumn demand from temperature and hour alone.
2. **One year only, with this split.** With a single year of data and
   our chosen 70/15/15 chronological split, the test set is a single
   season. Seasonal performance therefore cannot be measured — the
   segment table above has n = 0 for three seasons — and the test score
   is a score *for autumn 2018*, not a general accuracy claim. (A
   different split could put autumn in training, but only by shrinking
   the validation or test period; the split was fixed before training
   and was not changed to improve scores.)

A second, smaller observation: on the final test, Linear Regression
(286.4) is *worse* than the hourly-mean baseline (278.6), although it was
clearly better on validation (302.2 vs 420.3). The baseline is robust to
seasonal shift because it ignores weather entirely; the linear model
extrapolates its fitted weather coefficients into an unseen season and
pays for it. This is exactly why the baseline is in the comparison.

### 8.5 Trade-offs

| | Baseline | Linear Regression | Random Forest |
|---|---|---|---|
| Accuracy (val / test MAE) | 420 / 279 | 302 / 286 | **162 / 243** |
| Interpretability | trivial | coefficients readable | importances only; individual predictions opaque |
| Training cost | <0.01 s | 0.16 s | 6.4 s; 37 MB saved file |
| Prediction cost | instant | instant | instant (loaded once) |
| Robustness to unseen season | high (ignores weather) | low (linear extrapolation) | medium (under-predicts, but still best) |

Random Forest is selected because it is clearly the most accurate on both
validation and test, its training cost is trivial, and its opacity is
acceptable for a planning prototype — but if the requirement were "explain
every prediction to a stakeholder", Linear Regression would be the choice
despite its worse error.

## 9. Application

`python -m bike.app` serves a local Flask page at
http://127.0.0.1:5000/ (loopback only, no authentication, no external
services). The form asks for date, hour, the eight weather values with
units, holiday, and operating status, with sensible defaults
(15 Oct 2018, 18:00, 18 °C, 55% humidity — a busy evening hour). The page:

- validates every input (ranges from the dataset's physical limits, hour
  0–23, dew point ≤ temperature) and shows errors without crashing;
  submitted values stay visible after a prediction or an error;
- warns when a weather value is outside the training range;
- loads `artifacts/model.joblib` once at startup and never retrains
  (verified in tests: the model file's hash is unchanged after requests);
- shows the estimated hourly rentals, the measured test MAE/RMSE table
  read directly from `reports/metrics.json` (so the page cannot disagree
  with the evaluation outputs), and a clear label: *historical weather
  scenario, not a live forecast or a station-availability estimate*;
- returns exactly 0 when the service is not operating;
- offers a **Load example scenario** button, which fills in the
  documented example (15 Oct 2018, 18:00, 18 °C, 55% humidity) *without*
  calculating a prediction — the user still presses "Estimate hourly
  rentals" afterwards. Before any submission the prediction panel shows
  "—" with the text "No estimate yet — enter a scenario (or load the
  example) and press 'Estimate hourly rentals'." That empty state is
  intended, not a failed prediction.

**Smoke check (15 Nov 2026, live server, fresh environment):**
GET `/` → 200, showing "—" and "No estimate yet"; GET `/?example=1` →
200, example inputs loaded, still "—" (no prediction calculated);
POST default scenario → **2,100 rentals** (the real 15 Oct 2018 18:00
hour had 2,635 observed at 66% humidity; the model's own prediction for
that exact hour was 1,911 — the 2,100 figure is consistent with the known
under-prediction bias, not a bug); POST changed scenario (08:00, 30 °C)
→ **1,219 rentals**; operating = No → **0**; humidity 101 → rejected
with "Humidity must be between 0 and 100" and the submitted value 101
kept visible in the form.

## 10. Verification

### 10.1 Run log — distinguishing reported from reproduced results

| Run | When (15 Nov 2026) | Environment | What was done | Outcome |
|---|---|---|---|---|
| Run 1 — initial build | ~11:55 PKT | Original venv (Python 3.12.3) | Download, EDA, training, tests, live-server smoke check | Selected Random Forest; val MAE 162.2, test MAE 243.3; 12/12 tests passed |
| Run 2 — clean re-run | ~11:58 PKT | Same venv | Deleted `data/`, `artifacts/`, `reports/`; re-ran EDA + training + tests | Identical MAE/RMSE and dataset SHA-256; 12/12 tests passed |
| Run 3 — fresh-environment reproduction | ~12:36 PKT | **New venv, exact pinned `requirements.txt`** (Flask 3.1.3, scikit-learn 1.9.1, pandas 2.3.3, matplotlib 3.11.2, joblib 1.6.0, pytest 9.2.0, numpy 2.5.4) | Fresh `pip install`, deleted `data/`/`artifacts/`/`reports/`, re-ran EDA + training + tests + live-server smoke check | **Identical MAE/RMSE to four decimal places** and identical dataset SHA-256; per-model fit times differed (Linear 0.10 s vs 0.16 s; Random Forest 6.9 s vs 6.4 s), confirming timings vary; **21/21 tests passed** (9 new tests added in this revision) |

The tables in §8 and the saved `reports/metrics.json` in the archive are
from **Run 3** — the independently reproduced run, not only the first
build. Runs 1 and 2 are the earlier reported results; they agree with
Run 3 on every metric, which is why the headline numbers did not change
in this revision. Only the fit times, the test count, and the
documentation changed.

### 10.2 Checks (Run 3, fresh environment)

| Check | Result |
|---|---|
| Dependency installation in a fresh environment (`pip install -r requirements.txt`) | Passed; freeze confirms the pinned versions above |
| Dataset download + original-column/schema/sequence/missing/closure checks | Passed (8,760 rows, 0 missing, SHA-256 recorded in §6.3) |
| EDA (`python -m bike.eda`) | Passed in ~6 s; 6 charts + `eda_summary.json` |
| Training (`python -m bike.train`) | Passed in ~14 s; `metrics.json`, `test_predictions.csv`, 5 evaluation charts |
| Tests (`python -m pytest tests/ -q`) | **21 passed** — split integrity, original-column validation (4 cases), bad-input rejection (8 cases), calendar derivation, baseline/closure behaviour, saved-model loading, full app round-trip, example scenario, metrics-table agreement, no-retraining check |
| Flask smoke check (live server, curl) | Passed — see §9 |
| Charts inspected visually | `test_actual_vs_predicted.png`, `eda_hourly.png` render correctly |

To reproduce locally: follow the README quick start (PowerShell commands
included). Order matters: download/EDA → train → tests → app, because
the tests and the app need the trained model file.

### 10.3 Remaining unverified step

The prediction flow was verified with the Flask test client and with
`curl` against the live local server, but **not yet interactively in a
real browser**. The remaining manual check is: on your own laptop, open
http://127.0.0.1:5000/, press **Load example scenario**, then
**Estimate hourly rentals**, and confirm 2,100 appears. Everything else
in §10.2 ran and passed in this environment.

## 11. Ethics and responsible use

- **No personal data.** The dataset contains no individuals, locations
  more precise than the city, or anything that could identify a rider.
- **Observed ≠ wanted.** The target is rentals that happened. Using the
  model to claim "demand was low" in a neighbourhood or hour would be
  wrong if bikes were simply unavailable; supply limits are unobserved.
- **No transfer claim.** The model is trained on Seoul 2017–18. Applying
  it to Lahore, Karachi, or any Pakistani city without local data and
  validation would be misleading — climate, commuting patterns, holidays,
  and the bike system itself all differ.
- **Planning aid, not an operational promise.** An error of ~240
  rentals/hour (test MAE) is large relative to a quiet hour (~140
  rentals at 04:00). The model supports rough capacity planning, not
  exact staffing.
- **Attribution.** The dataset is CC BY 4.0; the source and DOI are cited
  in the README, the app footer, and §6.1.

## 12. Limitations

1. **One year, one city.** Seasonal generalisation is unproven; with
   this split the test set is a single season and training never saw
   autumn (§8.4).
2. **Under-prediction of busy hours.** Peak-hour test MAE is 482.3 and
   the model systematically under-predicts high-demand hours.
3. **No true forecasting.** Inputs are observed weather for the same hour,
   not forecasts available in advance; the app estimates scenarios.
4. **No supply, event, or station data.** Road closures, events, station
   outages, and bike availability are not in the data and cannot be
   predicted.
5. **Single Random Forest configuration.** No hyperparameter search was
   run; a tuned model might do better, but tuning on this short validation
   period risks overfitting it.
6. **Hourly-mean baseline beats Linear Regression on the test set** —
   a reminder that the ranking is period-dependent, not universal.

## 13. Future improvements

- Collect a second year (or use a multi-year dataset) so training can
  include every season and the test can cover more than one.
- Add lag features (previous hour's rentals) and a proper time-series
  validation scheme — carefully, since lags must not leak the target.
- Try gradient boosting and a tuned Random Forest, selected on a
  rolling-origin validation instead of a single split.
- Model the peak hours separately, or predict with a log-transformed
  target to handle the right-skewed distribution.
- Replace typed-in weather with a weather-forecast API for genuine
  next-hour forecasts, clearly labelled as forecast-based.
- Add a simple confidence/interval display (for example, quantile
  regression) so users see the uncertainty instead of a single number.

## 14. Expected outcomes (against the brief)

The project delivers the full lifecycle the brief requires: a justified
dataset with verified EDA, three compared models with a validation-based
selection, a working local application, 21 passing tests, and this
documentation. The headline result is a Random Forest with test MAE
243.3 rentals/hour — better than both alternatives, but with a documented
autumn under-prediction bias that a longer dataset would be needed to
remove. That limitation is part of the result, not a footnote to it.

## Appendix A — Reproduction commands (Windows PowerShell)

Run each command separately, in order, from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m bike.eda
python -m bike.train
python -m pytest tests/ -q
python -m bike.app
```

If PowerShell blocks the activation script, first run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

On Linux or macOS, replace the activation line with
`source .venv/bin/activate`.

## Appendix B — Files delivered

```
bike/data.py, bike/eda.py, bike/model.py, bike/train.py, bike/app.py
templates/index.html
tests/test_project.py
requirements.txt, pyproject.toml, .python-version, .gitignore
README.md
docs/Technical-Report.md            (this file)
reports/eda_*.png, reports/eda_summary.json
reports/hourly_profile.png, reports/monthly_profile.png
reports/test_week.png, reports/test_actual_vs_predicted.png,
reports/test_error_by_hour.png
reports/metrics.json, reports/test_predictions.csv,
reports/training_correlations.csv, reports/dataset_summary.csv
artifacts/model.joblib              (37 MB, regenerated by bike.train)
data/SeoulBikeData.csv              (downloaded; excluded from Git)
```
