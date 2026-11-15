# Weather-Aware Bike Rental Demand Prediction

## Technical Report

**Student:** Abdul Rafay Awan  
**SAP ID:** 72427  
**Programme:** BS Artificial Intelligence, Semester 3  
**Course:** Artificial Intelligence, Fall 2026  
**Section:** AI 3-1  
**Instructor:** Anees Tariq  
**Institution:** Riphah International University, Faculty of Computing, Islamabad  
**Submission deadline:** 21 December 2026  

**GitHub repository:** [weather-aware-bike-rental](https://github.com/AbdulrafayAwan-72427/weather-aware-bike-rental)

---

## Abstract

This project estimates the total number of observed rentals across Seoul’s public bike-sharing system during one hour. It uses weather measurements, calendar information, holiday status and operating information from the UCI Seoul Bike Sharing Demand dataset.

The dataset contains 8,760 hourly records covering December 2017 to November 2018. Three approaches are compared: a historical hourly-mean baseline, Linear Regression and a Random Forest Regressor. The data is separated chronologically into training, validation and test periods. All learned preprocessing is fitted on operating training hours only, and the model with the lowest validation mean absolute error is selected.

Random Forest achieves the lowest validation MAE of 162.2 rentals per hour. On the final test period, its MAE is 243.3 and RMSE is 346.4. The baseline achieves test MAE of 278.6, while Linear Regression achieves 286.4. Peak hours remain difficult, with operating-hour test MAE of 482.3.

A local Flask interface loads the saved model and estimates rentals for user-entered weather scenarios. The main limitation is seasonal coverage: training contains no autumn observations, while the final test period is entirely autumn. The project therefore demonstrates a historical scenario estimator rather than an evaluated operational forecasting service.

---

## 1. Problem Statement

Hourly bicycle rentals vary with time of day, weather, calendar patterns and holidays. A bike-system planner may need to estimate how rental activity changes under different conditions.

A fixed historical hourly average provides a useful reference, but it cannot adapt its prediction to temperature, rainfall or other weather inputs. This project investigates whether a learned model using those additional inputs improves on that reference.

### 1.1 Prediction Task

Predict the total number of observed rentals across Seoul’s public bike-sharing system during one hour.

The target column is `Rented Bike Count`.

The interface accepts:

- Date and hour.
- Temperature.
- Humidity.
- Wind speed.
- Visibility.
- Dew point temperature.
- Solar radiation.
- Rainfall.
- Snowfall.
- Holiday status.
- Operating status.

Date supplies calendar features. Operating status controls a closure rule rather than entering the learned estimator.

### 1.2 Intended User

The intended user is a planner exploring hourly rental scenarios for the Seoul system.

The project estimates a system-wide total. It does not identify which individual stations need bicycles or predict station-level inventory.

### 1.3 Meaning of Demand

The dataset measures completed rentals, not everyone who wanted to rent a bicycle.

If bicycles were unavailable, some desired trips would not appear in the target. A low rental count can therefore reflect low demand, limited supply or both.

The project uses “demand prediction” as its application context, but the measurable target is **observed hourly rentals**.

---

## 2. Objectives

1. Define a real-world regression problem with measurable evaluation criteria.
2. Acquire and justify an appropriate public dataset.
3. Validate the data and describe its main patterns through exploratory analysis.
4. Compare an hourly-mean baseline, Linear Regression and Random Forest.
5. Use chronological training, validation and test periods.
6. Fit learned preprocessing on training data only.
7. Select the model using validation MAE.
8. Report final held-out MAE and RMSE.
9. Analyse peak-hour, weather and seasonal errors with sample sizes.
10. Provide a local interface that uses the same saved preprocessing and prediction logic.
11. Document limitations, ethical considerations and reproduction steps.
12. Maintain the source, evaluation evidence and documentation in GitHub.

Success is defined by a reproducible comparison and a working demonstration with honest error reporting. No unsupported accuracy-percentage target is assumed.

---

## 3. Functional Requirements

| ID | Requirement | Implementation or evidence |
|---|---|---|
| FR1 | Estimate total hourly observed rentals from weather and calendar inputs | `bike/model.py`, `bike/app.py` |
| FR2 | Acquire and validate the dataset | `bike/data.py` |
| FR3 | Produce exploratory charts and summary statistics | `bike/eda.py`, `reports/eda_*.png`, `reports/eda_summary.json` |
| FR4 | Separate training, validation and test observations chronologically | `bike/train.py` |
| FR5 | Fit preprocessing on training data only | `ColumnTransformer` inside each learned model pipeline |
| FR6 | Compare the baseline, Linear Regression and Random Forest | `bike/model.py`, `reports/metrics.json` |
| FR7 | Select the model using validation MAE | `bike/train.py` |
| FR8 | Report held-out MAE, RMSE and segment errors | `reports/metrics.json`, `reports/test_predictions.csv` |
| FR9 | Return exactly zero for non-operating hours | Shared `predict()` function |
| FR10 | Reject invalid scenario inputs and warn about values outside training ranges | `validate()` and request handling in `bike/app.py` |
| FR11 | Load the saved model without retraining during requests | `create_app()` in `bike/app.py` |
| FR12 | Provide tests and reproduction instructions | `tests/test_project.py`, `requirements.txt`, README |
| FR13 | Provide professional documentation and repository evidence | `docs/`, README and GitHub repository |

Predictions remain decimal values during evaluation. The interface rounds the result to a non-negative integer for display.

---

## 4. Assumptions

- Weather measurements in the dataset describe the rental hour itself.
- A user-entered scenario represents conditions for the hour being estimated.
- A genuine advance forecast would require weather forecasts available before the rental hour and a separate evaluation.
- Closed-service hours have structurally zero rentals in this dataset.
- Models learn operating-hour rental patterns; a shared explicit rule handles closures.
- Calendar features can be derived without using the target.
- The dataset’s season labels are used during training. The interface derives season from calendar month.
- One historical year is sufficient to demonstrate the workflow, but it cannot establish reliable performance across future years or every season.
- Results from Seoul do not establish validity for another city.

---

## 5. Constraints

- Solo student project.
- Local CPU-based implementation without GPU requirements.
- Basic Python and scikit-learn concepts should remain understandable.
- A short implementation period followed by study and presentation preparation.
- One historical year from one city.
- No paid APIs or cloud services.
- No live weather feed.
- No station inventory, event or supply data.
- One Random Forest configuration rather than a large hyperparameter search.
- A local demonstration interface rather than a production deployment.

The teacher’s assessment weights are:

| Deliverable | Weight |
|---|---:|
| Requirements specification and dataset analysis | 4% |
| AI model design and implementation | 6% |
| Evaluation, documentation and GitHub repository | 5% |

---

## 6. Dataset

### 6.1 Source and Justification

**Dataset:** Seoul Bike Sharing Demand  
**Repository:** UCI Machine Learning Repository  
**Dataset number:** 560  
**DOI:** [10.24432/C5F62R](https://doi.org/10.24432/C5F62R)  
**License:** Creative Commons Attribution 4.0 International  

The dataset is suitable because it combines hourly system rental totals with weather, holiday and operating information. Its size supports a student implementation on a local computer.

No alternative dataset was experimentally compared, so this project makes no claim that other datasets would perform worse.

### 6.2 Coverage and Schema

The CSV contains 8,760 records from **1 December 2017 00:00 to 30 November 2018 23:00**.

The following table uses readable source-field labels. Exact original CSV headers are checked by `ORIGINAL_COLUMNS` in `bike/data.py`.

| Internal name | Source field | Meaning or unit |
|---|---|---|
| `date` | Date | Calendar date |
| `rentals` | Rented Bike Count | Target: completed rentals during the hour |
| `hour` | Hour | Hour of day, 0–23 |
| `temperature` | Temperature | Degrees Celsius |
| `humidity` | Humidity | Percent |
| `wind_speed` | Wind speed | Metres per second |
| `visibility` | Visibility | Units of 10 metres |
| `dew_point` | Dew point temperature | Degrees Celsius |
| `solar` | Solar Radiation | MJ/m² |
| `rainfall` | Rainfall | Millimetres |
| `snowfall` | Snowfall | Centimetres |
| `season` | Seasons | Winter, Spring, Summer or Autumn |
| `holiday` | Holiday | Holiday or No Holiday |
| `operating` | Functioning Day | Yes or No |

Dataset characteristics:

- No missing cells.
- Complete consecutive hourly timestamps.
- No duplicate hourly timestamps.
- 295 non-operating hours, all with zero rentals.
- 432 holiday hours.
- Rental count range: 0 to 3,556.
- Mean rental count: 704.6.
- Median rental count: 504.5.

### 6.3 Validation and Preparation

The loader reads the file using `cp1252`, which supports the degree symbol in the original CSV headers.

Before training, it checks:

1. Exact original column names and order.
2. A total of 8,760 records.
3. Missing cells.
4. Duplicate timestamps.
5. Consecutive timestamps separated by exactly 3,600 seconds.
6. Non-negative rental counts.
7. Operating flags restricted to `Yes` and `No`.
8. Zero rentals during every non-operating hour.

A missing, unexpected or reordered column raises a `ValueError` before renaming.

The preparation steps are:

- Rename validated columns.
- Parse dates from day/month/year.
- Construct hourly timestamps.
- Sort observations chronologically.
- Add deterministic calendar features.

No observations were dropped or values filled in to demonstrate cleaning. The supplied dataset did not require missing-value replacement.

The dataset SHA-256 is:

```text
373339b71a8935d69e9af0abf26a70744632119862eeb3919efb389a7b749c60
```

The checksum is recorded in `reports/metrics.json`.

### 6.4 Exploratory Analysis

The main EDA outputs are:

- `reports/eda_hourly.png`
- `reports/eda_season.png`
- `reports/eda_temperature.png`
- `reports/eda_distribution.png`
- `reports/eda_weather_holiday.png`
- `reports/eda_monthly.png`
- `reports/eda_summary.json`

Key findings:

- **Daily pattern:** operating-hour mean rentals reach approximately 1,554 at 18:00 and 1,050 at 08:00. The quietest hour is 04:00, at approximately 137.
- **Seasonal pattern:** operating-hour means are approximately 1,034 in summer, 924 in autumn, 746 in spring and 226 in winter.
- **Right-skewed target:** the mean exceeds the median because some hours have substantially higher counts.
- **Weather association:** temperature has a correlation of approximately +0.54 with rentals across the full dataset.
- **Wet conditions:** wet operating hours average approximately 178 rentals, compared with 797 during dry operating hours.
- **Holidays:** operating-hour rentals average approximately 529 on holidays and 739 on non-holidays.

These are descriptive associations, not causal estimates. Season, time of day and unobserved supply conditions may influence the relationships.

Full-year EDA is not independent test evidence and should not be used to justify changing the model after inspecting test outcomes.

---

## 7. Methodology

### 7.1 Chronological Splits

The observations are sorted by timestamp and separated without shuffling.

| Split | Exact period | Hours | Closed hours |
|---|---|---:|---:|
| Training | 1 Dec 2017 00:00 to 13 Aug 2018 11:00 | 6,132 | 48 |
| Validation | 13 Aug 2018 12:00 to 6 Oct 2018 23:00 | 1,308 | 151 |
| Test | 7 Oct 2018 00:00 to 30 Nov 2018 23:00 | 1,320 | 96 |

The implementation uses:

```python
train = frame.iloc[:6132]
validation = frame.iloc[6132:7440]
test = frame.iloc[7440:]
```

Chronological separation evaluates later observations using earlier observations for learning. A random split would mix past and future periods and answer a different evaluation question.

The training period contains:

- Winter: 2,160 hours.
- Spring: 2,208 hours.
- Summer: 1,764 hours.
- Autumn: 0 hours.

Validation contains summer and autumn. The final test period is entirely autumn.

### 7.2 Feature Engineering

The shared `features()` function derives:

- Weekday, where Monday is 0 and Sunday is 6.
- Month, from 1 to 12.
- `hour_sin`.
- `hour_cos`.

The cyclic hour features are:

```text
hour_sin = sin(2 × π × hour / 24)
hour_cos = cos(2 × π × hour / 24)
```

These features preserve the relationship between neighbouring hours around midnight.

Hour is also retained as a categorical input. This allows the model to assign separate levels to individual hours.

Season comes from the dataset during training. The interface derives it from month using:

- December–February: Winter.
- March–May: Spring.
- June–August: Summer.
- September–November: Autumn.

No feature uses rental-count lags, rolling target statistics or the current target.

### 7.3 Preprocessing

The learned models use a common preprocessing definition, fitted separately inside each model’s pipeline.

**Numeric inputs:**

- Eight weather measurements.
- Hour sine.
- Hour cosine.

Numeric preprocessing applies:

1. Median imputation.
2. Standard scaling.

The imputer is a defensive pipeline component. It does not mean that missing values were filled in the supplied dataset. The interface rejects missing or invalid required inputs.

**Categorical inputs:**

- Hour.
- Weekday.
- Month.
- Season.
- Holiday.

Categorical preprocessing uses:

```python
OneHotEncoder(handle_unknown="ignore", sparse_output=False)
```

An unseen category receives an all-zero encoding for that feature group.

The `ColumnTransformer` selects only the specified features. Although the training DataFrame contains other columns, `rentals`, raw `date`, `timestamp` and `operating` are excluded from estimator inputs.

All learned preprocessing is fitted on operating training rows only.

### 7.4 Models Compared

#### Hourly-Mean Baseline

The baseline calculates the mean rental count for each hour using operating training observations.

For example, an 18:00 scenario receives the training operating-hour mean for 18:00, regardless of its weather.

This approach provides a simple reference. An improvement over it shows that a more complex approach helped in the evaluated period, although it does not isolate which added feature caused the improvement.

#### Linear Regression

Linear Regression predicts a weighted sum of the scaled and encoded inputs plus an intercept.

Because hour is one-hot encoded, it can represent separate hourly levels, including commuting peaks.

Its limitations include:

- No automatic interactions between weather and hour.
- No automatic nonlinear weather relationships.
- Linear extrapolation outside observed input ranges.

Additional interaction or curve features would be needed to represent those relationships explicitly.

#### Random Forest Regressor

The Random Forest averages predictions from 250 decision trees.

Settings:

```python
RandomForestRegressor(
    n_estimators=250,
    min_samples_leaf=3,
    max_features=0.8,
    random_state=42,
    n_jobs=-1,
)
```

The forest can represent thresholds, nonlinear relationships and interactions without manually specifying them.

Only one forest configuration was compared. This keeps the experiment manageable, but it does not establish that these settings are optimal.

### 7.5 Closure and Non-Negative Output Rules

All candidates are fitted on operating training observations.

The same prediction function applies these rules to every approach:

```python
np.where(
    frame.operating.eq("No"),
    0.0,
    np.maximum(values, 0.0),
)
```

Therefore:

- Non-operating hours return exactly zero.
- Negative predictions are clipped to zero.
- Learned estimators handle operating-hour rental patterns.

These rules are applied consistently to the baseline and both learned models.

Evaluation uses unrounded predictions. The application rounds only for display.

### 7.6 Model Selection

The model with the lowest **validation MAE** is selected.

The selected training fit remains frozen. It is not refitted on validation or test observations.

The test set measures performance after selection. It does not select the model.

Refitting on training plus validation can also be valid under a fixed protocol that preserves an untouched test set. This experiment deliberately evaluates and deploys the frozen training fit instead.

---

## 8. Evaluation

### 8.1 Metrics

**Mean Absolute Error:**

```text
MAE = mean(|actual − predicted|)
```

MAE measures the average absolute error in rentals per hour.

**Root Mean Squared Error:**

```text
RMSE = sqrt(mean((actual − predicted)²))
```

RMSE gives larger errors more weight.

Lower values are better for both metrics. Neither gives a maximum-error guarantee for an individual prediction.

### 8.2 Main Results

The error scores and fitting times below match the latest saved `reports/metrics.json`.

| Model | Validation MAE | Validation RMSE | Test MAE | Test RMSE | Fit time |
|---|---:|---:|---:|---:|---:|
| Hourly-mean baseline | 420.3 | 566.3 | 278.6 | 397.5 | <0.01 s |
| Linear Regression | 302.2 | 422.9 | 286.4 | 387.2 | 0.07 s |
| **Random Forest selected** | **162.2** | **250.9** | **243.3** | **346.4** | **2.98 s** |

Errors are in rentals per hour.

The latest recorded fitting times are approximately:

- Baseline: 0.0055 seconds.
- Linear Regression: 0.0732 seconds.
- Random Forest: 2.9789 seconds.

Earlier reported environments recorded different times, including approximately 6–7 seconds for Random Forest. Those historical timings belong to their own runs. Fitting time depends on the machine, workload and run conditions.

Random Forest has the lowest validation MAE and therefore wins the selection.

It also has the lowest final test MAE and RMSE. Its test MAE is approximately 12.7% lower than the hourly-mean baseline in this period.

Linear Regression improves over the baseline on validation but has slightly worse test MAE. The ranking is therefore period-dependent.

### 8.3 Operating-Hour Test Results

The full test period includes 96 closed hours, for which every approach returns zero through the shared rule.

Operating-only scores provide an additional view of learned prediction quality.

| Model | Operating test MAE | Operating test RMSE |
|---|---:|---:|
| Hourly-mean baseline | 300.5 | 412.8 |
| Linear Regression | 308.8 | 402.1 |
| Random Forest | 262.4 | 359.7 |

There are 1,224 operating test hours.

### 8.4 Error Analysis

The selected model’s segment analysis uses operating test hours only.

Peak hours are defined as 07:00–09:00 and 17:00–19:00 inclusive.

Wet hours have rainfall greater than zero or snowfall greater than zero.

| Segment | Operating hours | MAE | RMSE |
|---|---:|---:|---:|
| Peak hours | 306 | 482.3 | 584.9 |
| Off-peak hours | 918 | 189.1 | 241.9 |
| Wet hours | 76 | 163.0 | 225.0 |
| Dry hours | 1,148 | 269.0 | 366.9 |
| Autumn | 1,224 | 262.4 | 359.7 |
| Winter | 0 | Not estimable | Not estimable |
| Spring | 0 | Not estimable | Not estimable |
| Summer | 0 | Not estimable | Not estimable |

The implementation reports segment metrics only when at least 30 operating observations are available.

Peak-hour errors are substantially larger than off-peak errors. These hours have higher rental counts and are particularly relevant to planning.

Wet-hour MAE is lower, but the segment contains only 76 hours and generally lower rental counts. This does not establish better relative accuracy in wet weather.

Evaluation outputs include:

- `reports/test_predictions.csv`
- `reports/test_week.png`
- `reports/test_actual_vs_predicted.png`
- `reports/test_error_by_hour.png`

### 8.5 Seasonal Coverage and Underprediction

On operating test hours:

- Mean observed rentals: approximately 847.7.
- Mean predicted rentals: approximately 614.5.
- Mean underprediction: approximately 233.2 rentals per hour.

The actual-versus-predicted chart shows substantial underprediction for many busy hours.

Training contains no autumn observations, while all test observations are autumn. Unknown autumn and month categories receive all-zero encodings in their respective feature groups.

This coverage gap is a plausible contributor to the performance difference between validation and test. The experiment does not isolate it as the sole cause.

The final score describes this autumn 2018 test period. It does not measure winter, spring or summer test performance, or establish performance in future years.

### 8.6 Technical Trade-Offs

| Aspect | Hourly mean | Linear Regression | Random Forest |
|---|---|---|---|
| Validation MAE | 420.3 | 302.2 | 162.2 |
| Test MAE | 278.6 | 286.4 | 243.3 |
| Interpretability | Direct hourly averages | Coefficients can be inspected | Individual predictions are harder to explain |
| Latest fit time | <0.01 s | 0.07 s | 2.98 s |
| Weather adaptation | None | Additive fitted effects | Nonlinear effects and interactions |
| Storage | Small dictionary | Relatively small pipeline | Approximately 37 MB saved artifact |
| Main limitation | Ignores weather and calendar differences beyond hour | Needs additional features for interactions | Busy-hour bias and uncertain generalisation outside training coverage |

Random Forest offers the best validation result among the compared approaches at a manageable fitting cost.

Its lower error does not remove the need to explain uncertainty, seasonal limitations and observed-rental constraints.

---

## 9. Application

### 9.1 Local Interface

The Flask application runs at:

```text
http://127.0.0.1:5000/
```

It binds to the local loopback address and is intended for demonstration on the user’s computer.

Uploading the repository to GitHub does not host the application publicly.

### 9.2 Prediction Flow

1. `create_app()` loads the saved model and metrics.
2. The browser displays the form with example values and no estimate.
3. A submitted scenario reaches `validate()`.
4. Validation checks the values and derives the app’s season.
5. `features()` derives weekday, month and hour-cycle features.
6. The saved pipeline transforms the row using its fitted preprocessing.
7. The shared `predict()` function applies the estimator and closure policy.
8. The application rounds the prediction and renders the result.

No model fitting occurs during a prediction request.

### 9.3 Input Validation

The server checks:

- A parseable date.
- Integer hour from 0 to 23.
- Finite numeric weather values.
- Values within the permitted interface ranges.
- Dew point no higher than temperature.
- Valid holiday and operating choices.

Training-range warnings appear for weather values outside the ranges recorded in `reports/metrics.json`.

Interface ranges are safeguards. They do not guarantee a physically consistent scenario or accurate extrapolation.

### 9.4 Form and Result States

- Initial page load shows no estimate.
- **Load example scenario** restores the documented inputs without calculating.
- Typing values does not recalculate.
- A valid submitted scenario replaces the displayed estimate.
- Browser validation can prevent an invalid submission from being sent.
- A browser-blocked submission leaves the previous displayed estimate unchanged.
- An invalid POST that reaches the server renders an error with no estimate.
- A valid closed-service submission returns exactly zero.

Browser rejection and server rejection are separate checks. A browser warning alone does not prove server-side validation.

### 9.5 Demonstration Examples

The preparation materials report:

| Scenario | Reported output |
|---|---:|
| Example: 15 Oct 2018, 18:00, 18 °C, humidity 55%, remaining documented defaults | 2,100 |
| Change hour to 08:00 and temperature to 30 °C, retaining other example values | 1,219 |
| Service operating set to No | 0 |

These are model outputs for specified scenarios, not measured rental counts.

During a live demonstration, read the actual displayed result. Changing any input or using a different saved model can change the output.

Application screenshots are included in:

```text
docs/screenshots/prediction.png
docs/screenshots/closed-service.png
docs/screenshots/validation.png
```

---

## 10. Verification and Reproducibility

### 10.1 Evidence Categories

The project distinguishes three forms of evidence:

1. **Reported build and runtime checks:** recorded in the implementation and preparation materials.
2. **Independent saved-file checks:** arithmetic and consistency checks performed on supplied CSV, metrics and repository files.
3. **Current GitHub evidence:** uploaded source, documents, screenshots, metrics and commit history.

Reading saved results does not establish that a fresh installation or live application run has been independently repeated.

### 10.2 Historical Build Run Log

The implementation materials record the following runs on 15 November 2026.

| Run | Reported environment and activity | Reported outcome |
|---|---|---|
| Run 1: initial build | Original environment using Python 3.12.3; dataset acquisition, EDA, training, tests and server checks | Random Forest selected; validation MAE 162.2; test MAE 243.3; 12 tests passed |
| Run 2: clean rerun | Same environment; generated data, model and reports regenerated | Matching error scores and dataset checksum; 12 tests passed |
| Run 3: fresh-environment reproduction | New environment using the pinned requirements; installation, EDA, training, tests and server checks | Matching MAE/RMSE to four decimal places; 21 tests passed; approximately 0.10 s Linear Regression and 6.9 s Random Forest fitting time |

These historical entries retain their original reported timings. They are separate from the latest saved run, whose fitting times appear in Section 8.2.

### 10.3 Independent Saved-File Checks

Read-only checks established:

- Dataset SHA-256 matches the recorded checksum.
- The supplied dataset contains 8,760 consecutive hourly records.
- No empty cells were found.
- The 295 non-operating hours match the reported count.
- Chronological split boundaries match the report.
- Baseline validation and test scores can be reproduced from the raw dataset.
- Random Forest test MAE and RMSE can be reproduced from saved predictions.
- Saved test targets match the corresponding raw dataset rows.
- Closed-service test predictions are zero.
- Peak-hour sample sizes and errors match the saved metrics.
- The mean operating-hour underprediction is approximately 233.2.

A complete comparison of the earlier and updated GitHub prediction files covered all 1,320 test rows. Headers, targets and metadata were identical. The maximum prediction difference was approximately `4.55 × 10⁻¹³`, consistent with negligible floating-point rounding.

The rounded evaluation results remain unchanged.

### 10.4 Test Coverage

`tests/test_project.py` contains 14 test functions producing 21 test cases through parameterisation.

Coverage includes:

- Invalid hour, humidity, date and numeric inputs.
- Dew point above air temperature.
- Calendar derivation.
- Hourly baseline and closure behavior.
- Exact original-header validation.
- Missing, extra and reordered columns.
- Saved-model and interface prediction agreement.
- Initial empty state.
- Example loading without prediction.
- Changed inputs producing changed predictions.
- Agreement between the interface table and saved metrics.
- Unchanged model-file checksum after requests.
- Chronological split boundaries.

The build materials report 21 passing tests. The later read-only repository verification did not rerun training or tests.

The unchanged artifact checksum shows that requests do not rewrite the model file. Source inspection separately shows that request handling does not call `fit()`.

### 10.5 Dependency Versions

The supplied `requirements.txt` pins:

| Package | Version |
|---|---|
| Flask | 3.1.3 |
| joblib | 1.6.0 |
| matplotlib | 3.11.2 |
| numpy | 2.5.4 |
| pandas | 2.3.3 |
| pytest | 9.2.0 |
| scikit-learn | 1.9.1 |

**Python 3.12 or newer is required for this pinned setup.** In particular, pinned NumPy 2.5.4 requires Python 3.12+.

Direct dependencies are pinned, but transitive dependencies are not fully locked. A future fresh installation should be verified before presentation.

### 10.6 GitHub and Submission Evidence

The repository is:

[https://github.com/AbdulrafayAwan-72427/weather-aware-bike-rental](https://github.com/AbdulrafayAwan-72427/weather-aware-bike-rental)

The repository contains source, tests, evaluation files, charts, screenshots and project documents.

The `.gitignore` excludes:

- Virtual environments.
- Python caches.
- Downloaded dataset files.
- Generated model artifacts.

A fresh clone therefore requires dataset acquisition and training before starting the application.

GitHub availability demonstrates that files were uploaded. It does not independently prove runtime correctness.

---

## 11. Ethics and Responsible Use

### 11.1 Aggregate Data

The dataset contains hourly aggregate rental counts and environmental information rather than personal rider records.

The project does not need individual identities or rider-level trip histories.

### 11.2 Observed Rentals and Unmet Demand

Observed rentals can be limited by bicycle supply.

A planner should not interpret a low prediction as proof that few people wanted to travel. The project cannot measure people who could not obtain a bicycle.

### 11.3 Geographic Transfer

The model uses Seoul data from 2017–2018.

It has not been validated for Lahore, Karachi, Islamabad or another city. Different weather, infrastructure, commuting patterns and holidays require local data and evaluation.

### 11.4 Decision Support

A test MAE of 243.3 rentals per hour is an average error, not a guarantee for a particular hour.

Peak-hour errors are substantially larger. The model should support exploratory analysis rather than precise staffing, station allocation or service guarantees.

### 11.5 Attribution and Artifacts

The dataset is attributed to UCI and its CC BY 4.0 license.

Serialized model files should come from a trusted source or be regenerated from inspected source code. The local demonstration is not presented as a production service.

---

## 12. Limitations

1. **One city and one year:** the experiment cannot establish long-term or cross-city generalisation.
2. **Autumn-only test:** winter, spring and summer test performance are unmeasured.
3. **No autumn in training:** unseen seasonal categories and changing relationships limit interpretation.
4. **Busy-hour underprediction:** peak-hour MAE is 482.3 rentals per hour.
5. **Historical weather inputs:** operational forecast performance has not been evaluated.
6. **No station or supply information:** the target cannot establish station availability or unmet demand.
7. **Limited model search:** one forest configuration was compared rather than an extensive tuning experiment.
8. **No prediction intervals:** the interface displays a point estimate without a calibrated uncertainty interval.
9. **Partial dependency locking:** direct dependencies are pinned, while transitive versions can vary.
10. **Period-dependent rankings:** Linear Regression performs better than the baseline on validation but worse on test MAE.

---

## 13. Future Improvements

### 13.1 Additional Years

The highest priority is obtaining more years of data so training includes every season and later periods can test recurring seasonal patterns.

### 13.2 Rolling-Origin Evaluation

Evaluate several chronological training and validation windows rather than relying on one split.

This would provide stronger evidence about stability across periods.

### 13.3 Additional Models and Features

Possible extensions include:

- Gradient boosting.
- Modest Random Forest tuning.
- Calendar interactions.
- Rental lags available before the prediction time.
- Explicit treatment of busy-hour errors.

Any model or feature selection must preserve a final untouched test period.

### 13.4 Forecast Inputs

A forecasting extension would use weather forecasts available before the rental hour and evaluate the full system with forecast errors included.

Replacing typed inputs with a weather API alone would not establish forecasting accuracy.

### 13.5 Uncertainty and Station-Level Planning

Prediction intervals could help users understand uncertainty.

Station-level allocation would require station-level inventory and rental data, a different target and separate validation.

These are future extensions rather than completed capabilities.

---

## 14. Conclusion

The project implements a complete regression workflow for estimating hourly observed rentals in Seoul.

It validates the dataset, derives calendar features, fits preprocessing on training data only, compares three approaches, selects by validation MAE and reports held-out errors. A local Flask interface uses the saved pipeline and shared prediction rules.

Random Forest has the strongest validation result and the lowest final test error among the compared approaches:

- Validation MAE: **162.2 rentals per hour**.
- Test MAE: **243.3 rentals per hour**.
- Test RMSE: **346.4 rentals per hour**.

The improvement over the hourly baseline is useful evidence for this period. However, substantial peak-hour errors and limited seasonal coverage prevent a general claim of year-round forecasting reliability.

The project’s contribution is a working, documented comparison with clearly stated limits and reproducible source.

---

## Appendix A: Reproduction Commands

### Windows PowerShell

Use Python 3.12 or newer.

For a fresh GitHub copy:

```powershell
git clone https://github.com/AbdulrafayAwan-72427/weather-aware-bike-rental.git
Set-Location weather-aware-bike-rental
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m bike.eda
.\.venv\Scripts\python.exe -m bike.train
.\.venv\Scripts\python.exe -m pytest tests/ -q
.\.venv\Scripts\python.exe -m bike.app
```

Run each command separately and check that it succeeds before continuing.

If the repository is already on the computer, start from its project folder and omit the clone step.

Directly invoking the environment’s Python avoids the need to activate it or change PowerShell execution policy.

After starting the app, open:

```text
http://127.0.0.1:5000/
```

Important setup details:

- Internet access is needed for dependency installation.
- The dataset downloads automatically when it is missing.
- An existing dataset file is reused.
- Training regenerates the model and evaluation outputs.
- The model is excluded from Git and must exist before starting the app.
- The tests that use the saved model require training first.

### Linux or macOS

With Python 3.12+ available as `python3`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m bike.eda
.venv/bin/python -m bike.train
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m bike.app
```

---

## Appendix B: Repository Structure

```text
bike/
    __init__.py
    data.py
    eda.py
    model.py
    train.py
    app.py

templates/
    index.html

tests/
    test_project.py

docs/
    Technical-Report.md
    Project Report.docx
    Project Presentation.pptx
    project-proposal.docx
    screenshots/
        prediction.png
        closed-service.png
        validation.png

reports/
    metrics.json
    test_predictions.csv
    eda_summary.json
    eda_hourly.png
    eda_season.png
    eda_temperature.png
    eda_distribution.png
    eda_weather_holiday.png
    eda_monthly.png
    hourly_profile.png
    monthly_profile.png
    test_week.png
    test_actual_vs_predicted.png
    test_error_by_hour.png
    training_correlations.csv
    dataset_summary.csv

.gitignore
.python-version
README.md
requirements.txt
pyproject.toml
```

Generated locally and excluded from Git:

```text
data/SeoulBikeData.csv
artifacts/model.joblib
.venv/
__pycache__/
.pytest_cache/
```

---

## References

1. **Seoul Bike Sharing Demand [Dataset].** UCI Machine Learning Repository, 2020. Dataset 560. [https://doi.org/10.24432/C5F62R](https://doi.org/10.24432/C5F62R). License: CC BY 4.0.
2. **Artificial Intelligence Semester Project Brief.** Riphah International University, BSAI Fall 2026. Supplied `Semester Project.docx`. Deadline: 21 December 2026.
3. **Project source and evaluation artifacts.** [weather-aware-bike-rental](https://github.com/AbdulrafayAwan-72427/weather-aware-bike-rental). Includes implementation, tests, metrics, predictions and charts.
4. **Project preparation materials.** `Bike-Rental-Presentation-Preparation.zip`, dated 15 November 2026. Source of reported historical runtime checks and demonstration examples.
5. **NumPy 2.5.4 package metadata.** [https://pypi.org/project/numpy/2.5.4/](https://pypi.org/project/numpy/2.5.4/). Python requirement: 3.12 or newer.