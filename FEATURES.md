# Features v1 (US-Accidents)

Format adapted from the Lab-6 `FEATURES.md`. The publish task saves 19 columns. This file defines them. It does not approve every column for model training.

**Target:** a severe accident, `label = 1` when `Severity` is 3 or 4.
**Prediction time:** when the accident is first reported (its `Start_Time`). Only information that exists at that moment may be used as a predictor.

| Column | Type | Definition and source | Available at prediction time? |
| --- | --- | --- | --- |
| ID | string | Accident ID from silver. The key (one row per ID). | Yes. Identifier only, not a predictor. |
| event_ts | timestamp | `Start_Time` from silver (local time of the accident). Used for time ordering, the time split and the feature store timestamp. | Yes. Not a predictor itself. |
| accident_year | int | Year from the ADF folder (`accident_year=`), equal to the year of `Start_Time`. | Yes. Use for the time split (train before 2022, test 2022 onwards), not as a predictor. |
| start_hour | int | Hour of `Start_Time`, 0–23. | Yes. |
| day_of_week | int | Day of `Start_Time`, 1 = Sunday … 7 = Saturday (Spark `dayofweek`). | Yes. |
| month | int | Month of `Start_Time`, 1–12. | Yes. |
| is_night | int | 1 if `Sunrise_Sunset` is "Night", 0 if "Day", null if missing. | Yes. |
| Temperature_F | double | Temperature at the nearest weather station (from silver; null if missing). | Yes, from the weather report at that time. |
| Visibility_mi | double | Visibility in miles (from silver; null if missing). | Yes. |
| Wind_Speed_mph | double | Wind speed (from silver; null if missing). | Yes. |
| Precipitation_in | double | Precipitation in inches (from silver; often null). | Yes, but many values are missing. Treat null as "unknown", not 0. |
| weather_condition | string | `Weather_Condition` text (for example "Clear", "Rain"); "Unknown" when missing. | Yes. Many distinct values; group or encode on training data only. |
| junction | int | 1 if a junction is nearby (`Junction` flag). | Yes, it describes the road. |
| crossing | int | 1 if a crossing is nearby. | Yes. |
| traffic_signal | int | 1 if a traffic signal is nearby. | Yes. |
| stop | int | 1 if a stop sign is nearby. | Yes. |
| county_prior_accidents | bigint | Number of strictly earlier accepted accidents in the same `State` + `County`. | Yes, if that history is kept. Cold start gives 0. |
| county_prior_severe_rate | double | Share of strictly earlier accidents in the same `State` + `County` with Severity 3–4. No history gives 0. | Only if the severity of earlier accidents was already known. This is also the basis of baseline B2. |
| label | int | 1 if `Severity` is 3 or 4; otherwise 0. | Target only. Never include in predictors. |

Both county columns use `Window.partitionBy("State", "County").orderBy(Start_Time seconds).rangeBetween(unboundedPreceding, -1)`, the same pattern as Lab-6. They exclude every accident at the same second, including the row itself. A zero rate with zero prior accidents is a missing-history marker, not a real "never severe" value.

## Leakage risks (clock test)

Columns that are only known **after** the accident are kept in silver/gold for reference but are **not** features:

- `End_Time`: when the traffic impact ended. Severity is defined by this impact, so it leaks the target.
- `Distance_mi`: length of road affected by the accident; measured after the accident.
- `Description`, `End_Lat`, `End_Lng`: dropped already in the clean step (written after the event).
- `Severity`: the target itself.

Other risks:

- A random split would let the county history use labels from the test period. Use the time split from the hypothesis.
- The county rate is computed on the full accepted data in time order. For training, it only uses earlier rows, so it is safe with a time split.
- `Source` (the data provider) is not a feature. If providers report severity differently (to be checked in EDA), it is a bias risk, not a predictor.
- `weather_condition` encodings must be fitted on training data only.

## Versions and paths

`curated/accidents_features_v1/` is the replaceable "latest" path. Every run also keeps its own snapshot under `curated/accidents_runs/RUN_ID/features/` (the exact `features_path` is in that run's `success.json`). Gold keeps all silver columns, one row per `ID`; accidents whose `End_Time` is missing or before `Start_Time` go to the run's quarantine with `reject_reason = end_before_start`.
