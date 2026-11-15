# Data contract, catalog and lineage

This file covers Phase 1 · II.3: columns, types, keys, assumptions, and how each output links back to the raw data.

## Catalog entries (Unity Catalog)

| Table | Type | Location | Written by | Key |
| --- | --- | --- | --- | --- |
| `dsai3202_databricks.s_60312217.silver_accidents` | Delta, partitioned by `accident_year` | `abfss://processed@amazonlake60312217.dfs.core.windows.net/silver_accidents` | `databricks/01_clean.py` (job task `clean`) | `ID` |
| `dsai3202_databricks.s_60312217.gold_accidents` | Delta | `abfss://curated@amazonlake60312217.dfs.core.windows.net/gold_accidents` | `databricks/04_publish.py` (job task `publish`) | `ID` |

The features are Parquet files, not a catalog table: the replaceable "latest" copy is at `curated/accidents_features_v1/`, and each run's own snapshot is at `curated/accidents_runs/<run_id>/features/`.

## Lineage (raw → features)

| Step | Input | Output | Code | Evidence |
| --- | --- | --- | --- | --- |
| 1. Ingest (raw, v1) | Kaggle `archive.zip` (US-Accidents, March 2023 release) | `raw/accidents/v1/US_Accidents_March23.csv` and `raw/accidents/v1/archive.zip` | `ingestion/ingest.sh` (AzCopy, `--overwrite=false`) | `ingestion/evidence/source-manifest.json` (SHA-256), receipts before/after the repeated upload |
| 2. Bronze (ADF) | `raw/accidents/v1/US_Accidents_March23.csv` | `processed/accidents/accident_year=YYYY/*.parquet` (8 year folders) | `adf/pl_transform_accidents.json`, `adf/df_accidents_partitioned.json` | `adf/evidence/transform-run-0282f9e1-output.json`: 7,728,394 rows read and written |
| 3. Silver (clean) | `processed/accidents/` (all year folders) | `silver_accidents` (each job run writes a new Delta version) | `databricks/01_clean.py` | `curated/quality/accidents/drop_report.json`, `clean_runs/<run_id>.json` |
| 4. Run input copy | `silver_accidents VERSION AS OF <silver_version in the drop report>` | `processed/accidents_runs/<run_id>/input` (Parquet) | `databricks/02_prepare.py` | `curated/quality/accidents/runs/<run_id>/input.json`, `profile.json` |
| 5. Gate | the run input copy | `gate.json`; failing rows → `processed/accidents_runs/<run_id>/quarantine` | `databricks/03_validate.py` | `runs/<run_id>/gate.json` |
| 6. Gold + features | input copy minus quarantine | `curated/accidents_runs/<run_id>/gold` (Delta), `.../features` (Parquet); current `gold_accidents` and `accidents_features_v1` | `databricks/04_publish.py` | `runs/<run_id>/success.json`, `coverage.json`, `comparison.json`; `latest.json` |

Every `success.json` records the exact `silver_version`, `gold_version`, `gold_path` and `features_path`, so any feature snapshot can be traced back to one silver version, and from there to the raw file hash.

## Column renames in bronze (ADF)

Parquet does not allow `(`, `)` or `%` in column names, so ADF renames them: `(` → `_`, `)` removed, `%` → `pct`. For example, `Distance(mi)` becomes `Distance_mi`, `Humidity(%)` becomes `Humidity_pct`, and `Wind_Chill(F)` becomes `Wind_Chill_F`. All bronze columns stay as text except `accident_year` (integer, from the year of `Start_Time`), which is also the folder name.

## Silver / gold schema

Gold has exactly the silver columns. One row per `ID`.

| Column | Type | Meaning / rule |
| --- | --- | --- |
| ID | string | Accident ID from the source. **Key**; never null, never repeated (gate rules `keys`, `unique_accident`). |
| Severity | int | 1 (least impact on traffic) to 4 (most). Must be 1–4. **Target source**: label = Severity ≥ 3. |
| Start_Time | timestamp | Local time when the accident started. Read from the first 19 characters (`yyyy-MM-dd HH:mm:ss`); must be inside 2016-01-01 to 2023-04-01. |
| End_Time | timestamp | Local time when the traffic impact ended. Known only after the accident (not a feature). Rows where it is missing or before `Start_Time` are quarantined. |
| Weather_Timestamp | timestamp | Time of the weather observation; may be null. |
| Start_Lat, Start_Lng | double | Start location; must be inside the contiguous US (lat 24–50, lng −125 to −66). |
| Distance_mi | double | Length of road affected. Known only after the accident (not a feature). |
| Temperature_F, Humidity_pct, Pressure_in, Visibility_mi, Wind_Speed_mph, Precipitation_in | double | Weather at the nearest station; null when missing. Missing values are kept, not deleted. |
| Source | string | Data provider (values and counts in the EDA). Not a feature (possible label bias; see EDA). |
| Street, City, County, State, Zipcode, Timezone | string | Location text; trimmed, empty → null. |
| Weather_Condition | string | Weather text (for example "Clear", "Rain"); null when missing (warn rule). |
| Sunrise_Sunset | string | "Day" or "Night"; null when missing. |
| Amenity, Bump, Crossing, Give_Way, Junction, No_Exit, Railway, Roundabout, Station, Stop, Traffic_Calming, Traffic_Signal, Turning_Loop | boolean | Nearby road features from the source. |
| accident_year | int | Year folder written by ADF. Must equal the year of `Start_Time`. Silver partition column. |

Dropped in the clean step: `End_Lat`, `End_Lng` and `Description` (written after the accident), `Country` (always "US"), `Airport_Code`, `Wind_Chill_F`, `Wind_Direction`, and the three twilight columns (`Sunrise_Sunset` is kept instead).

## Assumptions

- `Start_Time` is the local time of the accident (the source has no UTC offset). The Spark session time zone is UTC, so the hour is read exactly as written.
- `ID` identifies one accident. The clean step would keep one row per repeated `ID` (the lexically smallest full row, which is reproducible); in this dataset no repeated `ID` was found.
- Severity measures impact on traffic, as given by the source. Providers may score it differently (see EDA).
- Missing weather values mean "not reported", not 0.

## Feature columns

See [FEATURES.md](FEATURES.md) for the 19 feature columns, their availability at prediction time and the leakage risks.
