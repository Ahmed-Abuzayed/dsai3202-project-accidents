# Databricks notebook source
# MAGIC %md
# MAGIC # Shared rules for the accidents job
# MAGIC Adapted from Lab-6 `00_common`. This notebook holds everything the job tasks must agree on:
# MAGIC the data contract, the quality-gate rules, the file locations and the feature definitions.
# MAGIC
# MAGIC It is not a job task. `02_prepare`, `03_validate`, `04_publish` and `05_inspect` start with
# MAGIC `%run ./00_common`, so every task uses the same rules (one source of truth).
# MAGIC
# MAGIC **Changes from Lab-6:** silver accident columns instead of reviews; a one-column key (`ID`);
# MAGIC new gate rules for severity, time, location and end time; no product metadata join, so the
# MAGIC quarantine rule is "end time before start time" instead of "product missing"; new features
# MAGIC for the severe-accident task; the test flag is `test_bad_severity`.

# COMMAND ----------

# 1. Imports
import json
import math
import re
from datetime import datetime, timezone
from pyspark.sql import functions as F, Window

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data contract
# MAGIC `EXPECTED_TYPES` is the schema written by `01_clean`. `FEATURE_COLUMNS` is the exact output of
# MAGIC `make_features`. `QUARANTINE_TOLERANCE`: at most 0.5% of rows may be set aside by the quarantine
# MAGIC rule (same policy value as Lab-6); more than that stops the job.

# COMMAND ----------

KEY = ["ID"]
NUMERIC = ["Start_Lat", "Start_Lng", "Distance_mi", "Temperature_F", "Humidity_pct",
           "Pressure_in", "Visibility_mi", "Wind_Speed_mph", "Precipitation_in"]
TEXT = ["Source", "Street", "City", "County", "State", "Zipcode", "Timezone",
        "Weather_Condition", "Sunrise_Sunset"]
FLAGS = ["Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway",
         "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal", "Turning_Loop"]
SILVER_COLUMNS = (KEY + ["Severity", "Start_Time", "End_Time", "Weather_Timestamp"]
                  + NUMERIC + TEXT + FLAGS + ["accident_year"])
EXPECTED_TYPES = {"ID": "string", "Severity": "int", "Start_Time": "timestamp",
                  "End_Time": "timestamp", "Weather_Timestamp": "timestamp",
                  "accident_year": "int",
                  **{c: "double" for c in NUMERIC}, **{c: "string" for c in TEXT},
                  **{c: "boolean" for c in FLAGS}}
FEATURE_COLUMNS = KEY + ["event_ts", "accident_year", "start_hour", "day_of_week", "month",
    "is_night", "Temperature_F", "Visibility_mi", "Wind_Speed_mph", "Precipitation_in",
    "weather_condition", "junction", "crossing", "traffic_signal", "stop",
    "county_prior_accidents", "county_prior_severe_rate", "label"]
FIRST_TS, END_TS = "2016-01-01 00:00:00", "2023-04-01 00:00:00"
LAT_MIN, LAT_MAX, LNG_MIN, LNG_MAX = 24.0, 50.0, -125.0, -66.0
QUARANTINE_TOLERANCE = 0.005  # Lab-6 policy value: at most 0.5% of rows may be quarantined.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Job parameters and paths
# MAGIC `config()` reads the six job parameters and builds this run's paths.
# MAGIC
# MAGIC | Parameter | Value |
# MAGIC |---|---|
# MAGIC | `lake` | the Lab-3 storage account |
# MAGIC | `catalog`, `schema` | the Unity Catalog location of the tables |
# MAGIC | `run_id` | `{{job.run_id}}`, so every run gets its own folders |
# MAGIC | `test_bad_severity` | `true` only for the controlled-failure run |
# MAGIC | `compare_to` | empty, or a successful earlier run ID for the safe rerun |

# COMMAND ----------

def config():
    for name, default in [("lake", ""), ("catalog", ""), ("schema", ""),
                          ("run_id", ""), ("test_bad_severity", "false"), ("compare_to", "")]:
        dbutils.widgets.text(name, default)
    c = {name: dbutils.widgets.get(name).strip() for name in
         ["lake", "catalog", "schema", "run_id", "test_bad_severity", "compare_to"]}
    assert re.fullmatch(r"[a-z0-9]{3,24}", c["lake"]), "Enter your Lab-3 lake name."
    for name in ["catalog", "schema"]:
        assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", c[name]), "Enter your assigned catalog and schema."
    assert c["catalog"] != "hive_metastore", "Use the Unity Catalog."
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,80}", c["run_id"]), "Set run_id to the job run ID."
    assert c["test_bad_severity"] in ["true", "false"]
    assert not c["compare_to"] or re.fullmatch(r"[A-Za-z0-9_-]{1,80}", c["compare_to"])
    assert c["compare_to"] != c["run_id"], "Compare with an earlier run."
    c["processed"] = f"abfss://processed@{c['lake']}.dfs.core.windows.net"
    c["curated"] = f"abfss://curated@{c['lake']}.dfs.core.windows.net"
    c["reports"] = f"{c['curated']}/quality/accidents"
    c["table"] = f"{c['catalog']}.{c['schema']}.silver_accidents"
    c["gold_table"] = f"{c['catalog']}.{c['schema']}.gold_accidents"
    c["work"] = f"{c['processed']}/accidents_runs/{c['run_id']}"
    c["out"] = f"{c['curated']}/accidents_runs/{c['run_id']}"
    c["quality"] = f"{c['reports']}/runs/{c['run_id']}"
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    return c

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Small JSON helpers (unchanged from Lab-6)

# COMMAND ----------

def read_json(path):
    return json.loads(dbutils.fs.head(path, 1024 * 1024))

def write_json(path, value, overwrite=False):
    dbutils.fs.mkdirs(path.rsplit("/", 1)[0])
    dbutils.fs.put(path, json.dumps(value, indent=2, default=str, allow_nan=False), overwrite)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Schema check and content comparison (unchanged from Lab-6)

# COMMAND ----------

def schema_errors(frame):
    actual = dict(frame.dtypes)
    return [f"{name}: expected {kind}, got {actual.get(name, 'MISSING')}"
            for name, kind in EXPECTED_TYPES.items() if actual.get(name) != kind]

def same_content(left, right):
    """Compare all rows in both directions, including duplicates. Order is irrelevant."""
    assert set(left.columns) == set(right.columns), "Column names differ."
    cols = sorted(left.columns)
    a, b = left.select(*cols), right.select(*cols)
    assert a.schema.simpleString() == b.schema.simpleString(), "Column types differ."
    return a.exceptAll(b).limit(1).count() == 0 and b.exceptAll(a).limit(1).count() == 0

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Profile every input row
# MAGIC Describes the data: missing values, time range, visibility percentiles, counts per severity and
# MAGIC per year, and the correlation between severity and visibility. All rows, no sample.

# COMMAND ----------

def profile(frame):
    n = frame.count()
    assert n > 0, "Input is empty."
    bounds = frame.agg(F.min("Start_Time").alias("first"), F.max("Start_Time").alias("last"),
                       F.max("Visibility_mi").alias("max_visibility")).first()
    nulls = frame.agg(*[F.sum(F.col(c).isNull().cast("long")).alias(c)
                        for c in SILVER_COLUMNS]).first().asDict()
    paired = frame.select("Severity", "Visibility_mi").na.drop()
    stats = paired.agg(F.covar_pop("Severity", "Visibility_mi").alias("covariance"),
                       F.stddev_pop("Severity").alias("severity_sd"),
                       F.stddev_pop("Visibility_mi").alias("visibility_sd")).first()
    # Constant columns have undefined correlation. Avoid division by zero in ANSI mode.
    denominator = (stats["severity_sd"] or 0) * (stats["visibility_sd"] or 0)
    correlation = stats["covariance"] / denominator if denominator else None
    return {"rows": n, "population": "all input accidents; no sample",
        "null_counts": nulls, "earliest_start": bounds["first"], "latest_start": bounds["last"],
        "visibility_mi": {"q25_q50_q75_q95": frame.approxQuantile(
            "Visibility_mi", [0.25, 0.5, 0.75, 0.95], 0.01),
            "max": bounds["max_visibility"], "quantile_rank_error": 0.01},
        "severity": [r.asDict() for r in frame.groupBy("Severity").count().orderBy("Severity").collect()],
        "years": [r.asDict() for r in frame.groupBy("accident_year").count().orderBy("accident_year").collect()],
        "severity_visibility_correlation": correlation if correlation is not None
            and math.isfinite(correlation) else None}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Quality-gate rules
# MAGIC | Rule | Policy | Passes when |
# MAGIC |---|---|---|
# MAGIC | `schema` | fail | column names and types match `EXPECTED_TYPES` |
# MAGIC | `volume` | fail | rows = the `rows_out` of the clean report, and > 0 |
# MAGIC | `keys` | fail | no missing or empty `ID` |
# MAGIC | `severity` | fail | every `Severity` is 1, 2, 3 or 4 |
# MAGIC | `time_and_year` | fail | `Start_Time` is inside 2016-01-01 to 2023-04-01 and its year equals `accident_year` |
# MAGIC | `location` | fail | `Start_Lat` / `Start_Lng` are inside the contiguous United States |
# MAGIC | `unique_accident` | fail | no `ID` appears twice |
# MAGIC | `end_after_start` | quarantine | `End_Time` exists and is not before `Start_Time`; failing rows are set aside, and the rule fails only above 0.5% |
# MAGIC | `missing_weather` | warn | rows without `Weather_Condition` are counted; never stops the job |
# MAGIC
# MAGIC **fail** stops the job. **quarantine** keeps the run going but sets the rows aside with a reason.
# MAGIC **warn** only reports.

# COMMAND ----------

def evaluate_gate(silver, expected_rows, tolerance=QUARANTINE_TOLERANCE):
    errors = schema_errors(silver)
    if errors:
        return {"passed": False, "failed_rules": ["schema"], "schema_errors": errors}, None
    n = silver.count()
    results = []
    def record(name, policy, passed, detail):
        results.append({"rule": name, "policy": policy, "passed": bool(passed), "detail": detail})
    record("schema", "fail", True, "Clean column types match")
    record("volume", "fail", n == expected_rows and n > 0, {"actual": n, "expected": expected_rows})
    key_bad = F.col("ID").isNull() | (F.trim("ID") == "")
    severity_ok = F.coalesce(F.col("Severity").isin(1, 2, 3, 4), F.lit(False))
    time_ok = F.coalesce((F.col("Start_Time") >= F.lit(FIRST_TS).cast("timestamp")) &
                         (F.col("Start_Time") < F.lit(END_TS).cast("timestamp")), F.lit(False))
    year_ok = F.coalesce(F.year("Start_Time") == F.col("accident_year"), F.lit(False))
    location_ok = F.coalesce(F.col("Start_Lat").between(LAT_MIN, LAT_MAX) &
                             F.col("Start_Lng").between(LNG_MIN, LNG_MAX), F.lit(False))
    predicates = {"keys": ~key_bad, "severity": severity_ok,
                  "time_and_year": time_ok & year_ok, "location": location_ok}
    counts = silver.agg(*[F.sum((~good).cast("long")).alias(name)
                          for name, good in predicates.items()]).first().asDict()
    for name, count in counts.items():
        record(name, "fail", count == 0, {"invalid_rows": count or 0})
    duplicates = silver.groupBy(*KEY).count().filter("count > 1").count()
    record("unique_accident", "fail", duplicates == 0, {"duplicate_keys": duplicates})
    end_ok = F.coalesce(F.col("End_Time") >= F.col("Start_Time"), F.lit(False))
    quarantine = silver.filter(~end_ok)
    quarantine_count = quarantine.count()
    record("end_after_start", "quarantine", quarantine_count / max(n, 1) <= tolerance,
           {"quarantined_rows": quarantine_count, "fraction": quarantine_count / max(n, 1),
            "tolerance": tolerance})
    missing_weather = silver.filter("Weather_Condition IS NULL").count()
    record("missing_weather", "warn", missing_weather == 0, {"rows_without_weather": missing_weather})
    failed = [r["rule"] for r in results if not r["passed"] and r["policy"] != "warn"]
    return {"passed": not failed, "failed_rules": failed, "rules": results,
            "rows_in": n, "quarantined_rows": quarantine_count,
            "accepted_rows": n - quarantine_count, "quarantine_tolerance": tolerance}, quarantine

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Coverage: who was kept (unchanged from Lab-6)

# COMMAND ----------

def coverage(silver, gold, column):
    """Compare full-population groups. Missing after-gate groups get zero rows."""
    before = silver.groupBy(column).agg(F.count("*").alias("before"))
    after = gold.groupBy(column).agg(F.count("*").alias("after"))
    return (before.join(after, column, "left").fillna(0, ["after"])
            .withColumn("retained_pct", F.round(100.0 * F.col("after") / F.col("before"), 3)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Features (definitions are in `FEATURES.md`)
# MAGIC Every feature is known when the accident is first reported. `Start_Time` is the local time of
# MAGIC the accident (the session time zone is UTC, so the hour is read as written).
# MAGIC
# MAGIC The county history uses only **earlier** accidents in the same state and county. As in Lab-6, a
# MAGIC range frame ending at `-1` second excludes every accident at the same second, including this one.
# MAGIC `Distance_mi`, `End_Time` and `Description` are never used: they are only known after the accident.

# COMMAND ----------

def make_features(gold):
    ts = F.col("Start_Time").cast("long")
    county_history = (Window.partitionBy("State", "County").orderBy(ts)
                      .rangeBetween(Window.unboundedPreceding, -1))
    label = (F.col("Severity") >= 3).cast("int")
    return gold.select(*KEY, F.col("Start_Time").alias("event_ts"), "accident_year",
        F.hour("Start_Time").alias("start_hour"),
        F.dayofweek("Start_Time").alias("day_of_week"),
        F.month("Start_Time").alias("month"),
        F.when(F.col("Sunrise_Sunset").isNull(), None)
         .otherwise((F.col("Sunrise_Sunset") == "Night").cast("int")).alias("is_night"),
        "Temperature_F", "Visibility_mi", "Wind_Speed_mph", "Precipitation_in",
        F.coalesce(F.col("Weather_Condition"), F.lit("Unknown")).alias("weather_condition"),
        F.col("Junction").cast("int").alias("junction"),
        F.col("Crossing").cast("int").alias("crossing"),
        F.col("Traffic_Signal").cast("int").alias("traffic_signal"),
        F.col("Stop").cast("int").alias("stop"),
        F.count("*").over(county_history).alias("county_prior_accidents"),
        F.coalesce(F.avg(label).over(county_history), F.lit(0.0)).alias("county_prior_severe_rate"),
        label.alias("label"))

# COMMAND ----------

print(f"00_common loaded: {len(KEY)}-column accident key, {len(EXPECTED_TYPES)} contract columns, "
      f"{len(FEATURE_COLUMNS)} feature columns, quarantine tolerance {QUARANTINE_TOLERANCE:.1%}.")
