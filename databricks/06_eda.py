# Databricks notebook source
# MAGIC %md
# MAGIC # Exploratory analysis (Phase 1 · II.4)
# MAGIC Run by hand after a successful job run. It reads the **published** outputs named in
# MAGIC `curated/quality/accidents/latest.json` (gold table version and features snapshot).
# MAGIC
# MAGIC **Population:** every accepted accident (7,728,394 rows). **No sample is used**: all numbers are
# MAGIC Spark aggregations over the full table; percentiles use `approxQuantile` with 1% rank error.
# MAGIC Plots are drawn from these small aggregated tables.
# MAGIC
# MAGIC Results are saved to `curated/quality/accidents/eda.json` and summarised in the README.

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

# 1. Load the published gold version and the features of the same run.
lake = "amazonlake60312217"
spark.conf.set("spark.sql.session.timeZone", "UTC")
reports = f"abfss://curated@{lake}.dfs.core.windows.net/quality/accidents"
latest = read_json(f"{reports}/latest.json")
gold = spark.sql(f"SELECT * FROM {latest['gold_table']} VERSION AS OF {latest['gold_version']}")
features = spark.read.parquet(latest["features_path"])
gold = gold.withColumn("severe", (F.col("Severity") >= 3).cast("int")).cache()
n = gold.count()
assert n == latest["gold_rows"] and features.count() == n
print(f"Published run {latest['run_id']}, gold version {latest['gold_version']}: {n:,} accidents (full population, no sample)")
eda = {"run_id": latest["run_id"], "gold_version": latest["gold_version"], "population_rows": n,
       "sample": "none - all rows"}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Target distribution and the accuracy trap
# MAGIC Severity 3–4 is the positive class. A rule that always says "not severe" gets an accuracy equal
# MAGIC to the share of non-severe accidents, while finding none of the severe ones (Lec-6).

# COMMAND ----------

sev = [r.asDict() for r in gold.groupBy("Severity").count().orderBy("Severity").collect()]
positive = gold.agg(F.avg("severe")).first()[0]
eda["severity_counts"] = sev
eda["severe_share"] = positive
eda["always_not_severe_accuracy"] = 1 - positive
for r in sev:
    print(f"Severity {r['Severity']}: {r['count']:>9,} ({r['count'] / n:.1%})")
print(f"Severe (3-4) share: {positive:.2%}. 'Always not severe' accuracy = {1 - positive:.2%} with recall 0.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Severe share over time (matters for the time split: train < 2022, test ≥ 2022)

# COMMAND ----------

import matplotlib.pyplot as plt
by_year = gold.groupBy("accident_year").agg(F.count("*").alias("rows"), F.avg("severe").alias("severe_share")) \
              .orderBy("accident_year").toPandas()
eda["by_year"] = by_year.to_dict("records")
split = gold.groupBy((F.col("accident_year") >= 2022).alias("test_period")) \
            .agg(F.count("*").alias("rows"), F.avg("severe").alias("severe_share")).collect()
eda["split"] = {("test_2022_on" if r["test_period"] else "train_before_2022"):
                {"rows": r["rows"], "severe_share": r["severe_share"]} for r in split}
print(by_year.to_string(index=False))
print(json.dumps(eda["split"], indent=2))
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(by_year["accident_year"].astype(str), by_year["severe_share"])
ax.set_title("Share of severe accidents (Severity 3-4) by year, all rows")
ax.set_ylabel("severe share")
plt.tight_layout(); display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Data source bias risk
# MAGIC The dataset combines several data providers (`Source`). If they label severity differently,
# MAGIC the model could learn the provider instead of the accident.

# COMMAND ----------

by_source = gold.groupBy("Source").agg(F.count("*").alias("rows"), F.avg("severe").alias("severe_share")) \
                .orderBy("Source").toPandas()
source_year = gold.groupBy("accident_year", "Source").count().orderBy("accident_year", "Source").toPandas()
eda["by_source"] = by_source.to_dict("records")
eda["source_by_year"] = source_year.to_dict("records")
print(by_source.to_string(index=False))
print(source_year.pivot(index="accident_year", columns="Source", values="count").fillna(0).astype(int).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Severe share by time of day, night and weather (candidate predictors)

# COMMAND ----------

by_hour = features.groupBy("start_hour").agg(F.count("*").alias("rows"), F.avg("label").alias("severe_share")) \
                  .orderBy("start_hour").toPandas()
by_night = [r.asDict() for r in features.groupBy("is_night").agg(F.count("*").alias("rows"),
            F.avg("label").alias("severe_share")).orderBy("is_night").collect()]
top_weather = features.groupBy("weather_condition").agg(F.count("*").alias("rows"),
              F.avg("label").alias("severe_share")).orderBy(F.desc("rows")).limit(10).toPandas()
distinct_weather = features.select("weather_condition").distinct().count()
eda["by_hour"] = by_hour.to_dict("records")
eda["by_night"] = by_night
eda["top_weather"] = top_weather.to_dict("records")
eda["distinct_weather_conditions"] = distinct_weather
print(json.dumps(by_night, indent=2))
print(f"Distinct weather_condition values: {distinct_weather}")
print(top_weather.to_string(index=False))
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(by_hour["start_hour"], by_hour["severe_share"], marker="o")
ax.set_title("Share of severe accidents by start hour, all rows")
ax.set_xlabel("hour (local)"); ax.set_ylabel("severe share")
plt.tight_layout(); display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Numeric distributions, missing values and outliers
# MAGIC Percentiles over all rows. Values far outside physical ranges are flagged, not deleted:
# MAGIC they are a modelling decision (cap or treat as missing in Phase 2).

# COMMAND ----------

numeric = ["Temperature_F", "Humidity_pct", "Pressure_in", "Visibility_mi", "Wind_Speed_mph",
           "Precipitation_in", "Distance_mi"]
quantiles = {c: gold.approxQuantile(c, [0.0, 0.01, 0.5, 0.99, 1.0], 0.01) for c in numeric}
missing = gold.agg(*[F.sum(F.col(c).isNull().cast("long")).alias(c) for c in numeric]).first().asDict()
outlier_rules = {"Temperature_F": "Temperature_F < -40 OR Temperature_F > 130",
                 "Visibility_mi": "Visibility_mi > 30",
                 "Wind_Speed_mph": "Wind_Speed_mph > 100",
                 "Precipitation_in": "Precipitation_in > 5",
                 "Pressure_in": "Pressure_in < 25 OR Pressure_in > 32"}
outliers = {c: gold.filter(rule).count() for c, rule in outlier_rules.items()}
eda["numeric_quantiles_min_p1_p50_p99_max"] = quantiles
eda["numeric_missing"] = missing
eda["outlier_rules"] = outlier_rules
eda["outlier_counts"] = outliers
for c in numeric:
    q = quantiles[c] or [None] * 5   # an all-missing column has no percentiles
    print(f"{c:17s} min {q[0]} | p1 {q[1]} | median {q[2]} | p99 {q[3]} | max {q[4]} | missing {missing[c]:,}")
print(json.dumps(outliers, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Correlations with the label (point-biserial = Pearson with a 0/1 label)
# MAGIC Pairs with a missing value are dropped for each pair only.

# COMMAND ----------

corr_cols = ["start_hour", "is_night", "Temperature_F", "Visibility_mi", "Wind_Speed_mph", "Precipitation_in",
             "junction", "crossing", "traffic_signal", "stop", "county_prior_accidents", "county_prior_severe_rate"]
correlations = {}
for c in corr_cols:
    pair = features.select(F.col(c).cast("double").alias("x"), F.col("label").cast("double").alias("y")).na.drop()
    value = pair.stat.corr("x", "y")
    correlations[c] = value if value is not None and math.isfinite(value) else None   # constant column -> None
leak_check = gold.select(F.col("Distance_mi").cast("double").alias("x"), F.col("severe").cast("double").alias("y")) \
                 .na.drop().stat.corr("x", "y")
leak_check = leak_check if leak_check is not None and math.isfinite(leak_check) else None
no_history = features.filter("county_prior_accidents = 0").count()
eda["label_correlations"] = correlations
eda["distance_mi_label_correlation_excluded_leakage"] = leak_check
eda["rows_without_county_history"] = no_history
for c, v in sorted(correlations.items(), key=lambda kv: -abs(kv[1] or 0)):
    print(f"{c:25s} {v:+.4f}" if v is not None else f"{c:25s} undefined")
print(f"Distance_mi (excluded, known only after the accident): {leak_check}")
print(f"Rows with no earlier accident in their county: {no_history:,} ({no_history / n:.2%})")

# COMMAND ----------

# 8. Save the EDA evidence (one file per published run, plus a stable name).
write_json(f"{reports}/eda/{latest['run_id']}.json", eda, overwrite=True)
write_json(f"{reports}/eda.json", eda, overwrite=True)
gold.unpersist()
print(json.dumps(eda, indent=2, default=str)[:3000])
print("EDA saved:", f"{reports}/eda.json")
