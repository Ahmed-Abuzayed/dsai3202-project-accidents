# Databricks notebook source
# MAGIC %md
# MAGIC # Inspect the job runs
# MAGIC Adapted from Lab-6 `04_inspect`. Not a job task: run it by hand after each job run.
# MAGIC Paste a job run ID into the `RUN_ID` widget at the top and run the cells named below.
# MAGIC
# MAGIC | After | Run |
# MAGIC |---|---|
# MAGIC | first successful run | 3.1 → 4.1 |
# MAGIC | safe rerun (`compare_to` set) | 3.1 → 5.1 |
# MAGIC | controlled failure (`test_bad_severity = true`) | 3.1, then 6.1 |

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

# 3.1 Read one job run's saved input and profile. Paste the job run ID (not the job ID).
lake, catalog, schema = "amazonlake60312217", "dsai3202_databricks", "s_60312217"
dbutils.widgets.text("RUN_ID", "", "Job run ID")   # YOUR VALUE: job run ID from Jobs & Pipelines → Runs
RUN_ID = dbutils.widgets.get("RUN_ID").strip()
assert re.fullmatch(r"[A-Za-z0-9_-]{1,80}", RUN_ID), "Set RUN_ID: copy the job run ID from Jobs & Pipelines."
spark.conf.set("spark.sql.session.timeZone", "UTC")
curated = f"abfss://curated@{lake}.dfs.core.windows.net"
reports = f"{curated}/quality/accidents"
quality = f"{reports}/runs/{RUN_ID}"
gold_table = f"{catalog}.{schema}.gold_accidents"
manifest = read_json(f"{quality}/input.json")
summary = read_json(f"{quality}/profile.json")
print(json.dumps({k: manifest[k] for k in ["run_id", "test_bad_severity", "compare_to",
                  "silver_table", "silver_version", "expected_rows", "started_at"]}, indent=2))
total = summary["rows"]
print(f"\nAccidents: {total:,}, starting from {summary['earliest_start']} to {summary['latest_start']}.")
q25, q50, q75, q95 = summary["visibility_mi"]["q25_q50_q75_q95"]
print(f"Visibility (miles): median {q50}, q95 {q95}, maximum {summary['visibility_mi']['max']}.")
for r in summary["severity"]:
    print(f"  Severity {r['Severity']}: {r['count']:>9,} accidents ({r['count'] / total:.1%})")
print("Severity-visibility correlation:", summary["severity_visibility_correlation"])
display(spark.createDataFrame(summary["years"]).orderBy("accident_year"))
nulls = {k: v for k, v in summary["null_counts"].items() if v}
print("Columns with missing values:", json.dumps(nulls, indent=2))
print("All input rows were profiled. Quantiles have at most 1% rank error.")

# COMMAND ----------

# 4.1 Set RUN_ID in 3.1 to the first successful run and run 3.1 first.
gate = read_json(f"{quality}/gate.json")
success = read_json(f"{quality}/success.json")
eda = read_json(f"{quality}/coverage.json")
print(json.dumps(gate, indent=2))
print(json.dumps(success, indent=2))
display(spark.createDataFrame(eda["severity"]).orderBy("Severity"))
display(spark.createDataFrame(eda["years"]).orderBy("accident_year"))
print("Severe label share (Severity 3-4):", eda["positive_label_share"])
print("Gold rows missing weather:", eda["gold_rows_missing_weather"])
print("Gold rows missing visibility:", eda["gold_rows_missing_visibility"])
for label, column, rows in [("severity", "Severity", eda["severity"]), ("year", "accident_year", eda["years"])]:
    low = min(rows, key=lambda r: r["retained_pct"])
    print(f"Lowest retention by {label}: {low[column]} kept {low['retained_pct']}% ({low['after']:,} of {low['before']:,}).")
print(f"Balance: {gate['rows_in']:,} silver = {gate['accepted_rows']:,} gold + {gate['quarantined_rows']:,} quarantined.")
display(spark.read.parquet(success["features_path"]).orderBy(*KEY).limit(10))

# COMMAND ----------

# 5.1 Set RUN_ID in 3.1 to the rerun and run 3.1 first.
comparison = read_json(f"{quality}/comparison.json")
success = read_json(f"{quality}/success.json")
assert comparison["checked"], "This run had no compare_to. Use the rerun's ID."
baseline = read_json(f"{reports}/runs/{comparison['baseline_run']}/success.json")
assert comparison["same_gold"] and comparison["same_features"]
print(f"SAME CONTENT: run {RUN_ID} matches run {comparison['baseline_run']}")
print(f"  gold rows: {success['gold_rows']:,} vs {baseline['gold_rows']:,}")
print(f"  features rows: {success['features_rows']:,} vs {baseline['features_rows']:,}")
print(f"  silver version used: {success['silver_version']} vs {baseline['silver_version']}")

# COMMAND ----------

# 6.1 Set RUN_ID in 3.1 to the failed test run and run 3.1 first.
assert manifest["test_bad_severity"] == "true", "Use the controlled-failure run's ID."
gate = read_json(f"{quality}/gate.json")
assert not gate["passed"] and "severity" in gate["failed_rules"], gate["failed_rules"]
try:
    dbutils.fs.ls(f"{quality}/success.json")
    raise AssertionError("A test run wrote success.json.")
except Exception as error:
    if "A test run" in str(error):
        raise
latest = read_json(f"{reports}/latest.json")
assert latest == manifest["publication_before_test"], "latest.json changed during the test."
gold_version = int(spark.sql(f"DESCRIBE HISTORY {gold_table}").first()["version"])
assert gold_version == manifest["gold_version_before_test"], "gold_accidents changed during the test."
print("Failed rules:", gate["failed_rules"])
print(f"BLOCKED SAFELY: run {RUN_ID} published nothing; latest run is still {latest['run_id']}, "
      f"gold_accidents is still version {gold_version}.")
