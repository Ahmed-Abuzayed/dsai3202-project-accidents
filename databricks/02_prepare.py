# Databricks notebook source
# MAGIC %md
# MAGIC # Task 2 of 4 · Prepare the input
# MAGIC Adapted from Lab-6 `01_prepare`. Job order: **clean → prepare → validate → publish**. The job runs it.
# MAGIC
# MAGIC **Its job:** decide exactly which data this run will use, save a private copy of it, and describe it with a profile.
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | **Reads** | The clean report (`curated/quality/accidents/drop_report.json`) and the silver table at the exact version in that report. |
# MAGIC | **Writes** | This run's input copy in `processed/accidents_runs/<run_id>/`, plus `input.json` and `profile.json` in `curated/quality/accidents/runs/<run_id>/`. |
# MAGIC | **Stops when** | The clean report does not match the table, the counts do not balance, or the columns have the wrong types. |
# MAGIC
# MAGIC **Changes from Lab-6:** reads the accidents clean report and table; no product metadata; the
# MAGIC controlled-failure flag is `test_bad_severity` and changes one `Severity` to 9.

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 · Read the job parameters and check the clean report
# MAGIC - The report must belong to **this** silver table.
# MAGIC - Its accounting must balance: rows in = rows kept + rows dropped by each cleaning step.
# MAGIC - A controlled-failure test needs a successful earlier run in `compare_to`.

# COMMAND ----------

c = config()
started = datetime.now(timezone.utc).isoformat()
clean = read_json(f"{c['reports']}/drop_report.json")
assert clean["table"] == c["table"], "The clean report belongs to another table."
assert clean["rows_in"] == 7_728_394 and clean["rows_out"] > 0
assert clean["rows_in"] == clean["rows_out"] + sum(clean["dropped_by_step"].values())
assert c["test_bad_severity"] == "false" or c["compare_to"], "The test needs a successful baseline in compare_to."

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 · Choose this run's input
# MAGIC **Normal run (`compare_to` empty):** read silver with `VERSION AS OF` at the version in the clean report.
# MAGIC
# MAGIC **Comparison run (`compare_to` set):** reuse the input that the earlier run saved. Same input + same code must give the same output.

# COMMAND ----------

if c["compare_to"]:
    previous = read_json(f"{c['reports']}/runs/{c['compare_to']}/success.json")
    base = read_json(f"{c['reports']}/runs/{c['compare_to']}/input.json")
    assert previous["run_id"] == c["compare_to"]
    source = spark.read.parquet(base["input_path"])
    source_version, source_report = base["silver_version"], base["clean_report"]
    expected_rows = base["expected_rows"]
else:
    source_version, source_report = int(clean["silver_version"]), clean
    source = spark.sql(f"SELECT * FROM {c['table']} VERSION AS OF {source_version}")
    expected_rows = int(clean["rows_out"])
assert not schema_errors(source), schema_errors(source)
assert source.count() == expected_rows, "Saved silver count differs from the clean report."

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 · Controlled-failure test (only when `test_bad_severity = true`)
# MAGIC One accident's `Severity` is changed to 9, which is impossible (valid values are 1 to 4).
# MAGIC The change is made in **this run's copy only**. The silver table and earlier runs are never touched.
# MAGIC The validate task should catch it and stop the job. In a normal run this cell does nothing.

# COMMAND ----------

if c["test_bad_severity"] == "true":
    chosen = source.orderBy(*KEY).first()
    source = source.withColumn("Severity", F.when(F.col("ID") == F.lit(chosen["ID"]), F.lit(9))
                               .otherwise(F.col("Severity")))
    print("TEST INPUT ONLY: Severity of", chosen["ID"], "changed to 9.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 · Save this run's input copy and its manifest
# MAGIC `mode("error")` fails if the folder exists, so a run never overwrites another run's folder.
# MAGIC The copy is read back and counted. `input.json` records where the input came from.

# COMMAND ----------

input_path = f"{c['work']}/input"
source.select(*SILVER_COLUMNS).write.mode("error").parquet(input_path)
saved = spark.read.parquet(input_path)
assert saved.count() == expected_rows
manifest = {"run_id": c["run_id"], "test_bad_severity": c["test_bad_severity"],
    "compare_to": c["compare_to"], "silver_table": c["table"], "silver_version": source_version,
    "clean_report": source_report, "expected_rows": expected_rows,
    "input_path": input_path, "started_at": started}
if c["test_bad_severity"] == "true":
    manifest["publication_before_test"] = read_json(f"{c['reports']}/latest.json")
    manifest["gold_version_before_test"] = int(spark.sql(f"DESCRIBE HISTORY {c['gold_table']}").first()["version"])
write_json(f"{c['quality']}/input.json", manifest)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 · Profile every input row
# MAGIC A profile only **describes** the data. Deciding whether it may be published is the next task.

# COMMAND ----------

summary = profile(saved)
write_json(f"{c['quality']}/profile.json", summary)
print(json.dumps(summary, indent=2, default=str))
print("INPUT SAVED:", c["run_id"], input_path)
