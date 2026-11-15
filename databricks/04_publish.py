# Databricks notebook source
# MAGIC %md
# MAGIC # Task 4 of 4 · Publish accepted rows
# MAGIC Adapted from Lab-6 `03_publish`. It runs only if **validate** succeeded.
# MAGIC
# MAGIC **Its job:** build **gold** from the accepted accidents, build the **features**, and record the result.
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | **Reads** | This run's `input.json`, `gate.json`, input copy and quarantine. |
# MAGIC | **Writes** | This run's gold (Delta) and features (Parquet) in `curated/accidents_runs/<run_id>/`; the current table `gold_accidents` and `curated/accidents_features_v1/`; `coverage.json`, `comparison.json` and `success.json`. |
# MAGIC | **Stops when** | The saved gate did not pass, this is a test run, or the counts do not balance. |
# MAGIC
# MAGIC **Changes from Lab-6:** gold = accepted accidents (silver minus quarantine, an anti join on `ID`)
# MAGIC instead of a join with product metadata; coverage by severity and year; accident features.

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 · Check the saved gate again (defence in depth)

# COMMAND ----------

c = config()
manifest = read_json(f"{c['quality']}/input.json")
gate = read_json(f"{c['quality']}/gate.json")
assert gate["run_id"] == c["run_id"] and gate["passed"], "No passing gate for this run."
assert manifest["test_bad_severity"] == "false", "Test runs never publish."

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 · Build gold and check the balance
# MAGIC Gold keeps every accepted accident: silver rows whose `ID` is not in quarantine.
# MAGIC - gold rows = the rows that validate accepted;
# MAGIC - silver rows = gold rows + quarantined rows.

# COMMAND ----------

silver = spark.read.parquet(manifest["input_path"])
if gate["quarantined_rows"] > 0:
    quarantine_ids = spark.read.parquet(gate["quarantine_path"]).select(*KEY)
    gold = silver.join(quarantine_ids, KEY, "left_anti")
else:
    gold = silver
gold_rows = gold.count()
assert gold_rows > 0 and gold_rows == gate["accepted_rows"]
assert gate["rows_in"] == gold_rows + gate["quarantined_rows"], "Counts do not balance."

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 · Save this run's gold and features
# MAGIC Both are written into this run's own folder with `mode("error")`, then read back.

# COMMAND ----------

gold_path, features_path = f"{c['out']}/gold", f"{c['out']}/features"
gold.write.format("delta").mode("error").save(gold_path)
gold = spark.read.format("delta").load(gold_path)
features = make_features(gold)
assert features.columns == FEATURE_COLUMNS
features.coalesce(4).write.mode("error").parquet(features_path)
features = spark.read.parquet(features_path)
assert features.count() == gold_rows

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 · Measure who was kept
# MAGIC Accidents per severity and per year before (silver) and after (gold) the gate, the share of
# MAGIC severe labels, and the gold rows without weather or visibility.

# COMMAND ----------

eda = {"population": "all silver and accepted gold rows; no sample",
    "severity": [r.asDict() for r in coverage(silver, gold, "Severity").orderBy("Severity").collect()],
    "years": [r.asDict() for r in coverage(silver, gold, "accident_year").orderBy("accident_year").collect()],
    "positive_label_share": features.agg(F.avg("label")).first()[0],
    "gold_rows_missing_weather": gold.filter("Weather_Condition IS NULL").count(),
    "gold_rows_missing_visibility": gold.filter("Visibility_mi IS NULL").count()}
write_json(f"{c['quality']}/coverage.json", eda)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 · Rerun comparison (only when `compare_to` is set)
# MAGIC Same saved input, so gold and features must contain exactly the same rows, checked in both directions.

# COMMAND ----------

comparison = {"baseline_run": c["compare_to"], "checked": bool(c["compare_to"])}
if c["compare_to"]:
    previous = read_json(f"{c['reports']}/runs/{c['compare_to']}/success.json")
    comparison["same_gold"] = same_content(gold, spark.read.format("delta").load(previous["gold_path"]))
    comparison["same_features"] = same_content(features, spark.read.parquet(previous["features_path"]))
    assert comparison["same_gold"] and comparison["same_features"], "Rerun changed values."
    print("SAME CONTENT: gold and features match run", c["compare_to"])
write_json(f"{c['quality']}/comparison.json", comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 · Update the current outputs
# MAGIC `gold_accidents` (Unity Catalog) and `curated/accidents_features_v1/` always hold the latest
# MAGIC published result. Before replacing anything, the cell checks location and schema; after writing,
# MAGIC it checks that the current outputs match this run's snapshot exactly.

# COMMAND ----------

current_gold = f"{c['curated']}/gold_accidents"
current_features = f"{c['curated']}/accidents_features_v1"
if spark.catalog.tableExists(c["gold_table"]):
    detail = spark.sql(f"DESCRIBE DETAIL {c['gold_table']}").first()
    assert detail["format"] == "delta" and detail["location"].rstrip("/") == current_gold, \
        "gold_accidents points elsewhere. Ask the instructor before writing."
    assert detail["partitionColumns"] == [], "Unexpected gold partition layout."
    existing = spark.table(c["gold_table"])
    assert existing.schema.simpleString() == gold.schema.simpleString(), "Existing gold schema differs."
    gold.select(*existing.columns).write.mode("overwrite").insertInto(c["gold_table"])
else:
    gold.write.format("delta").mode("error").option("path", current_gold).saveAsTable(c["gold_table"])
features.coalesce(4).write.mode("overwrite").parquet(current_features)
assert same_content(gold, spark.table(c["gold_table"]))
assert same_content(features, spark.read.parquet(current_features))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 · Write the success record last
# MAGIC `success.json` exists only if the whole task finished. `latest.json` points to the latest published run.

# COMMAND ----------

gold_version = int(spark.sql(f"DESCRIBE HISTORY {c['gold_table']}").first()["version"])
finished = datetime.now(timezone.utc)
report = {"run_id": c["run_id"], "gold_table": c["gold_table"], "gold_version": gold_version,
    "gold_path": gold_path, "features_path": features_path, "gold_rows": gold_rows,
    "quarantined_rows": gate["quarantined_rows"], "silver_rows": gate["rows_in"],
    "features_rows": gold_rows, "features_columns": FEATURE_COLUMNS,
    "current_gold_path": current_gold, "current_features_path": current_features,
    "silver_version": manifest["silver_version"], "comparison": comparison,
    "finished_at": finished.isoformat(),
    "pipeline_elapsed_seconds": (finished - datetime.fromisoformat(manifest["started_at"])).total_seconds()}
write_json(f"{c['quality']}/success.json", report)
write_json(f"{c['reports']}/latest.json", report, overwrite=True)
print(json.dumps(report, indent=2))
print(f"BALANCED: {gate['rows_in']:,} silver = {gold_rows:,} gold + {gate['quarantined_rows']:,} quarantined")
print("PUBLISHED:", c["run_id"])
