# Databricks notebook source
# MAGIC %md
# MAGIC # Task 3 of 4 · Validate before publication
# MAGIC Adapted from Lab-6 `02_validate`. It runs only if **prepare** succeeded.
# MAGIC
# MAGIC **Its job:** decide whether this run's data may be published. A profile *describes* data; a **quality gate** *decides*.
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | **Reads** | This run's `input.json` and input copy (saved by prepare). |
# MAGIC | **Writes** | Accidents that end before they start to `processed/accidents_runs/<run_id>/quarantine/`, and the verdict to `curated/quality/accidents/runs/<run_id>/gate.json`. |
# MAGIC | **Stops when** | Any rule with policy **fail** is broken, or more than 0.5% of rows are quarantined. Rules: `00_common`, section 7. |
# MAGIC
# MAGIC **Changes from Lab-6:** no metadata input; the quarantine reason is `end_before_start`.

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 · Load this run's saved input
# MAGIC Validate reads the copy that prepare saved, not the live silver table.

# COMMAND ----------

c = config()
manifest = read_json(f"{c['quality']}/input.json")
assert manifest["run_id"] == c["run_id"]
silver = spark.read.parquet(manifest["input_path"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 · Apply the gate and set inconsistent rows aside
# MAGIC Rows whose `End_Time` is missing or before `Start_Time` are not deleted. They are saved to
# MAGIC **quarantine** with a reason, read back and counted.

# COMMAND ----------

gate, quarantine = evaluate_gate(silver, manifest["expected_rows"])
gate["run_id"] = c["run_id"]
gate["quarantine_path"] = f"{c['work']}/quarantine"
if quarantine is not None:
    (quarantine.withColumn("reject_reason", F.lit("end_before_start"))
        .write.mode("error").parquet(gate["quarantine_path"]))
    assert spark.read.parquet(gate["quarantine_path"]).count() == gate["quarantined_rows"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 · Save the verdict, then pass or fail the task
# MAGIC 1. **Save `gate.json` first**, so the evidence exists even when the gate fails.
# MAGIC 2. **Then raise an error if the gate failed.** A failed task is what stops publish (`Run if = All succeeded`).

# COMMAND ----------

write_json(f"{c['quality']}/gate.json", gate)
print(json.dumps(gate, indent=2))
if not gate["passed"]:
    raise ValueError("QUALITY GATE FAILED: " + ", ".join(gate["failed_rules"]))
print("QUALITY GATE PASSED")
