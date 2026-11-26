"""Evaluate the chosen run and both baselines ONCE, on one fixed untouched test view (2022).

    python3 train/evaluate_holdout.py

Adapted from Lab-9 evaluate_holdout.py. Same idea: rebuild the test rows from the retained
data, check they are exactly the run's test rows (membership hash), take a fixed sample,
score the model and the baseline on the same rows, log test_* metrics to the run.
What changed:
  * one model run (.chosen) instead of two;
  * two baselines from the hypothesis: B1 always "not severe" (the training majority) and
    B2 "severe" when county_prior_severe_rate > the training severe rate;
  * the test view is 20,000 accidents (Lab: 2,000), because severe accidents are rarer here;
  * it also prints representative errors and one error slice: the data provider (Source),
    as promised in Phase 1. Source is not a feature, so it is read from the Phase-1 gold table.
Test results are final evidence: do not change the settings chosen on validation after this.
"""
import argparse
import hashlib
import json
import re
import subprocess

import numpy as np
import pandas as pd

from aml_common import connect, evidence, read_file
from ml_contract import (DEPLOY_FROM, KEY_COLS, TEST_FROM, fixed_sample, membership_hash,
                         seconds, severity_scores)
from model_yml import data_asset_of

GOLD = "curated/accidents_runs/1100058089459956/gold"   # Phase-1 published run (write-once folder)


def lake_fs(uri):
    from adlfs import AzureBlobFileSystem
    from azure.identity import AzureCliCredential
    m = re.fullmatch(r"abfss://([^@]+)@([^.]+)\.dfs\.core\.windows\.net/(.*)", uri)
    if not m:
        raise SystemExit(f"data asset is not an ADLS path: {uri}")
    container, lake, rest = m.groups()
    return AzureBlobFileSystem(account_name=lake, credential=AzureCliCredential()), container+"/"+rest.rstrip("/")


def read_test_period(asset, columns):
    """Rows of 2021-2023 from the retained snapshot, after checking every file's SHA-256 (as in the Lab)."""
    import pyarrow.dataset as ds
    name, version = asset.split(":")
    uri = subprocess.check_output(["az", "ml", "data", "show", "-n", name, "-v", version,
                                   "--query", "path", "-o", "tsv"], text=True).strip()
    fs, folder = lake_fs(uri)
    manifest = json.loads(fs.cat_file(folder+"/_manifest.json"))
    payload = {k: v for k, v in manifest.items() if k != "snapshot_sha256"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if digest != manifest["snapshot_sha256"]:
        raise SystemExit("retained manifest hash differs")
    for part in manifest["parts"]:
        h = hashlib.sha256()
        with fs.open(folder+"/"+part["path"], "rb") as f:
            for chunk in iter(lambda: f.read(1024*1024), b""):
                h.update(chunk)
        if h.hexdigest() != part["sha256"]:
            raise SystemExit("retained snapshot bytes differ")
    dataset = ds.dataset(folder, filesystem=fs, format="parquet")
    # accident_year narrows the read; the exact cut below uses event_ts, like train.py.
    table = dataset.to_table(columns=list(dict.fromkeys(columns)),
                             filter=ds.field("accident_year") >= 2021)
    return table.to_pandas(), digest, fs


def provider_of(fs, test_ids):
    """Source for the test IDs, from the Phase-1 gold table (only this slice needs it)."""
    import pyarrow.dataset as ds
    gold = ds.dataset(GOLD, filesystem=fs, format="parquet")    # files starting with _ (the Delta log) are skipped
    t = gold.to_table(columns=["ID", "Source"], filter=ds.field("ID").isin(list(test_ids)))
    return t.to_pandas().drop_duplicates("ID").set_index("ID")["Source"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-rows", type=int, default=20000)
    a = ap.parse_args()
    mlflow, client = connect()
    run_id = read_file(".chosen")
    if not run_id:
        raise SystemExit("Write the chosen run id to .chosen first")
    run = client.get_run(run_id)
    p = run.data.params
    columns = p["features"].split(",")
    asset = data_asset_of(run_id, run.data.tags.get("mlflow.parentRunId", ""))
    df, digest, fs = read_test_period(asset, columns + KEY_COLS + ["label", "event_ts", "accident_year"])
    if digest != p["data_sha256"]:
        raise SystemExit("Data bytes differ from the training run")
    s = seconds(df)
    test_all = df[(s >= TEST_FROM) & (s < DEPLOY_FROM)]
    if membership_hash(test_all) != p["test_membership_sha256"]:
        raise SystemExit("These test rows are not the run's test rows (membership hash differs)")
    seed = int(p["seed"])
    test = fixed_sample(test_all, a.max_rows, seed).sort_values(KEY_COLS).reset_index(drop=True)
    view_hash = membership_hash(test)
    if p.get("test_view_sha256", view_hash) != view_hash:
        raise SystemExit("A different final test view was already recorded; keep its fixed membership")

    y = test["label"].astype(int).to_numpy()
    train_rate = float(p["train_severe_rate"])
    b1 = np.full(len(test), int(train_rate >= .5))                                      # B1: training majority
    b2 = (test["county_prior_severe_rate"].astype(float) > train_rate).astype(int).to_numpy()   # B2: county rule
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    X = test[columns].astype(float)
    prob = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    results = {"model": severity_scores(y, pred), "B1": severity_scores(y, b1), "B2": severity_scores(y, b2)}

    # Representative errors: the first five of each kind, in the fixed ID order
    show = ["ID", "label"] + [c for c in ["start_hour", "Visibility_mi", "county_prior_accidents",
                                          "county_prior_severe_rate", "crossing", "traffic_signal"] if c in test]
    err = test[show].copy()
    err["p_severe"] = prob.round(4)
    missed = err[(pred == 0) & (y == 1)].head(5)
    false_alarm = err[(pred == 1) & (y == 0)].head(5)

    # Error slice: data provider
    slices = {}
    try:
        source = test["ID"].map(provider_of(fs, test["ID"])).fillna("unknown").to_numpy()
        for name in sorted(set(source)):
            m = source == name
            slices[str(name)] = {"n": int(m.sum()), "severe_share": float(y[m].mean()),
                                 "model": severity_scores(y[m], pred[m]), "B2": severity_scores(y[m], b2[m])}
    except Exception as e:  # noqa: BLE001 - the main result does not depend on the slice
        print(f"(provider slice skipped: {type(e).__name__}: {e})")

    summary = {"run_id": run_id, "n": len(test), "test_rows_2022": len(test_all), "data_asset": asset,
               "data_sha256": digest, "split_id": p["split_id"], "test_membership_sha256": p["test_membership_sha256"],
               "test_view_sha256": view_hash, "threshold": p["threshold"], "train_severe_rate": train_rate,
               "results": results, "slice_by_source": slices,
               "missed_severe_examples": missed.to_dict("records"),
               "false_alarm_examples": false_alarm.to_dict("records")}
    out = evidence("holdout.json")
    out.write_text(json.dumps(summary, indent=2, default=float) + "\n")
    for name, r in results.items():
        prefix = "test_" if name == "model" else f"test_{name.lower()}_"
        for k, v in r.items():
            client.log_metric(run_id, prefix + k, v)
    client.log_param(run_id, "test_view_sha256", view_hash)
    client.log_artifact(run_id, str(out), "final_evaluation")

    print(f"\nThe same {len(test):,} untouched 2022 test accidents ({y.mean():.1%} severe):")
    print(f"{'':34}{'bal_acc':>8}{'recall_sev':>11}{'prec_sev':>9}{'f1_sev':>8}{'accuracy':>10}")
    for name, label in [("model", f"model ({p['model']}, t={p['threshold']})"),
                        ("B1", "B1 always 'not severe'"), ("B2", "B2 county rate > train rate")]:
        r = results[name]
        print(f"{label:<34}{r['balanced_accuracy']:>8.4f}{r['recall_severe']:>11.4f}{r['precision_severe']:>9.4f}"
              f"{r['f1_severe']:>8.4f}{r['accuracy']:>10.4f}")
    print("\nMissed severe accidents (first 5):")
    print(missed.to_string(index=False))
    print("\nFalse alarms (first 5):")
    print(false_alarm.to_string(index=False))
    if slices:
        print("\nError slice by provider (Source):")
        print(f"{'Source':<10}{'n':>8}{'severe':>8}{'model recall':>14}{'model prec':>12}{'B2 recall':>11}{'B2 prec':>9}")
        for name, sl in slices.items():
            print(f"{name:<10}{sl['n']:>8}{sl['severe_share']:>8.1%}{sl['model']['recall_severe']:>14.4f}"
                  f"{sl['model']['precision_severe']:>12.4f}{sl['B2']['recall_severe']:>11.4f}{sl['B2']['precision_severe']:>9.4f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
