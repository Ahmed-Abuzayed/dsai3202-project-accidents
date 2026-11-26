"""model_yml.py - write the registration file for one run, from what the run logged.

    python3 train/model_yml.py <run-id> <version> --commit <git-commit>     # writes train/model-v<version>.yml

Adapted from Lab-9 model_yml.py. Nothing is typed by hand: the features, threshold, metrics,
split and data hash come from the run itself, and the data asset from the job's inputs (or
its parent sweep's). Added for Phase 2: the model name accident-severity, the code commit,
the sampling facts and a limitations tag. You still run `az ml model create`.
"""
import argparse
import json
import re
import subprocess

from aml_common import WEEK, connect

LIMITATIONS = ("Trained on a fixed 150,000-row sample of accidents from 2016 up to val_from_ts; "
               "the severe share falls from 24.1% (before 2022) to 6.4% (2022) (Phase-1 EDA); "
               "providers report severity differently (Phase-1 EDA); "
               "the endpoint takes the 14 computed features, not raw records.")


def az_json(*args):
    out = subprocess.run(["az", "ml", "job", "show", "-n", *args, "-o", "json"], capture_output=True, text=True, timeout=120)
    return json.loads(out.stdout) if out.returncode == 0 and out.stdout.strip() else {}


def data_asset_of(run_id: str, parent: str) -> str:
    for name in (run_id, parent):
        if not name:
            continue
        path = str(((az_json(name).get("inputs") or {}).get("data") or {}).get("path") or "")
        m = re.search(r"azureml:([\w-]+):(\d+)$", path) or re.search(r"data/([\w-]+)/versions/(\d+)", path)
        if m:
            return f"{m.group(1)}:{m.group(2)}"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("version")
    ap.add_argument("--commit", required=True, help="the git commit of the code that trained this run")
    ap.add_argument("--limitations", default=LIMITATIONS)
    args = ap.parse_args()
    if not re.fullmatch(r"[0-9a-f]{7,40}", args.commit):
        raise SystemExit("--commit must be a git commit hash (git rev-parse HEAD)")
    _, client = connect()

    run = client.get_run(args.run)
    p, m, tags = run.data.params, run.data.metrics, run.data.tags
    parent = tags.get("mlflow.parentRunId", "")
    asset = data_asset_of(args.run, parent)
    if asset == "unknown" or not p.get("data_sha256") or not p.get("split_id"):
        raise SystemExit("Run lacks retained data/split lineage. Train it with train/train.py before registering")
    hp = {"logreg": f"C={p.get('C')}", "rf": f"n_estimators={p.get('n_estimators')}"}.get(
        p.get("model"), f"estimator={p.get('estimator', '?')}") + f", severe_weight={p.get('severe_weight', '1.0')}"
    rnd = lambda k, d=4: str(round(m[k], d)) if k in m else "not logged"
    t = {
        "git_commit": args.commit,
        "training_run": args.run,
        "sweep_run": parent or "none",
        "data_asset": asset,
        "data_sha256": p.get("data_sha256", ""),
        "feature_set": p.get("feature_set", ""),
        "features": p.get("features", ""),
        "excluded_features": p.get("excluded_features", ""),
        "split_id": p.get("split_id", ""),
        "split_method": p.get("split_method", ""),
        "test_membership_sha256": p.get("test_membership_sha256", ""),
        "val_from_ts": p.get("val_from_ts", "?"),
        "test_from_ts": p.get("test_from_ts", "?"),
        "deploy_from_ts": p.get("deploy_from_ts", "?"),
        "train_severe_rate": p.get("train_severe_rate", "?"),
        "model": p.get("model", "?"),
        "estimator": p.get("estimator", "?"),
        "hyperparameters": hp,
        "severe_weight": p.get("severe_weight", "1.0"),
        "threshold": p.get("threshold", "?"),
        "seed": p.get("seed", "?"),
        "sample_rows": p.get("sample_rows", "?"),
        "sample_method": p.get("sample_method", "?"),
        "val_accuracy": rnd("val_accuracy"),
        "val_balanced_accuracy": rnd("val_balanced_accuracy"),
        "val_severe_found": rnd("val_severe_found"),
        "val_precision_severe": rnd("val_precision_severe"),
        "baseline_accuracy": rnd("baseline_accuracy"),
        "latency_p95_ms": rnd("latency_p95_ms", 2),
        "limitations": args.limitations,
        "stage": "candidate",
    }
    desc = (f"Severe accident (Severity 3-4) on {t['feature_set']}, {hp}, threshold {t['threshold']}. "
            f"From run {args.run}" + (f", a trial of sweep {parent}." if parent else "."))
    lines = ["$schema: https://azuremlschemas.azureedge.net/latest/model.schema.json",
             "name: accident-severity", f"version: {args.version}", "type: mlflow_model",
             f"path: azureml://jobs/{args.run}/outputs/artifacts/paths/model/",
             f"description: {json.dumps(desc)}", "tags:"]
    lines += [f"  {k}: {json.dumps(str(v))}" for k, v in t.items()]
    out = WEEK / "train" / f"model-v{args.version}.yml"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out.relative_to(WEEK)}  - now run: az ml model create -f {out.relative_to(WEEK)}")


if __name__ == "__main__":
    main()
