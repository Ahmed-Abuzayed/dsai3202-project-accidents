"""Train the accident-severity classifier.

Adapted from the Lab-9 train.py (which extends Lab-8's). Same steps: read the retained
data asset, check its hash, split on time, fit a scikit-learn Pipeline on training rows
only, score validation, time single-row predictions, log everything to MLflow.

What changed for the project:
  1. Data: the Phase-1 feature snapshot (accidents-features), key ID, time event_ts.
  2. Split: train/val before 2022, test 2022, deploy 2023 (see ml_contract.py).
  3. Target: severe = 1 (Severity 3 or 4). Scores look for the severe class.
     The Lab's --unhappy_weight became --severe_weight (how much a missed severe counts).
  4. "mine" is fixed to HistGradientBoostingClassifier, from the Lab-9 menu, and my metric
     is val_flagged_share (the share of accidents the model calls severe).
  5. No placeholder fix: a county rate of 0 with no history is already marked by
     county_prior_accidents = 0, which is also a feature.
  6. Logs train_severe_rate (the full training partition), used by baseline B2.

Local test:
    python train.py --data ./some_folder --features start_hour,county_prior_severe_rate --feature_set test
"""

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import FixedThresholdClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_contract import (KEY_COLS, TIME_COL, approved_features, fixed_sample, seconds,
                         severity_scores, snapshot_fingerprint, split_frame, split_manifest)

FEATURES: list[str] = []
TARGET = "label"             # 1 if Severity is 3 or 4 (Phase-1 FEATURES.md)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the accident severity model.")
    p.add_argument("--data", required=True, help="folder of parquet files (mounted by the job)")
    p.add_argument("--features", required=True, help="comma-separated names, from FEATURES.md")
    p.add_argument("--feature_set", required=True, help="where the names came from")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_size", type=float, default=0.15, help="share of the pre-2022 rows for validation")
    p.add_argument("--model", choices=["logreg", "rf", "mine"], default="logreg")
    p.add_argument("--C", type=float, default=1.0, help="logreg: inverse regularisation strength")
    p.add_argument("--severe_weight", type=float, default=1.0,
                   help="how much a mistake on a severe accident (label 1) counts in training; 1 = like any other")
    p.add_argument("--n_estimators", type=int, default=1500, help="rf: number of trees")
    p.add_argument("--threshold", type=float, default=0.5, help="say 'severe' only when P(severe) >= threshold")
    p.add_argument("--sample_rows", type=int, default=150000, help="maximum training rows; 0 = all TRAINING rows")
    p.add_argument("--latency_rows", type=int, default=200, help="single-row predictions to time")
    return p.parse_args()


# --------------------------------------------------------------------------
# 1. Load: every parquet file under the folder, only the columns we need
# --------------------------------------------------------------------------
def load_folder(folder: str) -> pd.DataFrame:
    files = sorted(Path(folder).rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {folder} - is the data asset path right?")
    wanted = list(dict.fromkeys(FEATURES + KEY_COLS + [TARGET, TIME_COL]))
    df = pd.concat([pd.read_parquet(f, columns=wanted) for f in files], ignore_index=True)
    print(f"loaded {len(df):,} rows from {len(files)} parquet file(s)")
    return df


# --------------------------------------------------------------------------
# 2. Make every feature a number (they already are; nulls become NaN)
# --------------------------------------------------------------------------
def to_numbers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in FEATURES:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out[TARGET] = out[TARGET].astype("int64")
    out["event_s"] = seconds(out)
    return out


# --------------------------------------------------------------------------
# 3. The models. "mine" is the gradient boosting option from the Lab-9 menu.
# --------------------------------------------------------------------------
def my_model(args: argparse.Namespace):
    return HistGradientBoostingClassifier(max_iter=200, random_state=args.seed,
                                          class_weight={0: 1.0, 1: args.severe_weight})


def my_metric(y, pred) -> dict:
    # How many accidents get flagged severe: the alert load for whoever reads the flags.
    return {"val_flagged_share": float((pred == 1).mean())}


def build_model(args: argparse.Namespace) -> Pipeline:
    if args.model == "mine":
        clf = my_model(args)
    elif args.model == "rf":
        clf = RandomForestClassifier(n_estimators=args.n_estimators, max_leaf_nodes=128, min_samples_leaf=20,
                                     max_features=0.5, n_jobs=-1, random_state=args.seed,
                                     class_weight={0: 1.0, 1: args.severe_weight})
    else:
        clf = LogisticRegression(C=args.C, max_iter=1000, class_weight={0: 1.0, 1: args.severe_weight})
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),     # medians learned from training rows only
        ("scale", StandardScaler()),
        # predict() says 1 only when P(severe) >= threshold. The threshold travels with the model.
        ("model", FixedThresholdClassifier(clf, threshold=args.threshold, response_method="predict_proba")),
    ])


def time_single_rows(pipe: Pipeline, X: pd.DataFrame, n: int):
    """Model-compute latency only; the endpoint check measures client latency."""
    rows = X.iloc[: min(n, len(X))]
    pipe.predict(rows.iloc[[0]])                  # warm-up, not timed
    ms = []
    for i in range(len(rows)):
        one = rows.iloc[[i]]
        t0 = time.perf_counter()
        pipe.predict(one)
        ms.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(ms)), float(np.percentile(ms, 95)), len(ms)


def main() -> None:
    args = parse_args()
    FEATURES[:] = approved_features([f.strip() for f in args.features.split(",") if f.strip()])
    if not FEATURES:
        raise SystemExit("no features given")
    print(f"features from {args.feature_set}: {', '.join(FEATURES)}")

    data_hash = snapshot_fingerprint(args.data)
    df = to_numbers(load_folder(args.data))
    train, val, test, deploy = split_frame(df, args.val_size)
    splits = split_manifest((train, val, test, deploy), args.seed)
    splits["data_sha256"] = data_hash
    train_severe_rate = float(train[TARGET].mean())          # full training partition, before sampling
    train = fixed_sample(train, args.sample_rows, args.seed)  # sample TRAINING only
    val = fixed_sample(val, 22500, args.seed)                 # same fixed validation view for every run
    splits["train_used"] = len(train)
    splits["val_used"] = len(val)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    # The test and deploy slices are counted and set aside. Nothing below scores them.

    mlflow.log_params({
        "model": args.model, "C": args.C, "n_estimators": args.n_estimators,
        "severe_weight": args.severe_weight,
        "data_sha256": data_hash, "split_id": splits["split_id"],
        "split_method": splits["method"], "excluded_features": "weather_condition",
        "label_availability": splits["label_availability"],
        "val_membership_sha256": splits["membership_sha256"]["val"],
        "test_membership_sha256": splits["membership_sha256"]["test"],
        "threshold": args.threshold, "seed": args.seed, "sample_rows": args.sample_rows,
        "sample_method": "fixed_sample: first N training IDs by sha256(seed:ID)",
        "val_size": args.val_size,
        "features": ",".join(FEATURES), "n_features": len(FEATURES), "feature_set": args.feature_set,
        "n_rows": len(df), "n_train": len(train), "n_val": len(val), "n_test": len(test), "n_deploy": len(deploy),
        "train_severe_rate": round(train_severe_rate, 6),
        "val_from_ts": splits["boundaries"]["val_from_ts"],
        "test_from_ts": splits["boundaries"]["test_from_ts"],
        "deploy_from_ts": splits["deploy_from_ts"],     # the endpoint check scores only rows after this day
    })

    pipe = build_model(args)
    mlflow.log_param("estimator", type(pipe.named_steps["model"].estimator).__name__)
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0

    train_pred = pipe.predict(X_train)
    val_pred = pipe.predict(X_val)
    v, t = severity_scores(y_val, val_pred), severity_scores(y_train, train_pred)
    majority = int(y_train.mean() >= .5)                     # learn the majority from training only
    lat_mean, lat_p95, n_timed = time_single_rows(pipe, X_val, args.latency_rows)

    metrics = {
        "train_accuracy": t["accuracy"], "val_accuracy": v["accuracy"],
        "train_val_gap": t["accuracy"] - v["accuracy"],
        "baseline_accuracy": float((y_val == majority).mean()),
        "val_balanced_accuracy": v["balanced_accuracy"],     # the sweep's primary metric
        "val_severe_found": v["recall_severe"], "val_not_severe_found": v["recall_not_severe"],
        "val_precision_severe": v["precision_severe"], "val_f1_severe": v["f1_severe"],
        "latency_ms_per_row": lat_mean, "latency_p95_ms": lat_p95,
        "fit_seconds": fit_seconds,
    }
    metrics.update(my_metric(np.asarray(y_val), np.asarray(val_pred)))

    # The model first, the metrics last: the sweep reads its metric only after the model is saved.
    mlflow.sklearn.log_model(pipe, "model", input_example=X_val.head(5))
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    environment = {"python": platform.python_version(), "packages": {
        d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata.get("Name")}}
    (out / "environment.json").write_text(json.dumps(environment, indent=2))
    mlflow.log_artifact(str(out / "environment.json"))
    (out / "splits.json").write_text(json.dumps(splits, indent=2))
    mlflow.log_artifact(str(out / "splits.json"))
    mlflow.log_metrics(metrics)

    print("splits :", json.dumps(splits))
    print(f"model  : {args.model} ({type(pipe.named_steps['model'].estimator).__name__}) "
          f"severe_weight={args.severe_weight} threshold={args.threshold}   fit {fit_seconds:.1f}s")
    print(f"val_accuracy {metrics['val_accuracy']:.4f}   baseline_accuracy {metrics['baseline_accuracy']:.4f}")
    print(f"severe found {metrics['val_severe_found']:.4f}   not severe found {metrics['val_not_severe_found']:.4f}"
          f"   val_balanced_accuracy {metrics['val_balanced_accuracy']:.4f} (their average)")
    print(f"precision severe {metrics['val_precision_severe']:.4f}   flagged share {metrics['val_flagged_share']:.4f}")
    print(f"latency_ms_per_row {lat_mean:.2f}   p95 {lat_p95:.2f}   ({n_timed} single-row predictions)")


if __name__ == "__main__":
    main()
