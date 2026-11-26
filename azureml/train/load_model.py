"""load_model.py - a fresh Python process that knows only a name and a version.

    python3 train/load_model.py accident-severity 1

Copied from Lab-9 load_model.py (only the example row changed). It asks the registry for
that version, prints the tags, loads the model, predicts one row and writes
evidence/loaded.txt. No run id and no path: the name and the version are enough.
"""
import json
import subprocess
import sys

import pandas as pd

from aml_common import WEEK, connect, evidence

# One made-up accident, only to show the model loads and answers.
EXAMPLE = {"start_hour": 8, "day_of_week": 3, "month": 5, "is_night": 0, "Temperature_F": 70.0,
           "Visibility_mi": 10.0, "Wind_Speed_mph": 7.0, "Precipitation_in": 0.0, "junction": 0,
           "crossing": 0, "traffic_signal": 1, "stop": 0, "county_prior_accidents": 1500,
           "county_prior_severe_rate": 0.3}


def via_registry(mlflow, client, name, version):
    mv = client.get_model_version(name, version)
    tags = dict(mv.tags or {})
    model = mlflow.sklearn.load_model(f"models:/{name}/{version}")
    return tags, model, getattr(mv, "run_id", "") or ""


def via_cli(mlflow, name, version):
    """Fallback: the CLI fetches tags and downloads the files, MLflow loads them."""
    info = json.loads(subprocess.run(["az", "ml", "model", "show", "-n", name, "-v", str(version), "-o", "json"],
                                     capture_output=True, text=True, timeout=120, check=True).stdout)
    dl = WEEK / f"model-v{version}"
    if not list(dl.rglob("MLmodel")):
        subprocess.run(["az", "ml", "model", "download", "-n", name, "-v", str(version), "--download-path", str(dl)],
                       check=True, timeout=600)
    mlmodel = next(dl.rglob("MLmodel"), None)
    if mlmodel is None:
        raise SystemExit(f"downloaded {dl}, but found no MLmodel file inside it")
    return info.get("tags") or {}, mlflow.sklearn.load_model(str(mlmodel.parent)), info.get("job_name", "") or ""


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python3 train/load_model.py <model-name> <version>")
    name, version = sys.argv[1], sys.argv[2]
    mlflow, client = connect()
    how = "registry"
    try:
        tags, model, run_id = via_registry(mlflow, client, name, version)
    except Exception as e:  # noqa: BLE001
        print(f"(registry API path failed: {type(e).__name__}: {str(e)[:120]} - using the CLI instead)")
        how = "cli"
        tags, model, run_id = via_cli(mlflow, name, version)

    print(f"model    : {name} version {version}   (loaded via {how})")
    if run_id:
        print(f"from run : {run_id}")
    print("tags     :")
    for k in sorted(tags):
        print(f"  {k:<24}{tags[k]}")
    steps = getattr(model, "named_steps", {})
    last = list(steps.values())[-1] if steps else model
    print(f"object   : {type(model).__name__} -> {type(last).__name__}")
    cols = list(getattr(model, "feature_names_in_", []))
    if cols:
        one = pd.DataFrame([{c: EXAMPLE.get(c, 0.0) for c in cols}])[cols].astype(float)
        p = float(model.predict_proba(one)[0, 1])
        threshold = getattr(steps.get("model"), "threshold", 0.5) if steps else 0.5
        print(f"predict  : example accident -> P(severe) {p:.3f}, threshold {threshold} -> label {int(model.predict(one)[0])}")
    evidence("loaded.txt").write_text(f"{name}:{version} tags={len(tags)} via={how} object={type(last).__name__}\n")
    print("wrote evidence/loaded.txt")


if __name__ == "__main__":
    main()
