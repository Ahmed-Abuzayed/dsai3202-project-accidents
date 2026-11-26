"""score.py - the scoring contract for the accident-severity model.

Copied from Lab-10 score.py. Changed: the 14 project columns, the model name, and check 5:
a null value is accepted as "unknown" (the model's imputer fills it with the training
median, as FEATURES.md asks), while text is still refused.

Two functions are the whole contract:

  init()  runs ONCE, when the deployment starts. It pays the loading cost.
  run()   runs on EVERY request. It pays only for validation and prediction.

Azure ML mounts the registered model and tells you where through the
environment variable AZUREML_MODEL_DIR. Everything else in this file is
about refusing bad input politely, which is most of what a production
scoring script does.

The column list is NOT decided here. The model is a scikit-learn Pipeline
fitted on a DataFrame, so it remembers the columns it was trained on
(feature_names_in_). Those came from FEATURES.md (accidents_features_v1).
The list below is only used if a model does not remember its columns.
"""
import json
import logging
import os

import mlflow.sklearn
import pandas as pd
import numpy as np

log = logging.getLogger("score")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# The inference server sets up logging before this file loads, so basicConfig alone
# changes nothing there. Setting the level on our own logger keeps the INFO lines.
log.setLevel(logging.INFO)

# Fallback only: the 14 predictors from FEATURES.md. The model's own feature_names_in_ wins in init().
FEATURES = [
    "start_hour", "day_of_week", "month", "is_night",
    "Temperature_F", "Visibility_mi", "Wind_Speed_mph", "Precipitation_in",
    "junction", "crossing", "traffic_signal", "stop",
    "county_prior_accidents", "county_prior_severe_rate",
]

# One request scores at most this many accidents. Bigger than this is a batch
# job, not an HTTP call. The check scripts send 200 at a time.
MAX_ROWS = 1000

model = None          # the scikit-learn Pipeline, loaded once in init()
expected = FEATURES   # column order the model was fitted with
model_label = "accident-severity:?"


def _find_mlmodel(root):
    """Return the folder under root that holds the MLmodel file.

    The folder name depends on how the model was registered (a job output
    called 'model', a local folder with another name). Searching for the
    MLmodel file removes the guess.
    """
    for dirpath, _dirs, files in os.walk(root):
        if "MLmodel" in files:
            return dirpath
    return None


def _label_from_path(root):
    """/var/azureml-app/azureml-models/<name>/<version>/... -> 'name:version'."""
    parts = [p for p in root.split(os.sep) if p]
    if "azureml-models" in parts:
        i = parts.index("azureml-models")
        if len(parts) > i + 2:
            return f"{parts[i + 1]}:{parts[i + 2]}"
    return "accident-severity:?"


def init():
    """Load the model. Any error here means no request will ever succeed."""
    global model, expected, model_label
    root = os.getenv("AZUREML_MODEL_DIR", ".")
    path = _find_mlmodel(root)
    if path is None:
        raise FileNotFoundError(
            f"no MLmodel file under AZUREML_MODEL_DIR={root}; contents: {os.listdir(root)}"
        )
    model = mlflow.sklearn.load_model(path)
    # A Pipeline fitted on a DataFrame remembers its column order. Trust it
    # over our constant: it is the one thing that cannot drift from training.
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        expected = [str(c) for c in names]
    model_label = os.getenv("MODEL_LABEL") or _label_from_path(root)
    log.info("init: loaded %s from %s; expects %s", model_label, path, expected)


def _error(message, **extra):
    out = {"error": message, "expected_columns": expected}
    out.update(extra)
    log.warning("rejected request: %s", message)
    return out


def run(raw_data):
    """Turn one JSON request into predictions, or into a useful error.

    Request:   {"data": [{"start_hour": 8, "day_of_week": 3, ..., "county_prior_severe_rate": 0.3}, ...]}
               Each row may also carry an "id"; it is ignored by the model and
               returned in "ids", so the caller can match answers to rows.
    Response:  {"predictions": [1, 0], "probabilities": [0.71, 0.08],
                "model": "accident-severity:1", "n": 2}
               predictions: 1 = severe (Severity 3-4); probabilities: P(severe).
    On bad input the response has an "error" key and "expected_columns".
    """
    # 1. Is there a body at all?
    if raw_data is None or not str(raw_data).strip():
        return _error('empty body: send {"data": [{...}, ...]}')

    # 2. Is it JSON?
    try:
        payload = json.loads(raw_data)
    except (ValueError, TypeError) as exc:
        return _error(f"body is not valid JSON: {exc}")

    # 3. Is it the shape we published?
    if not isinstance(payload, dict) or "data" not in payload:
        return _error("missing top-level key 'data'")
    rows = payload["data"]
    if not isinstance(rows, list) or len(rows) == 0:
        return _error("'data' must be a non-empty list of objects")
    if len(rows) > MAX_ROWS:
        return _error(
            f"too many rows: {len(rows)} (max {MAX_ROWS} per request); split the request",
            rows_received=len(rows),
            max_rows=MAX_ROWS,
        )
    if not all(isinstance(r, dict) for r in rows):
        return _error("every item in 'data' must be an object of feature: value")

    # 4. Are all the columns there?
    missing = [c for c in expected if any(c not in r for r in rows)]
    if missing:
        return _error(f"missing field(s): {', '.join(missing)}", missing=missing)

    # 5. Are they numbers? (True/False count as numbers; null means "unknown"; "rain" does not)
    df = pd.DataFrame(rows)[expected]
    bad = {}
    for col in expected:
        coerced = pd.to_numeric(df[col], errors="coerce")
        invalid = (coerced.isna() & df[col].notna()) | np.isinf(coerced.astype(float))
        if invalid.any():
            first_bad = df[col][invalid].iloc[0]
            bad[col] = repr(first_bad)
        df[col] = coerced
    if bad:
        detail = "; ".join(f"{c} got {v}" for c, v in bad.items())
        return _error(f"wrong type, expected a number: {detail}", wrong_type=bad)
    df = df.astype(float)

    # 6. Predict. predict() applies the threshold stored inside the model (Lab-9),
    #    so the endpoint decides exactly like the run you chose.
    #    If this raises, the server answers 500 and the gateway 424.
    try:
        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1] if hasattr(model, "predict_proba") else None
    except Exception as exc:  # noqa: BLE001 - we want the message in the response
        log.exception("predict failed")
        raise RuntimeError("model prediction failed; inspect deployment logs") from exc

    out = {
        "predictions": [int(p) for p in preds],
        "probabilities": None if probs is None else [round(float(p), 4) for p in probs],
        "model": model_label,
        "n": int(len(rows)),
    }
    if any("id" in r for r in rows):
        out["ids"] = [r.get("id") for r in rows]
    # One log line per request: input size and output. This is what drift
    # monitoring reads later.
    log.info("scored n=%d positives=%d model=%s", out["n"], sum(out["predictions"]), model_label)
    return out
