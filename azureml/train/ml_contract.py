"""The split and the scores, shared by every Phase-2 script.

Adapted from the Lab-8/Lab-9 ml_contract.py. What changed:
  * The key is the accident ID (Lab: asin|reviewerID|unixReviewTime).
  * The time column is event_ts (the accident's Start_Time).
  * Test and deploy start on fixed dates, as the approved hypothesis says:
    test = 2022, deploy = 2023. Validation = the last 15% of the pre-2022 rows,
    cut on whole UTC days like the Lab (no day is divided).
  * The class we want to find is "severe" (label 1). The Lab looked for label 0.
Kept from the Lab: snapshot_fingerprint, membership_hash, fixed_sample, split_manifest.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DAY = 86400
KEY_COLS = ["ID"]
TIME_COL = "event_ts"
# Never predictors: the key, the time columns, the free-text weather (score.py takes numbers
# only), and the target. See FEATURES.md.
EXCLUDED = {"ID", "event_ts", "accident_year", "weather_condition", "label"}
METHOD = "fixed-date-time-split-v1"
TEST_FROM = 1640995200     # 2022-01-01 00:00 UTC: the hypothesis test years start here
DEPLOY_FROM = 1672531200   # 2023-01-01 00:00 UTC: the newest rows, kept for the endpoint (Lab-10 "unseen")


def snapshot_fingerprint(folder):
    """Verify the retained bytes before training or evaluating them (unchanged from the Lab)."""
    root = Path(folder)
    manifest = root/"_manifest.json"
    if not manifest.exists():
        raise ValueError("retained snapshot manifest missing: run retain_data.py and register a new data version")
    data = json.loads(manifest.read_text())
    actual = {str(p.relative_to(root)) for p in root.rglob("*.parquet")}
    if actual != {p["path"] for p in data["parts"]}:
        raise ValueError("snapshot part list differs from its retained manifest")
    for part in data["parts"]:
        p = root/part["path"]
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024*1024), b""):
                h.update(chunk)
        if h.hexdigest() != part["sha256"]:
            raise ValueError(f"snapshot bytes changed: {part['path']}")
    payload = {k: v for k, v in data.items() if k != "snapshot_sha256"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if digest != data["snapshot_sha256"]:
        raise ValueError("snapshot manifest hash differs")
    return digest


def approved_features(columns):
    return [c for c in columns if c not in EXCLUDED]


def seconds(frame):
    """event_ts as whole seconds since 1970 (UTC)."""
    ts = pd.to_datetime(frame[TIME_COL], utc=True)
    return ((ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).astype("int64")


def ids(frame):
    if "ID" not in frame.columns:
        raise ValueError("accident key ID missing")
    return frame["ID"].astype(str)


def membership_hash(frame):
    # Sorting makes this independent of file order and DataFrame row order.
    value = "\n".join(sorted(ids(frame))) + "\n"
    return hashlib.sha256(value.encode()).hexdigest()


def split_frame(frame, val_size=.15):
    """train / val / test / deploy, oldest to newest. Needs the column event_s (seconds).

    train + val = before 2022; val = its last ~15% of rows, starting at a whole UTC day.
    test = 2022. deploy = 2023.
    """
    if ids(frame).duplicated().any():
        raise ValueError("duplicate accident IDs: publish a unique table first")
    s = frame["event_s"]
    days = (s // DAY) * DAY
    before = days[s < TEST_FROM]
    counts = before.value_counts().sort_index()
    if len(counts) < 2 or not (s >= TEST_FROM).any() or not (s >= DEPLOY_FROM).any():
        raise ValueError("need rows before 2022, in 2022 and in 2023")
    ends = counts.cumsum().to_numpy()
    pos = int(np.searchsorted(ends, len(before)*(1-val_size), side="left")) + 1
    pos = max(1, min(pos, len(counts)-1))         # keep train and val nonempty
    val_from = int(counts.index[pos])
    masks = [days < val_from,
             (days >= val_from) & (s < TEST_FROM),
             (s >= TEST_FROM) & (s < DEPLOY_FROM),
             s >= DEPLOY_FROM]
    return tuple(frame.loc[m].sort_values(KEY_COLS, kind="stable").copy() for m in masks)


def fixed_sample(frame, n, seed=42):
    """A stable sample chosen by a hash of the ID, independent of training size (unchanged)."""
    if n < 0:
        raise ValueError("sample size must be nonnegative")
    if not n or n >= len(frame):
        return frame.copy()
    key = ids(frame)
    ranks = key.map(lambda x: hashlib.sha256(f"{seed}:{x}".encode()).hexdigest())
    return frame.loc[ranks.sort_values(kind="stable").index[:n]].copy()


def split_manifest(parts, seed=42):
    names = ["train", "val", "test", "deploy"]
    out = {"method": METHOD, "seed": seed,
           "label_availability": "county history uses only strictly earlier accidents (Phase-1 FEATURES.md)",
           "membership_sha256": {n: membership_hash(f) for n, f in zip(names, parts)}}
    out.update({n: len(f) for n, f in zip(names, parts)})
    out["boundaries"] = {"val_from_ts": int(parts[1].event_s.min()),
                         "test_from_ts": int(parts[2].event_s.min()),
                         "deploy_from_ts": int(parts[3].event_s.min())}
    out["deploy_from_ts"] = out["boundaries"]["deploy_from_ts"]
    encoded = json.dumps(out, sort_keys=True, separators=(",", ":")).encode()
    out["split_id"] = hashlib.sha256(encoded).hexdigest()
    return out


def severity_scores(y, pred):
    """Every rate with the SEVERE accident (label 1) as the class we want to find."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "recall_severe": tp/(tp+fn) if tp+fn else 0.,
            "precision_severe": tp/(tp+fp) if tp+fp else 0.,
            "f1_severe": float(f1_score(y, pred, pos_label=1, zero_division=0)),
            "recall_not_severe": tn/(tn+fp) if tn+fp else 0.,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}
