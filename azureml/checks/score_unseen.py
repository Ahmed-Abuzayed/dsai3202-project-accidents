"""score_unseen.py - score accidents the model has never seen, through the live endpoint.

    python3 checks/score_unseen.py                # accident-severity:1, 2,000 accidents from 2023

Adapted from Lab-10 score_unseen.py. train.py kept 2023 aside as the deploy slice and
logged where it starts (deploy_from_ts); model_yml.py put it on the model as a tag.
This keeps only accidents from the day AFTER that day, sends a seeded sample of 2,000 to
the endpoint (200 per request, IDs included) and compares the answers with the labels.
(The Lab also counted an "all of the latest year" shortcut; here the deploy slice is
exactly 2023, so that count is not needed.) Writes evidence/unseen.json.
"""
import argparse
import json
import time
from datetime import datetime, timezone

import pandas as pd

from common import DAY, Endpoint, as_json_rows, evidence, model_info, names, read_rows


def day(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=names().get("MODEL", "accident-severity"))
    ap.add_argument("--version", default=names().get("VERSION", "1"))
    ap.add_argument("--deployment", default="blue")
    ap.add_argument("--max-rows", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    tags = model_info(a.model, a.version).get("tags") or {}
    start = int(float(tags["deploy_from_ts"]))
    cutoff = (start // DAY + 1) * DAY          # midnight after the first day of the slice
    cols = [c for c in tags.get("features", "").split(",") if c]
    print(f"model    : {a.model}:{a.version}   deploy_from_ts {start} = {day(start)}")
    print(f"unseen   : accidents from {day(cutoff)} on (strictly after the day the deploy slice starts)")

    df, where = read_rows(["ID", "event_ts", "accident_year", "label"] + cols, tags.get("data_asset", ""),
                          tags.get("data_sha256", ""), [2023])
    s = (pd.to_datetime(df["event_ts"], utc=True) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)
    unseen = df[s >= cutoff]
    print(f"table    : {where}, {len(unseen):,} unseen accidents")
    sample = unseen.sample(min(a.max_rows, len(unseen)), random_state=a.seed).reset_index(drop=True)
    sample["id"] = sample["ID"].astype(str)
    sample[cols] = sample[cols].astype(float)

    ep, got, failed = Endpoint(), {}, 0
    t1 = time.time()
    for b, i in enumerate(range(0, len(sample), a.batch), 1):
        chunk = sample.iloc[i:i + a.batch]
        status, by, body = ep.post(as_json_rows(chunk, cols), deployment=a.deployment)
        if status != 200 or not isinstance(body, dict) or "error" in body \
                or body.get("model") != f"{a.model}:{a.version}" or len(body.get("ids") or []) != len(chunk):
            print(f"  batch {b}: HTTP {status}, invalid answer {str(body)[:120]}")
            failed += 1
            continue
        got.update(zip(body["ids"], body["predictions"]))
        if b <= 2 or i + a.batch >= len(sample):
            print(f"  batch {b}: {len(chunk)} accidents, HTTP {status}, served by {by or a.deployment} ({body['model']})")
        elif b == 3:
            print("  ...")
    sample["pred"] = sample["id"].map(got)
    scored = sample.dropna(subset=["pred"])
    y, p = scored["label"].astype(int).to_numpy(), scored["pred"].astype(int).to_numpy()
    rec_sev = float(((p == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
    rec_not = float(((p == 0) & (y == 0)).sum() / max((y == 0).sum(), 1))
    prec_sev = float(((p == 1) & (y == 1)).sum() / max((p == 1).sum(), 1))
    bal = (rec_sev + rec_not) / 2
    tag = lambda k: float(tags[k]) if tags.get(k, "not logged") != "not logged" else float("nan")
    print(f"\n                     unseen 2023   validation (model tag)")
    print(f"balanced accuracy    {bal:>11.4f}   {tag('val_balanced_accuracy'):>10.4f}")
    print(f"severe found         {rec_sev:>11.4f}   {tag('val_severe_found'):>10.4f}")
    print(f"precision severe     {prec_sev:>11.4f}   {tag('val_precision_severe'):>10.4f}")
    print(f"severe share         {y.mean():>11.4f}")
    print(f"scored               {len(scored):>11,}   ({failed} failed batches, {time.time() - t1:.0f}s)")
    out = {"passed": failed == 0 and len(scored) == len(sample), "model": f"{a.model}:{a.version}",
           "deployment": a.deployment, "deploy_from_ts": start, "unseen_from": day(cutoff),
           "unseen_rows": int(len(unseen)), "n": int(len(scored)), "failed_batches": failed,
           "severe_share": round(float(y.mean()), 4), "balanced_accuracy": round(bal, 4),
           "recall_severe": round(rec_sev, 4), "precision_severe": round(prec_sev, 4),
           "tag_val_balanced_accuracy": tag("val_balanced_accuracy")}
    path = evidence("unseen.json")
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"written  : {path}")


if __name__ == "__main__":
    main()
