"""same_input_check.py - does the deployment answer exactly as the registered model does?

    python3 checks/same_input_check.py              # accident-severity:1, deployment blue, 200 accidents

Adapted from Lab-10 same_input_check.py. The proof is the same rows on both sides:
  1. Take 200 fixed accidents from the 2022 test period, each with its ID. The same 200 every
     run: they are chosen by a fixed hash of the ID. (The Lab took them from all rows before
     the deploy slice; one year is enough and keeps the VM's memory low.) Rows with missing
     weather stay in, sent as null, so the imputer is checked too.
  2. Score them with the registered version, downloaded and loaded on this VM.
  3. Send exactly the same rows to the endpoint, IDs included.
  4. Match the answers by ID. Compare the probabilities and the labels.
score.py rounds probabilities to 4 decimals, so a difference up to 0.00005 is rounding and
nothing else. That is the declared tolerance. Labels must agree.
Writes evidence/same_input_<deployment>.json and the rows sent.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from common import Endpoint, as_json_rows, evidence, local_model, model_info, names, read_rows

TOLERANCE = 0.00005


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=names().get("MODEL", "accident-severity"))
    ap.add_argument("--version", default=names().get("VERSION", "1"))
    ap.add_argument("--deployment", default="blue")
    ap.add_argument("--n", type=int, default=200, help="how many fixed accidents (max 1000, one request)")
    a = ap.parse_args()
    if not 1 <= a.n <= 1000:
        raise SystemExit("--n must be between 1 and 1000")
    tags = model_info(a.model, a.version).get("tags") or {}
    model = local_model(a.model, a.version)
    cols = [str(c) for c in getattr(model, "feature_names_in_", [])]
    print(f"model    : {a.model}:{a.version}  columns {', '.join(cols)}")

    # 1. 200 fixed accidents from 2022, chosen by a fixed hash of the ID
    df, where = read_rows(["ID", "accident_year"] + cols, tags.get("data_asset", ""), tags.get("data_sha256", ""), [2022])
    h = pd.util.hash_pandas_object(df[["ID"]], index=False)
    rows = df.loc[h.nsmallest(a.n).index].copy().reset_index(drop=True)
    rows["id"] = rows["ID"].astype(str)
    rows[cols] = rows[cols].astype(float)
    print(f"rows     : {len(rows)} fixed accidents from {where} (2022), {int(rows[cols].isna().any(axis=1).sum())} with a null value")

    # 2. the registered model, on this VM
    X = rows[cols]
    p_local = model.predict_proba(X)[:, 1]
    y_local = model.predict(X)

    # 3. the same rows, through the endpoint
    payload = as_json_rows(rows, cols)
    evidence(f"same_input_{a.deployment}_rows.json").write_text(json.dumps(payload, indent=1) + "\n")
    t0 = time.time()
    status, served_by, body = Endpoint().post(payload, deployment=a.deployment)
    if status != 200 or not isinstance(body, dict) or "error" in body:
        raise SystemExit(f"the endpoint did not answer correctly: HTTP {status} {str(body)[:200]}")
    print(f"endpoint : HTTP {status}, served by {served_by or a.deployment} ({body.get('model', '?')}) in {time.time() - t0:.1f}s")

    # 4. match by ID, compare
    got = pd.DataFrame({"id": body.get("ids") or [], "p_endpoint": body.get("probabilities") or [],
                        "y_endpoint": body.get("predictions") or []})
    mine = pd.DataFrame({"id": rows["id"], "p_local": p_local, "y_local": y_local})
    both = mine.merge(got, on="id", how="inner")
    max_diff = float((both["p_local"] - both["p_endpoint"]).abs().max()) if len(both) else float("nan")
    labels_agree = int((both["y_local"] == both["y_endpoint"]).sum())
    ok = (body.get("model") == f"{a.model}:{a.version}" and len(both) == len(rows) == body.get("n")
          and max_diff <= TOLERANCE + 1e-9 and labels_agree == len(both))

    print(f"\n{'id':<16}{'local P(severe)':>16}{'endpoint':>10}{'|diff|':>10}  labels")
    for _, r in both.head(5).iterrows():
        print(f"{r['id'][:15]:<16}{r['p_local']:>16.6f}{r['p_endpoint']:>10.4f}{abs(r['p_local'] - r['p_endpoint']):>10.6f}"
              f"  {int(r['y_local'])} / {int(r['y_endpoint'])}")
    print("...")
    print(f"matched by id        : {len(both)} of {len(rows)}")
    print(f"largest |difference| : {max_diff:.6f}   (tolerance {TOLERANCE}, because score.py rounds to 4 decimals)")
    print(f"labels that agree    : {labels_agree} of {len(both)}")
    print("RESULT: " + ("SAME - the deployment answers as the registered model does" if ok else "DIFFERENT"))
    summary = {"model": f"{a.model}:{a.version}", "deployment": a.deployment, "served_by": served_by,
               "endpoint_model": body.get("model"), "n": len(rows), "matched": len(both),
               "max_abs_diff": round(max_diff, 8), "tolerance": TOLERANCE, "labels_agree": labels_agree,
               "rows_with_null": int(rows[cols].isna().any(axis=1).sum()), "passed": bool(ok), "columns": cols}
    out = evidence(f"same_input_{a.deployment}.json")
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"written  : {out}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
