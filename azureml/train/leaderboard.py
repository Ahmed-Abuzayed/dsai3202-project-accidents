"""leaderboard.py - the sweep's trials, sorted, with the latency ceiling applied.

Copied from Lab-9. Changed: severe/not-severe columns, severe_weight, output in evidence/.

    python3 train/leaderboard.py                       # sweep from .sweep, sorted on val_balanced_accuracy
    python3 train/leaderboard.py --sort val_accuracy   # the same trials, sorted on accuracy
    python3 train/leaderboard.py --ceiling-ms 50       # the p95 latency budget (default 50 ms)
    python3 train/leaderboard.py --sort val_flagged_share   # my extra metric

Reads the sweep's child runs from MLflow, prints one row per trial, marks the rows
over the ceiling, and writes evidence/leaderboard.json. The ceiling is judged
on latency_p95_ms: a latency budget is a p95, not an average.
It does not choose for you. The run id you pick goes into .chosen by hand.
"""
import argparse
import json
import pathlib
import subprocess

from aml_common import connect, evidence, read_file

METRICS = ["val_accuracy", "val_balanced_accuracy", "val_severe_found", "val_not_severe_found", "val_precision_severe",
           "val_f1_severe", "baseline_accuracy", "latency_ms_per_row", "latency_p95_ms", "train_accuracy"]
PARAMS = ["model", "estimator", "C", "n_estimators", "severe_weight", "threshold", "seed", "feature_set", "deploy_from_ts"]
# Everything the course's train.py logs. Any other metric of a trial is one a student added (step 2.2).
STANDARD = set(METRICS) | {"train_val_gap", "fit_seconds"}


def child_runs(client, sweep: str):
    parent = client.get_run(sweep)
    runs = client.search_runs(experiment_ids=[parent.info.experiment_id],
                              filter_string=f"tags.mlflow.parentRunId = '{sweep}'", max_results=200)
    if runs:
        return parent, list(runs)
    names = subprocess.run(["az", "ml", "job", "list", "--parent-job-name", sweep, "--query", "[].name", "-o", "tsv"],
                           capture_output=True, text=True, timeout=120).stdout.split()
    return parent, [client.get_run(n) for n in names]


def row_of(run):
    p, m = run.data.params, run.data.metrics
    config = {"logreg": "logreg", "rf": f"rf {p.get('n_estimators', '?')} trees"}.get(
        p.get("model"), f"mine: {p.get('estimator', '?')}")
    return {"run_id": run.info.run_id, "status": run.info.status, "config": config,
            **{k: p.get(k) for k in PARAMS}, **{k: float(v) for k, v in m.items()},
            **{k: (float(m[k]) if k in m else None) for k in METRICS},
            "extra_metrics": sorted(k for k in m if k not in STANDARD)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", nargs="?", default=read_file(".sweep"))
    ap.add_argument("--ceiling-ms", type=float, default=50.0)
    ap.add_argument("--sort", default="val_balanced_accuracy", help="metric to sort on, highest first")
    ap.add_argument("--out", default=str(evidence("leaderboard.json")))
    args = ap.parse_args()
    if not args.sweep:
        raise SystemExit("no sweep name: .sweep is empty. Submit the sweep first (echo <name> > .sweep).")

    _, client = connect()
    parent, runs = child_runs(client, args.sweep)
    rows = [row_of(r) for r in runs]
    if not rows:
        raise SystemExit(f"sweep {args.sweep} has no trials yet. Status: az ml job show -n {args.sweep} --query status")

    rows.sort(key=lambda r: (r.get(args.sort) is not None, r.get(args.sort) or 0), reverse=True)
    base = next((r["baseline_accuracy"] for r in rows if r.get("baseline_accuracy") is not None), None)
    f4 = lambda v: f"{v:.4f}" if v is not None else "-"
    f1 = lambda v: f"{v:.1f}" if v is not None else "-"
    w_id = max(len("run id"), *(len(r["run_id"]) for r in rows)) + 2

    print(f"sweep    : {args.sweep}   status {parent.info.status}")
    print(f"trials   : {len(rows)}   sorted on {args.sort}, highest first   ceiling {args.ceiling_ms:g} ms at p95")
    if base is not None:
        print(f"baseline : val_accuracy {base:.4f} if you always say 'not severe'; balanced accuracy 0.5000")
    print()
    shown = {"val_accuracy", "val_severe_found", "val_precision_severe", "val_balanced_accuracy", "latency_p95_ms"}
    extra_col = args.sort if args.sort not in shown else ""          # sorting on another metric adds its column
    w_cfg = max(len("model"), *(len(r["config"]) for r in rows)) + 2
    head = (f"{'#':<3}{'run id':<{w_id}}{'model':<{w_cfg}}{'weight':>7}{'thresh':>7}"
            f"{'val_acc':>9}{'severe':>9}{'precis':>8}{'bal_acc':>9}{'p95 ms':>9}"
            + (f"{extra_col[:22]:>24}" if extra_col else "") + "  ceiling")
    print(head)
    print("-" * len(head))
    for i, r in enumerate(rows, 1):
        lat = r.get("latency_p95_ms")
        mark = "no latency" if lat is None else ("OVER" if lat > args.ceiling_ms else "ok")
        if r["status"] != "FINISHED":
            mark += f" ({r['status']})"
        r["over_ceiling"] = lat is not None and lat > args.ceiling_ms
        w = r.get("severe_weight")
        w = f"{float(w):g}" if w not in (None, "") else "-"
        print(f"{i:<3}{r['run_id']:<{w_id}}{r['config']:<{w_cfg}}{w:>7}{str(r['threshold']):>7}"
              f"{f4(r['val_accuracy']):>9}{f4(r['val_severe_found']):>9}{f4(r['val_precision_severe']):>8}"
              f"{f4(r['val_balanced_accuracy']):>9}{f1(lat):>9}"
              + (f"{f4(r.get(extra_col)):>24}" if extra_col else "") + f"  {mark}")
    print()

    done = [r for r in rows if r["status"] == "FINISHED" and r.get("val_balanced_accuracy") is not None]
    under = [r for r in done if r.get("latency_p95_ms") is not None and not r["over_ceiling"]]
    best_bal = max(done, key=lambda r: r["val_balanced_accuracy"], default=None)
    best_acc = max(done, key=lambda r: r["val_accuracy"] or 0, default=None)
    best_under = max(under, key=lambda r: r["val_balanced_accuracy"], default=None)
    mine = [r for r in rows if r.get("model") not in ("logreg", "rf")]
    if parent.info.status not in ("FINISHED", "KILLED", "FAILED"):
        print(f"The sweep is still running ({len(rows)} trials so far). Run this again when it is Completed.")
    print(f"Trials over the {args.ceiling_ms:g} ms ceiling: {sum(r['over_ceiling'] for r in done)}.  Under it: {len(under)}.")
    if mine:
        added = sorted({k for r in mine for k in r["extra_metrics"]})
        print(f"Gradient boosting ('mine'): {len(mine)} trials, {sum(r in done for r in mine)} finished. "
              f"Extra metric: {', '.join(added) if added else 'none logged'}")
    if best_acc:
        print(f"Highest val_accuracy        : {best_acc['run_id']}  ({best_acc['val_accuracy']:.4f}, balanced {best_acc['val_balanced_accuracy']:.4f})")
    if best_bal:
        print(f"Highest balanced accuracy   : {best_bal['run_id']}  ({best_bal['val_balanced_accuracy']:.4f}, p95 {f1(best_bal.get('latency_p95_ms'))} ms)")
    if best_under:
        print(f"Best balanced, under ceiling: {best_under['run_id']}  ({best_under['val_balanced_accuracy']:.4f}, p95 {f1(best_under['latency_p95_ms'])} ms)")
    elif done:
        print("No finished trial is under the ceiling. Check the trials in ml.azure.com -> the sweep -> Trials.")
    print("Write the run id you would put behind an endpoint to .chosen:   echo <run-id> > .chosen")

    pathlib.Path(args.out).write_text(json.dumps({
        "sweep": args.sweep, "sweep_status": parent.info.status, "ceiling_ms": args.ceiling_ms,
        "sort": args.sort, "baseline_accuracy": base, "n_trials": len(rows), "n_finished": len(done),
        "n_over_ceiling": sum(r["over_ceiling"] for r in done),
        "best_by_accuracy": best_acc["run_id"] if best_acc else None,
        "best_by_balanced": best_bal["run_id"] if best_bal else None,
        "best_under_ceiling": best_under["run_id"] if best_under else None,
        "trials": rows}, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
