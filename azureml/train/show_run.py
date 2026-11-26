"""show_run.py - read one job back from MLflow: the pilot of "mine" (copied from Lab-9).

    python3 train/show_run.py              # the job named in .pilot
    python3 train/show_run.py <job-name>

The pilot is one real job with model 'mine': the same data, environment and settings as
one sweep trial. This prints what it logged and whether a sweep trial of it fits the limits,
then writes evidence/pilot.json.
"""
import json
import sys

from aml_common import connect, evidence, read_file
from leaderboard import STANDARD

TRIAL_TIMEOUT_S = 1800      # sweep.yml: trial_timeout
CEILING_MS = 50.0           # the latency budget in section 3


def main():
    job = sys.argv[1] if len(sys.argv) > 1 else read_file(".pilot")
    if not job:
        raise SystemExit("no job name: .pilot is empty. Submit the pilot first (echo <name> > .pilot).")
    _, client = connect()
    run = client.get_run(job)
    p, m = run.data.params, run.data.metrics
    extra = {k: v for k, v in m.items() if k not in STANDARD}
    model = p.get("model", "?")
    est = p.get("estimator", "?")
    print(f"run      : {job}   status {run.info.status}")
    print(f"model    : {model}: {est}   weight {p.get('severe_weight', '?')}   threshold {p.get('threshold', '?')}")
    if "val_balanced_accuracy" in m:
        print(f"balanced accuracy {m['val_balanced_accuracy']:.4f}   (severe found {m.get('val_severe_found', 0):.4f}, "
              f"not severe found {m.get('val_not_severe_found', 0):.4f}; always 'not severe' scores 0.5000)")
    print("extra metric: " + ("  ".join(f"{k} {v:.4f}" for k, v in extra.items()) if extra else "none logged"))
    fit, p95 = m.get("fit_seconds"), m.get("latency_p95_ms")
    too_slow = fit is not None and fit > TRIAL_TIMEOUT_S / 3
    if fit is not None and p95 is not None:
        print(f"training  : {fit:.1f} s     one accident: {p95:.1f} ms at p95 (budget {CEILING_MS:g} ms)")
        if too_slow:
            print(f"WARNING: training took {fit / 60:.1f} min. A sweep trial may run {TRIAL_TIMEOUT_S // 60} min in all."
                  " Make the model smaller (smaller max_iter) and run the pilot again.")
        if p95 > CEILING_MS:
            print("Note: over the latency budget. It can still run in the sweep, but the leaderboard will cross it out.")
    if run.info.status == "FINISHED" and model == "mine" and extra and not too_slow:
        print("\nOK: the pilot ran on the real data. Next: submit the sweep.")
    evidence("pilot.json").write_text(json.dumps({
        "run_id": job, "status": run.info.status, "model": model, "estimator": est,
        "extra_metrics": sorted(extra), "fit_seconds": fit, "too_slow": too_slow, "latency_p95_ms": p95,
        "val_balanced_accuracy": m.get("val_balanced_accuracy")}, indent=2) + "\n")


if __name__ == "__main__":
    main()
