"""latency_check.py - client-observed latency of the endpoint, one accident per request.

    python3 checks/latency_check.py --n 100 --target-ms 1000

Adapted from the "sample" mode of Lab-10 traffic_check.py (send_100.sh): it sends
checks/sample-request.json N times, one after another, measures each answer on this VM
(network + Azure gateway + model) and reports the p95 against the Lab-10 target of
1,000 ms. Every request must succeed. Writes evidence/latency.json and the request log.
"""
import argparse
import json
import math
import time

from common import ROOT, Endpoint, evidence, names


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--target-ms", type=float, default=1000)
    a = ap.parse_args()
    ep = Endpoint()
    expected = f"{names().get('MODEL', 'accident-severity')}:{names().get('VERSION', '1')}"
    data = (ROOT / "checks" / "sample-request.json").read_bytes()
    log = []
    for i in range(a.n):
        t0 = time.time()
        status, by, body = ep.post_bytes(data, timeout=15)
        ms = (time.time() - t0) * 1000
        ok = status == 200 and isinstance(body, dict) and body.get("model") == expected and body.get("n") == 1
        log.append({"i": i, "status": status, "deployment": by, "ok": bool(ok), "latency_ms": round(ms, 2)})
    good = sorted(r["latency_ms"] for r in log if r["ok"])
    p95 = good[max(0, math.ceil(.95 * len(good)) - 1)] if good else None
    p50 = good[max(0, math.ceil(.50 * len(good)) - 1)] if good else None
    out = {"model": expected, "requests": a.n, "successful": len(good), "failures": a.n - len(good),
           "client_p50_ms": p50, "client_p95_ms": p95, "client_target_ms": a.target_ms,
           "client_target_met": p95 is not None and p95 <= a.target_ms and len(good) == a.n}
    evidence("latency_requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
    path = evidence("latency.json")
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"successful {len(good)} of {a.n}; failures {a.n - len(good)}")
    print(f"client p50 {p50} ms; p95 {p95} ms; target {a.target_ms:.0f} ms; target met: {out['client_target_met']}")
    print(f"written  : {path}")


if __name__ == "__main__":
    main()
