"""Functional checks: one valid request and four invalid ones, against one deployment.

    python3 checks/check_deployment.py                 # deployment blue, accident-severity:1

Adapted from Lab-10 check_deployment.py. Changed: it also saves each input, the full
answer and the model version that answered, in evidence/functional_<deployment>.json
(the brief asks for inputs, outputs and the model version).
Valid = HTTP 200 with one prediction from the expected version.
Invalid = an answer with an "error" key, not a crash (score.py checks 1-5).
"""
import argparse
import json

from common import ROOT, Endpoint, evidence, names


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deployment", default="blue")
    ap.add_argument("--version", default=names().get("VERSION", "1"))
    a = ap.parse_args()
    ep = Endpoint()
    expected = f"{names().get('MODEL', 'accident-severity')}:{a.version}"
    here = ROOT / "checks"
    files = [here/"sample-request.json", here/"sample-request-2-rows.json"] + sorted((here/"bad-requests").glob("*.json"))
    results = []
    for p in files:
        data = p.read_bytes()
        status, by, body = ep.post_bytes(data, deployment=a.deployment, timeout=30)
        body = body if isinstance(body, dict) else {"raw": body}
        if p.parent.name != "bad-requests":
            n = len(json.loads(data)["data"])
            ok = status == 200 and body.get("model") == expected and body.get("n") == n and "error" not in body
        else:
            ok = status in (200, 400) and bool(body.get("error"))
        shown_input = data.decode()[:400] + ("..." if len(data) > 400 else "")   # too-long.json has 1,200 rows
        results.append({"input_file": p.name, "input": shown_input, "status": status, "served_by": by,
                        "output": body, "passed": bool(ok)})
        print(("PASS" if ok else "FAIL"), p.name, status, json.dumps(body)[:110])
    summary = {"deployment": a.deployment, "model": expected,
               "passed": all(r["passed"] for r in results), "checks": results}
    out = evidence(f"functional_{a.deployment}.json")
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"written  : {out}")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
