"""Shared by the endpoint checks. Adapted from Lab-10 lab10_common.py.

  * names()        the group, workspace and endpoint that scripts/setup.sh saved in azureml/.names
  * model_info()   a registered version's tags, from `az ml model show`
  * local_model()  the registered version loaded on this VM (downloaded once by az ml)
  * read_rows()    rows of one period from the retained data asset, after checking its hashes
  * Endpoint       the scoring URI and key, and one POST
Changed from the Lab: the key is ID, rows are read for one time period only (the table has
7.7 million rows), and null values are sent as JSON null. Nothing here creates or deletes anything.
"""
import hashlib
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # azureml/
EVIDENCE = ROOT / "evidence"
DAY = 86400


def names() -> dict:
    out = {}
    f = ROOT / ".names"
    if f.is_file():
        for line in f.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def evidence(name: str) -> Path:
    EVIDENCE.mkdir(exist_ok=True)
    return EVIDENCE / name


def az(*args, timeout=300) -> str:
    """Run one az command; return stdout, or '' if it failed."""
    n = names()
    extra = ["-g", n["RG"], "-w", n["WS"]] if args and args[0] == "ml" and n.get("RG") and n.get("WS") else []
    try:
        r = subprocess.run(["az", *args, *extra], capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def model_info(name: str, version: str) -> dict:
    out = az("ml", "model", "show", "-n", name, "-v", str(version), "-o", "json")
    if not out:
        sys.exit(f"cannot read {name}:{version} from the registry. Check: az ml model list -n {name} -o table")
    return json.loads(out)


def local_model(name: str, version: str):
    """Download the registered version once (azureml/model-v<version>/), then load it with MLflow."""
    import mlflow.sklearn
    dl = ROOT / f"model-v{version}"
    if not list(dl.rglob("MLmodel")):
        print(f"downloading {name}:{version} from the registry (once) ...")
        az("ml", "model", "download", "-n", name, "-v", str(version), "--download-path", str(dl), timeout=900)
    mlmodel = next(dl.rglob("MLmodel"), None)
    if mlmodel is None:
        sys.exit(f"no MLmodel file under {dl}. Delete the folder and run again: rm -rf {dl}")
    return mlflow.sklearn.load_model(str(mlmodel.parent))


def read_rows(columns, data_asset, expected_hash, years):
    """Rows whose accident_year is in years, from the retained snapshot (hashes checked as in the Lab)."""
    import pyarrow.dataset as ds
    from adlfs import AzureBlobFileSystem
    from azure.identity import AzureCliCredential
    name, _, ver = data_asset.partition(":")
    path = az("ml", "data", "show", "-n", name, "-v", ver or "1", "--query", "path", "-o", "tsv")
    m = re.match(r"^abfss://([^@]+)@([^.]+)\.dfs\.core\.windows\.net/(.*)$", path or "")
    if not m:
        sys.exit(f"cannot read the path of data asset {data_asset} (got: {path or 'nothing'})")
    folder = f"{m.group(1)}/{m.group(3).rstrip('/')}"
    fs = AzureBlobFileSystem(account_name=m.group(2), credential=AzureCliCredential())
    manifest = json.loads(fs.cat_file(folder+"/_manifest.json"))
    content = {k: v for k, v in manifest.items() if k != "snapshot_sha256"}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    if digest != manifest["snapshot_sha256"] or digest != expected_hash:
        sys.exit("data hash differs from the registered model; use the recorded retained data version")
    for part in manifest["parts"]:
        h = hashlib.sha256()
        with fs.open(folder+"/"+part["path"], "rb") as f:
            for chunk in iter(lambda: f.read(1024*1024), b""):
                h.update(chunk)
        if h.hexdigest() != part["sha256"]:
            sys.exit("retained data bytes differ")
    dataset = ds.dataset(folder, filesystem=fs, format="parquet")
    table = dataset.to_table(columns=list(dict.fromkeys(columns)), filter=ds.field("accident_year").isin(years))
    return table.to_pandas(), f"{m.group(2)}/{folder}"


def as_json_rows(frame, cols, id_col="id"):
    """DataFrame rows -> list of dicts for the request. NaN becomes null ("unknown")."""
    out = []
    for rec in frame[[id_col] + cols].itertuples(index=False, name=None):
        row = {id_col: rec[0]}
        for c, v in zip(cols, rec[1:]):
            row[c] = None if v != v else float(v)       # v != v only for NaN
        out.append(row)
    return out


class Endpoint:
    """The scoring URI and key of the endpoint, and one POST (copied from Lab-10)."""

    def __init__(self):
        ep = names().get("EP", "")
        self.name = ep
        self.uri = az("ml", "online-endpoint", "show", "-n", ep, "--query", "scoring_uri", "-o", "tsv")
        self.key = az("ml", "online-endpoint", "get-credentials", "-n", ep, "--query", "primaryKey", "-o", "tsv")
        if not self.uri or not self.key:
            sys.exit(f"cannot read the scoring URI or key of endpoint '{ep}'. Is it up? "
                     "Check: az ml online-endpoint show -n $EP --query provisioning_state")

    def post_bytes(self, data: bytes, deployment="", timeout=90):
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        if deployment:
            headers["azureml-model-deployment"] = deployment
        req = urllib.request.Request(self.uri, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                status, by, raw = r.status, r.headers.get("azureml-model-deployment", ""), r.read()
        except urllib.error.HTTPError as e:
            status, by, raw = e.code, "", e.read()
        try:
            body = json.loads(raw.decode() or "null")
            if isinstance(body, str):           # score.py output can arrive double-encoded
                body = json.loads(body)
        except ValueError:
            body = {"raw": raw[:300].decode(errors="replace")}
        return status, by, body

    def post(self, rows, deployment="", timeout=90):
        return self.post_bytes(json.dumps({"data": rows}).encode(), deployment, timeout)
