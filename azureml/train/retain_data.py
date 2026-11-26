"""Retain the exact Phase-1 feature bytes before registering the data asset.

    python3 train/retain_data.py $SA

Adapted from Lab-8 retain_data.py. The Lab read Lab-7's .source; this reads the
Phase-1 success.json of the published run, copies its feature Parquet files to a
content-addressed folder (curated/p2_snapshots/<sha256>/features), writes a
_manifest.json with one SHA-256 per file, and writes train/data-asset.yml.
It never overwrites an existing snapshot; a rerun only verifies it.
"""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent            # azureml/train
ROOT = HERE.parent                                # azureml
RUN_ID = "1100058089459956"                       # Phase-1 published run (databricks/evidence/job-runs.txt)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 train/retain_data.py <lake-account>")
    from adlfs import AzureBlobFileSystem
    from azure.identity import AzureCliCredential
    import pyarrow.parquet as pq
    lake = sys.argv[1]
    fs = AzureBlobFileSystem(account_name=lake, credential=AzureCliCredential())
    success = json.loads(fs.cat_file(f"curated/quality/accidents/runs/{RUN_ID}/success.json"))
    source = success["features_path"]                     # abfss://curated@<lake>.dfs.core.windows.net/accidents_runs/<run>/features
    folder = "curated/" + source.split(".dfs.core.windows.net/", 1)[1].strip("/")
    files = sorted(p for p in fs.find(folder) if p.endswith(".parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {folder}")
    with tempfile.TemporaryDirectory(prefix="p2-retain-") as tmp:
        cached, parts = [], []
        for i, remote in enumerate(files):
            rel = remote.removeprefix(folder+"/")
            local = Path(tmp)/rel; local.parent.mkdir(parents=True, exist_ok=True)
            fs.get_file(remote, str(local))
            rows = pq.ParquetFile(local).metadata.num_rows
            parts.append({"path": rel, "sha256": sha(local), "rows": rows, "bytes": local.stat().st_size})
            cached.append(local)
            print(f"hashed {i+1}/{len(files)}: {rel} ({rows:,} rows)")
        rows = sum(p["rows"] for p in parts)
        if rows != int(success["features_rows"]):
            raise SystemExit(f"row count {rows:,} differs from success.json ({success['features_rows']:,})")
        provenance = {"format": 1, "phase1_run_id": RUN_ID, "phase1_features_path": source,
                      "columns": success["features_columns"], "parts": parts, "rows": rows}
        digest = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        dst = f"curated/p2_snapshots/{digest}/features"
        marker = dst+"/_manifest.json"
        if fs.exists(marker):
            old = json.loads(fs.cat_file(marker))
            if old != {**provenance, "snapshot_sha256": digest}:
                raise SystemExit("Existing snapshot manifest differs; refusing to overwrite it")
            for part in parts:
                h = hashlib.sha256()
                with fs.open(dst+"/"+part["path"], "rb") as f:
                    for chunk in iter(lambda: f.read(1024*1024), b""):
                        h.update(chunk)
                if h.hexdigest() != part["sha256"]:
                    raise SystemExit("Retained snapshot was modified; do not reuse its asset version")
            print("verified existing retained snapshot")
        else:
            for local, part in zip(cached, parts):
                remote = dst+"/"+part["path"]
                if not fs.exists(remote):
                    fs.put_file(str(local), remote)
            fs.pipe_file(marker, json.dumps({**provenance, "snapshot_sha256": digest}, indent=2).encode())
    uri = f"abfss://curated@{lake}.dfs.core.windows.net/p2_snapshots/{digest}/features/"
    version = os.environ.get("DATA_VERSION", "1")
    yaml = (f"$schema: https://azuremlschemas.azureedge.net/latest/data.schema.json\n"
            f"name: accidents-features\nversion: {version}\ntype: uri_folder\n"
            f"description: Retained Phase-1 features from Databricks run {RUN_ID}\n"
            f"path: {uri}\ntags:\n  snapshot_sha256: {digest}\n"
            f"  phase1_run_id: \"{RUN_ID}\"\n  rows: \"{rows}\"\n")
    (HERE/"data-asset.yml").write_text(yaml)
    (ROOT/"evidence").mkdir(exist_ok=True)
    (ROOT/"evidence/data_source.json").write_text(json.dumps(
        {"phase1_run_id": RUN_ID, "phase1_features_path": source, "rows": rows,
         "files": len(parts), "snapshot_sha256": digest, "path": uri}, indent=2)+"\n")
    print(f"retained {rows:,} rows; snapshot SHA-256 {digest}")
    print("wrote train/data-asset.yml; next: az ml data create -f train/data-asset.yml")


if __name__ == "__main__":
    main()
