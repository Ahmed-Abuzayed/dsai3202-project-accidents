# Environment and dependencies

Everything runs on Azure services; nothing needs to be installed locally except the tools for ingestion.

## Ingestion (Lab-2 VM, Ubuntu)

| Tool | Used for | Check |
| --- | --- | --- |
| Azure CLI (`az`) | `az login`, `az storage blob list` (receipts) | `az version` |
| AzCopy v10 | upload to `raw/accidents/v1/` with `--overwrite=false`, read-back download | `azcopy --version` |
| Python 3 (standard library only: `zipfile`, `hashlib`, `json`) | unzip, SHA-256 manifest | `python3 --version` |
| bash, `sha256sum`, `diff` | `ingestion/ingest.sh` | — |

Environment variable: `SA=amazonlake60312217` (from `~/.config/dsai3202/lab3.env`, Lab-3). Login uses `AZCOPY_AUTO_LOGIN_TYPE=AZCLI`, so no keys are stored.

## Bronze (Azure Data Factory)

- Mapping Data Flow on the AutoResolve integration runtime, Small compute (data flow Spark 3.5).
- Linked service to ADLS Gen2 with the factory's managed identity (*Storage Blob Data Contributor* on the lake).
- Pipeline concurrency 1, schedule trigger stopped.
- Definitions: `adf/*.json`.

## Databricks (clean, quality gate, publish, EDA)

| Item | Value |
| --- | --- |
| Workspace | `dsai3202-databricks` (Qatar Central) |
| Compute | `dsai3202-lab5-60312217`, policy `dsai3202-lab5` |
| Runtime | Databricks Runtime 18 LTS (Scala 2.13, Spark 4.1), Photon on |
| Nodes | driver + 2 workers, Standard_D4s_v3 (16 GB, 4 cores each) |
| Auto-terminate | 60 minutes of inactivity |
| Data access | Unity Catalog, catalog `dsai3202_databricks`, schema `s_60312217` |
| Python libraries | only what the runtime provides: `pyspark`, `pandas`, `matplotlib` (EDA plots). No extra installs. |

Job definition: `databricks/job-export.yml` (exact export) and `databricks/job.json` (template with placeholders).
