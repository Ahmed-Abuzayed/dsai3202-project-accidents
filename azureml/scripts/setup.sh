#!/usr/bin/env bash
# Phase 2 setup on the lab VM. A short version of the Lab-8/9/10 bootstrap scripts:
# it reuses the names the labs saved on this VM, sets the az defaults, saves the MLflow
# tracking URI, and checks the Python packages the Lab-9 bootstrap installed.
# Safe to run again. It creates nothing in Azure.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"          # azureml/

[ -f "$HOME/.config/dsai3202/lab3.env" ] && . "$HOME/.config/dsai3202/lab3.env"
RG="${RG:-$(cat "$HOME/.dsai3202_rg" 2>/dev/null)}"
WS="$(cat "$HOME/.dsai3202_ws" 2>/dev/null)"
SA="${SA:-$(cat "$HOME/.dsai3202_lake" 2>/dev/null)}"
SID="$(printf '%s' "$SA" | tr -cd '0-9')"
EP="ep-accidents-$SID"                 # endpoint names are unique per region
[ -n "$RG" ] && [ -n "$WS" ] && [ -n "$SA" ] || { echo "Missing RG, WS or SA. Sign in (az login --use-device-code) and check the Lab-8 workspace."; exit 1; }

az account show -o none 2>/dev/null || { echo "Not signed in: az login --use-device-code"; exit 1; }
az extension show -n ml -o none 2>/dev/null || az extension add -n ml --only-show-errors
az configure --defaults group="$RG" workspace="$WS"
az ml workspace show -n "$WS" -g "$RG" --query mlflow_tracking_uri -o tsv > "$ROOT/.tracking_uri" \
  || { echo "Workspace $WS not found in $RG"; exit 1; }

cat > "$ROOT/.names" <<NAMES
RG=$RG
WS=$WS
SA=$SA
EP=$EP
MODEL=accident-severity
VERSION=1
NAMES
sed 's/^/export /' "$ROOT/.names" > "$ROOT/.env"

python3 -c "import mlflow, azureml.mlflow, sklearn, pandas, pyarrow, adlfs, azure.identity; assert sklearn.__version__ == '1.5.2'" 2>/dev/null \
  && echo "Python packages: OK (scikit-learn 1.5.2, the same as the training job)" \
  || echo "Python packages missing: run the Lab-9 bootstrap (step 5) or: python3 -m pip install --user mlflow-skinny==2.17.2 azureml-mlflow==1.58.0 scikit-learn==1.5.2 numpy==1.26.4 scipy==1.13.1 pandas==2.2.3 pyarrow==17.0.0 adlfs==2024.7.0 azure-identity"

echo "group     : $RG"
echo "workspace : $WS"
echo "lake      : $SA"
echo "endpoint  : $EP"
echo "Next: source $ROOT/.env"
