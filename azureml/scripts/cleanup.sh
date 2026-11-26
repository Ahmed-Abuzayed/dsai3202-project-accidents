#!/usr/bin/env bash
# Delete the endpoint (and with it the blue deployment). Adapted from Lab-10 cleanup.sh.
# Keeps the workspace, the registered model, the data asset and both environments,
# so the endpoint can be created again for the presentation (see README, Redeploy).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/.names"
state=$(az ml online-endpoint show -n "$EP" -g "$RG" -w "$WS" --query provisioning_state -o tsv 2>/dev/null)
if [ -n "$state" ] && [ "$state" != "Deleting" ]; then
  echo "\$ az ml online-endpoint delete -n $EP --yes"
  az ml online-endpoint delete -n "$EP" -g "$RG" -w "$WS" --yes --no-wait || { echo "delete failed; read the error"; exit 1; }
fi
for i in $(seq 1 40); do
  az ml online-endpoint show -n "$EP" -g "$RG" -w "$WS" -o none 2>/dev/null || break
  printf "\r  still deleting ... %s s " $((i*15)); sleep 15
done
echo
if az ml online-endpoint show -n "$EP" -g "$RG" -w "$WS" -o none 2>/dev/null; then
  echo "$EP is still there: run this again"; exit 1
fi
echo "endpoints left in the workspace:"; az ml online-endpoint list -g "$RG" -w "$WS" --query "[].name" -o tsv
echo "compute targets (must be empty: jobs used serverless):"; az ml compute list -g "$RG" -w "$WS" --query "[].name" -o tsv
rm -rf "$ROOT"/model-v*/
echo "$EP gone; model, data and environments kept."
