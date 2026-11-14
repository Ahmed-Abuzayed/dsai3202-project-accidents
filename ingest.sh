#!/usr/bin/env bash
# DSAI3202 Project Phase 1 - batch ingestion of US-Accidents into the raw zone.
# Adapted from Lab-3 (Parts 2, 4 and 5.3): source manifest with SHA-256,
# AzCopy upload with --overwrite=false, read-back hash check, and receipts
# before/after a repeated upload.
#
# Before running:
#   1. Download archive.zip from https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
#      (Kaggle login needed) and copy it to the VM:
#      scp -i <key> archive.zip azureuser@<vm-ip>:~/project/ingest/
#   2. source ~/.config/dsai3202/lab3.env   (sets SA = lake name)
#   3. az login --use-device-code
#
# Run on the VM from ~/project/ingest:  bash ingest.sh
set -euo pipefail

RAW_PREFIX="accidents/v1"
export AZCOPY_AUTO_LOGIN_TYPE=AZCLI
: "${SA:?Set SA (source ~/.config/dsai3202/lab3.env)}"

mkdir -p source evidence

# Step 1 - unzip and keep the original zip next to the extracted CSV (Lab-3 Part 2)
if [ -f archive.zip ]; then
  python3 -m zipfile -e archive.zip source/
  mv archive.zip source/
fi
ls -l source/

# Step 2 - source manifest: name, size and SHA-256 of every file
sha256sum source/* | tee evidence/sha256.txt
python3 - <<'EOF'
import json, os, hashlib
from datetime import datetime, timezone
files = []
for name in sorted(os.listdir("source")):
    path = f"source/{name}"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    files.append({"name": name, "bytes": os.path.getsize(path), "sha256": h.hexdigest()})
manifest = {
    "dataset": "US-Accidents: A Countrywide Traffic Accident Dataset (2016-2023), Sobhan Moosavi et al.",
    "source_url": "https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents",
    "license": "CC BY-NC-SA 4.0",
    "received_utc": datetime.now(timezone.utc).isoformat(),
    "raw_version": "v1",
    "files": files,
}
if not os.path.exists("evidence/source-manifest.json"):
    json.dump(manifest, open("evidence/source-manifest.json", "w"), indent=2)
print(json.dumps(manifest, indent=2))
EOF

# Step 3 - upload with overwrite disabled (Lab-3 step 4.1)
for f in US_Accidents_March23.csv archive.zip; do
  azcopy copy "source/$f" "https://$SA.blob.core.windows.net/raw/$RAW_PREFIX/$f" --overwrite=false
done

# Step 4 - list what arrived (Lab-3 step 4.2)
az storage blob list --container-name raw --account-name "$SA" --auth-mode login \
  --prefix "$RAW_PREFIX/" \
  --query "[].{Name:name,Bytes:properties.contentLength}" -o table

# Step 5 - read back and compare SHA-256 (Lab-3 step 4.3)
mkdir -p verify
azcopy copy "https://$SA.blob.core.windows.net/raw/$RAW_PREFIX/*" verify/
sha256sum verify/* | tee evidence/sha256-readback.txt
diff <(awk '{print $1}' evidence/sha256.txt) <(awk '{print $1}' evidence/sha256-readback.txt) \
  && echo "MATCH source <-> raw"
rm -r verify

# Step 6 - receipts (name, bytes, ETag) before the retry
receipts() {
  az storage blob list --container-name raw --account-name "$SA" --auth-mode login \
    --prefix "$RAW_PREFIX/" \
    --query "[].[name, properties.contentLength, properties.etag]" -o tsv
}
[ -f evidence/receipts-before.tsv ] || receipts | tee evidence/receipts-before.tsv

# Step 7 - repeat the upload; it must be skipped (Lab-3 step 5.3)
for f in US_Accidents_March23.csv archive.zip; do
  azcopy copy "source/$f" "https://$SA.blob.core.windows.net/raw/$RAW_PREFIX/$f" --overwrite=false
done
receipts | tee evidence/receipts-after.tsv
diff evidence/receipts-before.tsv evidence/receipts-after.tsv \
  && echo "SAME RECEIPTS: nothing duplicated or overwritten"
