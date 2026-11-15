#!/usr/bin/env bash
# Unpack CIC-IDS2017 TrafficLabelling CSVs into data/raw.
# Usage: scripts/get_data.sh ~/Downloads/GeneratedLabelledFlows.zip
set -euo pipefail
zip_path="${1:?Usage: scripts/get_data.sh path/to/GeneratedLabelledFlows.zip}"
dest="data/raw"
mkdir -p "$dest"
unzip -o -q "$zip_path" -d "$dest"
count=$(find "$dest" -name '*.csv' | wc -l | tr -d ' ')
find "$dest" -name '*.csv' -exec ls -lh {} \;
echo "Found ${count} CSV files under ${dest} (expected 8)."
