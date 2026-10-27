#!/usr/bin/env bash
# Run the full analysis. Requires data/Data.xlsx (and optionally data/weka_predictions_2023.csv).
set -euo pipefail
cd "$(dirname "$0")"
for s in scripts/0[1-9]_*.py; do
  echo "=== $s"
  python "$s"
done
echo "Done. Tables in results/, figures in results/figures/."
