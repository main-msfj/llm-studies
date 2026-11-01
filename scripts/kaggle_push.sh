#!/usr/bin/env bash
# Push notebooks/kaggle/GPT2.ipynb to Kaggle and run it there.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
cp kaggle/kernel-metadata.json "$STAGE/"
cp notebooks/kaggle/GPT2.ipynb "$STAGE/"
kaggle kernels push -p "$STAGE"
