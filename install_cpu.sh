#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_CPU_INDEX_URL="https://download.pytorch.org/whl/cpu"

cd "$PROJECT_ROOT"

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install \
  --index-url "$TORCH_CPU_INDEX_URL" \
  'torch>=2.5.0'

grep -vE '^[[:space:]]*torch([[:space:]]|$)' requirements.txt \
  | "$PYTHON_BIN" -m pip install -r /dev/stdin

echo "CPU dependencies installed successfully."