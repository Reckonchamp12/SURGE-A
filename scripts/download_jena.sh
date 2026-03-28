#!/usr/bin/env bash
# Download the Jena Climate dataset from Kaggle.
# Requires ~/.kaggle/kaggle.json with valid credentials.

set -euo pipefail

DEST="${1:-/kaggle/input/datasets/mnassrib/jena-climate}"
mkdir -p "$DEST"

echo "Downloading Jena Climate dataset..."
kaggle datasets download -d mnassrib/jena-climate -p "$DEST" --unzip
echo "✓ Jena Climate saved to $DEST"
