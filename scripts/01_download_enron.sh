#!/usr/bin/env bash
set -euo pipefail

# Downloads the Enron email dataset (tarball) into data/raw.
# Source: https://www.cs.cmu.edu/~./enron/

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
RAW_DIR="$ROOT_DIR/data/raw"
TARBALL="$RAW_DIR/enron_mail_20110402.tgz"
URL="https://www.cs.cmu.edu/~enron/enron_mail_20110402.tgz"

mkdir -p "$RAW_DIR"

if [ -f "$TARBALL" ]; then
  echo "Enron tarball already exists at $TARBALL"
  exit 0
fi

echo "Downloading Enron dataset to $TARBALL ..."
curl -L "$URL" -o "$TARBALL"

echo "Download complete. You can extract with:"
echo "  tar -xzf $TARBALL -C $RAW_DIR"
