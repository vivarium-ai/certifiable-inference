#!/bin/sh
set -eu

usage() {
  cat <<EOF
Usage: $(basename "$0") [BUILD_DIR]
  BUILD_DIR: Build directory (env: BUILD_DIR, default: ../<srcdir>-gcc)
EOF
  exit 1
}

SRCDIR="$(basename "$(pwd)")"
BUILD_DIR="${1:-${BUILD_DIR:-../$SRCDIR-gcc}}"
B="${B:-b}"

cd "$(dirname "$0")/../.." || exit 1

[ -d "$BUILD_DIR" ] || { echo "Build directory not found. Nothing to clean."; exit 0; }

echo "Cleaning build artifacts..."
"$B" clean: "$BUILD_DIR"
echo "Clean complete."
