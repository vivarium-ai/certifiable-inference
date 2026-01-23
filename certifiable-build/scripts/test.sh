#!/bin/sh
set -eu

usage() {
  cat <<EOF
Usage: $(basename "$0") [BUILD_DIR]
  BUILD_DIR: Build directory (env: BUILD_DIR, default: ../<srcdir>-gcc)
EOF
  exit 1
}

cd "$(dirname "$0")/../.." || exit 1
SRCDIR="$(basename "$(pwd)")"
BUILD_DIR="${1:-${BUILD_DIR:-../$SRCDIR-gcc}}"
B="${B:-b}"

[ -d "$BUILD_DIR" ] || { echo "Build directory not found. Run build first."; exit 1; }

echo "Running tests..."
"$B" test: "$BUILD_DIR"
