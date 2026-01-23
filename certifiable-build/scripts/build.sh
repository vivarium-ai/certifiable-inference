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

echo "Building..."
b
