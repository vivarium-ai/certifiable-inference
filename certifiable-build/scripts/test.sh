#!/bin/sh
set -eu

usage() {
  cat <<EOF
Usage: $(basename "$0") [BUILD_DIR]
  BUILD_DIR: Build directory (env: BUILD_DIR, default: build)
EOF
  exit 1
}

BUILD_DIR="${1:-${BUILD_DIR:-build}}"
CTEST="${CTEST:-ctest}"

[ -d "$BUILD_DIR" ] || { echo "Build directory not found. Run build first."; exit 1; }

echo "Running tests..."
$CTEST --test-dir "$BUILD_DIR" --output-on-failure
