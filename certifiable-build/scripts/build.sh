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
CMAKE="${CMAKE:-cmake}"

[ -d "$BUILD_DIR" ] || { echo "Build directory not found. Run config first."; exit 1; }

echo "Building..."
$CMAKE --build "$BUILD_DIR" --parallel
