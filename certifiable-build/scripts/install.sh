#!/bin/sh
set -eu

usage() {
  cat <<EOF
Usage: $(basename "$0") [BUILD_DIR] [PREFIX]
  BUILD_DIR: Build directory (env: BUILD_DIR, default: build)
  PREFIX:    Install prefix (env: PREFIX, default: /usr/local)
EOF
  exit 1
}

BUILD_DIR="${1:-${BUILD_DIR:-build}}"
PREFIX="${2:-${PREFIX:-/usr/local}}"
CMAKE="${CMAKE:-cmake}"

[ -d "$BUILD_DIR" ] || { echo "Build directory not found. Run build first."; exit 1; }

echo "Installing to $PREFIX..."
$CMAKE --install "$BUILD_DIR" --prefix "$PREFIX"
