#!/bin/sh
set -eu

usage() {
  cat <<EOF
Usage: $(basename "$0") [BUILD_DIR] [EXES] [DIST]
  BUILD_DIR: Build directory (env: BUILD_DIR, default: build)
  EXES:      Executables directory (env: EXES, optional)
  DIST:      Distribution directory (env: DIST, optional)
EOF
  exit 1
}

BUILD_DIR="${1:-${BUILD_DIR:-build}}"
EXES="${2:-${EXES:-}}"
DIST="${3:-${DIST:-}}"

echo "Cleaning build artifacts..."
rm -rf "$BUILD_DIR"
[ -n "$EXES" ] && rm -rf "$EXES"
[ -n "$DIST" ] && rm -rf "$DIST"
echo "Clean complete."
