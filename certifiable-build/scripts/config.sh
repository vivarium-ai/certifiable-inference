#!/bin/sh
set -eu

usage() {
  cat <<EOF
Usage: $(basename "$0") [BUILD_DIR] [BUILD_TYPE] [GENERATOR] [CCACHE]

Positional args override env vars.

  BUILD_DIR:   Build directory (env: BUILD_DIR, default: build)
  BUILD_TYPE:  Build type (env: BUILD_TYPE, default: Release)
  GENERATOR:   CMake generator (env: GENERATOR, default: Ninja if available else Unix Makefiles)
  CCACHE:      Compiler cache command (env: CCACHE, default: ccache; set to "" or "false" to disable)
  PREFIX:      Install prefix (env: PREFIX, default: /usr/local)
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac

BUILD_DIR="${1:-${BUILD_DIR:-build}}"
BUILD_TYPE="${2:-${BUILD_TYPE:-Release}}"
GENERATOR="${3:-${GENERATOR:-Ninja}}"
CCACHE="${4:-${CCACHE:-ccache}}"
PREFIX="${PREFIX:-/usr/local}"
CMAKE="${CMAKE:-cmake}"

# Allow disabling ccache via CCACHE="" or CCACHE=false
CCACHE_ARGS=""
if [ -n "$CCACHE" ] && [ "$CCACHE" != "false" ] && command -v "$CCACHE" >/dev/null 2>&1; then
  CCACHE_ARGS="-DCMAKE_C_COMPILER_LAUNCHER=$CCACHE -DCMAKE_CXX_COMPILER_LAUNCHER=$CCACHE"
fi

echo "Configuring: BUILD_DIR=$BUILD_DIR BUILD_TYPE=$BUILD_TYPE GENERATOR=$GENERATOR PREFIX=$PREFIX"

# shellcheck disable=SC2086
"$CMAKE" -S . -B "$BUILD_DIR" -G "$GENERATOR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  $CCACHE_ARGS
