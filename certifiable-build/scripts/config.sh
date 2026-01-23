#!/bin/sh
set -eu

usage() {
  cat <<EOF
Usage: $(basename "$0") [BUILD_DIR] [BUILD_TYPE] [CCACHE]

Positional args override env vars.

  BUILD_DIR:   Build directory (env: BUILD_DIR, default: ../<srcdir>-gcc)
  BUILD_TYPE:  Build type (env: BUILD_TYPE, default: release)
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

cd "$(dirname "$0")/../.." || exit 1
SRCDIR="$(basename "$(pwd)")"
BUILD_DIR="${1:-${BUILD_DIR:-../build2/$SRCDIR-default}}"
BUILD_TYPE="${2:-${BUILD_TYPE:-release}}"
CCACHE="${3:-${CCACHE:-ccache}}"
PREFIX="${PREFIX:-/usr/local}"
B="${B:-b}"

# Allow disabling ccache via CCACHE="" or CCACHE=false
if [ -n "$CCACHE" ] && [ "$CCACHE" != "false" ] && command -v "$CCACHE" >/dev/null 2>&1; then
  CC="${CC:-cc}"
  export CC="$CCACHE $CC"
fi

echo "Configuring: BUILD_DIR=$BUILD_DIR BUILD_TYPE=$BUILD_TYPE PREFIX=$PREFIX"

function build2_config_check {
  bdep config list @default > /dev/null 2>&1
}

if ! build2_config_check; then
  bdep init --wipe -C "$BUILD_DIR" \
    @default \
    cc \
    "config.config.mode=$BUILD_TYPE" \
    "config.install.root=$PREFIX"
fi
