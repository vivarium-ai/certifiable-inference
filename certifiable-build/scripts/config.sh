#!/bin/sh
set -eux

usage() {
  code=${1:-1}
  cat <<EOF
Usage: $(basename "$0") [BUILD_TYPE]

Defaults:
  BUILD_TYPE: release

Environment:
  PREFIX       Install prefix (default: /usr/local)
  PROJECT      Project name (default: basename of repo root)
  BDEP         bdep command (default: bdep)
  CONFIGS_ROOT Base configs dir (default: ../build2/configs)
EOF
  exit "$code"
}

case "${1:-}" in
  -h|--help) usage 0 ;;
esac

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH='' cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

PROJECT=${PROJECT:-$(basename "$REPO_ROOT")}
BDEP=${BDEP:-bdep}
CONFIGS_ROOT=${CONFIGS_ROOT:-../build2/configs}
BUILD_TYPE=${1:-${BUILD_TYPE:-release}}
PREFIX=${PREFIX:-/usr/local}

CC_GCC="$(command -v gcc)"
CXX_GCC="$(command -v g++)"
CC_CLANG="$(command -v clang)"
CXX_CLANG="$(command -v clang++)"

clang_dir="${CONFIGS_ROOT}/${PROJECT}-clang"
gcc_dir="${CONFIGS_ROOT}/${PROJECT}-gcc"

mkdir -p "$CONFIGS_ROOT"

"$BDEP" deinit --force -a @gcc @clang >/dev/null 2>&1 || true
"$BDEP" config remove @gcc @clang >/dev/null 2>&1 || true

"$BDEP" --no-default-options init --wipe -C "$gcc_dir" @gcc cc \
  "config.c=$CC_GCC" "config.cxx=$CXX_GCC" \
  "config.config.mode=$BUILD_TYPE" \
  "config.install.root=$PREFIX" || true

cfg_dir="$(bdep config list @gcc | awk '{print $2}')"
echo "CFG=$cfg_dir"
# cc module stores derived toolchain values in config.build
find "$cfg_dir" -maxdepth 3 -name config.build -print -exec sed -n '1,200p' {} \;

"$BDEP" --no-default-options init --wipe -C "$clang_dir" @clang cc \
  "config.c=$CC_CLANG" "config.cxx=$CXX_CLANG" \
  "config.config.mode=$BUILD_TYPE"
