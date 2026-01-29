#!/bin/sh
set -eu

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

clang_dir="${CONFIGS_ROOT}/${PROJECT}-clang"
gcc_dir="${CONFIGS_ROOT}/${PROJECT}-gcc"

mkdir -p "$clang_dir" "$gcc_dir"

"$BDEP" config list @clang >/dev/null 2>&1 || \
  "$BDEP" init -C "$clang_dir" @clang cc config.c=clang

"$BDEP" config list @gcc >/dev/null 2>&1 || \
  "$BDEP" init -C "$gcc_dir" @gcc cc config.c=gcc

"$BDEP" init @clang @gcc

"$BDEP" config set @clang \
  "config.config.mode=$BUILD_TYPE" \
  "config.install.root=$PREFIX" >/dev/null 2>&1 || true

"$BDEP" config set @gcc \
  "config.config.mode=$BUILD_TYPE" \
  "config.install.root=$PREFIX" >/dev/null 2>&1 || true
