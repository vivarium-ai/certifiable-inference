#!/bin/sh
set -eu

echo "Resolving system dependencies…"

OS="$(uname -s)"

case "$OS" in
  Linux)
    if command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update
      sudo apt-get install -y --no-install-recommends \
        build-essential \
        build2 \
        ccache
    else
      echo "Unsupported Linux distro (apt-get not found)"
      exit 1
    fi
    ;;
  Darwin)
    if ! command -v brew >/dev/null 2>&1; then
      echo "Homebrew not found. Install from https://brew.sh first."
      exit 1
    fi
    brew update
    brew install build2 ccache
    ;;
  *)
    echo "Unsupported OS: $OS"
    exit 1
    ;;
esac

echo "System dependencies installed successfully."
