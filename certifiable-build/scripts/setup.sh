#!/bin/sh
set -eu

OS="$(uname -s)"

case "$OS" in
  Linux)
    ;;
  Darwin)
    ;;
  *)
    echo "Unsupported OS: $OS"
    exit 1
    ;;
esac

if command -v brew >/dev/null 2>&1; then
  echo "Homebrew already installed."
else
  echo "Installing Homebrew..."

  NONINTERACTIVE=1 /bin/bash -c \
    "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

if ! command -v brew >/dev/null 2>&1; then
  if [ -x /home/linuxbrew/.linuxbrew/bin/brew ]; then
    eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
  else
    echo "brew installed but not found on PATH"
    exit 1
  fi
fi

echo "Homebrew version:"
brew --version

brew update
brew install build2

echo "System dependencies installed successfully."
