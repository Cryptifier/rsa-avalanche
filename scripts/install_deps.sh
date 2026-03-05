#!/usr/bin/env bash
set -euo pipefail

install_rustup() {
  if command -v cargo >/dev/null 2>&1; then
    return 0
  fi
  echo "Installing Rust toolchain via rustup..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  if [[ -f "${HOME}/.cargo/env" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/.cargo/env"
  fi
}

install_ubuntu_debian() {
  sudo apt-get update
  sudo apt-get install -y build-essential pkg-config libssl-dev ca-certificates curl
  install_rustup
}

install_arch() {
  sudo pacman -Sy --noconfirm base-devel pkgconf openssl ca-certificates curl
  install_rustup
}

install_macos() {
  if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is required on macOS. Install it from https://brew.sh/." >&2
    exit 1
  fi
  brew install pkg-config openssl@3
  install_rustup
}

detect_os() {
  local uname_out
  uname_out=$(uname -s)
  if [[ "${uname_out}" == "Darwin" ]]; then
    install_macos
    return 0
  fi

  if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    case "${ID:-}" in
      ubuntu|debian)
        install_ubuntu_debian
        return 0
        ;;
      arch)
        install_arch
        return 0
        ;;
    esac
    case "${ID_LIKE:-}" in
      *debian*)
        install_ubuntu_debian
        return 0
        ;;
      *arch*)
        install_arch
        return 0
        ;;
    esac
  fi

  if command -v apt-get >/dev/null 2>&1; then
    install_ubuntu_debian
    return 0
  fi
  if command -v pacman >/dev/null 2>&1; then
    install_arch
    return 0
  fi

  echo "Unsupported OS. Install Rust and system deps manually." >&2
  echo "Required: Rust toolchain, pkg-config, OpenSSL dev headers, build tools." >&2
  exit 1
}

detect_os

echo "Dependency installation complete. Running cargo build..."
if command -v cargo >/dev/null 2>&1; then
  cargo build
else
  echo "cargo not found after installation; skipping build." >&2
fi
