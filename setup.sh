#!/usr/bin/env bash
set -e

echo "Starting cross-platform setup..."

# -----------------------------
# Config
# -----------------------------
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=10

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

require_sudo() {
  if [ "$EUID" -ne 0 ]; then
    SUDO="sudo"
  else
    SUDO=""
  fi
}

require_sudo

confirm_update() {
  echo ""
  echo "About to install/update system packages."
  echo "This may modify system state (apt/brew/pacman/dnf updates)."
  echo "Type 'yes' to continue:"
  read -r RESPONSE
  if [ "$RESPONSE" != "yes" ]; then
    echo "Aborted."
    exit 1
  fi
  echo "Confirmed."
  echo ""
}

# -----------------------------
# Python version check
# -----------------------------
check_python_version() {
  PY_CMD=$1

  if ! command_exists "$PY_CMD"; then
    return 1
  fi

  VERSION=$($PY_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  MAJOR=$(echo "$VERSION" | cut -d. -f1)
  MINOR=$(echo "$VERSION" | cut -d. -f2)

  if [ "$MAJOR" -lt "$MIN_PYTHON_MAJOR" ] || { [ "$MAJOR" -eq "$MIN_PYTHON_MAJOR" ] && [ "$MINOR" -lt "$MIN_PYTHON_MINOR" ]; }; then
    return 1
  fi

  return 0
}

ensure_python() {
  if command_exists python3 && check_python_version python3; then
    PYTHON=python3
    return
  fi

  echo "Python >= 3.10 not found. Attempting installation..."
}

install_python_pip() {
  $PYTHON -m ensurepip >/dev/null 2>&1 || true
  $PYTHON -m pip install --upgrade pip
}

install_project() {
  $PYTHON -m pip install -e . --verbose
}

# -----------------------------
# Debian / Ubuntu
# -----------------------------
install_debian() {
  echo "Detected Debian/Ubuntu-like system"

  confirm_update

  $SUDO apt-get update
  $SUDO apt-get install -y python3 python3-pip python3-venv
  $SUDO apt-get upgrade -y

  ensure_python
  install_python_pip
  install_project
}

# -----------------------------
# RHEL / Fedora
# -----------------------------
install_rhel() {
  echo "Detected RHEL/Fedora-like system"

  confirm_update

  if command_exists dnf; then
    PKG_MGR="dnf"
  else
    PKG_MGR="yum"
  fi

  $SUDO $PKG_MGR install -y python3 python3-pip
  $SUDO $PKG_MGR upgrade -y

  ensure_python
  install_python_pip
  install_project
}

# -----------------------------
# Arch
# -----------------------------
install_arch() {
  echo "Detected Arch Linux"

  confirm_update

  $SUDO pacman -Sy --noconfirm python python-pip

  ensure_python
  install_python_pip
  install_project
}

# -----------------------------
# macOS
# -----------------------------
install_macos() {
  echo "Detected macOS"

  confirm_update

  if ! command_exists brew; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  fi

  brew update
  brew install python@3.11 || brew install python

  ensure_python
  install_python_pip
  install_project
}

# -----------------------------
# Windows
# -----------------------------
install_windows() {
  echo "Detected Windows environment"

  if command_exists python; then
    PYTHON=python
  elif command_exists python3; then
    PYTHON=python3
  else
    echo "Python not found. Install Python >= 3.10 manually or via winget."
    exit 1
  fi

  if ! check_python_version "$PYTHON"; then
    echo "Python version < 3.10 detected. Upgrade required."
    exit 1
  fi

  $PYTHON -m pip install --upgrade pip
  $PYTHON -m pip install -e . --verbose
}

# -----------------------------
# Dispatcher
# -----------------------------
OS_TYPE="$(uname -s | tr '[:upper:]' '[:lower:]')"

case "$OS_TYPE" in
  linux*)
    if [ -f /etc/os-release ]; then
      . /etc/os-release
      DISTRO="${ID,,}"

      case "$DISTRO" in
        ubuntu|debian) install_debian ;;
        fedora|rhel|centos|rocky|almalinux) install_rhel ;;
        arch|manjaro) install_arch ;;
        *)
          echo "Unknown Linux distro: $DISTRO"
          ensure_python
          install_python_pip
          install_project
          ;;
      esac
    fi
    ;;
  darwin*)
    install_macos
    ;;
  mingw*|msys*|cygwin*)
    install_windows
    ;;
  *)
    echo "Unsupported OS: $OS_TYPE"
    exit 1
    ;;
esac

echo "Setup complete."