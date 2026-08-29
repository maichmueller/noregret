#!/usr/bin/env bash
#
# Installs the build tooling the CI jobs need inside the compiler container.
#
# The container supplies the compiler and nothing else, so CMake, Ninja, the Python development
# headers the extension links against, and Conan all come from here. Conan lives in its own virtual
# environment because Debian's Python is externally managed.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    cmake \
    ninja-build \
    python3-dev \
    python3-venv

python3 -m venv /opt/conan
/opt/conan/bin/pip install --quiet --upgrade pip conan

# Only Conan is published onto PATH, not the whole virtual environment. Putting the venv's bin
# directory first would make CMake resolve Python to the venv interpreter and build the extension
# against that instead of the system Python the image ships.
ln -sf /opt/conan/bin/conan /usr/local/bin/conan

cmake --version | head -1
ninja --version
python3 --version
conan --version
