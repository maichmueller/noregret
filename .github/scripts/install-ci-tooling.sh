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

# Both the venv and the interpreter matter to later steps: CMake looks up Python through PATH.
echo "/opt/conan/bin" >> "${GITHUB_PATH}"

cmake --version | head -1
ninja --version
python3 --version
/opt/conan/bin/conan --version
