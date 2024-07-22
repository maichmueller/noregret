#!/bin/bash

use_conan=true
cmake_build_folder=build
cmake_source_folder=.
build_type=Release
cmake_cmd=cmake
cmake_extra_args=()
# Define valid CMake build types
valid_build_types=("Debug" "Release" "RelWithDebInfo" "MinSizeRel")

if [ -e "$cmake_source_folder/conanfile.py" ]; then
  toolchain_file="$cmake_build_folder/conan/build/$build_type/generators/conan_toolchain.cmake"
else
  toolchain_file="$cmake_build_folder/conan/conan_toolchain.cmake"
fi

# Loop through all arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
  "--noconan")
    use_conan=false
    ;;
    "--cmake_cmd="*)
    cmake_cmd="${1#*=}"
    ;;
  "--cmake_cmd")
    cmake_cmd="$2"
    shift # Move to the next argument
    ;;
  "--output="*)
    cmake_build_folder="${1#*=}"
    ;;
  "--output")
    cmake_build_folder="$2"
    shift # Move to the next argument
    ;;
  "--source="*)
    cmake_source_folder="${1#*=}"
    ;;
  "--source")
    cmake_source_folder="$2"
    shift # Move to the next argument
    ;;
  "--toolchain_file="*)
    toolchain_file="${1#*=}"
    ;;
  "--toolchain_file")
    toolchain_file="$2"
    shift # Move to the next argument
    ;;
  "--build_type="*)
    build_type="${1#*=}"
    ;;
  "--build_type")
    build_type="$2"
    shift # Move to the next argument
    ;;

  *)
    # If the argument doesn't match any known options, add it to cmake_extra_args
    cmake_extra_args+=("$1")
    ;;
  esac
  shift # Move to the next argument
done

# Function to find a near-match in valid build types
find_near_match() {
  local input=$1
  for valid_type in "${valid_build_types[@]}"; do
    if [[ "${valid_type,}" == "${input,}"* ]]; then
      echo "$valid_type"
      return 0
    fi
  done
  return 1
}

# Function to check if a build type is valid
is_valid_build_type() {
  local build_type=$1
  for valid_type in "${valid_build_types[@]}"; do
    if [ "$build_type" == "$valid_type" ]; then
      return 0 # Valid build type
    fi
  done
  return 1 # Invalid build type
}

# Check if the provided build type is valid
if is_valid_build_type "$build_type"; then
  echo "Selected build type: $build_type"
  # Perform further actions with the valid build type
else
  # Try to find a near-match
  near_match=$(find_near_match "$build_type")

  if [ -n "$near_match" ]; then
    echo "Did you mean '$near_match'? Auto-adapting..."
    input_build_type=$near_match
    echo "Selected build type: $input_build_type"
    # Perform further actions with the valid build type
  else
    echo "Error: Invalid build type. Valid build types are: ${valid_build_types[*]}"
    exit 1
  fi
fi

script_dir=$(dirname $0)

echo "Configuration script called with the args..."
echo "use_conan: $use_conan"
echo "cmake_build_folder: $cmake_build_folder"
echo "cmake_source_folder: $cmake_source_folder"
echo "build_type: $build_type"
echo "toolchain_file: $toolchain_file"
echo "cmake_extra_args: ${cmake_extra_args[*]}"
echo "Executing cmake configuration."

if [ "$use_conan" = true ]; then
  # install all dependencies defined for conan first
  conan install . -of="$cmake_build_folder/conan" --profile:host=default --profile:build=default --build=missing -g CMakeDeps
  # append the necessary cmake configuration to the cmake call
  cmake_extra_args="${cmake_extra_args[*]} \
  -DCMAKE_TOOLCHAIN_FILE=$toolchain_file \
  -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=conan_provider.cmake \
  -DCMAKE_POLICY_DEFAULT_CMP0091=NEW"
fi

cmake_run="${cmake_cmd} \
  -S $cmake_source_folder \
  -B $cmake_build_folder \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=$build_type \
  ${cmake_extra_args[*]}"

echo "Executing command: $cmake_run"
$cmake_run
