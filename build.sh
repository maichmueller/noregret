#!/bin/bash

if [[ -n $1 ]]; then
  cmake_build_folder=$1
else
  cmake_build_folder=build
fi


if [[ -n ${*:2} ]]; then
  build_targets=${*:2}
else
  build_targets="all"
fi

cmake --build "$cmake_build_folder" --target $build_targets
