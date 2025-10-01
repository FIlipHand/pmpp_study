#!/bin/bash

filename=$1
result="${filename%.*}"

mkdir -p ./build/${result%/*}/

nvcc $filename -o ./build/$result
