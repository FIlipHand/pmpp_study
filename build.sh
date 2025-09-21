#!/bin/bash

filename=$1
result="${filename%.*}"

echo $filename
echo $result

mkdir -p ./build/${result%/*}/

nvcc $filename -o ./build/$result
