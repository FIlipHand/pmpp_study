#!/bin/bash

if [[ ! -d ./dependencies ]]; then
    mkdir ./dependencies
fi

if [[ ! -f ./dependencies/stb_image_write.h ]]; then
    wget -O ./dependencies/stb_image_write.h https://raw.githubusercontent.com/nothings/stb/refs/heads/master/stb_image_write.h
fi

if [[ ! -f ./dependencies/stb_image.h ]]; then
    wget -O ./dependencies/stb_image.h https://raw.githubusercontent.com/nothings/stb/refs/heads/master/stb_image.h
fi

filename=$1
result="${filename%.*}"

mkdir -p ./build/${result%/*}/

nvcc $filename -o ./build/$result
