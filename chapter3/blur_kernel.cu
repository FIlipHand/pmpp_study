#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <vector_types.h>
#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

__global__ void blurKernel(unsigned char *A, unsigned char *B, int h, int w, int c, int blur_size) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    if (col > w || row > h) {
        return;
    }
    int pixValRed = 0;
    int pixValGreen = 0;
    int pixValBlue = 0;
    int numPixel = 0;
    int base = (row * w + col) * c;
    for (int i = -blur_size; i < blur_size + 1; ++i) {
        for (int j = -blur_size; j < blur_size + 1; ++j) {
            int blurRow = row + i;
            int blurCol = col + j;
            if (blurRow < h && blurCol < w && blurRow >= 0 && blurCol >= 0) {
                int pixLoc = ((blurRow)*w + blurCol) * c;
                pixValRed += A[pixLoc + 0];
                pixValGreen += A[pixLoc + 1];
                pixValBlue += A[pixLoc + 2];
                ++numPixel;
            }
        }
    }
    B[base + 0] = static_cast<unsigned char>(pixValRed / numPixel);
    B[base + 1] = static_cast<unsigned char>(pixValGreen / numPixel);
    B[base + 2] = static_cast<unsigned char>(pixValBlue / numPixel);
}

int main() {
    // load image
    int w, h, channels;
    unsigned char *data = stbi_load("./images/wildflowers_donoho.jpg", &w, &h, &channels, 0);
    if (!data) {
        std::cerr << "Failed loading image\n";
        return -1;
    }
    std::cout << "Loaded image: " << w << " x " << h << '\n';

    // code goes here
    unsigned char *bluredImage = new unsigned char[w * h * channels];
    unsigned char *data_d, *output_d;

    cudaMalloc((void **)&data_d, w * h * channels * sizeof(unsigned char));
    cudaMalloc((void **)&output_d, w * h * channels * sizeof(unsigned char));

    cudaMemcpy(data_d, data, w * h * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock = dim3(16, 16);
    dim3 blocksPerGrid = dim3((w + threadsPerBlock.x - 1) / threadsPerBlock.x,
                              (h + threadsPerBlock.y - 1) / threadsPerBlock.y);
    blurKernel<<<blocksPerGrid, threadsPerBlock>>>(data_d, output_d, h, w, channels, 4);
    cudaMemcpy(bluredImage, output_d, w * h * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    int ok = stbi_write_png("./images/blur.png", w, h, channels, bluredImage, w * channels);
    if (!ok) {
        std::cerr << "Failed saving image\n";
        return -1;
    }
    std::cout << "Saving image successful \n";

    // clean up
    delete[] bluredImage;
    delete[] data;

    cudaFree(data_d);
    cudaFree(output_d);
}