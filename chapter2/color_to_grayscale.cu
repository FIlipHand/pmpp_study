#include <iostream>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

__global__ void colorToGrayscale(float *A, float *B, int h, int w, int c) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    if (col > w || row > h) {
        return;
    }
    int base = (row * w + col) * c;
    B[col + row * w] = 0.21 * A[base] + 0.72 * A[base + 1] + 0.07 * A[base + 2];
}

int main() {
    // load image
    int w, h, channels;
    float *data = stbi_loadf("./images/wildflowers_donoho.jpg", &w, &h, &channels, 0);
    if (!data) {
        std::cerr << "Failed loading image\n";
        return -1;
    }
    std::cout << "Loaded image: " << w << " x " << h << '\n';

    // code goes here
    float *grayImage = new float[w * h];
    float *data_d, *output_d;

    cudaMalloc((void **)&data_d, w * h * channels * sizeof(float));
    cudaMalloc((void **)&output_d, w * h * sizeof(float));

    cudaMemcpy(data_d, data, w * h * channels * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock = dim3(16, 16);
    dim3 blocksPerGrid = dim3((w + threadsPerBlock.x - 1) / threadsPerBlock.x,
                              (h + threadsPerBlock.y - 1) / threadsPerBlock.y);
    colorToGrayscale<<<blocksPerGrid, threadsPerBlock>>>(data_d, output_d, h, w, channels);
    cudaMemcpy(grayImage, output_d, w * h * sizeof(float), cudaMemcpyDeviceToHost);

    // is that absolutely neccessary?
    std::vector<unsigned char> grayBytes(w * h);
    for (int i = 0; i < w * h; i++) {
        float v = grayImage[i];
        if (v < 0.0f)
            v = 0.0f;
        if (v > 1.0f)
            v = 1.0f;
        grayBytes[i] = static_cast<unsigned char>(v * 255.0f + 0.5f);
    }
    int ok = stbi_write_png("./images/grayscale.png", w, h, 1, grayBytes.data(), w);
    if (!ok) {
        std::cerr << "Failed saving image\n";
        return -1;
    }
    std::cout << "Saving image successful \n";

    // clean up
    delete[] grayImage;
    delete[] data;

    cudaFree(data_d);
    cudaFree(output_d);
}