#include <iostream>

void print2DMatrix(float *A, int h, int w) {
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            std::cout << A[i * w + j] << " ";
        }
        std::cout << "\n";
    }
}

void fillWithRandomFloats(float *A, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        A[i] = (float)rand() / (float)RAND_MAX;
    }
}