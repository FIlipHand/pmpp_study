#include <iostream>

void print2DMatrix(float *A, int h, int w) {
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            std::cout << A[i * w + j] << " ";
        }
        std::cout << "\n";
    }
}