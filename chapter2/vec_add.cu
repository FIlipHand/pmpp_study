#include <cstdio>
#include <cstdlib>

__global__ void vecAdd(float *A, float *B, float *C, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > n) {
        return;
    }
    C[idx] = A[idx] + B[idx];
}

int main() {

    int N = 100;
    float *A_h, *B_h, *C_h, *A_d, *B_d, *C_d;
    A_h = new float[N];
    B_h = new float[N];
    C_h = new float[N];

    for (int i = 0; i < N; ++i) {
        A_h[i] = float(i);
        B_h[i] = float(N - i);
    }

    cudaMalloc((void **)&A_d, N * sizeof(float));
    cudaMalloc((void **)&B_d, N * sizeof(float));
    cudaMalloc((void **)&C_d, N * sizeof(float));

    cudaMemcpy(A_d, A_h, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<threadsPerBlock, blocksPerGrid>>>(A_d, B_d, C_d, N);

    cudaMemcpy(C_h, C_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        printf("%.3f ", C_h[i]);
    }
    printf("\n");

    delete[] A_h;
    delete[] B_h;
    delete[] C_h;
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}