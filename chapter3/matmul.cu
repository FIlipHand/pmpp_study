#include "../utils/matrix_utils.cu"
__global__ void matMul(float *A, float *B, float *C, int h, int w, int w_out) {
    // A -> h x w
    // B -> w x w_out
    // C -> h x w_out
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (col < w_out && row < h) {
        float sum = 0.f;
        for (int i = 0; i < w; ++i) {
            sum += A[row * w + i] * B[w_out * i + col];
        }
        C[col + row * w_out] = sum;
    }
}

int main() {
    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;
    int M, N, K;
    M = 10;
    N = 5;
    K = 7;

    A_h = new float[M * N];
    B_h = new float[N * K];
    C_h = new float[M * K];

    for (int i = 0; i < M * N; ++i) {
        A_h[i] = i;
    }

    for (int i = 0; i < M * K; ++i) {
        B_h[i] = i;
    }
    print2DMatrix(A_h, M, N);
    print2DMatrix(B_h, N, K);

    cudaMalloc((void **)&A_d, M * N * sizeof(float));
    cudaMalloc((void **)&B_d, N * K * sizeof(float));
    cudaMalloc((void **)&C_d, M * K * sizeof(float));

    cudaMemcpy(A_d, A_h, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, N * K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock = dim3(16, 16);
    dim3 blocksPerGrid = dim3((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                              (K + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matMul<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, M, N, K);

    cudaMemcpy(C_h, C_d, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    print2DMatrix(C_h, M, K);
}