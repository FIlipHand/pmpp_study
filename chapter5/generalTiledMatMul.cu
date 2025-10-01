#include "../utils/matrix_utils.cu"

#define TILE_WIDTH 16

__global__ void generalTiledMatMul(float *A, float *B, float *C, int m, int n, int k) {
    // A: m x n
    // B: n x k
    // C: m x k

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = tx + blockDim.x * blockIdx.x;
    int row = ty + blockDim.y * blockIdx.y;

    extern __shared__ float sh_A_sh_B[];

    float *sh_A = sh_A_sh_B;
    float *sh_B = &sh_A_sh_B[TILE_WIDTH * TILE_WIDTH];

    float pVal = 0.f;

    for (int i = 0; i < ceil((float)n / TILE_WIDTH); ++i) {
        int a_col = i * TILE_WIDTH + tx;
        int b_row = i * TILE_WIDTH + ty;
        if ((col < m) && (a_col < n)) {
            sh_A[ty * TILE_WIDTH + tx] = A[row * n + a_col];
        } else {
            sh_A[ty * TILE_WIDTH + tx] = 0.f;
        }
        if ((row < k) && (b_row < m)) {
            sh_B[ty * TILE_WIDTH + tx] = B[b_row * k + col];
        } else {
            sh_B[ty * TILE_WIDTH + tx] = 0.f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            pVal += sh_A[tx * TILE_WIDTH + i] * sh_B[i * TILE_WIDTH + tx];
        }
        __syncthreads();
    }
    if (row < m && col < k) {
        C[row * m + col] = pVal;
    }
}

int main() {
    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;
    int M, N, K;
    M = 142;
    N = 213;
    K = 199;

    A_h = new float[M * N];
    B_h = new float[N * K];
    C_h = new float[M * K];

    fillWithRandomFloats(A_h, M * N);
    fillWithRandomFloats(B_h, N * K);

    std::cout << "Matrix A:\n";
    print2DMatrix(A_h, 10, 10);
    std::cout << "Matrix B:\n";
    print2DMatrix(B_h, 10, 10);

    cudaMalloc((void **)&A_d, M * N * sizeof(float));
    cudaMalloc((void **)&B_d, N * K * sizeof(float));
    cudaMalloc((void **)&C_d, M * K * sizeof(float));

    cudaMemcpy(A_d, A_h, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, N * K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock = dim3(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid = dim3((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                              (K + threadsPerBlock.y - 1) / threadsPerBlock.y);

    size_t shared_size = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float);
    generalTiledMatMul<<<blocksPerGrid, threadsPerBlock, shared_size>>>(A_d, B_d, C_d, M, N, K);

    cudaMemcpy(C_h, C_d, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Matrix C:\n";
    print2DMatrix(C_h, 10, 10);
}
