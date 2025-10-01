#include "../utils/matrix_utils.cu"

#define TILE_WIDTH 16

__global__ void tiledMatMul(float *A, float *B, float *C, int w) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = tx + blockDim.x * bx;
    int row = ty + blockDim.y * by;

    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    float pVal = 0.f;

    for (int i = 0; i < w / TILE_WIDTH; ++i) {
        sh_A[ty][tx] = A[row * w + TILE_WIDTH * i + tx];
        sh_B[ty][tx] = B[(i * TILE_WIDTH + ty) * w + col];

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            pVal += sh_A[tx][i] * sh_B[i][tx];
        }
        __syncthreads();
    }
    C[row * w + col] = pVal;
}

int main() {
    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;
    int width;
    width = 256 * 4;

    A_h = new float[width * width];
    B_h = new float[width * width];
    C_h = new float[width * width];

    fillWithRandomFloats(A_h, width * width);
    fillWithRandomFloats(B_h, width * width);

    std::cout << "Matrix A:\n";
    print2DMatrix(A_h, 10, 10);
    std::cout << "Matrix B:\n";
    print2DMatrix(B_h, 10, 10);

    cudaMalloc((void **)&A_d, width * width * sizeof(float));
    cudaMalloc((void **)&B_d, width * width * sizeof(float));
    cudaMalloc((void **)&C_d, width * width * sizeof(float));

    cudaMemcpy(A_d, A_h, width * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, width * width * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock = dim3(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid = dim3((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                              (width + threadsPerBlock.y - 1) / threadsPerBlock.y);
    tiledMatMul<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, width);

    cudaMemcpy(C_h, C_d, width * width * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Matrix C:\n";
    print2DMatrix(C_h, 10, 10);
}
