#include <cuda_runtime_api.h>
#include <iostream>
int main() {
    int deviceCount;
    cudaDeviceProp gpuInfo;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "I have exactly " << deviceCount << " on my machine!\n";

    cudaGetDeviceProperties(&gpuInfo, 0);

    std::cout << "Max threads per block: " << gpuInfo.maxThreadsPerBlock << "\n";
    std::cout << "Number of SMs: " << gpuInfo.multiProcessorCount << "\n";
    std::cout << "Max threds per block in x: " << gpuInfo.maxThreadsDim[0] << " y: " << gpuInfo.maxThreadsDim[1] << " z: " << gpuInfo.maxThreadsDim[2] << "\n";
    std::cout << "Max blocks per grid in x: " << gpuInfo.maxGridSize[0] << " y: " << gpuInfo.maxGridSize[1] << " z: " << gpuInfo.maxGridSize[2] << "\n";
    std::cout << "Number of registers per SM: " << gpuInfo.regsPerBlock << "\n";
    std::cout << "Max memory per block: " << gpuInfo.sharedMemPerBlock << "\n";
    std::cout << "Max memory per SM: " << gpuInfo.sharedMemPerMultiprocessor << "\n";

    // I have exactly 1 GPU on my machine!;
    // Max threads per block: 1024
    // Number of SMs: 80
    // Max threds per block in x: 1024 y: 1024 z: 64
    // Max blocks per grid in x: 2147483647 y: 65535 z: 65535
    // Number of registers per SM: 65536
    return 0;
}