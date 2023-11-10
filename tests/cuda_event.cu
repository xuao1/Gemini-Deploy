#include <cuda_runtime.h>
#include <iostream>

int main() {
    cudaEvent_t startEvent, stopEvent;
    cudaError_t error;

    // 创建 CUDA 事件
    error = cudaEventCreate(&startEvent);
    if (error != cudaSuccess) {
        std::cerr << "Failed to create start event: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    error = cudaEventCreate(&stopEvent);
    if (error != cudaSuccess) {
        std::cerr << "Failed to create stop event: " << cudaGetErrorString(error) << std::endl;
        cudaEventDestroy(startEvent); // 销毁已创建的事件
        return -1;
    }

    std::cout << "CUDA events created successfully." << std::endl;

    // 销毁事件
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return 0;
}
