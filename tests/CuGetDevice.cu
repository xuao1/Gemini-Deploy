#include <stdio.h>
#include <cuda.h>

int main() {
    CUresult res;
    CUdevice device;

    // 初始化 CUDA
    res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        printf("cuInit failed: res = %d\n", res);
        return -1;
    }

    // 获取当前的 CUDA 设备
    res = cuCtxGetDevice(&device);
    if (res != CUDA_SUCCESS) {
        printf("cuCtxGetDevice failed: res = %d\n", res);
        return -1;
    }

    // 输出设备 ID
    printf("Device ID: %d\n", device);

    return 0;
}
