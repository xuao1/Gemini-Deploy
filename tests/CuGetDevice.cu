#include <stdio.h>
#include <cuda.h>

int main() {
    CUresult res;
    CUdevice device;
    CUcontext context;

    // 初始化 CUDA
    res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        printf("cuInit failed: res = %d\n", res);
        return -1;
    }

    // 获取第一个可用的 CUDA 设备
    res = cuDeviceGet(&device, 0);
    if (res != CUDA_SUCCESS) {
        printf("cuDeviceGet failed: res = %d\n", res);
        return -1;
    }

    // 创建一个上下文
    res = cuCtxCreate(&context, 0, device);
    if (res != CUDA_SUCCESS) {
        printf("cuCtxCreate failed: res = %d\n", res);
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

    // 销毁上下文
    cuCtxDestroy(context);

    return 0;
}
