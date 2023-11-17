#define __USE_GNU
#include "hook.h"
#include "debug.h"

#include <arpa/inet.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <netinet/in.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <vector>

extern "C" {
void *__libc_dlsym(void *map, const char *name);
}
extern "C" {
void *__libc_dlopen_mode(const char *name, int mode);
}

#define STRINGIFY(x) #x
#define CUDA_SYMBOL_STRING(x) STRINGIFY(x)

typedef void *(*fnDlsym)(void *, const char *);

static void *real_dlsym(void *handle, const char *symbol) {
  static fnDlsym internal_dlsym =
      (fnDlsym)__libc_dlsym(__libc_dlopen_mode("libdl.so.2", RTLD_LAZY), "dlsym");
  return (*internal_dlsym)(handle, symbol);
}

struct hookInfo {
  int debug_mode;
  void *preHooks[NUM_HOOK_SYMBOLS];
  void *postHooks[NUM_HOOK_SYMBOLS];
  int call_count[NUM_HOOK_SYMBOLS];

  hookInfo() {
    const char *envHookDebug;

    envHookDebug = getenv("CU_HOOK_DEBUG");
    if (envHookDebug && envHookDebug[0] == '1')
      debug_mode = 1;
    else
      debug_mode = 0;
  }
};

static struct hookInfo hook_inf;

void *dlsym(void *handle, const char *symbol) {
  DEBUG("First place: In dlsym, symbol is %s", symbol);
  // Early out if not a CUDA driver symbol
  if (strncmp(symbol, "cu", 2) != 0) {
    return (real_dlsym(handle, symbol));
  }
  DEBUG("Second place: In dlsym, symbol is %s", symbol);
  if (strcmp(symbol, CUDA_SYMBOL_STRING(cuLaunchKernel)) == 0) {
    return (void *)(&cuLaunchKernel);
  } else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuLaunchCooperativeKernel)) == 0) {
    return (void *)(&cuLaunchCooperativeKernel);
  } else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuCtxSynchronize)) == 0) {
    return (void *)(&cuCtxSynchronize);
  } else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuCtxSetCurrent)) == 0) {
    return (void *)(&cuCtxSetCurrent);
  } else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuCtxDestroy)) == 0) {
    return (void *)(&cuCtxDestroy);
  } else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuCtxCreate)) == 0) {
    return (void *)(&cuCtxCreate);
  } else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuDevicePrimaryCtxReset)) == 0) {
    return (void *)(&cuDevicePrimaryCtxReset);
  } else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuDevicePrimaryCtxRelease)) == 0) {
    return (void *)(&cuDevicePrimaryCtxRelease);
  } else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuCtxPopCurrent)) == 0) {
    return (void *)(&cuCtxPopCurrent);
  } else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuCtxPushCurrent)) == 0) {
    return (void *)(&cuCtxPushCurrent);
  } else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuGetProcAddress)) == 0) {
    return (void *)(&cuGetProcAddress);
  }
  DEBUG("Third place: In dlsym, symbol is %s", symbol);
  return (real_dlsym(handle, symbol));
}

int get_current_device_id() {
  CUdevice device;
  CUresult rc = cuCtxGetDevice(&device);
  if (rc != CUDA_SUCCESS) {
    ERROR("failed to get current device: %d", rc);
  }

  DEBUG("Operation on device %d", device);
  return device;
}

CUresult cuLaunchKernel_prehook(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                                unsigned int gridDimZ, unsigned int blockDimX,
                                unsigned int blockDimY, unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream, void **kernelParams,
                                void **extra) {
  DEBUG("cuLaunchKernel_prehook +++++++++++++++++++++++++++++++++++");
  int device = get_current_device_id();
  DEBUG("hook kernel on device %d", device);
  return CUDA_SUCCESS;
}

CUresult cuLaunchCooperativeKernel_prehook(CUfunction f, unsigned int gridDimX,
                                           unsigned int gridDimY, unsigned int gridDimZ,
                                           unsigned int blockDimX, unsigned int blockDimY,
                                           unsigned int blockDimZ, unsigned int sharedMemBytes,
                                           CUstream hStream, void **kernelParams) {
  DEBUG("cuLaunchCooperativeKernel_prehook +++++++++++++++++++++++++++++++++++");
  return cuLaunchKernel_prehook(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                                sharedMemBytes, hStream, kernelParams, NULL);
}

CUresult cuCtxSynchronize_posthook(void) {
  DEBUG("cuCtxSynchronize_posthook +++++++++++++++++++++++++++++++++++");
  return CUDA_SUCCESS;
}

// CUresult cuGetProcAddress_prehook(CUfunction *hfunc, const char *name) {
//   DEBUG("cuGetProcAddress_prehook =============================================");
//   return CUDA_SUCCESS;
// }

void initialize() {
  DEBUG("Begin to initialize: in hook.cpp");
  // Init all variable in array

  // place post-hooks
  hook_inf.postHooks[CU_HOOK_CTX_SYNC] = (void *)cuCtxSynchronize_posthook;

  // place pre-hooks
  hook_inf.preHooks[CU_HOOK_LAUNCH_KERNEL] = (void *)cuLaunchKernel_prehook;
  hook_inf.preHooks[CU_HOOK_LAUNCH_COOPERATIVE_KERNEL] = (void *)cuLaunchCooperativeKernel_prehook;
  // hook_inf.preHooks[CU_HOOK_GET_PROC_ADDRESS] = (void *)cuGetProcAddress_prehook;

  DEBUG("Initialize done: in hook.cpp");
}

CUstream hStream;  // redundent variable used for macro expansion

#define CU_HOOK_GENERATE_INTERCEPT(hooksymbol, funcname, params, ...)                     \
  CUresult CUDAAPI funcname params {                                                      \
                                                                                          \
    static void *real_func = (void *)real_dlsym(RTLD_NEXT, CUDA_SYMBOL_STRING(funcname)); \
    CUresult result = CUDA_SUCCESS;                                                       \
                                                                                          \
    if (hook_inf.debug_mode) hook_inf.call_count[hooksymbol]++;                           \
                                                                                          \
    if (hook_inf.preHooks[hooksymbol])                                                    \
      result = ((CUresult CUDAAPI(*) params)hook_inf.preHooks[hooksymbol])(__VA_ARGS__);  \
    if (result != CUDA_SUCCESS) return (result);                                          \
    result = ((CUresult CUDAAPI(*) params)real_func)(__VA_ARGS__);                        \
    if (hook_inf.postHooks[hooksymbol] && result == CUDA_SUCCESS)                         \
      result = ((CUresult CUDAAPI(*) params)hook_inf.postHooks[hooksymbol])(__VA_ARGS__); \
                                                                                          \
    return (result);                                                                      \
  }

// cuda driver ctx/management
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_CTX_SYNC, cuCtxSynchronize, (void))
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_CTX_GET_CURRENT, cuCtxGetCurrent, (CUcontext * pctx), pctx)
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_CTX_SET_CURRENT, cuCtxSetCurrent, (CUcontext ctx), ctx)
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_CTX_DESTROY, cuCtxDestroy, (CUcontext ctx), ctx)
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_CTX_CREATE, cuCtxCreate,
                           (CUcontext * pctx, unsigned int flags, CUdevice dev), pctx, flags, dev)
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_CTX_POP_CURRENT, cuCtxPopCurrent, (CUcontext * pctx), pctx)
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_CTX_PUSH_CURRENT, cuCtxPushCurrent, (CUcontext ctx), ctx)
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_DEVICE_PRIMARY_CTX_RETAIN, cuDevicePrimaryCtxRetain,
                           (CUcontext * pctx, CUdevice dev), pctx, dev)
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_DEVICE_PRIMARY_CTX_RESET, cuDevicePrimaryCtxReset,
                           (CUdevice dev), dev)
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_DEVICE_PRIMARY_CTX_RELEASE, cuDevicePrimaryCtxRelease,
                           (CUdevice dev), dev)

// cuda driver kernel launch APIs
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_LAUNCH_KERNEL, cuLaunchKernel,
                           (CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                            unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                            unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
                            void **kernelParams, void **extra),
                           f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                           sharedMemBytes, hStream, kernelParams, extra)
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_LAUNCH_COOPERATIVE_KERNEL, cuLaunchCooperativeKernel,
                           (CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                            unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                            unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
                            void **kernelParams),
                           f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                           sharedMemBytes, hStream, kernelParams)
// CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_GET_PROC_ADDRESS, cuGetProcAddress,
//                            (CUfunction * hfunc, const char *name), hfunc, name)