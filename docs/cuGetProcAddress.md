# CuGetProcAddress

## 1 

Driver Entry Point Access APIs 提供了一种检索 CUDA 驱动程序函数地址的方法

从 CUDA 11.3 开始，用户可以使用**从这些 API 获取的函数指针**调用可用的 CUDA 驱动程序 API

## 2 

[How to hook CUDA runtime API in CUDA 11.4 - CUDA / CUDA Programming and Performance - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/how-to-hook-cuda-runtime-api-in-cuda-11-4/190050)

> And I observed a phenomenon : when I compile a CUDA program using NVCC v10.0, I can hook the underlying driver symbols of every CUDA runtime API, but if I compile a CUDA program using NVCC v11.4, I can’t hook the CUDA driver symbols of CUDA runtime API. I can only hook the CUDA driver symbols when I call the CUDA driver API in my program.

这个问题和我们遇到的相同：在 cuda10 时可以 hook 到很多 cuda API，并且可以在测试代码运行时 hook 住代码调用的 cuda API，而在 cuda 11.4 则不能。（实际应该是在 cuda 11.3 以后）

回答：

> A library, even a dynamically loaded one, can be “linked” to in more than one way. Using the usual method of a formal link, the link mechanism will be exposed at dynamic library load time, and these types of links can be hooked.

这段话可以看出一点是：

一个 library 被链接的方法有很多种。

使用 formal link，那么会在 load 时，被暴露出来。

所以在 cuda 10 以及 cuda 11.0 版本时，能够 hook 住的 cuda API 分成两部分：

+ 第一部分，有很多很多，是在 load 时 hook 住的
+ 第二部分，是 hook 住的测试程序调用的 cuda API 

在 CUDA 11.3 之后，第一部分没有了，或者说，只有 cuGetProcAddress

另一个回答：

> After CUDA 11.3, NVIDIA implement an driver API : cuGetProcAddress, to get CUDA driver symbols. Therefore, the symbol lookup of cuda driver APIs except cuGetProcAddress won’t happen during runtime. If you want to hook other cuda driver APIs, you need to hook the cuGetProcAddress first, and then let cuGetProcAddress return the modified APIs you want.

```c++
CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags)
```

## 3 Exploring the New Features of CUDA 11.3

> CUDA 11.3 also introduces a new driver and runtime API to query memory addresses for driver API functions. Previously, there was no direct way to obtain function pointers to the CUDA driver symbols. To do so, you had to call into `dlopen`, `dlsym`, or `GetProcAddress`. This feature implements a new **driver API**, `cuGetProcAddress`, and the corresponding new **runtime API** `cudaGetDriverEntryPoint`.
>
> This enables you to use the runtime API to call into driver APIs like `cuCtxCreate` and `cuModuleLoad` that do not have a runtime wrapper.

新引入了一个 driver API：cuGetProcAddress，一个 runtime API：cudaGetDriverEntryPoint

支持返回 driver API functions 的地址，这项功能在 cuda11.3 之前只能通过 dlopn、dlsym 等实现。

> The `Driver Entry Point Access APIs` provide a way to retrieve the address of a CUDA driver function. Starting from CUDA 11.3, users can call into available CUDA driver APIs using function pointers obtained from these APIs.
>
> These APIs provide functionality similar to their counterparts, dlsym on POSIX platforms and GetProcAddress on Windows. The provided APIs will let users:
>
> - Retrieve the address of a driver function using the `CUDA Driver API.`
> - Retrieve the address of a driver function using the `CUDA Runtime API.`
> - Request *per-thread default stream* version of a CUDA driver function. For more details, see [Retrieve per-thread default stream versions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#retrieve-per-thread-default-stream-versions)
> - Access new CUDA features on older toolkits but with a newer driver.

```c++
CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags)
```

作用：返回 the requested driver API function pointer.

> ###### Parameters
>
> - `symbol`
>
>   - The base name of the driver API function to look for. As an example, for the driver API cuMemAlloc_v2, `symbol` would be cuMemAlloc and `cudaVersion` would be the ABI compatible CUDA version for the _v2 variant.
>
> - `pfn`
>
>   - Location to return the function pointer to the requested driver function
>
> - `cudaVersion`
>
>   - The CUDA version to look for the requested driver symbol
>
> - `flags`
>
>   - Flags to specify search options.
>
> - `symbolStatus`
>
>   - Optional location to store the status of the search for `symbol` based on `cudaVersion`. See [CUdriverProcAddressQueryResult](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g4186e73ff4899ff0f2e750a09c5a9fb1) for possible values.

所以，返回的函数指针是 `pfn`，`cuGetProcAddress` 的返回值是查询成功的状态结果。

> ###### Description
>
> Returns in `**pfn` the address of the CUDA driver function for the requested CUDA version and flags.
>
> The CUDA version is specified as (1000 * major + 10 * minor), so CUDA 11.2 should be specified as 11020. For a requested driver symbol, if the specified CUDA version is greater than or equal to the CUDA version in which the driver symbol was introduced, this API will return the function pointer to the corresponding versioned function.
>
> The pointer returned by the API should be cast to a function pointer matching the requested driver function's definition in the API header file. The function pointer typedef can be picked up from the corresponding typedefs header file. For example, cudaTypedefs.h consists of function pointer typedefs for driver APIs defined in cuda.h.
>
> The API will return [CUDA_SUCCESS](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d) and set the returned `pfn` to NULL if the requested driver function is not supported on the platform, no ABI compatible driver function exists for the specified `cudaVersion` or if the driver symbol is invalid.

上面第二段的最后没读懂，是返回与什么 corresponding 的函数指针。

## 4 dlsym

### 动态链接库的基本概念

1. **动态加载（Dynamic Loading）**:
   - 程序在运行时（而不是在启动时）加载所需的库。这意味着只有当程序实际需要库中的功能时，该库才被加载到内存中。
2. **符号解析（Symbol Resolution）**:
   - 程序使用库中的函数或变量时，需要知道这些符号在内存中的地址。动态链接库通过符号解析来实现这一点。
3. **`dlopen` 和 `dlsym`**:
   - 这些是在 Unix-like 系统中用于动态加载库（`dlopen`）和获取库中符号地址（`dlsym`）的标准函数。

### Gemini 中的写法

使用 `__libc_dlopen_mode` 动态加载 `libdl.so.2`。`libdl` 是处理动态链接库的标准库

使用 `__libc_dlsym` 获取 `dlsym` 函数的地址

**重写 `dlsym` 函数**：首先检查请求的符号是否是 CUDA 符号：

+ 如果不是，它使用之前获取的 `dlsym` 的实际地址来解析符号（通过 `real_dlsym` 函数），这意味着这些符号的处理方式与标准动态链接库的处理方式相同。
+ 如果请求的符号是一个 CUDA 符号（如 `cuLaunchKernel`），则代码直接返回该 CUDA 函数的地址。实际是在 `hook.cpp` 中使用宏定义的 hook 函数，基本包含三部分：prehook 的处理，`real_dlsym` 返回的原本的 CUDA 函数，posthook 的处理。 