# NVCC Compiler Error with Extended Lambda Functions [![Windows](https://github.com/Ahdhn/Lambda/actions/workflows/Windows.yml/badge.svg)](https://github.com/Ahdhn/Lambda/actions/workflows/Windows.yml) [![Ubuntu](https://github.com/Ahdhn/Lambda/actions/workflows/Ubuntu.yml/badge.svg)](https://github.com/Ahdhn/Lambda/actions/workflows/Ubuntu.yml)

The purpose of this code is to show an NVCC compiler error that happens if a captured variable is wrapped around `__CUDA_ARCH__` in an extended lambda function. 
The following code will give `nvcc internal error: unexpected number of captures in __host__ __device__ lambda!`.
```c++
template <typename LambdaT>
__global__ void kernel(LambdaT func)
{
    func(threadIdx.x);
}

int main(int argc, char** argv)
{
    int d = 1;

    auto lambda = [=] __host__ __device__(int i) {
#ifdef __CUDA_ARCH__
        printf("\n d= %d", d * i);
#endif
    };

    kernel<<<1, 1>>>(lambda);

    return 0;
}
```

The code has been tested on [Windows VS2019](https://github.com/Ahdhn/Lambda/actions/runs/4164711607/jobs/7206713444#step:7:95) and [Ubuntu](https://github.com/Ahdhn/Lambda/actions/runs/4164711603/jobs/7206713465#step:7:85) with CUDA 11.7. Both show the same error. 




## Build 
```
mkdir build
cd build 
cmake ..
```

Depending on the system, this will generate either a `.sln` project on Windows or a `make` file for a Linux system. 

# Explanation
This error is due to a limiation in CUDA explained [here](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#extended-lambda-restrictions) (point 15 under Extended Lambda Restrictions) 
