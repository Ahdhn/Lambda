#include <cuda_runtime.h>
#include <stdio.h>

#include "helper.h"

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
