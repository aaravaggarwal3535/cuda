#include <iostream>
#include <cuda_runtime.h>

__global__ void predicateKernel(int *output) {

    int x = threadIdx.x;
    int result;

    asm volatile(
        "{                           \n\t"
        " .reg .pred p;              \n\t"

        " setp.lt.s32 p, %1, 4;      \n\t"

        " @p mov.s32 %0, 111;        \n\t"
        " @!p mov.s32 %0, 222;       \n\t"

        "}"
        : "=r"(result)
        : "r"(x)
    );

    output[x] = result;
}

int main() {

    const int N = 8;

    int h_output[N];
    int *d_output;

    cudaMalloc(&d_output, N * sizeof(int));

    predicateKernel<<<1, N>>>(d_output);

    cudaDeviceSynchronize();

    cudaMemcpy(
        h_output,
        d_output,
        N * sizeof(int),
        cudaMemcpyDeviceToHost
    );

    for (int i = 0; i < N; i++) {
        std::cout << "Thread " << i
                  << " -> " << h_output[i] << '\n';
    }

    cudaFree(d_output);

    return 0;
}