#include <iostream>
#include <cuda_runtime.h>

__global__ void predicateKernel(int *output) {
    int x = threadIdx.x;

    if (x < 5) {
        output[x] = 100;
    } else {
        output[x] = 200;
    }
}

int main() {
    const int N = 8;

    int h_output[N];
    int *d_output;

    cudaMalloc(&d_output, N * sizeof(int));

    predicateKernel<<<1, N>>>(d_output);

    cudaMemcpy(
        h_output,
        d_output,
        N * sizeof(int),
        cudaMemcpyDeviceToHost
    );

    for (int i = 0; i < N; i++) {
        std::cout << "Thread " << i
                  << " -> " << h_output[i]
                  << std::endl;
    }

    cudaFree(d_output);

    return 0;
}