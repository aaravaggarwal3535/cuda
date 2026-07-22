#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

// 1. Define a CUDA kernel to run on the GPU
__global__ void MyCuteKernel(float* d_data) {
    // Recreate the layout inside the kernel
    auto layout = make_layout(make_shape(4, 4));
    
    // Create the tensor mapping to the GPU memory pointer
    auto tensor = make_tensor(d_data, layout);

    // Coordinate assignments are safe to perform here on the device
    tensor(2, 3) = 5.0f;

    // Use printf inside the kernel (supported by CUDA)
    printf("From Device: tensor(2,3) = %f\n", tensor(2, 3));
}

int main() {
    float* d_A;
    // Allocate Device Memory
    cudaMalloc(&d_A, 4 * 4 * sizeof(float));

    // 2. Launch the kernel on the GPU
    MyCuteKernel<<<1, 1>>>(d_A);

    // 3. Synchronize the CPU with the GPU to wait for the printf to finish
    cudaDeviceSynchronize();

    // Clean up memory
    cudaFree(d_A);
    return 0;
}
