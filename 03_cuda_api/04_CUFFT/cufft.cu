#include <iostream>
#include <cuda_runtime.h>
#include <cufft.h>

int main()
{
    const int N = 8;

    cufftComplex h_data[N];

    // Initialize input signal
    for(int i=0;i<N;i++)
    {
        h_data[i].x = i;      // Real part
        h_data[i].y = 0.0f;   // Imaginary part
    }

    // Allocate GPU memory
    cufftComplex* d_data;

    cudaMalloc(&d_data, N*sizeof(cufftComplex));

    cudaMemcpy(
        d_data,
        h_data,
        N*sizeof(cufftComplex),
        cudaMemcpyHostToDevice);

    // Create FFT plan
    cufftHandle plan;

    cufftPlan1d(
        &plan,
        N,
        CUFFT_C2C,
        1);

    // Execute FFT
    cufftExecC2C(
        plan,
        d_data,
        d_data,
        CUFFT_FORWARD);

    cudaMemcpy(
        h_data,
        d_data,
        N*sizeof(cufftComplex),
        cudaMemcpyDeviceToHost);

    std::cout << "FFT Output\n\n";

    for(int i=0;i<N;i++)
    {
        std::cout
            << h_data[i].x
            << " + "
            << h_data[i].y
            << "i\n";
    }

    cufftDestroy(plan);

    cudaFree(d_data);

    return 0;
}