#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>

int main()
{
    const int N = 20;

    float* dData;

    cudaMalloc(&dData, N*sizeof(float));

    curandGenerator_t generator;

    // Create generator
    curandCreateGenerator(
        &generator,
        CURAND_RNG_PSEUDO_DEFAULT);

    // Set seed
    curandSetPseudoRandomGeneratorSeed(
        generator,
        1234ULL);

    // Generate numbers in [0,1]
    curandGenerateUniform(
        generator,
        dData,
        N);

    float hData[N];

    cudaMemcpy(
        hData,
        dData,
        N*sizeof(float),
        cudaMemcpyDeviceToHost);

    std::cout << "Random Numbers\n\n";

    for(int i=0;i<N;i++)
    {
        std::cout << hData[i] << std::endl;
    }

    curandDestroyGenerator(generator);

    cudaFree(dData);

    return 0;
}