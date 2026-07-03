#include <iostream>
#include <cuda_runtime.h>

#define N (1 << 24)// 2**24-1 elements
#define BLOCK_SIZE 256
#define UNROLL 8 // when rolling each kernel perform 8 additions per thread

#define CHECK(call)                                                   \
{                                                                     \
    cudaError_t err = call;                                           \
    if(err != cudaSuccess){                                           \
        std::cout << cudaGetErrorString(err) << std::endl;            \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
}
// normal vector addition kernel
__global__ void vectorAddNormal(const float *A,const float *B,float *C,int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        C[idx] = A[idx] + B[idx];
    }
}
// unrolled vector addition kernel
__global__ void vectorAddUnrolled(const float *A,const float *B,float *C,int n){
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * UNROLL;
    #pragma unroll
    for(int i = 0; i < UNROLL; i++)
    {
        if(idx + i < n){
            C[idx + i] = A[idx + i] + B[idx + i];
        }
    }
}

int main(){
    size_t bytes = N * sizeof(float);
    float *hA = new float[N];
    float *hB = new float[N];
    for(int i = 0; i < N; i++)
    {
        hA[i] = 1.0f;
        hB[i] = 2.0f;
    }
    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, bytes));
    CHECK(cudaMalloc(&dB, bytes));
    CHECK(cudaMalloc(&dC, bytes));
    CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Warmup
    for(int i = 0; i < 100; i++)
    {
        vectorAddNormal<<<(N + BLOCK_SIZE - 1)/BLOCK_SIZE,BLOCK_SIZE>>>(dA,dB,dC,N);
    }
    CHECK(cudaDeviceSynchronize());
    // Benchmark Normal
    cudaEventRecord(start);
    for(int i = 0; i < 100; i++)
    {
        vectorAddNormal<<<(N + BLOCK_SIZE - 1)/BLOCK_SIZE,BLOCK_SIZE>>>(dA,dB,dC,N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float normalTime;
    cudaEventElapsedTime(&normalTime,start,stop);
    std::cout<<"Normal kernel : "<<normalTime/100000.0f<<" s"<<std::endl;
    // Warmup Unrolled
    int grid =(N + BLOCK_SIZE * UNROLL - 1) /(BLOCK_SIZE * UNROLL);
    for(int i = 0; i < 100; i++)
    {
        vectorAddUnrolled<<<grid,BLOCK_SIZE>>>(dA,dB,dC,N);
    }
    CHECK(cudaDeviceSynchronize());
    // Benchmark Unrolled
    cudaEventRecord(start);
    for(int i = 0; i < 100; i++)
    {
        vectorAddUnrolled<<<grid,BLOCK_SIZE>>>(dA,dB,dC,N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float unrolledTime;
    cudaEventElapsedTime(&unrolledTime,start,stop);
    std::cout<<"Unrolled kernel : "<<unrolledTime/100000.0f<<" s"<<std::endl;
    std::cout<<"Speedup : "<<normalTime/unrolledTime<<"x"<<std::endl;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    delete[] hA;
    delete[] hB;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}