#include <algorithm>
#include <cassert>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <assert.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

// Tensor Core WMMA GEMM
__global__ void sgemm_wmma(int M, int N,int K,float alpha,const half* A,const half* B,float beta,float* C){
    // One warp computes one 16x16 tile
    const int warpId = threadIdx.x / 32;
    const int tileRow = blockIdx.y * blockDim.y + warpId;
    const int tileCol = blockIdx.x;
    if (tileRow * 16 >= M || tileCol * 16 >= N){
        return;
    }
    // Declare fragments
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator,16,16,16,float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    // Iterate over K
    for (int k = 0; k < K; k += 16){
        const half* tileA =A + tileRow * 16 * K + k;
        const half* tileB =B + k * N + tileCol * 16;
        wmma::load_matrix_sync(a_frag,tileA,K);
        wmma::load_matrix_sync(b_frag,tileB,N);
        wmma::mma_sync(c_frag,a_frag,b_frag,c_frag);
    }
    // Load previous C
    float cTile[16 * 16];
    float* tileC =C + tileRow * 16 * N + tileCol * 16;
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            cTile[i * 16 + j] =tileC[i * N + j];
        }
    }
    // Apply alpha and beta
    for (int i = 0; i < c_frag.num_elements; i++)
    {
        c_frag.x[i] =alpha * c_frag.x[i]+ beta * cTile[i];
    }
    // Store result
    wmma::store_matrix_sync(tileC,c_frag,N, wmma::mem_row_major);
}

int main(int argc, char* argv[])
{
    // Matrix dimensions
    int M = 4096;
    int N = 4096;
    int K = 4096;

    // Allocate host memory
    half* h_A = new half[M * K];
    half* h_B = new half[K * N];
    float* h_C = new float[M * N];

    // Initialize matrices A and B with random values
    for (int i = 0; i < M * K; i++)
        h_A[i] = static_cast<half>(rand() % 10);
    for (int i = 0; i < K * N; i++)
        h_B[i] = static_cast<half>(rand() % 10);
    for (int i = 0; i < M * N; i++)
        h_C[i] = static_cast<float>(rand() % 10);

    // Allocate device memory
    half* d_A;
    half* d_B;
    float* d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(32,4);
    dim3 grid((N + 15) / 16,(M + 16 * block.y - 1) / (16 * block.y));
    
    //warmup runs
    for (int i = 0; i < 10; i++) {
        sgemm_wmma<<<grid, block>>>(M, N, K, 1.0f, d_A, d_B, 1.0f, d_C);
    }
    // benchmark runs
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        sgemm_wmma<<<grid, block>>>(M, N, K, 1.0f, d_A, d_B, 1.0f, d_C);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time = milliseconds / 100.0f;
    // calculate glops
    long long int total_ops = 2LL * M * N * K;
    float sec = avg_time / 1000.0f;
    float gflops = (float)total_ops / 1e9f / sec;
    printf("Average time: %f s, GFLOPS: %f\n", sec, gflops);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}