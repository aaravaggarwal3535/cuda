#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// Initialize matrix with random values
void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// naive matrix multiplication kernel
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // if statement is necessary to make things work under tile quantization
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

// Global memory coalescing kernel
template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
  const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  // if statement is necessary to make things work under tile quantization
  if (cRow < M && cCol < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[cRow * K + i] * B[i * N + cCol];
    }
    C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
  }
}

int main(int argc, char **argv) {
    int M = 4096;
    int N = 4096;
    int K = 4096;

    float *A, *B;
    float *d_A, *d_B, *d_C;

    A = (float *)malloc(M * K * sizeof(float));
    B = (float *)malloc(K * N * sizeof(float));

    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));

    init_matrix(A, M, K);
    init_matrix(B, K, N);

    // Copy matrices from host to device
    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
  
    dim3 threads_per_block(32, 8);
    dim3 blocks_per_grid((M + threads_per_block.x - 1) / threads_per_block.x, (N + threads_per_block.y - 1) / threads_per_block.y);
  
    // warmup runs for naive kernel
    float alpha = 1.0f;
    float beta = 0.0f;
    for (int i = 0; i < 100; ++i) {
        sgemm_naive<<<blocks_per_grid, threads_per_block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    // benchmark kernels for naive kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float naive_time = 0;
    for (int i = 0; i < 100; ++i) {
        cudaEventRecord(start);
        sgemm_naive<<<blocks_per_grid, threads_per_block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        naive_time += elapsed_time;
    }
    float avg_naive_time = naive_time / 100.0f;

    //calculating glops
    long long total_ops = 2LL * M * N * K; 
    float seconds = avg_naive_time / 1000.0f;
    float giga_ops = (float)total_ops / 1e9f / seconds;
    printf("\nNaive kernel average time: %f s, GLOPS: %f\n", seconds, giga_ops);

    // warmup runs for global memory coalescing kernel
    for (int i = 0; i < 100; ++i) {
        sgemm_global_mem_coalesce<16><<<blocks_per_grid, threads_per_block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    // benchmark kernels for global memory coalescing kernel
    float coalescing_time = 0;
    for (int i = 0; i < 100; ++i) {
        cudaEventRecord(start);
        sgemm_global_mem_coalesce<16><<<blocks_per_grid, threads_per_block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        coalescing_time += elapsed_time;
    }
    float avg_coalescing_time = coalescing_time / 100.0f;

    //calculating glops
    seconds = avg_coalescing_time / 1000.0f;
    giga_ops = (float)total_ops / 1e9f / seconds;
    printf("Global memory coalescing kernel average time: %f s, GLOPS: %f\n", seconds, giga_ops);

    free(A);
    free(B);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}