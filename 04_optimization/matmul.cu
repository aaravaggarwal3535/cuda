#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <assert.h>

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

// Shared memory kernel with tiling and block-level caching
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  // the output block that we want to compute in this threadblock
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // allocate buffer for current block in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // the inner row & col that we're accessing in this thread
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // advance pointers to the starting positions
  A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
  B += cCol * BLOCKSIZE;                        // row=0, col=cCol
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads();
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }
  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
}

// Shared memory kernel with tiling and block-level caching, but with 1D block tiling
template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  // If we flip x and y here we get ~30% less performance for large matrices.
  // The current, 30% faster configuration ensures that blocks with sequential
  // blockIDs access columns of B sequentially, while sharing the same row of A.
  // The slower configuration would share columns of A, but access into B would
  // be non-sequential. So the faster configuration has better spatial locality
  // and hence a greater L2 hit rate.
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // each warp will calculate 32*TM elements, with 32 being the columnar dim.
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  // allocate space for the current blocktile in SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // todo: adjust this to each thread to load multiple entries and
  // better exploit the cache sizes
  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);
  const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
  const uint innerRowB = threadIdx.x / BN;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM] = {0.0};

  // outer loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    // advance blocktile
    A += BK;
    B += BK * N;

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // we make the dotproduct loop the outside loop, which facilitates
      // reuse of the Bs entry, which we can cache in a tmp var.
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta * C[(threadRow * TM + resIdx) * N + threadCol];
  }
}

// Shared memory kernel with tiling and block-level caching, but with 2D block tiling
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint totalResultsBlocktile = BM * BN;
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x);

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const uint strideA = numThreadsBlocktile / BK;
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const uint strideB = numThreadsBlocktile / BN;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  // register caches for As and Bs
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          alpha * threadResults[resIdxM * TN + resIdxN] +
          beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
    }
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
  
    dim3 threads_per_block(32, 32);
    dim3 blocks_per_grid((M + threads_per_block.x - 1) / threads_per_block.x, (N + threads_per_block.y - 1) / threads_per_block.y);
  
    float alpha = 1.0f;
    float beta = 0.0f;
    // warmup runs for naive kernel
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
    int BLOCKSIZE = 32;
    dim3 threads_per_block_coalescing(BLOCKSIZE * BLOCKSIZE);
    dim3 blocks_per_grid_coalescing((M + BLOCKSIZE - 1)/BLOCKSIZE,(N + BLOCKSIZE - 1)/BLOCKSIZE);
    for (int i = 0; i < 100; ++i) {
        sgemm_global_mem_coalesce<32><<<blocks_per_grid_coalescing, threads_per_block_coalescing>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    // benchmark kernels for global memory coalescing kernel
    float coalescing_time = 0;
    for (int i = 0; i < 100; ++i) {
        cudaEventRecord(start);
        sgemm_global_mem_coalesce<32><<<blocks_per_grid_coalescing, threads_per_block_coalescing>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
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

    // warmup runs for shared memory kernel
    BLOCKSIZE = 32;
    dim3 threads_per_block_shared(BLOCKSIZE * BLOCKSIZE);
    dim3 blocks_per_grid_shared((M + BLOCKSIZE - 1)/BLOCKSIZE,(N + BLOCKSIZE - 1)/BLOCKSIZE);
    for (int i = 0; i < 100; ++i) {
        sgemm_shared_mem_block<32><<<blocks_per_grid_shared, threads_per_block_shared>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    // benchmark kernels for shared memory kernel
    float shared_time = 0;
    for (int i = 0; i < 100; ++i) {
        cudaEventRecord(start);
        sgemm_shared_mem_block<32><<<blocks_per_grid_shared, threads_per_block_shared>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        shared_time += elapsed_time;
    }
    float avg_shared_time = shared_time / 100.0f;
    //calculating glops
    seconds = avg_shared_time / 1000.0f;
    giga_ops = (float)total_ops / 1e9f / seconds;
    printf("Shared memory kernel average time: %f s, GLOPS: %f\n", seconds, giga_ops);

    // warmup runs for 1D block tiling kernel
    const int BM = 64;
    const int BN = 64;
    const int BK = 8;
    const int TM = 8;
    dim3 gridDim_1d_block_tiling(CEIL_DIV(M, BN), CEIL_DIV(N, BM));
    dim3 blockDim_1d_block_tiling((BM * BN )/ TM);
    for (int i = 0; i < 100; ++i) {
        sgemm1DBlocktiling<BM, BN, BK, TM><<<gridDim_1d_block_tiling, blockDim_1d_block_tiling>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    // benchmark kernels for 1D block tiling kernel
    float blocktiling_time = 0;
    for (int i = 0; i < 100; ++i) {
        cudaEventRecord(start);
        sgemm1DBlocktiling<BM, BN, BK, TM><<<gridDim_1d_block_tiling, blockDim_1d_block_tiling>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        blocktiling_time += elapsed_time;
    }
    float avg_blocktiling_time = blocktiling_time / 100.0f;
    //calculating glops
    seconds = avg_blocktiling_time / 1000.0f;
    giga_ops = (float)total_ops / 1e9f / seconds;
    printf("1D block tiling kernel average time: %f s, GLOPS: %f\n", seconds, giga_ops);

    
    // warmup runs for 2D block tiling kernel
    // BK = 8; declared above
    // TM = 8; declared above
    const uint TN = 8;
    float blocktiling2D_time = 0;
    if (M >= 128 and N >= 128) {
      const uint BM = 128;
      const uint BN = 128;
      dim3 gridDim_2D_block_tiling(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
      dim3 blockDim_2D_block_tiling((BM * BN) / (TM * TN));
      for (int i = 0; i < 100; ++i) {
          sgemm2DBlocktiling<BM, BN, BK, TM, TN><<<gridDim_2D_block_tiling, blockDim_2D_block_tiling>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
      }
      // benchmark kernels for 2D block tiling kernel
      for (int i = 0; i < 100; ++i) {
          cudaEventRecord(start);
          sgemm2DBlocktiling<BM, BN, BK, TM, TN><<<gridDim_2D_block_tiling, blockDim_2D_block_tiling>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
          cudaDeviceSynchronize();
          cudaEventRecord(stop);
          cudaEventSynchronize(stop);
          float elapsed_time;
          cudaEventElapsedTime(&elapsed_time, start, stop);
          blocktiling2D_time += elapsed_time;
      }
    }
    else{
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
      const uint BM = 64;
      const uint BN = 64;
      dim3 gridDim_2D_block_tiling(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
      dim3 blockDim_2D_block_tiling((BM * BN) / (TM * TN));
      for (int i = 0; i < 100; ++i) {
          sgemm2DBlocktiling<BM, BN, BK, TM, TN><<<gridDim_2D_block_tiling, blockDim_2D_block_tiling>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
      }
      // benchmark kernels for 2D block tiling kernel
      for (int i = 0; i < 100; ++i) {
          cudaEventRecord(start);
          sgemm2DBlocktiling<BM, BN, BK, TM, TN><<<gridDim_2D_block_tiling, blockDim_2D_block_tiling>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
          cudaDeviceSynchronize();
          cudaEventRecord(stop);
          cudaEventSynchronize(stop);
          float elapsed_time;
          cudaEventElapsedTime(&elapsed_time, start, stop);
          blocktiling2D_time += elapsed_time;
      }
    }
    float avg_blocktiling2D_time = blocktiling2D_time / 100.0f;
    //calculating glops
    seconds = avg_blocktiling2D_time / 1000.0f;
    giga_ops = (float)total_ops / 1e9f / seconds;
    printf("2D block tiling kernel average time: %f s, GLOPS: %f\n", seconds, giga_ops);
    
    free(A);
    free(B);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}