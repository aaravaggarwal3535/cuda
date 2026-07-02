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

// Macro to check for cuBLAS errors
#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

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

// Shared memory kernel with tiling and block-level caching, but with 2D block tiling and vectorized loads
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmVectorize(int M, int N, int K, float alpha, float *A,
                               float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

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
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    // transpose A while loading it
    float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
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
  for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      // load C vector into registers
      float4 tmp = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      // perform GEMM update in reg
      tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
      tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
      tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
      tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
      // write back
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
          tmp;
    }
  }
}

// bank conflict resolution kernel using padding
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmResolveBankExtra_Padding(int M, int N, int K, float alpha,
                                         float *A, float *B, float beta,
                                         float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  const int extraCols = 5;
  __shared__ float Bs[BK * (BN + extraCols)];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    // transpose A while loading it
    float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    tmp = reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 0] = tmp.x;
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 1] = tmp.y;
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 2] = tmp.z;
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 3] = tmp.w;
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * (BN + extraCols) + threadCol * TN + i];
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
  for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      // load C vector into registers
      float4 tmp = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      // perform GEMM update in reg
      tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
      tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
      tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
      tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
      // write back
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
          tmp;
    }
  }
}

// Shared memory kernel with tiling and block-level caching, but with 2D block tiling and vectorized loads, but with bank conflict resolution
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmResolveBankConflicts_swizzling(int M, int N, int K, float alpha,
                                          float *A, float *B, float beta,
                                          float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

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
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    // transpose A while loading it
    float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    // "linearize" Bs while storing it
    tmp = reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 0) * 16 + innerColB / 2] = tmp.x;
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 1) * 16 + innerColB / 2] = tmp.y;
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 2) * 16 + innerColB / 2] = tmp.z;
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 3) * 16 + innerColB / 2] = tmp.w;
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[(dotIdx * 8 + i) * 16 + threadCol];
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
  for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      // load C vector into registers
      float4 tmp = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      // perform GEMM update in reg
      tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
      tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
      tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
      tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
      // write back
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
          tmp;
    }
  }
}

// Autotuned kernel with tiling and block-level caching, but with 2D block tiling and vectorized loads, but with bank conflict resolution
const int Autotuned_NUM_THREADS = 256;
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(Autotuned_NUM_THREADS)
    sgemmAutotuned(int M, int N, int K, float alpha, float *A, float *B,
                   float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // size of warptile
  constexpr int WM = TM * 16;
  constexpr int WN = TN * 16;
  // iterations of warptile
  constexpr int WMITER = CEIL_DIV(BM, WM);
  constexpr int WNITER = CEIL_DIV(BN, WN);

  // Placement of the thread in the warptile
  const int threadCol = threadIdx.x % (WN / TN);
  const int threadRow = threadIdx.x / (WN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (Autotuned_NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = Autotuned_NUM_THREADS / (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[WMITER * WNITER * TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
      float4 tmp = reinterpret_cast<float4 *>(
          &A[(innerRowA + offset) * K + innerColA * 4])[0];
      // transpose A while storing it
      As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
      As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
      As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
      As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }

    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
      reinterpret_cast<float4 *>(
          &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
          reinterpret_cast<float4 *>(
              &B[(innerRowB + offset) * N + innerColB * 4])[0];
    }
    __syncthreads();

    for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
      for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
          // block into registers
          for (uint i = 0; i < TM; ++i) {
            regM[i] = As[dotIdx * BM + (wmIdx * WM) + threadRow * TM + i];
          }
          for (uint i = 0; i < TN; ++i) {
            regN[i] = Bs[dotIdx * BN + (wnIdx * WN) + threadCol * TN + i];
          }
          for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
              threadResults[(wmIdx * TM + resIdxM) * (WNITER * TN) +
                            wnIdx * TN + resIdxN] +=
                  regM[resIdxM] * regN[resIdxN];
            }
          }
        }
      }
    }
    __syncthreads();
    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
  }

  // write out the results
  for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
    for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
      float *C_interim = C + (wmIdx * WM * N) + (wnIdx * WN);
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // load C vector into registers
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRow * TM + resIdxM) * N + threadCol * TN +
                         resIdxN])[0];
          // perform GEMM update in reg
          const int i =
              (wmIdx * TM + resIdxM) * (WNITER * TN) + wnIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          // write back
          reinterpret_cast<float4 *>(&C_interim[(threadRow * TM + resIdxM) * N +
                                                threadCol * TN + resIdxN])[0] =
              tmp;
        }
      }
    }
  }
}

/*
===============================================================================
                            Wraptiling
===============================================================================
*/
const int WARPSIZE = 32; // warpSize is not constexpr

namespace wt {
template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                             float *As, float *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB) {
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    const float4 tmp = reinterpret_cast<const float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    // float4 tmp;
    // asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
    //     : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
    //     : "l"(&A[(innerRowA + offset) * K + innerColA * 4]));
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }

  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
    // asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
    //     : "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 0]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 1]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 2]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 3])
    //     : "l"(&B[(innerRowB + offset) * N + innerColB * 4]));
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // populate registers for whole warptile
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] =
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }

    // execute warptile matmul
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        // calculate per-thread results
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          (wSubColIdx * TN) + resIdxN] +=
                regM[wSubRowIdx * TM + resIdxM] *
                regN[wSubColIdx * TN + resIdxN];
          }
        }
      }
    }
  }
}

} // namespace wt

/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam WMITER The number of subwarp tiling steps in M dimension.
 * @tparam WNITER The number of subwarp tiling steps in N dimension.
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 */
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B,
                    float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // Placement of the warp in the threadblock tile
  const uint warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  // size of the warp subtile
  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER; // 64/2=32
  constexpr uint WSUBN = WN / WNITER; // 32/2=16

  // Placement of the thread in the warp subtile
  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4

  // allocate space for the current blocktile in SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  // Move C_ptr to warp's output tile
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[WMITER * TM * WNITER * TN] = {0.0};
  // we cache into registers on the warptile level
  float regM[WMITER * TM] = {0.0};
  float regN[WNITER * TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads();
    wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                        TN>(regM, regN, threadResults, As, Bs, warpRow, warpCol,
                            threadRowInWarp, threadColInWarp);
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
    __syncthreads();
  }

  // write out the results
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // move C pointer to current warp subtile
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // load C vector into registers
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0];
          // perform GEMM update in reg
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          // write back
          reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0] = tmp;
        }
      }
    }
  }
}

// double buffring
// version 1 software double buffering
namespace db {

template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem_V1(const int N, const int K, float *A, float *B,
                             float *As, float *Bs, const int innerRowA,
                             const int innerColA, const int innerRowB,
                             const int innerColB) {
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    float4 tmp = reinterpret_cast<float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    // transpose A while storing it
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }

  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem_V1(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // populate registers for whole warptile
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] =
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }

    // execute warptile matmul
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        // calculate per-thread results
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          (wSubColIdx * TN) + resIdxN] +=
                regM[wSubRowIdx * TM + resIdxM] *
                regN[wSubColIdx * TN + resIdxN];
          }
        }
      }
    }
  }
}

} // namespace db

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemmDoubleBuffering_V1(const int M, const int N, const int K,
                         const float alpha, float *A, float *B, float beta,
                         float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // Placement of the warp in the threadblock tile
  const uint warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  // size of the warp subtile
  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER; // 64/2=32
  constexpr uint WSUBN = WN / WNITER; // 32/2=16

  // Placement of the thread in the warp subtile
  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4

  // allocate space for the current blocktile in SMEM
  __shared__ float As[2 * BM * BK];
  __shared__ float Bs[2 * BK * BN];

  // setup double buffering split
  bool doubleBufferIdx = threadIdx.x >= (NUM_THREADS / 2);

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  // Move C_ptr to warp's output tile
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // calculating the indices that this thread will load into SMEM
  // for the loading, we're pretending like there's half as many threads
  // as there actually are
  const uint innerRowA = (threadIdx.x % (NUM_THREADS / 2)) / (BK / 4);
  const uint innerColA = (threadIdx.x % (NUM_THREADS / 2)) % (BK / 4);
  constexpr uint rowStrideA = ((NUM_THREADS / 2) * 4) / BK;
  const uint innerRowB = (threadIdx.x % (NUM_THREADS / 2)) / (BN / 4);
  const uint innerColB = (threadIdx.x % (NUM_THREADS / 2)) % (BN / 4);
  constexpr uint rowStrideB = (NUM_THREADS / 2) / (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[WMITER * TM * WNITER * TN] = {0.0};
  // we cache into registers on the warptile level
  float regM[WMITER * TM] = {0.0};
  float regN[WNITER * TN] = {0.0};

  if (doubleBufferIdx == 0) {
    // load first (B0)
    db::loadFromGmem_V1<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
  }
  __syncthreads();

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += 2 * BK) {
    if (doubleBufferIdx == 0) {
      // process current (B0)
      db::processFromSmem_V1<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                          TN>(regM, regN, threadResults, As, Bs, warpRow,
                              warpCol, threadRowInWarp, threadColInWarp);
      __syncthreads();

      // process current+1 (B1)
      if (bkIdx + BK < K) {
        db::processFromSmem_V1<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN,
                            TM, TN>(regM, regN, threadResults, As + (BM * BK),
                                    Bs + (BK * BN), warpRow, warpCol,
                                    threadRowInWarp, threadColInWarp);
      }
      __syncthreads();

      // load current + 2 (B0)
      if (bkIdx + 2 * BK < K) {
        db::loadFromGmem_V1<BM, BN, BK, rowStrideA, rowStrideB>(
            N, K, A + 2 * BK, B + 2 * BK * N, As, Bs, innerRowA, innerColA,
            innerRowB, innerColB);
      }
    } else {
      // load current + 1 (B1)
      if (bkIdx + BK < K) {
        db::loadFromGmem_V1<BM, BN, BK, rowStrideA, rowStrideB>(
            N, K, A + BK, B + BK * N, As + (BM * BK), Bs + (BK * BN), innerRowA,
            innerColA, innerRowB, innerColB);
      }
      __syncthreads();

      // process current (B0)
      db::processFromSmem_V1<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                          TN>(regM, regN, threadResults, As, Bs, warpRow,
                              warpCol, threadRowInWarp, threadColInWarp);
      __syncthreads();

      // process current+1 (B1)
      if (bkIdx + BK < K) {
        db::processFromSmem_V1<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN,
                            TM, TN>(regM, regN, threadResults, As + (BM * BK),
                                    Bs + (BK * BN), warpRow, warpCol,
                                    threadRowInWarp, threadColInWarp);
      }
    }

    A += 2 * BK;     // move BK columns to right
    B += 2 * BK * N; // move BK rows down
    __syncthreads();
  }

  // write out the results
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // move C pointer to current warp subtile
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // load C vector into registers
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0];
          // perform GEMM update in reg
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          // write back
          reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0] = tmp;
        }
      }
    }
  }
}

// doble buffering version 2 : hardware double buffering using async copy
namespace {
template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB, typename T>
__device__ void loadFromGmem_V2(int N, int K, float *A, float *B, float *As,
                             float *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB, T &barrier) {

  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    cuda::memcpy_async(&As[(innerColA * 4 + 0) * BM + innerRowA + offset],
                       &A[(innerRowA + offset) * K + innerColA * 4],
                       cuda::aligned_size_t<sizeof(float)>(sizeof(float)),
                       barrier);
    cuda::memcpy_async(&As[(innerColA * 4 + 1) * BM + innerRowA + offset],
                       &A[(innerRowA + offset) * K + innerColA * 4 + 1],
                       cuda::aligned_size_t<sizeof(float)>(sizeof(float)),
                       barrier);
    cuda::memcpy_async(&As[(innerColA * 4 + 2) * BM + innerRowA + offset],
                       &A[(innerRowA + offset) * K + innerColA * 4 + 2],
                       cuda::aligned_size_t<sizeof(float)>(sizeof(float)),
                       barrier);
    cuda::memcpy_async(&As[(innerColA * 4 + 3) * BM + innerRowA + offset],
                       &A[(innerRowA + offset) * K + innerColA * 4 + 3],
                       cuda::aligned_size_t<sizeof(float)>(sizeof(float)),
                       barrier);
  }

  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    cuda::memcpy_async(&Bs[(innerRowB + offset) * BN + innerColB * 4],
                       &B[(innerRowB + offset) * N + innerColB * 4],
                       cuda::aligned_size_t<sizeof(float4)>(sizeof(float4)),
                       barrier);
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem_V2(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // populate registers for whole warptile
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] =
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }

    // execute warptile matmul
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        // calculate per-thread results
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          (wSubColIdx * TN) + resIdxN] +=
                regM[wSubRowIdx * TM + resIdxM] *
                regN[wSubColIdx * TN + resIdxN];
          }
        }
      }
    }
  }
}

} // namespace

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    runSgemmDoubleBuffering_V2(int M, int N, int K, float alpha, float *A,
                             float *B, float beta, float *C) {
  auto block = cooperative_groups::this_thread_block();
  __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> frontBarrier;
  __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> backBarrier;
  auto frontBarrierPtr = &frontBarrier;
  auto backBarrierPtr = &backBarrier;
  if (block.thread_rank() == 0) {
    init(&frontBarrier, block.size());
    init(&backBarrier, block.size());
  }
  __syncthreads();

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // Placement of the warp in the threadblock tile
  const uint warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  // size of the warp subtile
  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER; // 64/2=32
  constexpr uint WSUBN = WN / WNITER; // 32/2=16

  // Placement of the thread in the warp subtile
  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4

  // allocate space for the current blocktile in SMEM
  __shared__ float As[2 * BM * BK];
  __shared__ float Bs[2 * BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  // Move C_ptr to warp's output tile
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[WMITER * TM * WNITER * TN] = {0.0};
  // we cache into registers on the warptile level
  float regM[WMITER * TM] = {0.0};
  float regN[WNITER * TN] = {0.0};

  int As_offset = 0;
  int Bs_offset = 0;

  // double-buffering: load first blocktile into SMEM
  loadFromGmem_V2<BM, BN, BK, rowStrideA, rowStrideB>(
      N, K, A, B, As + As_offset * BM * BK, Bs + Bs_offset * BK * BN, innerRowA,
      innerColA, innerRowB, innerColB, (*frontBarrierPtr));

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K - BK; bkIdx += BK) {
    // double-buffering: load next blocktile into SMEM
    loadFromGmem_V2<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A + BK, B + BK * N, As + (1 - As_offset) * BM * BK,
        Bs + (1 - Bs_offset) * BK * BN, innerRowA, innerColA, innerRowB,
        innerColB, (*backBarrierPtr));

    // compute the current blocktile
    (*frontBarrierPtr).arrive_and_wait();
    processFromSmem_V2<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
        regM, regN, threadResults, As + As_offset * BM * BK,
        Bs + Bs_offset * BK * BN, warpRow, warpCol, threadRowInWarp,
        threadColInWarp);
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    As_offset = 1 - As_offset;
    Bs_offset = 1 - Bs_offset;
    // swap the front and back barriers
    auto tmp = frontBarrierPtr;
    frontBarrierPtr = backBarrierPtr;
    backBarrierPtr = tmp;

    __syncthreads();
  }

  // compute the last blocktile
  (*frontBarrierPtr).arrive_and_wait();
  processFromSmem_V2<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
      regM, regN, threadResults, As + As_offset * BM * BK,
      Bs + Bs_offset * BK * BN, warpRow, warpCol, threadRowInWarp,
      threadColInWarp);

  // write out the results
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // move C pointer to current warp subtile
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // load C vector into registers
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0];
          // perform GEMM update in reg
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          // write back
          reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0] = tmp;
        }
      }
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
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
        sgemm_naive<<<blocks_per_grid, threads_per_block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
      }
      cudaEventRecord(stop);
      cudaDeviceSynchronize();
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&naive_time, start, stop);
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
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
        sgemm_global_mem_coalesce<32><<<blocks_per_grid_coalescing, threads_per_block_coalescing>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
      }
      cudaEventRecord(stop);
      cudaDeviceSynchronize();
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&coalescing_time, start, stop);
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
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
      sgemm_shared_mem_block<32><<<blocks_per_grid_shared, threads_per_block_shared>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&shared_time, start, stop);
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
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
        sgemm1DBlocktiling<BM, BN, BK, TM><<<gridDim_1d_block_tiling, blockDim_1d_block_tiling>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
      }
      cudaEventRecord(stop);
      cudaDeviceSynchronize();
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&blocktiling_time, start, stop);
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
      cudaEventRecord(start);
      for (int i = 0; i < 100; ++i) {
          sgemm2DBlocktiling<BM, BN, BK, TM, TN><<<gridDim_2D_block_tiling, blockDim_2D_block_tiling>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
        }
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&blocktiling2D_time, start, stop);
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
      cudaEventRecord(start);
      for (int i = 0; i < 100; ++i) {
          sgemm2DBlocktiling<BM, BN, BK, TM, TN><<<gridDim_2D_block_tiling, blockDim_2D_block_tiling>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
        }
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&blocktiling2D_time, start, stop);
    }
    float avg_blocktiling2D_time = blocktiling2D_time / 100.0f;
    //calculating glops
    seconds = avg_blocktiling2D_time / 1000.0f;
    giga_ops = (float)total_ops / 1e9f / seconds;
    printf("2D block tiling kernel average time: %f s, GLOPS: %f\n", seconds, giga_ops);

    //2D block tiling kernel with vectorized loads
    float vectorized_time = 0;
    if (M >= 128 and N >= 128) {
      const uint BM = 128;
      const uint BN = 128;
      dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
      dim3 blockDim((BM * BN) / (TM * TN));
      // warmup runs for 2D block tiling kernel with vectorized loads
      for (int i = 0; i < 100; ++i) {
        sgemmVectorize<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
      }
      // benchmark kernels for 2D block tiling kernel with vectorized loads
      cudaEventRecord(start);
      for (int i = 0; i < 100; ++i) {
        sgemmVectorize<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
      }
      cudaEventRecord(stop);
      cudaDeviceSynchronize();
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&vectorized_time, start, stop);
    }
    else {
      // this is a hacky solution to the underlying problem
      // of not having proper bounds checking in the kernel
      const uint BM = 64;
      const uint BN = 64;
      dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
      dim3 blockDim((BM * BN) / (TM * TN));
      // warmup runs for 2D block tiling kernel with vectorized loads
      for (int i = 0; i < 100; ++i) {
        sgemmVectorize<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
      }
      // benchmark kernels for 2D block tiling kernel with vectorized loads
      cudaEventRecord(start);
      for (int i = 0; i < 100; ++i) {
        sgemmVectorize<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
      }
      cudaEventRecord(stop);
      cudaDeviceSynchronize();
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&vectorized_time, start, stop);
      float avg_vectorized_time = vectorized_time / 100.0f;
      //calculating glops
      seconds = avg_vectorized_time / 1000.0f;
      giga_ops = (float)total_ops / 1e9f / seconds;
    }
    printf("2D block tiling kernel with vectorized loads average time: %f s, GLOPS: %f\n", seconds, giga_ops);
    
    // bank conflict resolution kernel using swizzling
    // const uint BK = 8;
    // const uint TM = 8;
    // const uint TN = 8;
    float bank_conflict_time_swizzling = 0;
    if (M >= 128 and N >= 128) {
      const uint BM = 128;
      const uint BN = 128;
      dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
      dim3 blockDim((BM * BN) / (TM * TN));
      // warmup runs for 2D block tiling kernel with bank conflict resolution
      for (int i = 0; i < 100; ++i) {
        sgemmResolveBankConflicts_swizzling<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
      }
      // benchmark kernels for 2D block tiling kernel with bank conflict resolution
      cudaEventRecord(start);
      for (int i = 0; i < 100; ++i) {
        sgemmResolveBankConflicts_swizzling<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
      }
      cudaEventRecord(stop);
      cudaDeviceSynchronize();
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&bank_conflict_time_swizzling, start, stop);
    } else {
      // this is a hacky solution to the underlying problem
      // of not having proper bounds checking in the kernel
      const uint BM = 64;
      const uint BN = 64;
      dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
      dim3 blockDim((BM * BN) / (TM * TN));
      // warmup runs for 2D block tiling kernel with bank conflict resolution
      for (int i = 0; i < 100; ++i) {
        sgemmResolveBankConflicts_swizzling<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
      }
      // benchmark kernels for 2D block tiling kernel with bank conflict resolution
      cudaEventRecord(start);
      for (int i = 0; i < 100; ++i) {
        sgemmResolveBankConflicts_swizzling<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
      }
      cudaEventRecord(stop);
      cudaDeviceSynchronize();
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&bank_conflict_time_swizzling, start, stop);
    }
    float avg_bank_conflict_time = bank_conflict_time_swizzling / 100.0f;
    //calculating glops
    seconds = avg_bank_conflict_time / 1000.0f;
    giga_ops = (float)total_ops / 1e9f / seconds;
    printf("Bank conflict resolution by swizzling average time: %f s, GLOPS: %f\n", seconds, giga_ops);
    printf("this can be slow then previous kernels if there was no bank conflict in the previous kernels\n");

    // Resolve bank conflicts using padding
    // const uint BK = 8;
    // const uint TM = 8;
    // const uint TN = 8;
    float bank_conflict_time_padding = 0;
    if (M >= 128 and N >= 128) {
      const uint BM = 128;
      const uint BN = 128;
      dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
      dim3 blockDim((BM * BN) / (TM * TN));
      // warmup runs for 2D block tiling kernel with bank conflict resolution
      for (int i = 0; i < 100; ++i) {
        sgemmResolveBankExtra_Padding<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
      }
      // benchmark kernels for 2D block tiling kernel with bank conflict resolution
      cudaEventRecord(start);
      for (int i = 0; i < 100; ++i) {
        sgemmResolveBankExtra_Padding<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
      }
      cudaEventRecord(stop);
      cudaDeviceSynchronize();
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&bank_conflict_time_padding, start, stop);
    } else {
      // this is a hacky solution to the underlying problem
      // of not having proper bounds checking in the kernel
      const uint BM = 64;
      const uint BN = 64;
      dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
      dim3 blockDim((BM * BN) / (TM * TN));
      // warmup runs for 2D block tiling kernel with bank conflict resolution
      for (int i = 0; i < 100; ++i) {
        sgemmResolveBankExtra_Padding<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
      }
      // benchmark kernels for 2D block tiling kernel with bank conflict resolution
      cudaEventRecord(start);
      for (int i = 0; i < 100; ++i) {
        sgemmResolveBankExtra_Padding<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
      }
      cudaEventRecord(stop);
      cudaDeviceSynchronize();
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&bank_conflict_time_padding, start, stop);
    }
    float avg_bank_conflict_time_padding = bank_conflict_time_padding / 100.0f;
    //calculating glops
    seconds = avg_bank_conflict_time_padding / 1000.0f;
    giga_ops = (float)total_ops / 1e9f / seconds;
    printf("bank conflict resolution by padding average time: %f s, GLOPS: %f\n", seconds, giga_ops);
    printf("this can be slow then previous kernels if there was no bank conflict in the previous kernels\n");

    // autotuning kernel
    const uint Autotuned_BK = 16;
    const uint Autotuned_TM = 8;
    const uint Autotuned_TN = 8;
    const uint Autotuned_BM = 128;
    const uint Autotuned_BN = 128;
    dim3 blockDim(Autotuned_NUM_THREADS);
    static_assert(
        (Autotuned_NUM_THREADS * 4) % Autotuned_BK == 0,
        "NUM_THREADS*4 must be multiple of Autotuned_BK to avoid quantization issues "
        "during GMEM->SMEM tiling (loading only parts of the final row of Bs "
        "during each iteraion)");
    static_assert(
        (Autotuned_NUM_THREADS * 4) % Autotuned_BN == 0,
        "NUM_THREADS*4 must be multiple of Autotuned_BN to avoid quantization issues "
        "during GMEM->SMEM tiling (loading only parts of the final row of As "
        "during each iteration)");
    static_assert(
        Autotuned_BN % (16 * Autotuned_TN) == 0,
        "Autotuned_BN must be a multiple of 16*Autotuned_TN to avoid quantization effects");
    static_assert(
        Autotuned_BM % (16 * Autotuned_TM) == 0,
        "Autotuned_BM must be a multiple of 16*Autotuned_TM to avoid quantization effects");
    static_assert((Autotuned_BM * Autotuned_BK) % (4 * Autotuned_NUM_THREADS) == 0,
                  "Autotuned_BM*Autotuned_BK must be a multiple of 4*256 to vectorize loads");
    static_assert((Autotuned_BN * Autotuned_BK) % (4 * Autotuned_NUM_THREADS) == 0,
                  "Autotuned_BN*Autotuned_BK must be a multiple of 4*256 to vectorize loads");
    dim3 gridDim(CEIL_DIV(N, Autotuned_BN), CEIL_DIV(M, Autotuned_BM));
    // warmup runs for autotuned kernel
    for (int i = 0; i < 100; ++i) {
      sgemmAutotuned<Autotuned_BM, Autotuned_BN, Autotuned_BK, Autotuned_TM, Autotuned_TN><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    // benchmark kernels for autotuned kernel
    float autotuned_time = 0;
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
      sgemmAutotuned<Autotuned_BM, Autotuned_BN, Autotuned_BK, Autotuned_TM, Autotuned_TN><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&autotuned_time, start, stop);
    //calculating glops
    float avg_autotuned_time = autotuned_time / 100.0f;
    seconds = avg_autotuned_time / 1000.0f;
    giga_ops = (float)total_ops / 1e9f / seconds;
    printf("Autotuned kernel average time: %f s, GLOPS: %f\n", seconds, giga_ops);

    // wraptiling kernel
    const uint wraptilling_NUM_THREADS = 128;
    const uint wraptilling_BN = 128;
    const uint wraptilling_BM = 128;
    const uint wraptilling_BK = 16;
    const uint wraptilling_WN = 64;
    const uint wraptilling_WM = 64;
    const uint wraptilling_WNITER = 4;
    const uint wraptilling_TN = 4;
    const uint wraptilling_TM = 8;
    dim3 blockDim_wraptilling(wraptilling_NUM_THREADS);
    constexpr uint NUM_WARPS = wraptilling_NUM_THREADS / 32;
    // warptile in threadblocktile
    static_assert((wraptilling_BN % wraptilling_WN == 0) and (wraptilling_BM % wraptilling_WM == 0));
    static_assert((wraptilling_BN / wraptilling_WN) * (wraptilling_BM / wraptilling_WM) == NUM_WARPS);
    // threads in warpsubtile
    static_assert((wraptilling_WM * wraptilling_WN) % (WARPSIZE * wraptilling_TM * wraptilling_TN * wraptilling_WNITER) ==0);
    constexpr uint wraptilling_WMITER =
        (wraptilling_WM * wraptilling_WN) / (32 * wraptilling_TM * wraptilling_TN * wraptilling_WNITER);
    // warpsubtile in warptile
    static_assert((wraptilling_WM % wraptilling_WMITER == 0) and (wraptilling_WN % wraptilling_WNITER == 0));
    static_assert((wraptilling_NUM_THREADS * 4) % wraptilling_BK == 0,
                  "NUM_THREADS*4 must be multiple of wraptilling_BK to avoid quantization "
                  "issues during GMEM->SMEM tiling (loading only parts of the "
                  "final row of Bs during each iteraion)");
    static_assert((wraptilling_NUM_THREADS * 4) % wraptilling_BN == 0,
                  "NUM_THREADS*4 must be multiple of wraptilling_BN to avoid quantization "
                  "issues during GMEM->SMEM tiling (loading only parts of the "
                  "final row of As during each iteration)");
    static_assert(wraptilling_BN % (16 * wraptilling_TN) == 0,
                  "BN must be a multiple of 16*TN to avoid quantization effects");
    static_assert(wraptilling_BM % (16 * wraptilling_TM) == 0,
                  "BM must be a multiple of 16*TM to avoid quantization effects");
    static_assert((wraptilling_BM * wraptilling_BK) % (4 * wraptilling_NUM_THREADS) == 0,
                  "BM*BK must be a multiple of 4*256 to vectorize loads");
    static_assert((wraptilling_BN * wraptilling_BK) % (4 * wraptilling_NUM_THREADS) == 0,
                  "BN*BK must be a multiple of 4*256 to vectorize loads");
    dim3 gridDim_wraptilling(CEIL_DIV(N, wraptilling_BN), CEIL_DIV(M, wraptilling_BM));
    // warmup runs for wraptilling kernel
    for (int i = 0; i < 100; ++i) {
      sgemmWarptiling<wraptilling_BM, wraptilling_BN, wraptilling_BK, wraptilling_WM, wraptilling_WN, wraptilling_WNITER, wraptilling_TM,wraptilling_TN, wraptilling_NUM_THREADS><<<gridDim_wraptilling, blockDim_wraptilling>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    // benchmark kernels for wraptilling kernel
    float wraptilling_time = 0;
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
      sgemmWarptiling<wraptilling_BM, wraptilling_BN, wraptilling_BK, wraptilling_WM, wraptilling_WN, wraptilling_WNITER, wraptilling_TM,wraptilling_TN, wraptilling_NUM_THREADS><<<gridDim_wraptilling, blockDim_wraptilling>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&wraptilling_time, start, stop);
    float avg_wraptilling_time = wraptilling_time / 100.0f;
    //calculating glops
    seconds = avg_wraptilling_time / 1000.0f;
    giga_ops = (float)total_ops / 1e9f / seconds;
    printf("Wraptilling kernel average time: %f s, GLOPS: %f\n", seconds, giga_ops);

    // double buffering kernel V1
    const uint dbv1_NUM_THREADS = 256;
    const uint dbv1_BN = 256;
    const uint dbv1_BM = 128;
    const uint dbv1_BK = 16;
    const uint dbv1_WN = 32;
    const uint dbv1_WM = 128;
    const uint dbv1_WNITER = 1;
    const uint dbv1_TN = 8;
    const uint dbv1_TM = 8;
    dim3 blockDim_bdv1(dbv1_NUM_THREADS);

    constexpr uint dbv1_NUM_WARPS = dbv1_NUM_THREADS / 32;

    // warptile in threadblocktile
    static_assert((dbv1_BN % dbv1_WN == 0) and (dbv1_BM % dbv1_WM == 0));
    static_assert((dbv1_BN / dbv1_WN) * (dbv1_BM / dbv1_WM) == dbv1_NUM_WARPS);

    // threads in warpsubtile
    static_assert((dbv1_WM * dbv1_WN) % (WARPSIZE * dbv1_TM * dbv1_TN * dbv1_WNITER) ==
                  0);
    constexpr uint dbv1_WMITER =
        (dbv1_WM * dbv1_WN) / (32 * dbv1_TM * dbv1_TN * dbv1_WNITER);
    // warpsubtile in warptile
    static_assert((dbv1_WM % dbv1_WMITER == 0) and (dbv1_WN % dbv1_WNITER == 0));

    static_assert((dbv1_NUM_THREADS / 2 * 4) % dbv1_BK == 0,
                  "NUM_THREADS*4 must be multiple of BK to avoid quantization "
                  "issues during GMEM->SMEM tiling (loading only parts of the "
                  "final row of Bs during each iteraion)");
    static_assert((dbv1_NUM_THREADS / 2 * 4) % dbv1_BN == 0,
                  "NUM_THREADS*4 must be multiple of BN to avoid quantization "
                  "issues during GMEM->SMEM tiling (loading only parts of the "
                  "final row of As during each iteration)");
    static_assert(dbv1_BN % (16 * dbv1_TN) == 0,
                  "BN must be a multiple of 16*TN to avoid quantization effects");
    static_assert(dbv1_BM % (16 * dbv1_TM) == 0,
                  "BM must be a multiple of 16*TM to avoid quantization effects");
    static_assert((dbv1_BM * dbv1_BK) % (4 * dbv1_NUM_THREADS / 2) == 0,
                  "BM*BK must be a multiple of 4*256 to vectorize loads");
    static_assert((dbv1_BN * dbv1_BK) % (4 * dbv1_NUM_THREADS / 2) == 0,
                  "BN*BK must be a multiple of 4*256 to vectorize loads");

    dim3 gridDim_bdv1(CEIL_DIV(N, dbv1_BN), CEIL_DIV(M, dbv1_BM));
    // warmup runs for double buffering kernel V1
    for (int i = 0; i < 100; ++i) {
      sgemmDoubleBuffering_V1<dbv1_BM, dbv1_BN, dbv1_BK, dbv1_WM, dbv1_WN, dbv1_WNITER,dbv1_TM, dbv1_TN, dbv1_NUM_THREADS><<<gridDim_bdv1, blockDim_bdv1>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    // benchmark kernels for double buffering kernel V1
    float dbv1_time = 0;
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
      sgemmDoubleBuffering_V1<dbv1_BM, dbv1_BN, dbv1_BK, dbv1_WM, dbv1_WN, dbv1_WNITER,dbv1_TM, dbv1_TN, dbv1_NUM_THREADS><<<gridDim_bdv1, blockDim_bdv1>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&dbv1_time, start, stop);
    float avg_dbv1_time = dbv1_time / 100.0f;
    //calculating glops
    seconds = avg_dbv1_time / 1000.0f;
    giga_ops = (float)total_ops / 1e9f / seconds;
    printf("Double buffering kernel V1 average time: %f s, GLOPS: %f\n", seconds, giga_ops);
    printf("can be slow as it only helps when the load latency > compute time\n");
    
    // double buffering kernel V2
    const uint dbv2_NUM_THREADS = 128;
    const uint dbv2_BN = 128;
    const uint dbv2_BM = 128;
    const uint dbv2_BK = 16;
    const uint dbv2_WN = 64;
    const uint dbv2_WM = 64;
    const uint dbv2_WNITER = 4;
    const uint dbv2_TN = 4;
    const uint dbv2_TM = 8;
    dim3 blockDim_dbv2(dbv2_NUM_THREADS);
    
    constexpr uint NUM_WARPS_dbv2 = dbv2_NUM_THREADS / 32;
    
    // warptile in threadblocktile
    static_assert((dbv2_BN % dbv2_WN == 0) and (dbv2_BM % dbv2_WM == 0));
    static_assert((dbv2_BN / dbv2_WN) * (dbv2_BM / dbv2_WM) == NUM_WARPS_dbv2);
    
    // threads in warpsubtile
    static_assert((dbv2_WM * dbv2_WN) % (WARPSIZE * dbv2_TM * dbv2_TN * dbv2_WNITER) ==
    0);
    constexpr uint dbv2_WMITER =
    (dbv2_WM * dbv2_WN) / (32 * dbv2_TM * dbv2_TN * dbv2_WNITER);
    // warpsubtile in warptile
    static_assert((dbv2_WM % dbv2_WMITER == 0) and (dbv2_WN % dbv2_WNITER == 0));
    
    static_assert((dbv2_NUM_THREADS * 4) % dbv2_BK == 0,
    "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
    "issues during GMEM->SMEM tiling (loading only parts of the "
    "final row of Bs during each iteraion)");
    static_assert((dbv2_NUM_THREADS * 4) % dbv2_BN == 0,
    "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
    "issues during GMEM->SMEM tiling (loading only parts of the "
    "final row of As during each iteration)");
    static_assert(dbv2_BN % (16 * dbv2_TN) == 0,
    "BN must be a multiple of 16*TN to avoid quantization effects");
    static_assert(dbv2_BM % (16 * dbv2_TM) == 0,
    "BM must be a multiple of 16*TM to avoid quantization effects");
    static_assert((dbv2_BM * dbv2_BK) % (4 * dbv2_NUM_THREADS) == 0,
    "BM*BK must be a multiple of 4*256 to vectorize loads");
    static_assert((dbv2_BN * dbv2_BK) % (4 * dbv2_NUM_THREADS) == 0,
    "BN*BK must be a multiple of 4*256 to vectorize loads");
    
    dim3 gridDim_dbv2(CEIL_DIV(N, dbv2_BN), CEIL_DIV(M, dbv2_BM));
    // warmup runs for double buffering kernel V2
    for (int i = 0; i < 100; ++i) {
      runSgemmDoubleBuffering_V2<dbv2_BM, dbv2_BN, dbv2_BK, dbv2_WM, dbv2_WN, dbv2_WNITER,dbv2_TM, dbv2_TN, dbv2_NUM_THREADS><<<gridDim_dbv2, blockDim_dbv2>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    // benchmark kernels for double buffering kernel V2
    float dbv2_time = 0;
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
      runSgemmDoubleBuffering_V2<dbv2_BM, dbv2_BN, dbv2_BK, dbv2_WM, dbv2_WN, dbv2_WNITER,dbv2_TM, dbv2_TN, dbv2_NUM_THREADS><<<gridDim_dbv2, blockDim_dbv2>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&dbv2_time, start, stop);
    float avg_dbv2_time = dbv2_time / 100.0f;
    //calculating glops
    seconds = avg_dbv2_time / 1000.0f;
    giga_ops = (float)total_ops / 1e9f / seconds;
    printf("Double buffering kernel V2 average time: %f s, GLOPS: %f\n", seconds, giga_ops);
    printf("can be slow as it only helps when the load latency > compute time\n");
    
    // cublas testing on cuda cores
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    //warmup runs for cublas kernel
    for (int i = 0; i < 100; ++i) {
      CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    }
    // benchmark kernels for cublas kernel
    float cublas_time = 0;
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
      }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cublas_time, start, stop);
    float avg_cublas_time = cublas_time / 100.0f;
    //calculating glops
    seconds = avg_cublas_time / 1000.0f;
    giga_ops = (float)total_ops / 1e9f / seconds;
    printf("cuBLAS kernel on CUDA cores average time: %f s, GLOPS: %f\n", seconds, giga_ops);

    // cublas on tensor cores
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    //warmup runs for cublas kernel
    for (int i = 0; i < 100; ++i) {
      CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    }
    // benchmark kernels for cublas kernel
    float cublas_time_tensor = 0;
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
      }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cublas_time_tensor, start, stop);
    float avg_cublas_time_tensor = cublas_time_tensor / 100.0f;
    //calculating glops
    seconds = avg_cublas_time_tensor / 1000.0f;
    giga_ops = (float)total_ops / 1e9f / seconds;
    printf("cuBLAS kernel on TENSOR cores average time: %f s, GLOPS: %f\n", seconds, giga_ops);
    
    free(A);
    free(B);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CHECK_CUBLAS(cublasDestroy(handle));
}