#include <iostream>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = float;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

using Gemm = cutlass::gemm::device::Gemm<
    ElementInputA,
    LayoutA,

    ElementInputB,
    LayoutB,

    ElementOutput,
    LayoutC,

    ElementAccumulator
>;

int main()
{
    int M = 4096;
    int N = 4096;
    int K = 4096;

    cutlass::half_t *A;
    cutlass::half_t *B;
    float *C;

    // Allocate memory
    cudaMalloc(&A, sizeof(cutlass::half_t) * M * K);
    cudaMalloc(&B, sizeof(cutlass::half_t) * K * N);
    cudaMalloc(&C, sizeof(float) * M * N);

    Gemm gemm;

    Gemm::Arguments args(
        {M, N, K},
        {A, K},
        {B, N},
        {C, N},
        {C, N},
        {1.0f, 0.0f}
    );

    // 1. Warmup Runs
    int warmup_runs = 100;
    for (int i = 0; i < warmup_runs; ++i) {
        cutlass::Status status = gemm(args);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "CUTLASS failed during warmup run " << i << "\n";
            return -1;
        }
    }
    
    // Synchronize to ensure all warmup operations are complete before timing
    cudaDeviceSynchronize();

    // 2. Create CUDA Events for Benchmarking
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int benchmark_runs = 100;

    // Record the start event
    cudaEventRecord(start);

    // Execute the benchmark loop
    for (int i = 0; i < benchmark_runs; ++i) {
        gemm(args);
    }

    // Record the stop event
    cudaEventRecord(stop);
    cudaDeviceSynchronize(); // Ensure all operations are complete before measuring time
    
    // Wait for the stop event to be completed by the GPU
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float total_time_ms = 0.0f;
    cudaEventElapsedTime(&total_time_ms, start, stop);

    // 3. Calculate Metrics
    float avg_time_ms = total_time_ms / benchmark_runs;
    
    // GEMM ops count = 2 * M * N * K (1 multiply + 1 accumulate per element)
    long long int total_ops = 2LL * M * N * K;
    
    // GFLOPS = (Ops / (Avg Time in Seconds)) / 1e9 
    // Which simplifies to: Ops / (Avg Time in ms * 1e6)
    double gflops = (float)total_ops / (double(avg_time_ms) * 1.0e6);

    // Display Results
    std::cout << "================ Benchmarking Results ================\n";
    std::cout << "Matrix Dimensions (M x N x K): " << M << " x " << N << " x " << K << "\n";
    std::cout << "Warmup Iterations: " << warmup_runs << "\n";
    std::cout << "Benchmark Iterations: " << benchmark_runs << "\n";
    std::cout << "Average Execution Time: " << avg_time_ms << " ms\n";
    std::cout << "Performance: " << gflops << " GFLOPS\n";
    std::cout << "======================================================\n";

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}