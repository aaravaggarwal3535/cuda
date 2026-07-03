#include <iostream>
#include <cuda_runtime.h>

// Universal 3.x/4.x Runtime Drivers
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal.h> 
#include <cutlass/epilogue/thread/linear_combination.h>

int main() {
    int M = 4096;
    int N = 4096;
    int K = 4096;

    // 1. Map Core Configuration Parameters
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = float;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using ArchTag = cutlass::arch::Sm80; // Covering Sm80-Sm89 native mma instructions

    // 2. Define Components (Hierarchy Layout)
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>; // BM, BN, BK
    using WarpShape        = cutlass::gemm::GemmShape<64, 64, 64>;   // Warp Tile Size
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;    // Tensor Core MMA Instruction

    // 3. Define Thread-Level Epilogue
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementC,
        128 / cutlass::sizeof_bits<ElementC>::value, // Alignment = 4 floats (16 bytes)
        ElementAccumulator,
        ElementAccumulator
    >;

    // 4. Assemble explicitly into GemmUniversal 
    // This configures the multistage cp.async execution loop directly for SM80/SM89
    using Gemm = cutlass::gemm::device::GemmUniversal<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator,
        OperatorClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3 // StageCount: 3 stages of shared memory double/triple buffering
    >;

    // --- Execution and Benchmarking Routine ---
    size_t sizeA = (size_t)M * K * sizeof(ElementA);
    size_t sizeB = (size_t)K * N * sizeof(ElementB);
    size_t sizeC = (size_t)M * N * sizeof(ElementC);

    ElementA *dA, *dB;
    ElementC *dC;
    cudaMalloc(&dA, sizeA);
    cudaMalloc(&dB, sizeB);
    cudaMalloc(&dC, sizeC);

    // Initialize Arguments mapping to modern layout requirements
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, // Mode Selection
        {M, N, K},                               // Problem size
        1,                                       // Batch Count / Split-K factor
        {1.0f, 0.0f},                            // Alpha, Beta parameters
        dA, dB, dC, dC,                          // Direct Device Pointers
        M * K, K * N, M * N, M * N,              // Batch Strides
        K, N, N, N                               // Leading Dimensions
    };

    Gemm gemm_op;
    
    // Check if configuration is natively supported by current device
    cutlass::Status status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Initialization failed!\n";
        return -1;
    }

    // Warmup Loop (100 Iterations)
    for (int i = 0; i < 100; ++i) {
        gemm_op();
    }
    cudaDeviceSynchronize();

    // Benchmarking Loop (100 Iterations)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int benchmark_runs = 100;
    cudaEventRecord(start);
    for (int i = 0; i < benchmark_runs; ++i) {
        gemm_op();
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    // Calculations
    double avg_time_sec = (elapsed_ms / 1000.0) / benchmark_runs;
    double total_ops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = total_ops / (avg_time_sec * 1e9);

    std::cout << "============= CUTLASS 4.5 UNIVERSAL BENCHMARK ============\n";
    std::cout << "Target Hardware:   SM89 Compatible Pipeline\n";
    std::cout << "Staging Strategy:  " << 3 << "-Stage Asynchronous Loop\n";
    std::cout << "Avg Time:          " << avg_time_sec << " seconds\n";
    std::cout << "Performance:       " << gflops << " GFLOPS\n";
    std::cout << "========================================================\n";

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}