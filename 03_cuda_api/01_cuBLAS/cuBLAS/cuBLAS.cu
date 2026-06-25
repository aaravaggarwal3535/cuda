// dedicated for small handwritten matrices
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <chrono> // Added for CPU timing

#define M 1024
#define K 512
#define N 1024

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

#undef PRINT_MATRIX
#define PRINT_MATRIX(mat, rows, cols) \
    for (int i = 0; i < rows; i++) { \
        for (int j = 0; j < cols; j++) \
            printf("%8.3f ", mat[i * cols + j]); \
        printf("\n"); \
    } \
    printf("\n");

// Initialize vector with random values
void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

void cpu_matmul(float *A, float *B, float *C) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

int main() {
    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float));
    float *C_cpu = (float*)malloc(M * N * sizeof(float));
    float *C_cublas_s = (float*)malloc(M * N * sizeof(float));
    float *C_cublas_h = (float*)malloc(M * N * sizeof(float));

    init_vector(A, M * K);
    init_vector(B, K * N);

    // --- CPU TIMING ---
    float c_net = 0.0f;
    for (int i = 0; i < 100; i++) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_matmul(A, B, C_cpu);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;
        c_net += cpu_duration.count();
    }
    c_net /= 100.0f; // Average over 100 runs

    // CUDA setup
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // --- SGEMM TIMING ---
    float alpha = 1.0f, beta = 0.0f;
    printf("temprory runs");
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));

    float s_net = 0.0f;
    for (int i = 0; i < 100; i++) {
        auto sgemm_start = std::chrono::high_resolution_clock::now();
        
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
        CHECK_CUDA(cudaDeviceSynchronize()); // CRITICAL: Wait for GPU to finish math
        
        auto sgemm_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> sgemm_duration = sgemm_end - sgemm_start;
        s_net += sgemm_duration.count();
    }
    s_net /= 100.0f; // Average over 100 runs

    CHECK_CUDA(cudaMemcpy(C_cublas_s, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // cuBLAS HGEMM Setup
    half *d_A_h, *d_B_h, *d_C_h;
    CHECK_CUDA(cudaMalloc(&d_A_h, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B_h, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C_h, M * N * sizeof(half)));

    half *A_h = (half*)malloc(M * K * sizeof(half));
    half *B_h = (half*)malloc(K * N * sizeof(half));
    half *C_h = (half*)malloc(M * N * sizeof(half));
    for (int i = 0; i < M * K; i++) A_h[i] = __float2half(A[i]);
    for (int i = 0; i < K * N; i++) B_h[i] = __float2half(B[i]);

    CHECK_CUDA(cudaMemcpy(d_A_h, A_h, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_h, B_h, K * N * sizeof(half), cudaMemcpyHostToDevice));

    // --- HGEMM TIMING ---
    __half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
    printf("temprory runs");
    CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_h, d_B_h, N, d_A_h, K, &beta_h, d_C_h, N));
    CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_h, d_B_h, N, d_A_h, K, &beta_h, d_C_h, N));
    CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_h, d_B_h, N, d_A_h, K, &beta_h, d_C_h, N));
    CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_h, d_B_h, N, d_A_h, K, &beta_h, d_C_h, N));

    float h_net = 0.0f;
    for (int i = 0; i < 100; i++) {
        auto hgemm_start = std::chrono::high_resolution_clock::now();
        CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_h, d_B_h, N, d_A_h, K, &beta_h, d_C_h, N));
        CHECK_CUDA(cudaDeviceSynchronize());
        auto hgemm_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> hgemm_duration = hgemm_end - hgemm_start;
        h_net += hgemm_duration.count();
    }
    h_net /= 100.0f; // Average over 100 runs

    // Copy result back to host and convert to float
    CHECK_CUDA(cudaMemcpy(C_h, d_C_h, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    for (int i = 0; i < M * N; i++) {
        C_cublas_h[i] = __half2float(C_h[i]);
    }

    // Print results
    // printf("Matrix A (%dx%d):\n", M, K);
    // PRINT_MATRIX(A, M, K);
    // printf("Matrix B (%dx%d):\n", K, N);
    // PRINT_MATRIX(B, K, N);
    // printf("CPU Result (%dx%d):\n", M, N);
    // PRINT_MATRIX(C_cpu, M, N);
    // printf("cuBLAS SGEMM Result (%dx%d):\n", M, N);
    // PRINT_MATRIX(C_cublas_s, M, N);
    // printf("cuBLAS HGEMM Result (%dx%d):\n", M, N);
    // PRINT_MATRIX(C_cublas_h, M, N);

    // Print the timing results
    printf("----------------------------------------\n");
    printf("CPU Execution Time:        %f ms\n", c_net);
    printf("cuBLAS SGEMM Execution:    %f ms\n", s_net);
    printf("cuBLAS HGEMM Execution:    %f ms\n", h_net);
    printf("----------------------------------------\n");

    // Clean up
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_A_h));
    CHECK_CUDA(cudaFree(d_B_h));
    CHECK_CUDA(cudaFree(d_C_h));
    free(A);
    free(B);
    free(C_cpu);
    free(C_cublas_s);
    free(C_cublas_h);
    free(A_h);
    free(B_h);
    free(C_h);
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}