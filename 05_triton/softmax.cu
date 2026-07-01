#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <stdlib.h>

__global__ void softmax_cuda(float* input, float* output, int B, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.y;
    
    if (tid < N && bid < B) {
        int offset = bid * N;
        float max_val = input[offset];
        for (int i = 1; i < N; i++) {
            max_val = max(max_val, input[offset + i]);
        }
        
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += expf(input[offset + i] - max_val);
        }
        
        for (int i = 0; i < N; i++) {
            output[offset + i] = expf(input[offset + i] - max_val) / sum;
        }
    }
}

void softmax(float *x, int N) {
    float max = x[0];
    for (int i = 1; i < N; i++) {
        if (x[i] > max) {
            max = x[i];
        }
    }
    float sum = 0.0;
    for (int i = 0; i < N; i++) {
        x[i] = exp(x[i] - max);
        sum += x[i];
    }
    for (int i = 0; i < N; i++) {
        x[i] /= sum;
    }
}

int main() {
    const int B = 4096;  // Batch size
    const int N = 4096;  // Row length
    float *x_cpu = (float*)malloc(B * N * sizeof(float));
    float *x_gpu = (float*)malloc(B * N * sizeof(float));
    float *d_input, *d_output;

    // Initialize input vector
    for (int i = 0; i < B * N; i++) {
        x_cpu[i] = (float)rand() / RAND_MAX;  // Random values between 0 and 1
        x_gpu[i] = x_cpu[i];  // Copy to GPU input
    }

    // Allocate device memory
    cudaMalloc((void**)&d_input, B * N * sizeof(float));
    cudaMalloc((void**)&d_output, B * N * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, x_gpu, B * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid_x = (N + threadsPerBlock - 1) / threadsPerBlock;
    dim3 gridDim(blocksPerGrid_x, B);
    // warmup runs for the kernel to avoid cold start overhead
    for (int i = 0; i < 100; i++) {
        softmax_cuda<<<gridDim, threadsPerBlock>>>(d_input, d_output, B, N);
    }
    // benchmark runs for the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime = 0.0f;
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        softmax_cuda<<<gridDim, threadsPerBlock>>>(d_input, d_output, B, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    float avgTime = elapsedTime / 1000.0f;
    printf("Average time for softmax kernel: %f ms\n", avgTime);

    // Copy result back to host
    cudaMemcpy(x_gpu, d_output, B * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute softmax on CPU (for one batch as an example)
    float cpu_start = clock();
    softmax(x_cpu, N);
    float cpu_time = (clock() - cpu_start) / (float)CLOCKS_PER_SEC * 1000.0f;
    printf("CPU time for softmax: %f ms\n", cpu_time);

    // Compare results (for the first batch as an example)
    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(x_cpu[i] - x_gpu[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    printf("Maximum difference between CPU and GPU results (first batch): %e\n", max_diff);

    // Clean up
    free(x_cpu);
    free(x_gpu);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}