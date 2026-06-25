#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

__global__ void kernel1(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

__global__ void kernel2(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1.0f;
    }
}

void CUDART_CB myStreamCallback(cudaStream_t stream, cudaError_t status, void *userData) {
    printf("Stream callback: Operation completed\n");
}

int main(void) {
    const int N = 1000000;
    size_t size = N * sizeof(float);
    float *h_data, *d_data;
    cudaStream_t stream1, stream2, stream3;
    cudaEvent_t event;
    std::cout << event << std::endl;

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Allocate host and device memory
    CHECK_CUDA_ERROR(cudaMallocHost(&h_data, size));  // Pinned memory for faster transfers
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, size));

    // Initialize data
    for (int i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(i);
    }

    // Create streams with different priorities
    int leastPriority, greatestPriority;
    CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, leastPriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, greatestPriority));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream3)); // Default priority stream

    // Create event
    CHECK_CUDA_ERROR(cudaEventCreate(&event));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream1));
    // Asynchronous memory copy and kernel execution in stream1
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream1));
    kernel1<<<(N + 255) / 256, 256, 0, stream1>>>(d_data, N);

    // Record event in stream1
    CHECK_CUDA_ERROR(cudaEventRecord(event, stream1));

    // Make stream2 wait for event
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream2, event, 0));

    // Execute kernel in stream2
    kernel2<<<(N + 255) / 256, 256, 0, stream2>>>(d_data, N);

    // Add callback to stream2
    CHECK_CUDA_ERROR(cudaStreamAddCallback(stream2, myStreamCallback, NULL, 0));

    // Asynchronous memory copy back to host
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream2));
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream2));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    // Synchronize streams
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));

    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Time taken: %f ms\n", milliseconds);


    // Verify result
    for (int i = 0; i < N; ++i) {
        float expected = (static_cast<float>(i) * 2.0f) + 1.0f;
        if (fabs(h_data[i] - expected) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    cudaEvent_t event1, event2;
    CHECK_CUDA_ERROR(cudaEventCreate(&event1));
    CHECK_CUDA_ERROR(cudaEventCreate(&event2));

    for (int i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(i);
    }


    cudaEventRecord(event1, stream3);
    cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream3);
    kernel1<<<(N + 255) / 256, 256, 0, stream3>>>(d_data, N);
    kernel2<<<(N + 255) / 256, 256, 0, stream3>>>(d_data, N);
    cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream3);
    cudaEventRecord(event2, stream3);
    cudaEventSynchronize(event2);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, event1, event2);
    printf("Time taken for without streams: %f ms\n", elapsedTime);
    printf("Result time are almost same as here the stream version is alo getting synchronus in a way as we are using the event to synchronize the streams\n");

    // Clean up
    CHECK_CUDA_ERROR(cudaFreeHost(h_data));
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream3));
    CHECK_CUDA_ERROR(cudaEventDestroy(event1));
    CHECK_CUDA_ERROR(cudaEventDestroy(event2));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    return 0;
}