#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

#define NUM_THREADS 1000
#define NUM_BLOCKS 1000

// Kernel without atomics (incorrect)
__global__ void incrementCounterNonAtomic(int* counter) {
    // not locked
    int old = *counter;
    int new_value = old + 1;
    // not unlocked
    *counter = new_value;
}

// Kernel with atomics (correct)
__global__ void incrementCounterAtomic(int* counter) {
    atomicAdd(counter, 1);
}

//kernal with custom mutex add operations
// __device__ int globalLock = 0;

// __global__ void softwareAtomicAdd(int* address, int increment) {

//     while (atomicCAS(&globalLock, 0, 1) != 0);

//     *address += increment;

//     __threadfence();

//     atomicExch(&globalLock, 0);
// }

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    int h_counterNonAtomic = 0;
    int h_counterAtomic = 0;
    int h_counterSoftwareAtomic = 0;
    int *d_counterNonAtomic, *d_counterAtomic, *d_counterSoftwareAtomic;

    // Allocate device memory
    cudaMalloc((void**)&d_counterNonAtomic, sizeof(int));
    cudaMalloc((void**)&d_counterAtomic, sizeof(int));
    cudaMalloc((void**)&d_counterSoftwareAtomic, sizeof(int));

    // Copy initial counter values to device
    cudaMemcpy(d_counterNonAtomic, &h_counterNonAtomic, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counterAtomic, &h_counterAtomic, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counterSoftwareAtomic, &h_counterSoftwareAtomic, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernels
    double start_nonAtomic = get_time()*1e6; // Convert to microseconds
    incrementCounterNonAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterNonAtomic);
    cudaDeviceSynchronize(); // Ensure all threads have finished
    double end_nonAtomic = get_time()*1e6; // Convert to microseconds
    double nonAtomicTime = (end_nonAtomic - start_nonAtomic); // Convert to microseconds
    cudaMemcpy(&h_counterNonAtomic, d_counterNonAtomic, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Non-atomic counter value: %d, Time: %.2f ms\n", h_counterNonAtomic, nonAtomicTime / 1000.0);

    double start_atomic = get_time()*1e6; // Convert to microseconds
    incrementCounterAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterAtomic);
    cudaDeviceSynchronize(); // Ensure all threads have finished
    double end_atomic = get_time()*1e6; // Convert to microseconds
    double atomicTime = (end_atomic - start_atomic); // Convert to microseconds
    cudaMemcpy(&h_counterAtomic, d_counterAtomic, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Atomic counter value: %d, Time: %.2f ms\n", h_counterAtomic, atomicTime / 1000.0);

    // double start_softwareAtomic = get_time()*1e6; // Convert to microseconds
    // softwareAtomicAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterSoftwareAtomic, 1);
    // cudaDeviceSynchronize(); // Ensure all threads have finished
    // double end_softwareAtomic = get_time()*1e6; // Convert to microseconds
    // double softwareAtomicTime = (end_softwareAtomic - start_softwareAtomic); // Convert to microseconds
    // cudaMemcpy(&h_counterSoftwareAtomic, d_counterSoftwareAtomic, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("Software atomic counter value: %d, Time: %.2f ms\n", h_counterSoftwareAtomic, softwareAtomicTime / 1000.0);

    // Free device memory
    cudaFree(d_counterNonAtomic);
    cudaFree(d_counterAtomic);
    cudaFree(d_counterSoftwareAtomic);

    return 0;
}