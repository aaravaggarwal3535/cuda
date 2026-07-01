#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void polynomial_activation_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ output,
    size_t size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t val = x[idx];
        output[idx] = val * val + val + 1; // x^2 + x + 1
    }
}

torch::Tensor polynomial_activation_cuda(torch::Tensor x) {
    auto output = torch::empty_like(x); // it is used so that it automatically matches the input tensor's shape and type for the output tensor
    int threads = 1024;
    int blocks = (x.numel() + threads - 1) / threads;

    // main calling of the kernel that can be done by torch 
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "polynomial_activation_cuda", ([&] {
        polynomial_activation_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            x.numel()
        );
    }));

    return output;
}

// this is the main function that is used to bind the C++ code with python code so that it can be called from python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("polynomial_activation", &polynomial_activation_cuda, "Polynomial activation (CUDA)");
}