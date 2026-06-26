# cuDNN Convolution and Activation API

## Objective

The objective of these programs is to understand how NVIDIA's **cuDNN (CUDA Deep Neural Network Library)** performs deep learning operations efficiently on NVIDIA GPUs.

This project compares three different implementations of common neural network operations:

* **CPU Implementation**
* **Custom CUDA Kernel**
* **NVIDIA cuDNN Implementation**

The project demonstrates two operations:

* **Tanh Activation**
* **2D Convolution**

while measuring:

* Correctness
* Performance
* Speedup
* GPU utilization

---

# What is cuDNN?

cuDNN (CUDA Deep Neural Network Library) is NVIDIA's highly optimized GPU library designed specifically for deep learning workloads.

Instead of writing CUDA kernels manually for every neural network layer, cuDNN provides optimized implementations for:

* Convolution
* Matrix Multiplication (GEMM)
* Pooling
* Batch Normalization
* Layer Normalization
* Softmax
* Activation Functions
* Tensor Transformations
* RNNs
* Multi-head Attention
* Fusion Operations

Internally, cuDNN utilizes:

* Tensor Cores
* Shared Memory
* Register Blocking
* Warp-Level Programming
* Winograd Convolution
* FFT Convolution
* Implicit GEMM
* Kernel Fusion
* Memory Scheduling

to maximize throughput while minimizing memory bandwidth usage.

---

# Overall Program Flow

The program follows the same execution pipeline used internally by deep learning frameworks such as PyTorch and TensorFlow.

```text
Host Memory
      │
      ▼
Allocate GPU Memory
      │
      ▼
Copy Input Tensor to GPU
      │
      ▼
Create cuDNN Handle
      │
      ▼
Create Tensor Descriptors
      │
      ▼
Create Filter Descriptor
      │
      ▼
Create Convolution / Activation Descriptor
      │
      ▼
Find Best cuDNN Algorithm
      │
      ▼
Allocate Workspace
      │
      ▼
Execute cuDNN Operation
      │
      ▼
Copy Output Back
      │
      ▼
Compare with CPU & Custom CUDA
      │
      ▼
Benchmark Performance
```

---

# Theory

## Why does cuDNN use Descriptors?

GPU memory is simply a flat linear array.

Example:

```text
1 2 3 4 5 6 7 8
```

The GPU has no knowledge of whether this represents:

* A Vector
* A Matrix
* An Image
* A Tensor
* A Batch of Images

Therefore, cuDNN requires **Descriptors**.

Descriptors tell cuDNN:

* Data Type
* Dimensions
* Layout
* Strides
* Memory Organization

Example:

Raw Memory

```text
1 2 3 4 5 6 7 8
```

Descriptor

```text
N = 1
C = 2
H = 2
W = 2
```

cuDNN interprets it as

```text
Batch 0

Channel 0
1 2
3 4

Channel 1
5 6
7 8
```

Without descriptors, cuDNN cannot understand the structure of the memory.

---

# Main Components Used in the Code

## 1. cudnnHandle_t

```cpp
cudnnHandle_t cudnn;
cudnnCreate(&cudnn);
```

### Purpose

Creates the cuDNN execution context.

Every cuDNN function requires this handle.

Internally it stores:

* CUDA Stream
* GPU Context
* Internal Buffers
* Algorithm Cache
* Execution State

### API

```cpp
cudnnCreate(cudnnHandle_t* handle)
```

### Input

* Pointer to a cuDNN handle

### Output

* Initialized cuDNN context

Destroy using

```cpp
cudnnDestroy(handle);
```

---

## 2. Tensor Descriptor

Represents an input or output tensor.

```cpp
cudnnTensorDescriptor_t
```

### Create

```cpp
cudnnCreateTensorDescriptor(&tensorDesc);
```

Creates an empty descriptor.

---

### Configure

```cpp
cudnnSetTensor4dDescriptor(
    tensorDesc,
    layout,
    datatype,
    N,
    C,
    H,
    W
);
```

### Parameters

**tensorDesc**

Descriptor to configure.

**layout**

Tensor memory layout.

Usually

```cpp
CUDNN_TENSOR_NCHW
```

which means

```text
Batch
Channels
Height
Width
```

**datatype**

Usually

```cpp
CUDNN_DATA_FLOAT
```

Possible types

* FLOAT
* DOUBLE
* HALF
* INT8

**N**

Batch size

**C**

Number of Channels

**H**

Height

**W**

Width

Example

```cpp
256
32
224
224
```

means

* 256 Images
* 32 Channels
* Resolution 224 × 224

---

## 3. Filter Descriptor

Represents convolution kernels.

```cpp
cudnnFilterDescriptor_t
```

Create

```cpp
cudnnCreateFilterDescriptor(&filterDesc);
```

Configure

```cpp
cudnnSetFilter4dDescriptor(
    filterDesc,
    datatype,
    layout,
    outChannels,
    inChannels,
    kernelHeight,
    kernelWidth
);
```

### Parameters

* Output Channels
* Input Channels
* Kernel Height
* Kernel Width

Example

```text
64 filters
3 input channels
3 × 3 kernel
```

---

## 4. Convolution Descriptor

Represents the convolution operation itself.

```cpp
cudnnConvolutionDescriptor_t
```

Create

```cpp
cudnnCreateConvolutionDescriptor(&convDesc);
```

Configure

```cpp
cudnnSetConvolution2dDescriptor(
    convDesc,
    padH,
    padW,
    strideH,
    strideW,
    dilationH,
    dilationW,
    mode,
    datatype
);
```

### Parameters

Padding

```text
padH
padW
```

Stride

```text
strideH
strideW
```

Dilation

```text
dilationH
dilationW
```

Mode

Usually

```cpp
CUDNN_CROSS_CORRELATION
```

Datatype

Usually

```cpp
CUDNN_DATA_FLOAT
```

Example

```text
Padding = 1
Stride = 1
Dilation = 1
```

Produces a same-sized output.

---

## 5. Activation Descriptor

Used in the activation example.

```cpp
cudnnActivationDescriptor_t
```

Create

```cpp
cudnnCreateActivationDescriptor()
```

Configure

```cpp
cudnnSetActivationDescriptor(
    activationDesc,
    activationMode,
    nanMode,
    coefficient
);
```

### Parameters

Activation Mode

Examples

* ReLU
* Tanh
* Sigmoid
* ELU
* Clipped ReLU

NaN Mode

Usually

```cpp
CUDNN_PROPAGATE_NAN
```

Coefficient

Used only by some activation functions.

For Tanh

```cpp
0.0
```

---

## 6. Finding the Best Convolution Algorithm

One of the most important APIs.

```cpp
cudnnGetConvolutionForwardAlgorithm_v7()
```

cuDNN contains multiple convolution algorithms.

Examples include

* IMPLICIT GEMM
* PRECOMP GEMM
* FFT
* FFT TILING
* WINOGRAD
* DIRECT

Each algorithm has different characteristics.

Some

* use more memory
* execute faster
* work better for small images
* work better for large images

This API benchmarks the available algorithms and returns their performance.

### Inputs

* Handle
* Input Descriptor
* Filter Descriptor
* Convolution Descriptor
* Output Descriptor
* Maximum algorithms requested

### Outputs

For every algorithm:

* Execution Time
* Memory Usage
* Status
* Algorithm ID

The fastest valid algorithm is then selected.

---

## 7. Workspace

Many high-performance convolution algorithms require temporary GPU memory.

Query required memory

```cpp
cudnnGetConvolutionForwardWorkspaceSize()
```

### Inputs

* Handle
* Input Descriptor
* Filter Descriptor
* Convolution Descriptor
* Output Descriptor
* Selected Algorithm

### Output

Workspace Size

Allocate using

```cpp
cudaMalloc(&workspace, workspaceSize);
```

This workspace is later passed to the convolution function.

---

## 8. Forward Convolution

Main execution API

```cpp
cudnnConvolutionForward(
    handle,
    &alpha,
    inputDesc,
    input,
    filterDesc,
    filter,
    convDesc,
    algorithm,
    workspace,
    workspaceSize,
    &beta,
    outputDesc,
    output
);
```

### Parameter Explanation

#### Handle

cuDNN execution context.

---

#### Alpha

Scaling factor applied to the convolution result.

Formula

```text
Output = alpha × Conv(Input, Filter)
       + beta × ExistingOutput
```

Usually

```cpp
alpha = 1
```

---

#### Input Descriptor

Metadata describing the input tensor.

---

#### Input Pointer

Pointer to GPU memory.

Example

```cpp
float* d_input;
```

---

#### Filter Descriptor

Metadata describing convolution kernels.

---

#### Filter Pointer

GPU memory containing kernel weights.

---

#### Convolution Descriptor

Contains

* Padding
* Stride
* Dilation
* Convolution Mode

---

#### Algorithm

Algorithm selected by

```cpp
cudnnGetConvolutionForwardAlgorithm_v7()
```

---

#### Workspace

Temporary GPU memory.

---

#### Workspace Size

Size of temporary memory.

---

#### Beta

Scaling factor for existing output.

Usually

```cpp
0
```

---

#### Output Descriptor

Metadata describing output tensor.

---

#### Output Pointer

Destination GPU memory.

Internally

```text
Input
   ↓
Padding
   ↓
Sliding Window
   ↓
Multiply
   ↓
Accumulate
   ↓
Output
```

---

## 9. Activation Forward

Used in the activation example.

```cpp
cudnnActivationForward(
    handle,
    activationDesc,
    &alpha,
    inputDesc,
    input,
    &beta,
    outputDesc,
    output
);
```

Depending on the activation descriptor, cuDNN computes

```text
Output = tanh(Input)
```

or

```text
Output = relu(Input)
```

or any supported activation function.

---

# Benchmarking

The project compares

```text
CPU
      │
      ▼
Naive CUDA Kernel
      │
      ▼
Self-Made CUDA Kernel
      │
      ▼
cuDNN
```

Each implementation is benchmarked using CUDA Events.

Metrics collected include

* Average execution time
* Speedup
* Numerical correctness

---

# Why is cuDNN Faster?

A naive CUDA kernel repeatedly accesses global memory.

```text
Global Memory
      ↓
Multiply
      ↓
Store
      ↓
Repeat
```

cuDNN instead uses:

* Tensor Cores
* Shared Memory
* Register Blocking
* Instruction Scheduling
* Vectorized Memory Access
* Winograd Convolution
* FFT Convolution
* Kernel Fusion
* Pipeline Optimization

These techniques dramatically reduce memory traffic while maximizing arithmetic throughput.

---

# Summary of Important cuDNN APIs

| API                                         | Purpose                                                 |
| ------------------------------------------- | ------------------------------------------------------- |
| `cudnnCreate()`                             | Creates the cuDNN execution context                     |
| `cudnnDestroy()`                            | Destroys the cuDNN handle                               |
| `cudnnCreateTensorDescriptor()`             | Creates a tensor descriptor                             |
| `cudnnSetTensor4dDescriptor()`              | Defines tensor shape and layout                         |
| `cudnnCreateFilterDescriptor()`             | Creates a filter descriptor                             |
| `cudnnSetFilter4dDescriptor()`              | Defines convolution kernel dimensions                   |
| `cudnnCreateConvolutionDescriptor()`        | Creates a convolution descriptor                        |
| `cudnnSetConvolution2dDescriptor()`         | Defines padding, stride, dilation, and convolution mode |
| `cudnnCreateActivationDescriptor()`         | Creates an activation descriptor                        |
| `cudnnSetActivationDescriptor()`            | Configures activation type (ReLU, Tanh, etc.)           |
| `cudnnGetConvolutionForwardAlgorithm_v7()`  | Finds the fastest forward convolution algorithm         |
| `cudnnGetConvolutionForwardWorkspaceSize()` | Returns required workspace memory size                  |
| `cudnnConvolutionForward()`                 | Executes forward convolution                            |
| `cudnnActivationForward()`                  | Executes activation functions                           |

---

# Key Takeaways

* GPU memory is just a flat array; **Descriptors** tell cuDNN how to interpret it as tensors.
* A **`cudnnHandle_t`** stores the execution context for every cuDNN call.
* cuDNN provides multiple optimized convolution algorithms and automatically selects the best one for the given tensor shape and GPU.
* High-performance algorithms often require additional **workspace memory**.
* `cudnnConvolutionForward()` performs optimized forward convolution using the selected algorithm.
* `cudnnActivationForward()` applies highly optimized activation functions without requiring a custom CUDA kernel.
* Comparing **CPU**, **Naive CUDA**, **Custom CUDA**, and **cuDNN** implementations demonstrates both correctness and the significant performance improvements achieved by NVIDIA's deep learning library.
