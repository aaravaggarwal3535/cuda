# Tensor Cores
**There are three ways:**
- WMMA API (high level, easiest)
- mma.sync PTX assembly (low level, faster)
- CUTLASS (production quality)

## WMMA
In base CUDA cores we can perform 32 FMAs per instruction in a thread 
But by using CUDA Cores we can have : 
4096 FMAs per instruction's.

This is executed by the a wrap thus it is called: Warp Matrix Multiply Accumulate(WMMA)

There is no threadId.x because it assumes that all thread participated and handles that automatically

**Results for naive kernel: time: 0.018984 s, GFLOPS: 7239.621582**

significant improvement from the previous naive kernel.

```bash
#include <mma.h>
using namespace nvcuda;
```
this is a import for WMMA

```bash
int warpId = threadIdx.y;
```
in tensor core we dont move thread wise instead a wrap is computed as a unit so we calculate wrap id for traversal
```bash
tileRow = blockIdx.y * blockDim.y + warpId
tileCol = blockIdx.x
```
title row gives the adress of current Row that is being executed
title col gives the adress of current col that is being executed
```bash
wmma::fragment
```
This line of code declares a special variable (called a fragment) that holds a 16 × 16 mini-matrix tile inside a GPU warp's registers
```bash
wmma::fragment<
        wmma::matrix_a,
        16,16,16,
        half,
        wmma::row_major> a_frag;
```
- wmma::fragment: The template class used by CUDA to store matrix data across a warp (a group of 32 threads).
- wmma::matrix_a: Specifies that this specific fragment will act as the first input matrix (A) in a matrix multiplication operation (D = A × B + C)
- 16, 16, 16: Configures the geometry of the target matrix multiply-accumulate operation (M, N, K). This means it will handle a 16 × 16 tile of matrix A.
- half: Sets the data type of the matrix elements to FP16 (16-bit half-precision floating-point), which is the standard format Tensor Cores use for speed.
- wmma::row_major: Defines how the data layout looks in the raw memory. Elements are arranged row by row continuously
- a_frag: The actual name given to this variable instance.
- c_frag: the type is float as FP16xFP16 = FP32
```bash
fill_fragment(c_frag,0)
```
simmilar to all value of C_frag will be set to 0
```bash
for(k=0;k<K;k+=16)
```
the actual loop that is used for matrix multiplication
```bash
wmma::load_matrix_sync(a_frag,tileA,K)
```
it does:
Global Memory

↓

Warp registers

↓

Tensor Core fragment
```bash
wmma::mma_sync(c_frag,a_frag,b_frag,c_frag);
```
this is the actual matrix multiplication performed c_frag = a_frag * b_frag + c_frag
```bash
wmma::store_matrix_sync(tileC,c_frag,N, wmma::mem_row_major);
```
this is used to store the data back to the global memory

## Inline ptx(mm.sync)
WMMA it was the wrapper around PTX
but
WMMA is intentionally restrictive.

For example

You cannot choose
- register layout
- exact instruction
- operand layout
- new Tensor Core modes

Everything is hidden.

so using inline instructions remove all those restrictions.

in this we write
```bash
asm volatile(
"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
"{%0,%1,%2,%3},"
"{%4,%5,%6,%7},"
"{%8,%9},"
"{%10,%11,%12,%13};"
:
...
);
```
instead of 
```bash
wmma::mma_sync(c,a,b,c);
```

so that we can takl directly to othe tensor cores
fragments are slow as there are in c++ so compiler need to spend time in packing and unpacking fragments

but with ptx we can directly talk to registers thus saves time 

**what does the instruction mean :**
- mma.sync.aligned : Execute Tensor Core MMA(Matrix Multiply Acumalate)
- m16 : means output rows 16 A(16 x K)
- n8 : means the output column B(K x 8)
- k16 : reduction dimension C (M X N)
- row.col : means A is row major and B is column major 
- f32.f16.f16.f32 : Accumalator(float), A(half), B(half), output(float)
- register : we manually own every register
        <br>A register = a0, a1, a2, a3
        <br>B register = b0, b1
        <br>C register = c0, c1, c2, c3
        <br>Tensor core instruction : a0 a1 a2 a3 x b0 b1 + c0 c1 c2 c3
        <br>ldmatrix : this is used to rearange the data stored on the shared memory to that on the register of tensor core as the tensor core wants the data in a weird way so this rearange it automatically
        <br>more instruction types : FP16, BF16, TF32, INT8, FP8

We can also overlap cp.async, ldmatrix, mma.sync 