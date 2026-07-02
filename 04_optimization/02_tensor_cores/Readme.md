# Tensor Cores
**There are three ways:**
- WMMA API (high level, easiest)
- mma.sync PTX assembly (low level, faster)
- CUTLASS (production quality) (already in previous sections)

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