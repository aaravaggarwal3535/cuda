# CUTLASS (version 4.5)

stands for : CUDA Templates for Linear Algebra Subroutines and Solvers

in cuda api when we use cublas we can say that there can be many diffrent kernel like
GEMM 128×128×32
GEMM 128×64×32
GEMM 64×64×32
GEMM FP32
GEMM FP16
GEMM BF16
Tensor Core
SIMT
...

instead of writing a kernel for all opof them where the algorithm is mostly simmilar the only thing that changes is
tile sizes
pipeline depth
instruction type
Tensor Core vs CUDA Core
layouts

so NVIDIA genralizes it 
CUTLASS is like a GEMM genrator
we tell cutlass:
Tile size = 128×128×32

Warp tile = 64×64×32

Instruction = Tensor Core

Layout = Row Major

Pipeline = 3 stages

Architecture = SM89
and it genrates the optimized kernel for it 

**Cutlas Architechture**
```
Application
      │
      ▼
CUTLASS API
      │
      ▼
Device GEMM
      │
      ▼
Threadblock GEMM
      │
      ▼
Warp GEMM
      │
      ▼
MMA Instruction
      │
      ▼
Tensor Core
```

- Application
    ```bash
    cutlass::gemm(...)
    ```
    this is what we use in here
- Device kernel : this decides the size of the grid and block
- Threadblock GEMM : decides the process of the single block operations like shared memory, double buffring, cp.async etc.
- wrap GEMM : this decide the operation of the wrap inside the block, wrap tiling happen here
- instructions : the ptx instructions comes here to utilize tensor core
- tensor core : the finall implementation of the tens core


# how to install and implement cutlass:
```bash
git clone https://github.com/NVIDIA/cutlass.git
```
```bash
nvcc -I ./cutlass/include -std=c++17 --expt-relaxed-constexpr -o cutlass_app cutlass.cu
```
```bash
./cutlass
```

```bash
using Gemm = cutlass::gemm::device::Gemm<...>;
```
this is kernel declaration

```bash
Gemm::Arguments args(
    {M,N,K},
    {A,K},
    {B,N},
    {C,N},
    {C,N},
    {1.0f,0.0f}
);
```
- MNK : define the dimensions
- AK : A is pointer to the matrix, K is the number of column in A
- BN : pointer to B, N is number of column in B
- CN : pointer to C, N is number of column in C for the beta*C operations
- CN : pointer to the output matrix, N is number of column in C
- 1,0 : alpha and beta

```bash
gemm(args);
```
use to run the actual kernel

# advance implementation of CUTLASS(v3x) concept remain same the syntax can change in future
To run
```bash
nvcc -I ./cutlass/include -std=c++17 --expt-relaxed-constexpr -o cutlass_test cutlass_adv.cu
./cutlass_test
```
CUTLASS hierarchy
```
Entire Matrix

        │
        ▼
+--------------------------+
| Threadblock Tile         |   (CTA)
| 128×128×32               |
+--------------------------+

        │
        ▼
+-----------+-----------+
| Warp Tile | Warp Tile |
| 64×64×32  | 64×64×32  |
+-----------+-----------+

        │
        ▼
+----------------------+
| MMA Tile             |
| 16×8×16              |
+----------------------+

        │
        ▼
Tensor Core
```
Grid --> ThreadblockShape --> WarpShape --> InstructionShape --> Tensor Core

like previously we have block tiling BM, BM, BK here we have ThreadBlockShape
suppose we have block 128x128 here we have Warpshape
instruction shape : it is a tensorcore instruction usually : 16x8x16

datatype should be half for matrix and float for accumalator and output matrix
A --> Row major, B --> column Major, C --> Row major

operator class : 
- cuda cores :  cutlass::arch::OpClassSimt
- tensor cores : cutlass::arch::OpClassTensorOp

architechture : depend on your device for mine it is cutlass::arch::Sm89

COllective builder : place where cuda start as it contains all the parameter that a cuda instruction needs

```bash
CollectiveBuilder<
    Arch,
    OperatorClass,

    ElementA,
    LayoutA,
    AlignmentA,

    ElementB,
    LayoutB,
    AlignmentB,

    Accumulator,

    TileShape,
    ClusterShape,

    StageCount,

    Schedule
>
```
- arch : cutlass::arch::Sm89(for me yours can be anything else)
- class : OpClassTensorOp
- element : it type
- layout : row or column major
- Alignment = 8: beans 16byte means 128 bit load
- tile shape : Shape<_128,_128,_64> BM, BN, BK
- cluster shape : 128x128
- stagecount : 2 : means double buffring
- schedule : warp sheduling or cp.async or prefetch etc

if you want to calculate a epilogue like Relu of the output that can be computed simultaneously 
C = alpha * A * B + beta * C
will then be
C = Relu(alpha * A * B + beta * C)

- LinearCombination : D = alpha* Accumulator + beta *C
- LinearCombinationRelu : D = ReLU(alpha* Accumulator + beta *C)
- LinearCombinationClamp : D = Clamp(...)
- LinearCombinationBias : D = Accumulator + Bias

To execute epilogue:
```bash
cutlass::epilogue::thread::LinearCombination<
    float,
    1,
    float,
    float
>
```

if we want a custom epilogue:

```bash
struct MyEpilogue
{
    CUTLASS_HOST_DEVICE
    float operator()(float acc,float old)
    {
        float x=acc+old;
        return 1.f/(1.f+exp(-x));
    }
};
```

Iterator : they are used to maintain a track of where we actually are simmilar to the threadId.x etc
```bash
iterator.load(fragment);
```
this till load only the specified memory to the tensor register and left the rest in the global memory only

Type of iterator:
- global iterator : global memry --> register
- shared iterator : shared memory --> register
- output iterator : accumalator --> global memory

data alwas load from global memory --> fragments --> tensor core, fragments are actually registers that are associated by a thread

iterator has a thread map that fix which thread registor loads what

we use iterator as it genrates reproducability when a new architechture changes we only need to change iterator rest all remain same

kernel scheduling : this is the component that decides how are the operations performed and scheduled one after the other for proper traversal of a GEMM
- naive cuda gemm : load a --> load b --> compute --> write c
- warp specialization : procedure warp --> consumer warp
    - procedure warp : global to shared memory
    - consumer warp : shared to tensorcore then writing to accumalator

CUTLASS schedule:
- kernel multistage : it is basic load and then compute but in this serveral stages are buffered while one computes other loads
- KernelCpAsyncWarpSpecialized : this is the one that usees the procedure and consumer architechture, procedure usses cp.async, and consumer uses mma.sync first loads then computes but ecerthing is overlapped
- KernelTmaWarpSpecialized : for advance hopper GPU sma90+ where there are dedicated hardware components in gpu to copy memory from global to shared much faster called TMA(Tensor Memory Accelerator).

stage count means how many stage can go in parallel
- StageCount<2> : stage 0 compute while stage 1 loads
- StageCount<3> : stage 0 compute while stage 1 and 2 loads 

but more stage more shared memory get consumed so lower occupancy so there is a trade off that can be auto tuned

StageCountAuto : it is used as it can seledct the best count to set for stage 

let say we have a huge matrix to multiply that can consume the work of all the SMs that are on the GPU

but for small mamtrix all the SMs cant work thus most of the gpu do nothing this is the problem.

so to solve this the scheduler always launch the number of thread block same as the number of the SMs and then distribute the thread in it such that they dont go to ideal state until the process is done this is called persistent kernel

stream K : in case of huge K dimention a single block cant compute K so instead it is distributed such that multiple K block perform same tile compuitation all in parallel, this allocation is dynamic

scheduler type that do all of this is: KernelMultistage, KernelCpAsyncWarpSpecialized
