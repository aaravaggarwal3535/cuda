# Introduction 
## Scalable data parallel computing using GPU's
Mordern GPU's are made with multiple parallel operations performing threads.
PTX is defined as a virtual machine and a ISA for genral purpose parallel thread execution. 


## Goal of PTX
- Provide Stable ISA that provide multiple GPU genrations. 
- provide machine independent ISA for the C/C++ to target
- Facilitate hand-coding of libraries, performance kernels, and architecture tests.
- Provide a scalable programming model that spans GPU sizes from a single unit to many parallel
units.

# Programing Model
## A Highly Multithread Co-Processor
It act as a coprocessot to the host(CPU), as CPU can schedule its tasks to the GPU that required parallel execution of the thread, meanwhile ost is free to do other processes. 

## Thread Hirarchy
The batch of threads that execute a kernel is organized as grid. The grid consist either coopretive thread or cluster of coopretive thread. 

## Cooperative thread array
CTA is used for the parallel execution of the threads, the threads can comunicate with each other. to cordinate the communication we use synchronization. each thread has a unique id which is a collection of three parameter tid.x, tid.y, tid.z within a 1D, 2D, 3D CTA. Threads in the CTA execute in SIMT(Single Insturuction Multiple Thread) in groups called warps. A warp has 32 thread. PTC includes a run-time immediate constant, WARP_SZ, Which may be used in any opreand where a immediate operand is allowed.

## Cluster(Blocks) of cooperative Thread Array
Cluster is a group od CTA that can run together in parallel and can communicate with each other via shared memory. Cluster wise barrier can be used to synchronize all the thread within the cluster. each cluster in CTAs can has 1d, 2d, 3d shape specified by the parameter cluster_nctaid, each CTA in th cluster have a unique CTA  identifier that is cluster_ctarank across all dimension. and each CTA in a cluster has a unique CTA identifier that is cluster_ctaid. Threads may read and use these values through predefined, read-only special registers <b>%cluster_ctaid, %cluster_nctaid, %cluster_ctarank, %cluster_nctarank</b>. This can hold max 1024 threads.

## Grid of Clusters
Grid is a combination of cluster of CTAs, these are made so that multiple cluster of CTA can be fired for a single kernel but this comes at a cost that you cant communicate and synchronize threads in diffrent clusters. Each cluster has a unique identifier in a grid that is clusterid, Each grid of clusters has a 1D, 2D , or 3D shape specified by the parameter nclusterid, Each grid also has a unique temporal grid identifier (gridid). Threads may read and use these values through predefined, read-only special registers <b>%tid, %ntid, %clusterid, %nclusterid, and %gridid</b>

Each CTA has a unique identifier (ctaid) within a grid. Each grid of CTAs has 1D, 2D, or 3D shape specified by the parameter nctaid. Thread may use and read these values through predefined, read only special registers <b>%ctaid and %nct</b>.

## Memory Hierarchy
![Grid With Clusters](./assets/grid_with_clusters.png)
![Memory Hierarchy](./assets/memory_hierarchy.png)

# PTX Machine Model

## A set of WIMT Multiprocessors:
NVIDIA GPU architecture is built around a scalable array of multithreaded Streaming Multiprocessors (SMs). When a host program invokes a kernel grid, the blocks of the grid are enumerated and distributed to multiprocessors with available execution capacity. The threads of a thread block execute concurrently on one multiprocessor. 

**NOTE**
```A GPU has only fixed number of SM's
Each SM has a fixed number of cores
Each Core have a few resident Threads

A block is given to SM for execution

ex: of H100
there are 144 SM's

each SM have 128 CUDA core, 64 FP64 cores, 4 Int32 cores, 4tensor core

A single SM of H100 can have 2048 core max

if block of 256 is used then 8 block can work on a single SM.

warp has max 32 thread, combine to make cluster of CTA's(blocks) that has max 1024 thread combine to make Grid
```

![hardware model](./assets/hardware_model.png)

A SM consists of multiple Scalar Processor (SP) cores, a multithreaded instruction unit, and on-chip shared memory. 

To manage hundreds of threads running several different programs, the multiprocessor employs an architecture we call SIMT (single-instruction, multiple-thread). 

SIMT select a warp that is ready to execute, the warp execute one common execution at a time so full efficiency is realized when all thread of a warp agree on there execution path, otherwise if they diverge with a data driven condition branch then warp serially execute all and disabling thread that are not on the path then after execution is done then they converge back to the same execution path.

![SIMT vs SIMD](./assets/simt_vs_simd.png)

## Independent Thread Scheduling
Starting from the Volta architechture threads now contain independent program counter and call stack. and a schedule optimizer determine how to group active threads.

## On-chip Shared Memory(as in above image)
- One set of local 32-bit registers per processor.
- A parallel data cache or shared memory that is shared by all scalar processor cores and is where the shared memory space resides.(actual shared memory)
- A read-only constant cache that is shared by all scalar processor cores and speeds up reads from the constant memory space, which is a read-only region of device memory.
- A read-only texture cache that is shared by all scalar processor cores and speeds up reads from the texture memory space, which is a read-only region of device memory; each multiprocessor accesses the texture cache via a texture unit that implements the various addressing modes and data filtering.

# Syntax
