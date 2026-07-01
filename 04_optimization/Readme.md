# Optimizing kernels:

## naive kernel
```
the most basic kernel that we write in the basic kernel section it is the slowest but we will try to optimize it.
such that we can reach the performance of cuBLAS.

in the naive kernel we fix a column of matrix B and then traverse through all the rows of the matrix A then we move to next column in matrix B and then traverse through all the rows again in matrix A
```

## Global memory coalesce
```
in this the orow remain the same and we instead traverse through the column bu this make it a bit faster as columns are laid out adjacent in memory and we dont need to skip through certain elements until we reach the next row thus making the operation s a bit faster.

in naive kernel:
the wrap with 32 thread with each thread finding one element of c
Thread0   row0 col0
Thread1   row0 col1
...
Thread15  row0 col15

Thread16  row1 col0
Thread17  row1 col1
...
Thread31  row1 col15

so in this 16 thread read the same adress and 16 thread read another adress

in this version we dont use threadId.y this means we dont divide wrap to calculate 2 rows instead we make a single wrap to calculate single row

GPU combines into one memory transaction this is called Memory Coalescing.

```
<b>Arrange threads so that every warp accesses contiguous global memory (coalesced memory access).</b>

## Shared memory chached blocking

```
register(8TB/s)  --> shared(1.5TB/s) --> global/local(200GB/s) --> host(5GB/s)

__shared__ : this is how we can use shared memory(program wise) and in hardware(L1 cache)

so will be using the method of tiling and as the tile blocks are short so we can easily put them on shared memory which is small but a lot fast thus making the process very fast

tiling : 

A = 1, 2, 3, 4                           B = 1, 2, 3, 4
    5, 6, 7, 8                               5, 6, 7, 8
    9, 10, 11, 12                            9, 10, 11, 12
    13, 14, 15, 16                           13, 14, 15, 16
                    
                       tile size = 2x2
tile 1        from A                              from B
              1, 2           \/                    1, 2    =   11, 14  
              5, 6           /\                    5, 6        35, 46

tile 2        from A                              from B
              3, 4           \/                   9, 10    =   79, 86
              7, 8           /\                   13, 14       167, 182

now the tile for C is:
addition of tile 1 and tile 2 output and more if they exist:
tile 1  for C is :  90, 100
                    202, 228

by this current calculation C matrix will be :  90 , 100 , _ , _
                                                202 , 228 , _ , _
                                                _ , _ , _ , _
                                                _ , _ , _ , _

lets verify this by base method 
for C00 = (1 * 1) + (2 * 5) + (3 * 9) + (4 * 13) = 90 # which is correct

now in this we see that we can find the matmul of two bigger matrices by doing matmul of the small matrices that are part of those bigger matrices by this we can do it faster as the whole bigger matrix cant be loaded on the L1 shared memory but these small matrices can be and are processed faster thus making the matmul faster.

and the first thread load it from global memory then store it in the shared memory other threads use it thus making the matmul faster also

in this a single thread computes one element of C and the block compute a complete tile of the C
```

## 1D Blocktiling for calculating Multiple Results per Thread
```
one block computes
BM x BN = 64*64 = 4096 outputs
but this dont mean it has 4096 threads
because TM = 8
this mean one thread compute 1 element 
this means instead of a single thread computing C10,5
it computes:
C10,5

C11,5

C12,5

C13,5

C14,5

C15,5

C16,5

C17,5

in this the row change but the column remain the same, TM(Thread Multiply) = 8 output/thread
so this means for number of thread per block = BM x BN / TM

now in this example we can see the the column dont change for a thread but the row change, in previous kernel we have 8 thread to calculate this and all of them readthe value which take more time now here we make it easy for it as one thread reads the value once and process 8 output thus reducing the read and write from the memory each time 
and this memory is also stored in register which is the fastest memory among the all whis makes its processing even faster

thread 0 --> C00                 thread 1 --> C01
             C10                              C11
             C20                              C21
             C30                              C31
             C40                              C41
             C50                              C51
             C60                              C61
             C70                              C71
```c
nvcc -ptx matmul.cu -o kernel.ptx
```
```
this command is used to see the ptx code that will be genrated by out script 
just view it using
```
```c
nvim kernel.ptx
```
```
we can also see the shader assembly file which actually run on gpu as ptx get compiled down to assembly code first by the command
```
```c
nvcc -cubin matmul.cu -o kernel.cubin
```
```
view it again using:
cuobjdump --dump-sass kernel.cubin
```
```
there we can see that there are a few LDS 32 which mean these ones are not coleased because for 4 column LDS 128 should be there 
this is a problem by 2d Block tiling
```

## Increasing Arithmetic Intensity via 2D Blocktiling
```
in the last implementation the problem was that in a thread we load one column of the B once but still the A is loaded multiple time like
A0 * B0
A1 * B0
A2 * B0
A3 * B0
A4 * B0
A5 * B0
A6 * B0
A7 * B0

this still calls the shread memory multiple time by this causing a slowness in the running of the kernel

now in the 2d tiling one thread calculate 8x8(TM x TN) results.

Lets visualize : 

so if the tiles are like

tile A : 1 2                                tile B : 5 6 
         3 4                                         7 8

then in 1D tiling the operation thread 0 will be            |           but in 2d tiling the operations will be 
                                                            |
        1×5 + 2×7                                           |                   1×5 + 2×7
        3×5 + 4×7                                           |                   1×6 + 2×8
                                                            |
thread 1 will do                                            |                   3×5 + 4×7
                                                            |                   3×6 + 4×8
        1×6 + 2×8                                           |                   
        3×6 + 4×8                                           |                   

so the benefit is that in 1d two thread loads the same data [[1, 2],[3, 4]] in both the thread this take extra time as the shared memory is slow

but in 2d tiling both of them loaded and a single thread perform everthing thus saving time as read and write operations are only done once.
```
```
outermost loop loops the tiles of the blocks and then .
second loops are used to matrix multiple of these tiles.
```

## Vectorize SGEMM and GEMM access
```
now the next bottle neck that we have is the movment of data from global memory to shred memory

suppose we have 
1 2 3 4

without vectorization every thread loads
Thread 0 -> 1

Thread 1 -> 2

Thread 2 -> 3

Thread 3 -> 4

GPU executes 
load float
load float
load float
load float

4 times same operation

instead GPU support float4 which is litrally where 1, 2, 3, 4 are packed together 
so now

Thread 0

↓

load float4

1 isntruction instead of 4

so this reduces the instructions by 75% which provides the speed increase

so now everywhere we just load it such that we have 4 item combined together 

as if a = float4 then it will have
a.x, a.y, a.z, a.w --> consist the memory adress of the 4 memory that are packed together.
```

## 