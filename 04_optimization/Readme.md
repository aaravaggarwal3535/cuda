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

## 