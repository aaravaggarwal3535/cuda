# Unrolling

a normal loop in called rolled loop

in unrolling loop compiler transform :
```bash
for(int i=0;i<4;i++)
{
    sum+=A[i];
}
```
into
```bash
sum += A[0];
sum += A[1];
sum += A[2];
sum += A[3];
```
there is no loop, no increment no compare no branch
here the additional cost of maintaining i, checking each time etc is nullified

there is also GPU level parallelism where all the load can happen simultaneously and then all the multiple can happen simultaneously thus hiding the latency.

small loops are automatically unrolled by the compiler

but we can set manual instruction for compiler to unroll a loop by 

```bash
#pragma unroll
or
#pragma unroll 8
or disable it by 
#pragma unroll 1
```
Partial unroll useful for larger loops, like 16 iteration become 4 iteration each perform 4 operation
```bash
#pragma unroll 4
```
we should do this for :
smaller loops
inner loops