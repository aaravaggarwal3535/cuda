# Indexing
```
we have a specific kernal that do the same task but on diffrent threads depending on how many threads we define
```
## __global__
```
it is a function that can be called by both gpu and cpu
```

## __device__
```
it is a function that can be called only be the gpu
```

## __host__
```
it is a function that can only be called by a cpu
```

## dim3
```
it is a cuda specific datatype used to define the dimensions of block or threads
```

## cuda anology
```
gpu consist of (grids)
each grid have (blocks)
each block have (threads)
```

## device syncronization
```
it will majke all the threads to reach this point and then allow the cpu to proceed with the further code
```