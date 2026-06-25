# Profiling
## nvtxRangePush()
```
this is used to gie names on the timeline
```

## nvtxRangePop()
```
this is used to stop the most rescent nvtxRangePush() command
```

## to compile
```
nvcc -o 00 naive_matmul.cu -lnvToolsExt
nsys profile --stats=true ./00

this will provide you with a report which you can open using the 
ncu report1.sdys-rep
```