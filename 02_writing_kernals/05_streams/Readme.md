# streams
```
it is used to make the code more asynchronus assume it like we are calculating one batch on gpu and simultaneously loading another batch on the gpu thus saving time and making it asynchronus
```

## defining a stream
```c
cudaStream_t stream1
```

## getting priority for the stream
```c
cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority)
```

## initializing a stream
### without priority
```c
cudaStreamCreate(&stream3)
```
### with priority
```c
cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, greatestPriority)
```

## cudaMemcpyAsync()
```c
cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream1);
// used to copy data in a stream a synchronously
```

## callback
```c
cudaStreamAddCallback(stream2, myStreamCallback, NULL, 0)
// used to call a fuction ones the stream ends mostly used to do logging
```

## cudaStreamSynchronize()
```c
cudaStreamSynchronize(stream1)
// make the compile to first end the work on the specified stream end and then proceed further
```

## cudaStreamDestroy()
```
used to free the memory of the stream
```

## cudaMallocHost()
```
this is used to fix a memory on the os that os cant change this memory is readed faster by the device.
```

# Events

## defining event
```c
cudaEvent_t start, stop;
```

## initializing the event
```c
cudaEventCreate(&start)
```

## putting event on a stream
```c
cudaEventRecord(start, stream1)
```

## cudaStreamWaitEvent(stream2, event, 0)
```
make the stream 2 continue when the event 2 has occured
```

## cudaEventSynchronize(stop)
```
make everthing else dont go further ion processing until the stop event is executed
```

## time between 2 interval
```c
cudaEventElapsedTime(&elapsedTime, event1, event2)
```