# atomic threads
```
this means the operation performed by the memory should be atomic are there should not be any dirty read or write and every thread must maintain the memory intact for any other thread to use it
```

## predefine atomic operations
```
atomicAdd(int* address, int val): Atomically adds val to the value at address and returns the old value.
atomicSub(int* address, int val): Atomically subtracts val from the value at address and returns the old value.
atomicExch(int* address, int val): Atomically exchanges the value at address with val and returns the old value.
atomicMax(int* address, int val): Atomically sets the value at address to the maximum of the current value and val.
atomicMin(int* address, int val): Atomically sets the value at address to the minimum of the current value and val.
atomicAnd(int* address, int val): Atomically performs a bitwise AND of the value at address and val.
atomicOr(int* address, int val): Atomically performs a bitwise OR of the value at address and val.
atomicXor(int* address, int val): Atomically performs a bitwise XOR of the value at address and val.
atomicCAS(int* address, int compare, int val): Atomically compares the value at address with compare, and if they are equal, replaces it with val. The original value is returned
```

## atomic operation from scratch
```
You can think of atomics as a very fast, hardware-level mutex operation(operations that apply mutual exclusion). It's as if each atomic operation does this: 
lock(memory_location)
old_value = *memory_location
*memory_location = old_value + increment
unlock(memory_location)
return old_value
```

Code:
```c
__device__ int softwareAtomicAdd(int* address, int increment) {
    __shared__ int lock;
    int old;
    
    if (threadIdx.x == 0) lock = 0;
    __syncthreads();
    
    while (atomicCAS(&lock, 0, 1) != 0);  // Acquire lock
    
    old = *address;
    *address = old + increment;
    
    __threadfence();  // Ensure the write is visible to other threads
    
    atomicExch(&lock, 0);  // Release lock
    
    return old;
}
```
```
atomicCAS() → it is used to make the principle of the lock if atomicCAS(lock,0,1); execute then the lock is acquired if not this means someone else has locked it before hand, 0 is unlocked and 1 is locked
```

```
But we will still prefer atomicAdd() instead of our own custom mutex as the atomic add is actually considered to be one operation and mutex is considered to be multiple operations
```

```
__shared__ it create a variable that can be accessed by all the thread inside a block having a block level scope 
Int old stores the previous value
```

```c
if (threadIdx.x == 0) lock = 0; //only thread 0 initializes the lock
```

```c
__syncthreads(); // every thread much reach here first before they can continue
```

```c
atomicCAS(address, compare, value) means:
if (*address == compare)
{
    *address = value;
}
```

```
Suppose thread A lock then thread B comes it will not be able to lock as it is already locked and need to wait and we use while loop as we want all the thread to try unlit they actually acquire a lock
```

```c
__threadfence(); // this ensure that all the read and write operation of the current thread gets finished before the unlock and the data is available for the rest this is essential as sometime GPU can do the unlocking first and then the read or write operation for optimization purposes but this ensure that this dont happen
```

```c
atomicExch(&lock,0); : release the lock looks like:
    *lock = 0;
```