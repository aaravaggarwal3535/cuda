# CuTe

A matrix like 
A = [1 2 3]
    [4 5 6]

so it get stored in memory like : 
1 2 3 4 5 6
and we have a pointer to the first element of the array

but in CuTe we have 
Data + Layout = Tensor

so instead of a tensor we get 

Tensor A

## The three core object of CuTe are:
- Shape  -> What exists
- Layout -> Where it lives
- Tensor -> Actual data

### shape
auto shape = make_shape(4, 8); --> 4 row and 8 column 
there is no memory allocated yet and a tensor is not instead we just described the shape what it will be.

### layout
it is the cordinate map of the CuTe 
like it tells that is it row major or column major.

### tensor
it contains the pointer and layout

## why was cute created
every kernel can have diffrent shape and for them we need to write the kernel seprately which will be a headache s nvidia created cutlas so that this process can be centralized

```cpp
#include <cute/tensor.hpp>

using namespace cute;

int main() {

    auto shape = make_shape(4, 8);

    auto layout = make_layout(shape);

    print(shape);
    print(layout);

    return 0;
}
```

To run this:
```bash
nvcc -std=c++17 -I./cutlass/include/ -Xcompiler "/Zc:preprocessor" cute.cu -o cute
cute.exe
```

use microsoft x64 compiler for this

## Layout 
layout contain shape and stride 
let say:
shape --> (4,4)
stride --> (4,1)

we will often write it as --> (4,4):(4,1)

the is means the mat rix is like (4,4):
[1 2 3 4
 5 6 7 8
 9 1 2 3
 4 5 6 7]

and stride will look like (4,1):
which mean to move to next row traverse 4 element and to move to next column move 1 element.

Row major --> (N,1)
Column major --> (1,M)

CuTe syntax: 
```cpp
auto layout = make_layout(make_shape(4,8));
```

it is smart enough to make stride automatically but if you need to make it manually:
```cpp
auto layout = make_layout(
    make_shape(4,8),
    make_stride(8,1)
);
```

now what will below code 
```cpp
layout(2,5)
```
it will find = 2*8+5 = return 21st element 


## Hierarchical Layouts
let say we have a matrix of size 8x8 but gpu will store it in 4 tiles of 4x4

+---------+---------+
| Tile 0  | Tile 1  |
| 4 × 4   | 4 × 4   |
+---------+---------+
| Tile 2  | Tile 3  |
| 4 × 4   | 4 × 4   |
+---------+---------+

so now we have 2 cordinate system means we need a hierarcial system

```cpp
make_shape(

    make_shape(2,2),

    make_shape(4,4)

)
```

this will be 2x2 tiles of 4x4 element each

## Tensor
it is a combination of engien + layout
in this engien stores the actual pointer

to create a tensor:
```cpp
// allocate memory
float* A;
cudaMalloc(&A,1024*1024*sizeof(float));

// create layout
auto layout = cute::make_layout(
    cute::make_shape(1024,1024)
);

// create tensor
auto tensor = cute::make_tensor(A,layout);
```

```cpp
// accessing tensor
tensor(row,col)
```

an proper example:
```cpp
#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

// 1. Define a CUDA kernel to run on the GPU
__global__ void MyCuteKernel(float* d_data) {
    // Recreate the layout inside the kernel
    auto layout = make_layout(make_shape(4, 4));
    
    // Create the tensor mapping to the GPU memory pointer
    auto tensor = make_tensor(d_data, layout);

    // Coordinate assignments are safe to perform here on the device
    tensor(2, 3) = 5.0f;

    // Use printf inside the kernel (supported by CUDA)
    printf("From Device: tensor(2,3) = %f\n", tensor(2, 3));
}

int main() {
    float* d_A;
    // Allocate Device Memory
    cudaMalloc(&d_A, 4 * 4 * sizeof(float));

    // 2. Launch the kernel on the GPU
    MyCuteKernel<<<1, 1>>>(d_A);

    // 3. Synchronize the CPU with the GPU to wait for the printf to finish
    cudaDeviceSynchronize();

    // Clean up memory
    cudaFree(d_A);
    return 0;
}
```

this make it resuable just cahnge the layout and now the same kernenlworks for diffrent size

## Tensor slicing

```cpp
auto A = make_tensor(
    ptr,
    make_layout(make_shape(8,8))
);
```
this created a matrix A completely and load it all at once

### local tile : the most common slicing function

```cpp
auto tile =
    local_tile(
        A,
        make_shape(Int<4>{},Int<4>{}),
        make_coord(0,0)
    );
```

This means split tensor A into 4x4 tiles 

and tile(2,1) means row 2 and column 1 inside the tile

to change the tile view without changing the tensor pointer just change the position in make_coord

much cleaner version:
```cpp
auto blockTile =
local_tile(
A,
make_shape(Int<128>{},Int<128>{}),
make_coord(blockIdx.y,blockIdx.x)
);
```

### local partition 

this will instead of giving a tile to threadblock it give a tile to each thread

```cpp
auto threadTile =
local_partition(
blockTile,
layout,
threadIdx.x
);
```

## copy atom

it describe a particular copy operation

### basic operation

```cpp
cute::copy(src, dst)
```

it is used to copy data from global tensor to shared tensor

### basic example

```cpp
auto gA = make_tensor(
    global_ptr,
    global_layout
);

auto sA = make_tensor(
    shared_ptr,
    shared_layout
);

copy(gA, sA);
```

### proper example
```cpp
//creating global tensor
auto gA =
    make_tensor(
        gmem_ptr,
        make_layout(
            make_shape(1024,1024)
        )
    );
//creating tile
auto gA_tile =
    local_tile(
        gA,
        make_shape(
            Int<128>{},
            Int<128>{}
        ),
        make_coord(blockIdx.y, blockIdx.x)
    );
//creating shared memory tensor
auto sA =
    make_tensor(
        smem_ptr,
        make_layout(
            make_shape(Int<128>{},Int<128>{})
        )
    );
//copying
copy(gA_tile, sA);
```

## MMA/Tensor core operation

              GPU
               │
       ┌───────┴───────┐
       │               │
   Global Memory   Global Memory
       │               │
       │ copy          │ copy
       ▼               ▼
   Shared A         Shared B
       │               │
       │ copy          │ copy
       ▼               ▼
   Register A       Register B
       │               │
       └───────┬───────┘
               │
               ▼
          Tensor Core
               │
               ▼
          Accumulator

