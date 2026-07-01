# Torch

we will be writing a kernel in cuda then using it in torch 

## scaler_t
it is used as it handles all the diffrent type that exist in the torch to be easily processed in the cuda

## __restrict__
it is used so that we dont overlap memory access

## pybind
This section uses pybind11 to create a Python module for the CUDA extension:
- PYBIND11_MODULE is a macro that defines the entry point for the Python module.
- TORCH_EXTENSION_NAME is a macro defined by PyTorch that expands to the name of the extension (usually derived from the setup.py file).
- m is the module object being created.
- m.def() adds a new function to the module:
  - The first argument "polynomial_activation" is the name of the function in Python.
  - &polynomial_activation_cuda is a pointer to the C++ function to be called.
  - The last argument is a docstring for the function.

> we essentially tell the compiler that the arrays are not overlapping
> this way the compiler can make assumptions about the memory layout and 
> aggressively optimize
- notice in the top line how this is saved to `/home/elliot/.cache/torch_extensions/py311_cu121` (you can remove stuff in the .cache directory if it gets flooded with binaries)

## workflow
```bash
$ python ./setup.py install
```
```bash
$ python polynomial_activation.py
```