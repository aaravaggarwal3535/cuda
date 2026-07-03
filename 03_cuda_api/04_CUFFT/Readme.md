# CUFFT

it is api by nvidia for calculating FAST FOURIER TRANSFORM

it is used to transfer signal from time domain to signel domain

To execute:
```bash
nvcc cufft.cu -lcufft -o fft
./fft
```

cufftPlan1d(...) : Creates a 1D FFT.
cufftExecC2C(...) : Complex number → Complex FFT.
cufftDestroy(plan); : release storage

