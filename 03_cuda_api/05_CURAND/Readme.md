# CURAND

it is in nvidia api that is used to genrate random numbers

To run:
```bash
nvcc curand.cu -lcurand -o curand_example
./curand_example
```

Diffrent types:
- curandGenerateUniform(...)
- curandGenerateUniform(...)
- curandGenerateLogNormal(...)
- curandGenerate(...)

create genrator : curandCreateGenerator(...)
seed : curandSetPseudoRandomGeneratorSeed(...)
uniform : curandGenerateUniform(...)
Destroy : curandDestroyGenerator(...)