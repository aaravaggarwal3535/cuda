# best way to learn ptx will be done shortly
```
Phase 1 — PTX Fundamentals (Become comfortable reading PTX)

This phase is about understanding how CUDA C++ becomes PTX.

Chapter 1: What PTX really is
CUDA Compilation Pipeline
CUDA C++
NVVM IR
PTX
SASS
Driver JIT
Why PTX exists
PTX vs SASS
Virtual ISA vs Machine ISA
Chapter 2: PTX Syntax

Everything about

.reg
.pred
.global
.shared
.local
.const
.param

ld
st
mov
add
mul
mad

You'll learn to read PTX like C.

Chapter 3: Registers

Everything about

.reg .f32
.reg .b32
.reg .u64
.reg .pred

How registers are allocated.

Register lifetime.

Register pressure.

Why occupancy depends on registers.

Chapter 4: Predicates

Instead of

if(x)

PTX uses

setp

@p

You'll learn

setp.lt

setp.eq

setp.ne

setp.gt
Chapter 5: Memory Instructions

Everything about

ld.global

ld.shared

ld.const

ld.local

st.global

st.shared
Chapter 6: Address Arithmetic

How

A[row*K+col]

becomes

mul

add

cvta

ld.global
Phase 2 — Writing Inline PTX

This is where we begin replacing CUDA code.

Chapter 7

Inline Assembly

asm volatile(
"..."
);

Everything

constraints

outputs

inputs

clobbers

volatile

register classes

Chapter 8

Writing arithmetic

Instead of

a+b

we'll write

add.f32

Instead of

a*b+c
fma.rn.f32
Chapter 9

Reading thread IDs

Instead of

threadIdx.x

we'll write

mov.u32

%tid.x
Chapter 10

Loading memory manually

Instead of

A[idx]

we'll write

cvta

add

ld.global
Phase 3 — Tensor Core PTX

Now we begin Tensor Core programming.

Chapter 11

Tensor Core Hardware

What actually happens inside

mma.sync
Chapter 12

ldmatrix

Probably the most confusing instruction.

You'll learn

why it exists

how it distributes data

how every lane gets registers

lane mapping

bank conflicts

Chapter 13

mma.sync

Every single operand explained.

Example

mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32

We'll decode every word.

Chapter 14

Tensor Core Register Layout

Exactly

which lane owns

which registers

why

how

Chapter 15

Building Tensor Core GEMM manually

No WMMA.

Pure PTX.

Phase 4 — Ampere Pipeline
Chapter 16

cp.async

Everything

pipeline

stages

wait_group

commit_group

barriers

Chapter 17

Software Pipeline

Hide latency

Overlap

Load

Compute

Store

Chapter 18

Building CUTLASS Pipeline

Exactly how NVIDIA does it.

Phase 5 — Modern Tensor Core Kernels
Chapter 19

Warp Specialization

Producer warps

Consumer warps

Chapter 20

Persistent GEMM

Chapter 21

Hopper WGMMA

Chapter 22

Tensor Memory Accelerator (TMA)

Chapter 23

CUTLASS Internals

Chapter 24

FlashAttention
```