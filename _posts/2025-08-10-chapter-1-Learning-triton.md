---
layout: post
title: "What is Triton and Getting Started - Day 1"
date: 2025-08-10
description: "What is Triton, How to get started and Why I started?"
tags: ["llm Inference","Triton" , "Inference optimization"]
author: Sahibpreet Singh
---
## Learning Triton from Scratch: Chapter 1 - Diving into the World of Parallel Computing

*Welcome to my journey into GPU kernel programming. This isn't just another tutorial—it's my real-time exploration of how parallel computing works, driven by curiosity and the dream of creating blazing-fast ML kernels.*

## Why I Started This Journey

I wasn't trying to solve a specific performance problem. I was driven by pure curiosity: **How does the world of parallel computing actually work?**

Every time I see libraries like Unsloth achieving 2x-5x speedups on language models with their "custom Triton kernels," I wonder: *What magic are they doing under the hood?* How do they make GPUs sing in ways that standard PyTorch can't?

I wanted to peek behind the curtain of modern AI acceleration. I wanted to understand how thousands of GPU cores work together to crunch numbers at incredible speeds. And maybe—just maybe—I could learn to create those crown jewel optimizations myself.

That curiosity led me to Triton, OpenAI's Python-like language for GPU programming.

## What Exactly Is Triton?

Triton sits in the sweet spot between "easy but limited" and "powerful but complex":

```
Easy & Limited  ←→  Powerful & Complex
   PyTorch     ←  Triton  →    CUDA
```

**Triton is a domain-specific language** that lets you write GPU kernels using Python-like syntax, but with explicit control over parallelism and memory access. It compiles to highly optimized GPU code while hiding the most painful low-level details.

Think of it as "NumPy for GPUs" with superpowers.

## My First Encounter with Parallel Thinking

Let me show you the exact moment everything clicked for me. Here's the simplest possible Triton kernel—vector addition:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Step 1: Figure out which chunk of work I'm responsible for
    pid = tl.program_id(0)
    
    # Step 2: Calculate which elements I need to process
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Step 3: Create a safety mask to avoid reading garbage
    mask = offsets < N
    
    # Step 4: Load my data, compute, and store the result
    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)
    c = a + b
    tl.store(C_ptr + offsets, c, mask=mask)

# Usage
N = 1024
BLOCK_SIZE = 256

A = torch.randn(N, device='cuda')
B = torch.randn(N, device='cuda')
C = torch.empty_like(A)

# Launch 4 programs in parallel
vector_add_kernel[(N // BLOCK_SIZE,)](A, B, C, N, BLOCK_SIZE=BLOCK_SIZE)

print(torch.allclose(C, A + B))  # True!
```

Simple, right? But there's a **profound shift in thinking** happening here. Let me break down every concept that confused me at first.

## The Mental Model: Programs, Not Threads

### What's a Program vs. a Thread?

This is the first thing that trips people up. In traditional CPU programming, you think about **threads**—independent execution contexts that run your code.

In Triton, you think about **programs**—independent workers that each process a chunk of your data.

```python
# When you launch this:
vector_add_kernel[(4,)](A, B, C, N, BLOCK_SIZE=256)

# You're creating 4 parallel programs:
Program 0: processes elements 0-255
Program 1: processes elements 256-511
Program 2: processes elements 512-767  
Program 3: processes elements 768-1023
```

Each program runs the exact same code, but on different data. This is called **SPMD** (Single Program, Multiple Data).

### What Does `tl.program_id(0)` Actually Do?

```python
pid = tl.program_id(0)
```

This is how each program figures out **"Which worker am I?"**

- Program 0 gets `pid = 0`
- Program 1 gets `pid = 1`  
- Program 2 gets `pid = 2`
- And so on...

The `0` in `program_id(0)` refers to the first dimension of your launch grid. We'll see multi-dimensional grids later.

### The Curious Case of Offsets

```python
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)
```

**Offsets are the bridge between program ID and actual array indices.**

Let's trace through this:
- Program 0: `block_start = 0 * 256 = 0`, `offsets = [0, 1, 2, ..., 255]`
- Program 1: `block_start = 1 * 256 = 256`, `offsets = [256, 257, 258, ..., 511]`

The `tl.arange(0, BLOCK_SIZE)` creates a vector `[0, 1, 2, ..., BLOCK_SIZE-1]`. When you add it to `block_start`, each program gets its own unique set of array indices.

## Tiles: The Secret to GPU Efficiency

Here's where it gets interesting. **A tile is a rectangular chunk of data that a program processes together.**

In our vector example, each program processes a 1D tile of 256 elements. But tiles really shine in 2D operations like matrix multiplication:

```
Matrix A (1024x1024) broken into 64x64 tiles:

[Tile 0,0] [Tile 0,1] [Tile 0,2] ...
[Tile 1,0] [Tile 1,1] [Tile 1,2] ...
[Tile 2,0] [Tile 2,1] [Tile 2,2] ...
...
```

**Why tiles?** GPUs are optimized for processing chunks of data together. A tile is the right size to:
1. Fit in fast GPU memory (shared memory/cache)
2. Keep all GPU cores busy
3. Minimize memory bandwidth bottlenecks

## The Mystery of the Mask

```python
mask = offsets < N
a = tl.load(A_ptr + offsets, mask=mask)
```

**Why do we need masks?** Here's the problem:

Imagine you have an array of 1000 elements, but your BLOCK_SIZE is 256. You'll launch 4 programs:
- Programs 0, 1, 2: each process exactly 256 elements ✅
- Program 3: should process 1000 - 768 = 232 elements ⚠️

But Program 3's offsets are `[768, 769, ..., 1023]`—it's trying to read 256 elements when only 232 exist!

The mask says: **"Only read/write where offsets < N"**. For out-of-bounds indices, Triton loads zeros and ignores stores.

```python
# Without mask: might read garbage memory or crash
# With mask: safe, predictable behavior
mask = offsets < N  # [True, True, ..., True, False, False, ...]
```

## Load and Store: The Memory Dance

### What's Really Happening in `tl.load()`?

```python
a = tl.load(A_ptr + offsets, mask=mask)
```

This single line is doing something remarkable:
1. **Takes a base pointer** (`A_ptr`) to your array in GPU memory
2. **Adds offsets** to get the exact memory addresses you want
3. **Reads multiple values at once** (vectorized memory access)
4. **Applies the mask** to handle boundary conditions safely

In traditional programming, you'd write a loop:
```python
# Sequential version
for i in offsets:
    if i < N:
        a[i] = A[i]  # One memory access at a time
```

Triton does this **all at once, in parallel**, across multiple memory controllers.

### The Power of `tl.store()`

```python
tl.store(C_ptr + offsets, c, mask=mask)
```

Similarly, store writes multiple values simultaneously. The GPU's memory system is designed for this—it's much more efficient to write 256 values together than 256 individual writes.

## Questions Every Curious Reader Asks

### "Why not just use a for loop?"

You *could* write:
```python
for i in range(N):
    C[i] = A[i] + B[i]
```

But this processes elements **one at a time, sequentially**. Modern GPUs have thousands of cores—why use just one?

Triton lets you use **all available cores simultaneously**. That's the difference between driving alone on a highway vs. having a 1000-car convoy all moving in parallel.

### "What if my array size doesn't divide evenly by BLOCK_SIZE?"

That's exactly why we need masks! The last program might get fewer elements than BLOCK_SIZE, and the mask handles this gracefully.

### "How does Triton know where my data is?"

When you pass PyTorch tensors to the kernel, Triton gets:
- **Pointer**: Memory address where your data starts
- **Device**: Which GPU the data lives on  
- **Strides**: How elements are laid out in memory

The `A_ptr + offsets` arithmetic computes the exact memory addresses to read from.

### "Could this really be faster than PyTorch?"

For simple operations like vector addition, probably not—PyTorch is already highly optimized. But when you start **fusing operations** (like `(A + B) * C + D` in a single kernel), or implementing **custom algorithms** that don't exist in standard libraries, that's where Triton shines.

## The "Aha!" Moment

Here's what clicked for me: **Parallel programming isn't about making one thing go faster—it's about doing many things simultaneously.**

Instead of:
```
Thread 1: Process all 1024 elements sequentially (slow)
```

You get:
```
Program 0: Process elements 0-255    }
Program 1: Process elements 256-511  } All happening
Program 2: Process elements 512-767  } at the same
Program 3: Process elements 768-1023 } time!
```

This is the foundation of modern GPU computing, and why libraries like Unsloth can achieve such dramatic speedups.

## What's Next

Now that we understand the fundamentals—programs, tiles, offsets, masks, and memory operations—we can start building more complex kernels.


The journey into parallel computing has just begun. Every time I write a Triton kernel, I'm getting closer to understanding how the masters of optimization think.

---
