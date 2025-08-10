---
layout: post
title: "What is Triton and Getting Started - Day 1"
date: 2025-08-10
description: "What is Triton, How to get started and Why I started?"
tags: ["llm Inference","Triton" , "Inference optimization"]
author: Sahibpreet Singh
---
## Learning Triton from Scratch: Chapter 1 - Diving into the World of Parallel Computing
Welcome to my journey into GPU kernel programming. This isn't just another tutorial—it's my real-time exploration of how parallel computing works, driven by curiosity and the dream of creating blazing-fast ML kernels.

Why I Started This Journey
I wasn't trying to solve a specific performance problem. I was driven by pure curiosity: How does the world of parallel computing actually work?
Every time I see libraries like Unsloth achieving 2x-5x speedups on language models with their "custom Triton kernels," I wonder: What magic are they doing under the hood? How do they make GPUs sing in ways that standard PyTorch can't?
I wanted to peek behind the curtain of modern AI acceleration. I wanted to understand how thousands of GPU cores work together to crunch numbers at incredible speeds. And maybe—just maybe—I could learn to create those crown jewel optimizations myself.
That curiosity led me to Triton, OpenAI's Python-like language for GPU programming.
What Exactly Is Triton?
Triton sits in the sweet spot between "easy but limited" and "powerful but complex":

Easy & Limited  ←→  Powerful & Complex
   PyTorch     ←  Triton  →    CUDA

   Triton is a domain-specific language that lets you write GPU kernels using Python-like syntax, but with explicit control over parallelism and memory access. It compiles to highly optimized GPU code while hiding the most painful low-level details.
Think of it as "NumPy for GPUs" with superpowers.
My First Encounter with Parallel Thinking
Let me show you the exact moment everything clicked for me. Here's the simplest possible Triton kernel—vector addition:


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