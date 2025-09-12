---
layout: post
title: "I'm Teaching Myself Triton - Here's What's Actually Happening"
date: 2025-09-12
description: "Part-1 Describes how I got started with triton, why you should and How to get started"
tags: [Nvidia, Triton, GPU Programming]
author: Sahibpreet Singh
pinned: true
---
Starting from start, Month of July-2025 super scared and tensed. I was doing everything left right and centre to secure a job and in the middle of all this felt super helpless and during that time only got the hit to make myself learn something which I always feared GPU programming. Yes actually how GPU's can be made to work and understand the depth of how things work and not just **.cuda()** and work done.

Truly saying, Since I was new made a choice to teach myself CUDA but after couple of weeks motivation died a slow death just because it was too time consuming. Then came to my mind the name of **Unsloth** and how they made use of **Triton** for making our beloved LLM's faster and smaller by programming things in **Triton** and this is exactly was my first motivation to learn Triton.

### The Hard Part
The hard part of learning Triton was not the language but the ceremony around it and lack of structured resources around it. ChatGPT and Claude were my two partners.

Enough of back story and I feel I have explained how i got started. Let's see why it matters for you(The Honest Answer) :-

Look, I won't sell you dreams. Triton isn't going to magically 10x your career overnight. But here's what it actually did for me: it removed the fear. That intimidating gap between "I use PyTorch" and "I understand what my GPU is doing"? Triton atleast helps to bridge that.

1. It allows you to write custom fucntions a.k.a `Kernels` and you can check and control certain aspects of your GPU and prevent it from sitting IDLE for most of the time.
2. The Code Is Actually Readable
Here's the shocking part - Triton kernels look like NumPy code. Not 200 lines of CUDA with thread management and synchronization barriers. We're talking 20-30 lines that you can actually understand, modify, and debug. When something goes wrong, you can actually figure out why.
3. Big Question? "Will you always beat Pytorch in performance"- Answer is *Yes* and *No*. If we are talking of standard operations beating Pytorch is little sloppy because you are fighting with `cuBLAS` and `cuDNN` which are written by NVIDIA wizards and will destroy your amateur matrix multiplication every time but if you come to a territory where hardware is something you have assembled and you are trying to exceute a custom version of Attention for faster inference `Triton holds your hand`.

4. More Importantly - Since while learning, you will try and fight with Pytorch to see how far you came you will get in depths of all the **Standard Operations** so your understanding of concepts gets another soothing touch.

Covered Why You should learn. Now  ***Terminology Alert*** :-

When I started, and saw people kept throwing around words like "blocks," "tiles," and "warps" like I was supposed to just know. Here's what they actually mean, explained like you're five (because that's what I needed):

To Easy visualise - think of Big Warehouse :-

1. Threads - The Individual Workers
A thread is the smallest unit of execution on a GPU. Think of it as one worker following the same instruction manual as everyone else, but working on their own piece of data. 
Your GPU runs thousands of threads in parallel - they all execute the same code simultaneously, just on different numbers.

Simple example: If you're adding two arrays of 10,000 elements, you might have 10,000 threads where thread #0 adds element[0], thread #1 adds element[1], and so on. All doing the same operation (addition), just on different bits of data.

2. Warps - The Synchronized Squad 

Note - In Next Blogs, We will dive deeper in **Warps** when we will Understand `Coalesced vs Uncoalesced Memory Access`

Here's where it gets interesting. Threads don't work alone - they move in groups of 32 called "warps." Think of it like a rowing team - all 32 threads in a warp execute the same instruction at the same time. They're perfectly synchronized. If one thread in a warp takes a different code path (like in an if-statement), the others wait. This is why branching in GPU code can kill performance.

3. Blocks - The Team with Shared Memory

A block is a group of threads (multiple warps) that can talk to each other through shared memory. Imagine a team of workers who share a whiteboard. In Triton, when you write `tl.program_id(0)`, you're asking "which team (block) am I?" Typically 256-1024 threads per block.

The crucial part: threads in the same block can share data quickly. Threads in different blocks? They can't talk directly - they'd have to write notes in global memory (beleive me that's very slow).

Bigger Perspective - Whenever we try and write code in Triton our goal is to write such way that shared memory is used the most.

4. Tiles - The Chunk of Work (Why Triton is "Tile Programming")
This is where Triton gets its identity. A **Tile** is just a chunk of your data that one block processes. Instead of thinking "thread 457 processes element 457," you think "block 3 processes this 32×32 tile of the matrix."

Sample Code
```python
BLOCK_SIZE = 1024
pid = tl.program_id(0)
tile_start = pid * BLOCK_SIZE
tile_elements = tl.arange(0, BLOCK_SIZE)  # This is your tile!
```
The magic: Triton automatically maps your tile operations to threads. You say "process this tile," Triton figures out which threads do what. You think in chunks, not individual threads.

CUDA - This is where allows you to manage individual workers in that warehouse but on otherside
Triton - Team A do work - A and Team B - Work B.

5. Grid - All The Blocks
The grid is just all your blocks together. If you have 1 million elements and each block handles 1024 elements, your grid has ~1000 blocks. Simple as that.

Perfect- Enough of theory. Let's write code and then break it :-

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,  # pointer to first input array
    y_ptr,  # pointer to second input array  
    output_ptr,  # pointer to output array
    n_elements,  # total number of elements
    BLOCK_SIZE: tl.constexpr  # this is a compile-time constant
):
    # Which block am I?
    pid = tl.program_id(axis=0)
    
    # What's my starting position?
    block_start = pid * BLOCK_SIZE
    
    # What elements do I process? (my tile)
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Don't read past the end of the array
    mask = offsets < n_elements
    
    # Load my tile of data
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Do the actual work
    output = x + y
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)
```