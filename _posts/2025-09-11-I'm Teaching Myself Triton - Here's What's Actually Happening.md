---
layout: post
title: "I'm Teaching Myself Triton - Here's What's Actually Happening"
date: 2025-09-12
description: "Part-1 Describes how I got started with triton, why you should and How to get started"
tags: [Nvidia, Triton, GPU Programming]
author: Sahibpreet Singh
pinned: true
---
## My Journey: From Fear to Understanding

Starting from the beginning, July 2025 - super scared and tense. I was doing everything left, right, and center to secure a job. In the middle of all this, I felt super helpless, and during that time I got the idea to teach myself something I always feared: GPU programming. Yes, actually understanding how GPUs work at a deeper level, not just slapping `.cuda()` on everything and calling it done.

Honestly, since I was new, I made a choice to teach myself CUDA. But after a couple of weeks, motivation died a slow death because it was too time-consuming. Then I remembered **Unsloth** and how they used **Triton** to make our beloved LLMs faster and smaller by programming in **Triton** - and this was exactly my first motivation to learn Triton.

### The Hard Part
The hard part of learning Triton wasn't the language itself, but the ceremony around it and the lack of structured resources. ChatGPT and Claude were my two partners.

Enough backstory - I feel I've explained how I got started. Let's see why it matters for you.

## Why You Should Care (The Honest Answer)

Look, I won't sell you dreams. Triton isn't going to magically 10x your career overnight. But here's what it actually did for me: it removed the fear. That intimidating gap between "I use PyTorch" and "I understand what my GPU is doing"? Triton at least helps bridge that.

### 1. Write Custom GPU Kernels
You can write custom functions (kernels) and control certain aspects of your GPU, preventing it from sitting idle most of the time.

### 2. The Code Is Actually Readable
Here's the shocking part - Triton kernels look like NumPy code. Not 200 lines of CUDA with thread management and synchronization barriers. We're talking 20-30 lines that you can actually understand, modify, and debug. When something goes wrong, you can actually figure out why.

### 3. Performance Reality Check
**Big Question: "Will you always beat PyTorch in performance?"**

Answer: *Yes* and *No*.

- **Standard operations**: Beating PyTorch is tricky because you're fighting `cuBLAS` and `cuDNN`, written by NVIDIA wizards who will destroy your amateur matrix multiplication every time
- **Custom operations**: When you need custom hardware optimizations or specialized operations like custom attention for faster inference, Triton holds your hand

### 4. Deeper Understanding of Operations
Since you'll try to compete with PyTorch while learning, you'll dive deep into all the **standard operations**, giving your understanding of concepts another solid foundation.

## GPU Terminology Decoded (Warehouse Analogy)

***Terminology Alert***

When I started, I saw people throwing around words like "blocks," "tiles," and "warps" like I was supposed to just know them. Here's what they actually mean, explained like you're five (because that's what I needed):

To easily visualize - think of a big warehouse:

### 1. Threads - The Individual Workers
A thread is the smallest unit of execution on a GPU. Think of it as one worker following the same instruction manual as everyone else, but working on their own piece of data.

Your GPU runs thousands of threads in parallel - they all execute the same code simultaneously, just on different numbers.

**Simple example**: If you're adding two arrays of 10,000 elements, you might have 10,000 threads where thread #0 adds element[0], thread #1 adds element[1], and so on. All doing the same operation (addition), just on different bits of data.

### 2. Warps - The Synchronized Squad

**Note**: In future blogs, we'll dive deeper into **warps** when we understand `Coalesced vs Uncoalesced Memory Access`.

Here's where it gets interesting. Threads don't work alone - they move in groups of 32 called "warps." Think of it like a rowing team - all 32 threads in a warp execute the same instruction at the same time. They're perfectly synchronized. If one thread in a warp takes a different code path (like in an if-statement), the others wait. This is why branching in GPU code can kill performance.

### 3. Blocks - The Team with Shared Memory

A block is a group of threads (multiple warps) that can talk to each other through shared memory. Imagine a team of workers who share a whiteboard. In Triton, when you write `tl.program_id(0)`, you're asking "which team (block) am I?" Typically 256-1024 threads per block.

**The crucial part**: Threads in the same block can share data quickly. Threads in different blocks? They can't talk directly - they'd have to write notes in global memory (believe me, that's very slow).

**Bigger perspective**: Whenever we write code in Triton, our goal is to write it so that shared memory is used the most.

### 4. Tiles - The Chunk of Work (Why Triton is "Tile Programming")
This is where Triton gets its identity. A **tile** is just a chunk of your data that one block processes. Instead of thinking "thread 457 processes element 457," you think "block 3 processes this 32×32 tile of the matrix."

**Sample Code:**
```python
BLOCK_SIZE = 1024
pid = tl.program_id(0)
tile_start = pid * BLOCK_SIZE
tile_elements = tl.arange(0, BLOCK_SIZE)  # This is your tile!
```

**The magic**: Triton automatically maps your tile operations to threads. You say "process this tile," Triton figures out which threads do what. You think in chunks, not individual threads.

**The difference**:
- **CUDA**: Manage individual workers in the warehouse  
- **Triton**: "Team A, do work A" and "Team B, do work B"

### 5. Grid - All The Blocks
The grid is just all your blocks together. If you have 1 million elements and each block handles 1024 elements, your grid has ~1000 blocks. Simple as that.

## Your First Triton Kernel: Step by Step

Perfect - enough theory. Let's write code and then break it down:

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
Perfect! We'll now use the above code as our sample to understand Triton terminology.

### Breaking Down Each Part

#### 1. Program ID - Finding Your Team

```python
pid = tl.program_id(axis=0) # "What's my team number?"
```

In our warehouse analogy, say today we have to ship 1 million parcels and we have 1000 teams of workers (blocks). Each team needs to know which section of packages to handle:

- Team 0 (pid=0) goes to Section A (packages 0-999)
- Team 1 (pid=1) goes to Section B (packages 1000-1999)

**But what does `axis` mean here?**

Imagine the warehouse has multiple floors and aisles:
- `program_id(0)` = Which aisle is my team assigned to? (X-axis)
- `program_id(1)` = Which floor is my team on? (Y-axis)  
- `program_id(2)` = Which building section? (Z-axis)

For our simple addition, we just have one long aisle of packages, so we only need `program_id(0)` - just the aisle number.

**Note**: This is mainly an analogy to help. You can have more than 3 axes, but since most GPU architectures are designed with 3 dimensions in mind, most problems can be solved with 3 axes. There can be scenarios where you might use **batch_no**, so **axis=3** is possible.

![warehouse](/assets/blog-1-Triton/blog-1-team-diag-1.png)
#### 2. Block Start - Where Does My Team Start?

In our warehouse, each team gets an assignment that tells them **WHERE** to work. This assignment can have multiple parts depending on how complex the warehouse layout is.

**Simple Warehouse (1D) - One Long Aisle:**
- 1 million packages in one long row
- 1000 teams needed
- Each team gets a simple number: Team 0, Team 1, Team 2...
- `program_id(0)` returns this team number
- Example: Team 523 handles packages 523,000-523,999

We'll get into how Triton handles program_id with axis=1 and axis=2 in later programs.

So in our program: **`block_start = pid * BLOCK_SIZE`**

#### 3. Offsets - Getting Your Exact Package List
```python
offsets = block_start + tl.arange(0, BLOCK_SIZE)
```

Remember, each team has workers (i.e., threads).

`tl.arange(0, BLOCK_SIZE)` creates [0, 1, 2, ..., 1023] - like handing out clipboard numbers to each worker in the team.

For Team 2 (starting at 2048):
- Worker 0 gets package 2048 (2048 + 0)
- Worker 1 gets package 2049 (2048 + 1)  
- Worker 1023 gets package 3071 (2048 + 1023)

These `offsets` are the exact package numbers this team will handle.

#### 4. Masks - The Safety Checklist (Why Masks Save Your Kernel)
```python
mask = offsets < n_elements
```

**The Problem**: What if you have 5000 packages but Team 4 tries to process packages 4096-5119? Packages 5000-5119 don't exist!

![warehouse](/assets/blog-1-Triton/blog-1-warehouse-diag-2.png)

The mask is your safety checklist:
- Packages 4096-4999: ✓ (exist, process them)
- Packages 5000-5119: ✗ (don't exist, skip)

When I was learning, there were numerous times I tried to access positions that didn't follow the masking condition. These were my observations:
- Kernel crashes immediately and gives "Bad memory" error
- Reads garbage memory and gives wrong results (this happened mostly when practicing on [leetgpu](http://leetgpu.com))

#### 5. Load, Compute, Store - Actually Moving the Packages
```python
x = tl.load(x_ptr + offsets, mask=mask)
y = tl.load(y_ptr + offsets, mask=mask)
output = x + y
tl.store(output_ptr + offsets, output, mask=mask)
```

This is the actual work:
- **Load**: "Team, grab packages from shelves X and Y (but check your safety list first!)"
- **Compute**: "Add contents of matching packages together"  
- **Store**: "Put results on the output shelf (again, check safety list!)"

The mask in `tl.load` returns 0 for non-existent packages. The mask in `tl.store` prevents writing to non-existent locations.

**BOOM!** This is your first baby Triton kernel up and running.

## Testing Your Kernel

Combine the above code with the cell below and you can see the actual piece in action:

```python
def add(x: torch.Tensor, y: torch.Tensor):
    # Create output tensor
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda, "Tensors must be on GPU!"
    
    # Get total number of elements
    n_elements = output.numel()
    
    # Define grid dimensions (how many blocks/teams we need)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch the kernel!
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output

# Test time
x = torch.randn(1_000_000, device='cuda')
y = torch.randn(1_000_000, device='cuda')

triton_result = add(x, y)
pytorch_result = x + y

# Check if it's working
print(f"Results match: {torch.allclose(triton_result, pytorch_result)}")
print(f"Max difference: {(triton_result - pytorch_result).abs().max().item()}")
```

## Common Mistakes That Cost Me Time

### 1. BLOCK_SIZE Must Be Power of 2
Triton always runs with block sizes that are powers of 2 (256, 512, 1024, etc.).

### 2. Forgetting Masks
I can't stress enough how simple yet tricky it is to set masks correctly:
- **No mask** = kernel crashes or garbage results
- **Wrong mask** = silent corruption of results

### 3. Grid Calculation
`triton.cdiv` divides and rounds UP. This is critical for handling arrays not perfectly divisible by BLOCK_SIZE.

## Try This Yourself

- Write a kernel for `multiplication` and `division`
- Try changing the block size to 512
- Experiment with different array sizes to see masking in action
