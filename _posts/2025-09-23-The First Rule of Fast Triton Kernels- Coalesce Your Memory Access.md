---
layout: post
title: "The First Rule of Fast Triton Kernels: Coalesce Your Memory Access"
date: 2025-09-23
description: "Part-2 The First Rule of Fast Triton Kernels: Coalesce Your Memory Access"
tags: [Triton, Nvidia, GPU Programming]
author: Sahibpreet Singh
pinned: true
published: true
---

## Step 1 - How to make "Triton" faster using "Memory Coalescing"
Last time we built our triton kernel a simple Vector Addition. Now on normal operations we might not feel but the real use of Triton is to make 
"kernels" faster without getting into hassle of CUDA (which can be little overwhelming) to start with. This is where first concept comes in called as "Memory Coalescing" comes in.

## Deep Dive - What's the Problem?
**Pop Quiz** : What's the #1 performance killer in GPU programming? 
![gpu-quiz]({{ site.baseurl }}/assets/blog-2-memory-coalsecing/pop_quiz.png)

🙅‍♂️ Nope. It's something so mundane you probably never thought about it: *where your data lives in memory.* Move your data wrong, and your lightning-fast GPU becomes slower than your laptop's CPU. Move it right, and you unlock 10x-100x speedups. It is just like a **Ferrari** is sitting in a shopping cart.

But Why Does Memory Matter So Much?

Example -  **Your RTX 4090 can perform 83 trillion operations per second, but it can only fetch 1,008 GB of data per second**
Let me put that in perspective:
- **GPU Compute**: 83 TFLOPS (83 trillion FP32 operations/second)  
- **Memory Bandwidth**: 1,008 GB/s
- **The Math**: That's roughly a **82:1 mismatch!**

This means your $1,600 GPU spends **98% of its time waiting** for data to arrive from memory. 
So, Even if you upgrade to the flagship H100 ($30k), the ratio gets `*worse* - 590:1`! More horsepower, same traffic jam.

Simple Solution ( We will dive deeper ahead ) - Keep Things In order so that our beloved GPU does not spends lot of time picking things (Data Points).

Goal: By the end of this post, you will understand the first most important concept for writing high-performance GPU code.

But Before diving in further. Let's understand diff b/w :-

Concept Alert🚨  

### **Memory Bound Operations** vs **Compute Bound Operations**

We will use same Warehouse Analogy we used in previous blog.

- **Memory Bound Operation**: In our warehouse where a worker who spends most time walking between parcels rather than actually picking them up. The bottleneck is moving around, not the picking itself.

- **Compute Bound Operation**: In the same warehouse another worker at a crafting station who spends most time assembling complex items from materials already at hand. The bottleneck is the actual work, not fetching materials.

- **But What is GPU Reality**: Most operations (including our vector addition) are memory bound - we spend more time waiting for data than computing with it.

How to classify `The Arithmetic Intensity Test` i.e

- Operations per byte < 1 = Memory Bound (focus on access patterns)

- Operations per byte > 10 = Compute Bound (focus on algorithms)

![gpu-quiz]({{ site.baseurl }}/assets/blog-2-memory-coalsecing/mem-comp.png)

Example - Reality check with vector addition:
Load 2 integers (8 bytes) + Store 1 result (4 bytes) = 12 bytes
1 operation ÷ 12 bytes = 0.08 operations/byte → `Memory bound!`

## Memory Coalescing 
### Back to Our Warehouse: The Memory Access Challenge

### Quick Recap 📦

In our warehouse we have:
- **Teams (Programs)** = Different kernel launches  
- **Workers (Threads)** = Individual threads in each team
- **Parcels** = Data elements in memory
- **Pick-up lists** = Memory access patterns

### The Slow Way: Uncoalesced Access 🐌
Now If I had been a Worker here pain would have been to pickup items from completely far locations. Say on a rack having 1-1000 boxes arranged horizontally 

I am picking things like boxes numbered  `234, 568, 900` etc.

This is where I would have spent **95%** of time running around, **5%** actually picking parcels. Sound familiar? 🤔

### The Fast Way: Coalesced Access ⚡
Easisest would have been if I had to pick(same) boxes placed consecutively at `234,235,236` **coalesced**.

**In GPU terms:** This is **coalesced memory access** - threads accessing consecutive memory locations.

Now Since we have understood the concept of **coalescing** the hero that supports this concept is called `Warps`. 

But What is Warp?🤔

In our same Warehouse terminology **A Warp = A Small Team of 32 Workers Who Must Move Together**

- **GPU Reality**: Threads don't like to work independently but they love to be grouped into **warps** of 32 threads each.
- **The Catch**: All 32 threads in a warp **must execute the same instruction at the same time**
- **Memory Implication**: When one thread in the warp needs data, ALL 32 threads pause until that memory request is fulfilled.

Disclaimer - We will get into details of Warps of what and how they are useful in further blogs if you feel any vaccum. Just take the above definition as memory point to be used in this blog further.

#### Proof by Code: Let's See the Difference! 
Theory is good but let's see the practical numbers to solidify the case and make it air-tight.

```python
# === Memory Coalescing with Multiplication ===
<span style="color: #1E90FF ; font-weight: bold;"> 
import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt

@triton.jit
def multiply_coalesced(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """
    COALESCED: Threads access consecutive memory addresses
    Thread 0→Addr 0, Thread 1→Addr 1, Thread 2→Addr 2... ⚡
    """
    pid = tl.program_id(0) #Which aisle is my team assigned to?
    offsets = pid * BLOCK + tl.arange(0, BLOCK) #`offsets` are the exact package numbers a team will handle.
    mask = offsets < N
    
    # Load consecutive addresses - So it should be FAST!
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # creating and stroing the results
    result = x * y
    tl.store(out_ptr + offsets, result, mask=mask)

@triton.jit
def multiply_scattered(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """
    NON-COALESCED: Threads access scattered memory addresses
    Thread 0→Addr 0, Thread 1→Addr 64, Thread 2→Addr 128... 🐌
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    
    # We have intenionally Scattered addresses by jumping 64 positions each time - To show SLOWNESS!

    scattered_offsets = (offsets * 64) % N
    
    x = tl.load(x_ptr + scattered_offsets, mask=mask)
    y = tl.load(y_ptr + scattered_offsets, mask=mask)
    result = x * y
    tl.store(out_ptr + offsets, result, mask=mask)  # Store still coalesced

def run_benchmark():
    # Setup
    N = 1024 * 1024  # 1M elements
    x = torch.randn(N, device='cuda')
    y = torch.randn(N, device='cuda')
    out = torch.empty_like(x)
    
    grid = (triton.cdiv(N, 256),)
    BLOCK = 256
    
    # Warmup
    multiply_coalesced[grid](x, y, out, N, BLOCK)
    multiply_scattered[grid](x, y, out, N, BLOCK)
    torch.cuda.synchronize()
    
    # Time coalesced access
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # calling our coalesced access
    start.record()
    multiply_coalesced[grid](x, y, out, N, BLOCK)
    end.record()
    torch.cuda.synchronize()
    time_coalesced = start.elapsed_time(end)
    
    # calling our non coalesced access
    start.record()
    multiply_scattered[grid](x, y, out, N, BLOCK)
    end.record()
    torch.cuda.synchronize()
    time_scattered = start.elapsed_time(end)
    
    # Results time 
    print(f"Coalesced access:     {time_coalesced:.3f} ms")
    print(f"Non-coalesced access: {time_scattered:.3f} ms") 
    print(f"Slowdown factor:      {time_scattered/time_coalesced:.1f}×")
    
    return time_coalesced, time_scattered

def plot_comparison(time_coalesced, time_scattered):
    """Simple bar chart showing the performance difference"""
    
    methods = ['Coalesced\n(Fast)', 'Non-Coalesced\n(Slow)']
    times = [time_coalesced, time_scattered]
    colors = ['green', 'red']
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(methods, times, color=colors, alpha=0.8)
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/4, bar.get_height(), 
                f'{time:.2f}ms', ha='center', fontweight='bold')
    
    plt.title('Memory Coalescing Performance Impact', fontsize=16, fontweight='bold')
    plt.ylabel('Execution Time (ms)')
    plt.grid(axis='y', alpha=0.3)
    
    #  speedup annotation
    speedup = time_scattered / time_coalesced
    plt.text(0.2, max(times) * 0.4, f'{speedup:.1f}× faster!', 
             ha='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('memory_coalescing_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Memory Coalescing Benchmark")
    print("=" * 40)
    
    coalesced_time, scattered_time = run_benchmark()
    plot_comparison(coalesced_time, scattered_time)
    
    print(f"\nThe takeaway: Memory access patterns matter MORE than the actual computation!")
</span>
```

We will not get into basic details of code of what is `progarma_id`, `mask`, `offsets` etc since they have been already covered in previous blog [here](https://sahibpreetsingh12.github.io/posts/i-m-teaching-myself-triton-here-s-what-s-actually-happening/)

but the line that is causing all the difference is `scattered_offsets = (offsets * 64) % N` 

Let me break down why `scattered_offsets = (offsets * 64) % N` creates chaos:

- Thread 0: accesses address `0 * 64 = 0`
- Thread 1: accesses address `1 * 64 = 64` 
- Thread 2: accesses address `2 * 64 = 128`

And instead of consecutive addresses (0,1,2,3...), threads now jump 64 positions apart! 

**Here's why this breaks the warp's efficiency:**

Remember, all `32 threads in a warp must wait for memory together`. With coalesced access, the GPU can satisfy all 32 threads with **one memory transaction**. But with scattered access:

❌ **Warp efficiency is destroyed** - GPU needs **multiple separate memory requests**

❌ All 32 threads sit idle waiting for each scattered memory fetch

❌ The warp's coordinated movement becomes a liability instead of an advantage

It's like our 32 chained warehouse workers trying to grab items from 32 different aisles - they're forced to move together, but now that coordination makes them **slower** instead of faster!

When I ran the above code on colab (free Tier - T4 GPU) below are my results.
![results]({{ site.baseurl }}/assets/blog-2-memory-coalsecing/coal-noncoal.png)

you can try `scattered_offsets = (offsets * 97) % N`  # As 97 is prime to see more devastating effects of this phenomenon.


In later blogs will show you how this simple trick can make your inference faster on open source llm's like Deepseek, Qwen etc and when i was learning of this trick I felt why would someone intenionally do this but because of how we write our regular code our code writing habits push us to Non-coalesced patterns of data loading and hence causing Under-utilisation of GPU.

##  The Golden Rule of Fast GPU Code

> **"Threads in a warp should access contiguous blocks of memory"**

This isn't just advice - it's the **fundamental principle** that separates fast GPU code from slow GPU code.

**Why `tl.arange(0, BLOCK_SIZE)` is everywhere:**
Now you understand why this pattern appears in every Triton kernel - it's specifically designed to ensure consecutive memory access!


## Conclusion: It's Not the GPU, It's the Memory Access

Today you learned the **single most important concept** in GPU optimization:

1. **GPUs are Memory Bound**: They wait for data more than they compute.

2. **Warps are Key**: 32 threads move together as a single unit.

3. **Coalesced Access is the Goal**: Ensure threads in a warp access consecutive memory.

Mastering this principle is the difference between code that **runs** and code that **flies**.


### Key Takeaways: The TL;DR for Fast GPUs

•  The Real Bottleneck: It's not the math, it's the memory waiting game. Your GPU is a Ferrari engine stuck in traffic.

•  The Warp Squad 👨‍👩‍👧‍👦: Threads are a team of 32 chained together. If one waits, they all wait.

•  The Golden Rule ✨: Keep your warp squad happy by making them pick up consecutive parcels (contiguous memory).

•  Coalesced vs. Scattered ⚡: One efficient trip vs. 32 chaotic trips. The difference? A 5x-10x speedup from a single line of code!

•  Triton's Magic Wand 🪄: The `tl.arange` pattern isn't just common, it's a direct instruction to your GPU to "go fast and coalesce!"

### **Next Time... Tiling!**

Now that you understand coalescing, the next logical step is **tiling** - the strategy we use to apply this concept to bigger problems like matrix multiplication.

Stay tuned as we break down how to optimize large-scale operations in Triton!
