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
Funny but --> Even if you upgrade to the flagship H100 ($30k), the ratio gets `*worse* - 590:1`! More horsepower, same traffic jam.

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
Now If I had been a Worker here pain would have been to pickup items from completely far locations. Say on a rack having 1-1000 boxes aranged horizontally 

I am picking thing boxes numbered  `234, 568, 900` etc.

This is where I would have spent **95%** of time running around, **5%** actually picking parcels. Sound familiar? 🤔

### The Fast Way: Coalesced Access ⚡
Easisest would have been if I had to pick(same) boxes placed consecutively at `234,235,236` **coalesced**.

**In GPU terms:** This is **coalesced memory access** - threads accessing consecutive memory locations.
