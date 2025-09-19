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
Last time we built our triton kernel a simple Vector Addition. Now on normal operations we might not feel but the real usi of Triton is to make 
"kernels" faster without getting into hassle of CUDA (which can be little overwhelming) to start with. This is where first concept comes in called as "Memory Coalescing" comes in.

## Deep Dive - What's the Problem?
**Pop Quiz** : What's the #1 performance killer in GPU programming? 
![gpu-quiz]({{ site.baseurl }}/assets/blog-2-memory-coalsecing/pop_quiz.png)

Notes:

🙅‍♂️ Nope. It's something so mundane you probably never thought about it: *where your data lives in memory.* Move your data wrong, and your lightning-fast GPU becomes slower than your laptop's CPU. Move it right, and you unlock 10x-100x speedups. It is just like a **Ferrari** is sitting in a shopping cart.

But Why Does Memory Matter So Much?

Example -  **Your RTX 4090 can perform 83 trillion operations per second, but it can only fetch 1,008 GB of data per second**
Let me put that in perspective:
- **GPU Compute**: 83 TFLOPS (83 trillion FP32 operations/second)  
- **Memory Bandwidth**: 1,008 GB/s
- **The Math**: That's roughly a **82:1 mismatch!**

This means your $1,600 GPU spends **98% of its time waiting** for data to arrive from memory. 
Funny but --> Even if you upgrade to the flagship H100 ($30k), the ratio gets *worse* - 590:1! More horsepower, same traffic jam.