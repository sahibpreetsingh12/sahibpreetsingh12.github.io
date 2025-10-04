---
layout: post
title: " 'For' Loops Are Killing Your Triton Kernel - Here's How Vectorized Loading Saves It"
date: 2025-10-06
description: "Part-4: How vectorized loading turns your sleepy GPU into a computational beast - from 1024 memory trips to just 2"
category: Triton
tags: [Triton, Attention, GPU, Performance]
author: "Sahibpreet Singh"
published: true
pinned: false
---

## Recap 
In Our last blog we discovered [How to write your own attention kernel](https://sahibpreetsingh12.github.io/posts/attention-isn-t-magic-building-the-kernel-that-powers-llms-from-scratch/) but we saw as sequence length and dimenion increased it resulted in upto 40x slower than PyTorch? I spent a whole weekend staring at GPU profiler results, convinced something was broken. The GPU utilization graph looked like a flatline with occasional tiny spikes. My RTX 4090, capable of 83 TFLOPS, was operating at roughly the speed of a 2005 calculator.

Then I saw it in the profiler: **"Memory stall: 97.3%"**.

My GPU wasn't computing—it was WAITING. Like a Formula 1 driver stuck behind a cyclist on a narrow road. 

Today, we're going to fix that with one simple trick that will blow your mind: **<span style="color: #9ACD32; font-weight: bold;">vectorized loading a.k.a broadcasting</span>**. We'll turn our sleepy kernel from making 1,024 separate trips to memory into making just 2 efficient ones. The result? A 3-5x speedup with roughly the same amount of code.

### The Sequential Nightmare 🐌

In our last attention kernel, here's what each program was doing:

```python
# THE SLOW WAY - What we did last time
for k_idx in range(1024):  # For EACH of 1024 spice jars...
    key = tl.load(K_ptr + k_idx * d_model + offsets)  # Walk to jar k_idx
    score = compute_similarity(query, key)             # Taste it
    # Walk back to workbench, repeat 1023 more times 
```
The above approach is good when we think from First principles persepctive but it's hurting the core principle we discussed here <span style="color: #9ACD32;">[Most GPU Ops are Memory Bound](https://sahibpreetsingh12.github.io/posts/the-first-rule-of-fast-triton-kernels-coalesce-your-memory-access/#:~:text=But%20What%20is%20GPU%20Reality%3A%20Most%20operations%20(including%20our%20vector%20addition)%20are%20memory%20bound%20%2D%20we%20spend%20more%20time%20waiting%20for)</span>

