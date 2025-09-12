---
layout: post
title: "I'm Teaching Myself Triton - Here's What's Actually Happening"
date: 2025-09-12
description: "Part-1 Describes how I got started with triton, why you should and How to get started"
tags: [Nvidia, Triton, GPU Programming]
author: Sahibpreet Singh
pinned: true
---
Starting from start month of July-2025 super scared and tensed. I was doing everything left right and centre to secure a job and in the middle of all this felt super helpless and during that time only got the hit to make myself learn something which I always feared GPU programming. Yes actually how GPU's can be made to work and understand the depth of how things work and not just **.cuda()** and work done.

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

When I started, people kept throwing around words like "blocks," "tiles," and "warps" like I was supposed to just know. Here's what they actually mean, explained like you're five (because that's what I needed):