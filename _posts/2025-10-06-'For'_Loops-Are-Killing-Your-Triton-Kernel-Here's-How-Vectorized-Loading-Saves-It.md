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
In Our last blog we discovered [How to write your own attention kernel](https://sahibpreetsingh12.github.io/posts/attention-isn-t-magic-building-the-kernel-that-powers-llms-from-scratch/) but we saw as we increased sequence length and dimenion increased it resulted in upto 40x slower than PyTorch? This si where I spent a whole weekend staring at GPU profiler results, convinced something was broken. The GPU utilization graph looked like a flatline with occasional tiny spikes. My RTX 4090, capable of 83 TFLOPS, was operating at roughly the speed of a 2005 calculator.

Then I saw it in the profiler: **"Memory stall: 97.3%"**.

My GPU wasn't computing—it was WAITING. Like a Formula 1 driver stuck behind a cyclist on a narrow road. 

Today, we're going to fix that with one simple trick that will blow your mind: **<span style="color: #9ACD32; font-weight: bold;">vectorized loading a.k.a broadcasting</span>**. We'll turn our sleepy kernel from making 1,024 separate trips to memory into making just 2 efficient ones. The result? A 3-5x speedup with roughly the same amount of code.

### The Sequential Nightmare 🐌

In our last attention kernel, here's what each program was doing:

```python
# THE SLOW WAY 
for k_idx in range(1024):  # For EACH of 1024 spice jars...
    key = tl.load(K_ptr + k_idx * d_model + offsets)  # Walk to jar k_idx
    score = compute_similarity(query, key)             # Taste it
    # Walk back to workbench, repeat 1023 more times 
```
The above approach is good when we think from First principles persepctive but it's hurting the core principle we discussed here <span style="color: #9ACD32;">[Most GPU Ops are Memory Bound](https://sahibpreetsingh12.github.io/posts/the-first-rule-of-fast-triton-kernels-coalesce-your-memory-access/#:~:text=But%20What%20is%20GPU%20Reality%3A%20Most%20operations%20(including%20our%20vector%20addition)%20are%20memory%20bound%20%2D%20we%20spend%20more%20time%20waiting%20for)</span>

<span style="color: #9ACD32;">Pop Quiz</span>: If walking to get one jar takes 100 nanoseconds, how long for 1024 jars?

<span>Answer</span>: 102,400 nanoseconds of JUST WALKING. Not cooking, not mixing—just fetching ingredients!

And this is where our GPU is wasting a lot of time 🚚

## Vectorized Loading 

```python
# THE FAST WAY 
# Load ALL 1024 programs in ONE  trip!
all_keys = tl.load(K_ptr + seq_offsets[:, None] * d_model + dim_offsets[None, :])
# Now compute ALL similarities at once
scores = tl.sum(query[None, :] * all_keys, axis=1) * scale
```
Now we have done exactly same work but in 512x fewer trips. Your GPU finally gets to compute instead of wait.

But what <span style="color: #9ACD32; font-weight: bold;">[:, None]</span> and <span style="color: #9ACD32; font-weight: bold;">[None, :]</span> actually DO? Let's see

```python
import torch

seq_offsets = torch.tensor([0, 1, 2, 3])           # Shape: [4]
print(seq_offsets[:, None]) 
# [[0],         # Shape: [4, 1] - Column vector!
#  [1],
#  [2],
#  [3]]

dim_offsets = [0, 1, 2, 3]           # Shape: [4]
dim_offsets[None, :] = [[0, 1, 2, 3]] # Shape: [1, 4] - Row vector!
```

Now If we do 
```python
print(seq_offsets + dim_offsets)

# [[0, 1, 2, 3],
#  [4, 5, 6, 7],
#  [8, 9, 10, 11],
#  [12, 13, 14, 15]]

# and Thats broadcasting done
```
If I talk in chef and cooking analogy we took in last blog. Instead of our chef(program) going to kitchen and coming back 1024 times to take all jars.

This time chef will go and put all of these 1024 jars in one forklift and will come back and start cooking. We will it's impact when we will run our code.


## Code Time

```python
@triton.jit
def vectorized_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Output_ptr,
    seq_len, d_model, scale,
    BLOCK_SIZE_SEQ: tl.constexpr,  # How many jars our forklift can carry
    BLOCK_SIZE_DIM: tl.constexpr   # How big each jar is
):
    # Step 1: 
    query_idx = tl.program_id(0)  # example say we are chef #42, handling order #42"
    
    # Boundary check - some chefs might not have orders
    if query_idx >= seq_len:
        return
    
    # Step 2: Create our coordinate system for the warehouse
    # Think of this as a grid map of the warehouse floor
    seq_offsets = tl.arange(0, BLOCK_SIZE_SEQ)  # Aisle numbers: [0,1,2,...,127]
    dim_offsets = tl.arange(0, BLOCK_SIZE_DIM)  # Shelf positions: [0,1,2,...,127]
    
    # Safety checks - don't try to grab from non-existent shelves
    seq_mask = seq_offsets < seq_len
    dim_mask = dim_offsets < d_model
    
    # Step 3: Load our customer's order (same as before)
    # This is our chef's recipe card - what flavor profile we're aiming for
    q_ptrs = Q_ptr + query_idx * d_model + dim_offsets
    query = tl.load(q_ptrs, mask=dim_mask, other=0.0)
    
    # Step 4: THE GAME CHANGER - Load ALL Keys (spice jars) at once!
    
    # Create a 2D grid of addresses using broadcasting
    # seq_offsets[:, None] = Column of aisle numbers [[0],[1],[2],...]
    # dim_offsets[None, :] = Row of shelf positions [[0,1,2,3,...]]
    # Result: 2D grid where [i,j] = address of jar i, position j
    k_ptrs = K_ptr + seq_offsets[:, None] * d_model + dim_offsets[None, :]
    
    # Load the ENTIRE spice collection in one go!
    # all_keys has shape [BLOCK_SIZE_SEQ, BLOCK_SIZE_DIM]
    # This single line replaces 1024 individual tl.load() calls!
    all_keys = tl.load(k_ptrs, mask=seq_mask[:, None] & dim_mask[None, :], other=0.0)
    
    # Step 5: Taste ALL spices at once (parallel computation!)
    # query[None, :] broadcasts our taste profile to match each jar
    # Shape: [1, d_model] * [seq_len, d_model] = [seq_len, d_model]
    # Then sum along dim=1 to get similarity score for each sequence position
    scores = tl.sum(query[None, :] * all_keys, axis=1) * scale
    
    # Apply sequence mask to scores (ignore padding positions)
    scores = tl.where(seq_mask, scores, -float('inf'))
    
    # Step 6: Softmax - Convert scores to probabilities (our recipe percentages)
    # This part remains mostly unchanged from basic version
    
    # Find the spiciest score (for numerical stability)
    max_score = tl.max(scores, axis=0)
    
    # Subtract max before exponential (prevents overflow)
    # Like adjusting all spice levels relative to the strongest one
    exp_scores = tl.exp(scores - max_score)
    
    # Zero out the invalid positions
    exp_scores = tl.where(seq_mask, exp_scores, 0.0)
    
    # Calculate total to normalize (sum to 100%)
    sum_exp = tl.sum(exp_scores, axis=0)
    
    # Get our final recipe percentages
    attn_weights = exp_scores / sum_exp
    
    # Step 7: Load ALL Values (ingredient jars) at once - second forklift trip!
    # Same broadcasting pattern as keys
    v_ptrs = V_ptr + seq_offsets[:, None] * d_model + dim_offsets[None, :]
    all_values = tl.load(v_ptrs, mask=seq_mask[:, None] & dim_mask[None, :], other=0.0)
    
    # Step 8: Mix ingredients according to recipe (attention weights)
    # attn_weights[:, None] broadcasts weights to each ingredient dimension
    # Shape: [seq_len, 1] * [seq_len, d_model] = [seq_len, d_model]
    # Sum along dim=0 to blend all ingredients into final dish
    output = tl.sum(attn_weights[:, None] * all_values, axis=0)
    
    # Step 9: Serve the final dish (store output)
    o_ptrs = Output_ptr + query_idx * d_model + dim_offsets
    tl.store(o_ptrs, output, mask=dim_mask)


# Python wrapper to run the function

def vectorized_attention(Q, K, V, scale=None):
    """
    Vectorized attention - loads everything at once instead of loop-by-loop
    
    The key insight: Replace sequential for loops with vectorized loads
    Old: for k in keys: load(k)  # 1024 separate loads
    New: load(all_keys)          # 1 vectorized load
    
    Args:
        Q: Query tensor [seq_len, d_model]
        K: Key tensor [seq_len, d_model]  
        V: Value tensor [seq_len, d_model]
        scale: Scaling factor (default: 1/sqrt(d_model))
    
    Returns:
        Output tensor [seq_len, d_model]
    """
    seq_len, d_model = Q.shape
    
    # Validate inputs
    assert K.shape == (seq_len, d_model), f"K shape mismatch: {K.shape}"
    assert V.shape == (seq_len, d_model), f"V shape mismatch: {V.shape}"
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "Tensors must be on GPU"
    
    # Default scaling for numerical stability
    if scale is None:
        scale = 1.0 / math.sqrt(d_model)
    
    # Create output tensor
    output = torch.empty_like(Q)
    
    # Choose block sizes (must be powers of 2 for Triton)
    # Make blocks large enough to fit our data
    BLOCK_SIZE_SEQ = triton.next_power_of_2(seq_len)
    BLOCK_SIZE_DIM = triton.next_power_of_2(d_model)
    
    # Launch configuration - still one program per query
    # But each program now loads ALL keys/values at once!
    grid = (seq_len,)
    
    # Fire up our army of forklift-driving chefs!
    vectorized_attention_kernel[grid](
        Q, K, V, output,
        seq_len, d_model, scale,
        BLOCK_SIZE_SEQ=BLOCK_SIZE_SEQ,
        BLOCK_SIZE_DIM=BLOCK_SIZE_DIM
    )
    
    return output
```

The triton kernel the whole code above is same the changed bits as compared to last blog are `Step-4` and `Step-7` for explanation of other steps I would recommend checking my last blog.
