---
layout: post
title: "Attention Isn't Magic: Building the Kernel That Powers LLMs From Scratch"
date: 2025-09-29
description: "From abstract math to a real GPU kernel, this post demystifies the attention mechanism by building it from scratch in Triton."
tags: [Triton, Attention, GPU, Performance]
author: "Sahibpreet Singh"
published: true
pinned: false
---

## From Theory to Photons: Applying Our First Rule
In our last post, we learned the most important rule of GPU programming: ***coalesce your memory access***. But theory is one thing—applying it is another. 
Now, it's time to tackle the Everest of modern AI: the attention mechanism. Our goal isn't to be fast (yet). 

Our goal is to translate the abstract math of `softmax(QKᵀ/√d)V` into a real, working GPU kernel, line by line how normally we would in python and then in later blogs we will learn what and why different methods like `tiling` and `vectorised loading`, `online` algorithms make a lot of sense.

### <span style="color: #3256cdff; font-weight: bold;">A 60-Second Refresher on Attention</span>
<div align="center">
  <img src="{{ site.baseurl }}/assets/blog-3-simple-attention/cooking.png" alt="cooking-analogy" style="max-width: 100%; height: auto;">
</div>

Before we write the code, we need a quick mental model.
Let's go with basic analogy of quick receipe builder.

Imagine you want to create a new flavor (the Output).
-  <span style="color: #9ACD32; font-weight: bold;">Query</span>: You have a specific taste profile in mind. "I want something that's 80% spicy and 20% smoky." This is your Query vector.

-  <span style="color: #9ACD32; font-weight: bold;">Keys</span>: You have a shelf full of spice jars, each with a label describing its essence ("Cayenne Pepper," "Smoked Paprika," "Cinnamon"). These labels are the Key vectors.

-  <span style="color: #9ACD32; font-weight: bold;">Values</span>: Inside each jar is the actual spice itself. This is the Value vector.

Our kernel's(again saying it's Just like a FUNCTION in python) job is to perform these steps for each Query:
-  <span style="color: #9ACD32; font-weight: bold;">Step 1</span>: Find the Matches. It compares your desired taste profile (Query) to the label on every single jar (Key) to see how well they match. This generates your similarity scores.

-  <span style="color: #9ACD32; font-weight: bold;">Step 2</span>: Create the Recipe. It runs these scores through a softmax to create the final recipe: "Use 75% from the Cayenne jar, 23% from the Smoked Paprika jar, and 2% from the Cinnamon jar."

-  <span style="color: #9ACD32; font-weight: bold;">Step 3</span>: Mix the Ingredients. It then takes the actual spices (Values) from inside those jars according to the recipe and mixes them together to create your final, complex flavor (Output).

From the GPU's perspective, we're just building <span style="color: #3256cdff; font-weight: bold;">millions of tiny, unique recipes in parallel</span>. Our "slow" kernel today will do this one recipe at a time. Later, we'll learn how to do it much more efficiently.

If still not convinced with my basic analogy the best place to read about this is - [Jay Alammar](https://jalammar.github.io/illustrated-transformer/)

## The "Slow but Simple" Kernel: A Line-by-Line Breakdown

Our goal is clarity, not speed. We will write a kernel where one GPU program handles exactly one query. This is wonderfully simple to understand and a perfect starting point.

Recommended - Use Colab with T4 GPU that's enough for today's blog.

```python
import triton
import math
import torch
import triton.language as tl

@triton.jit
def basic_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Output_ptr,
    seq_len, d_model, scale,
    BLOCK_SIZE_SEQ: tl.constexpr,
    BLOCK_SIZE_DIM: tl.constexpr
):
    # 1. getting the query
    # Each chef (program) handles exactly ONE customer's order (query)
    query_idx = tl.program_id(0)

    # 2. Load Query
    # Load our customer's complete taste profile and keep it on our workbench
    dim_offsets = tl.arange(0, BLOCK_SIZE_DIM)
    dim_mask = dim_offsets < d_model  # Don't use empty spice jars - only real ingredients!
    q_ptrs = Q_ptr + query_idx * d_model + dim_offsets
    query = tl.load(q_ptrs, mask=dim_mask, other=0.0)

    # Compute Score
    # Initialize our score clipboard - start with "impossible" scores for unused slots
    scores = tl.full([BLOCK_SIZE_SEQ], value=-float('inf'), dtype=tl.float32)

    # Walk through every spice jar in the warehouse, one at a time (inefficient!)
    for k_idx in range(seq_len):
        if k_idx < BLOCK_SIZE_SEQ:
            # Read the label on this spice jar (key vector)
            k_ptrs = K_ptr + k_idx * d_model + dim_offsets
            key = tl.load(k_ptrs, mask=dim_mask, other=0.0)

            # How well does this spice match our customer's taste?
            score = tl.sum(query * key) * scale

            # Write the score on our clipboard at the right position
            scores = tl.where(tl.arange(0, BLOCK_SIZE_SEQ) == k_idx, score, scores)

    #4. Softmax
    # Turn raw scores into proper percentages that sum to 100%
    seq_mask = tl.arange(0, BLOCK_SIZE_SEQ) < seq_len
    scores = tl.where(seq_mask, scores, -float('inf'))  # Cross out unused recipe slots
    max_score = tl.max(scores, axis=0)                  # Find highest score (numerical stability)
    attn_weights = tl.exp(scores - max_score)           # Convert to positive weights
    attn_weights = tl.where(seq_mask, attn_weights, 0.0)  # Zero out unused slots
    attn_weights = attn_weights / tl.sum(attn_weights, axis=0)  # Normalize to 100%

    
    output = tl.zeros([BLOCK_SIZE_DIM], dtype=tl.float32)

    # 5. Second trip through warehouse - collect ingredients per recipe
    output = tl.zeros([BLOCK_SIZE_DIM], dtype=tl.float32)

    for v_idx in range(seq_len):
        if v_idx < BLOCK_SIZE_SEQ:
            # Get the actual spice from this jar (value vector)
            v_ptrs = V_ptr + v_idx * d_model + dim_offsets
            value = tl.load(v_ptrs, mask=dim_mask, other=0.0)
            
            # How much of this spice does our recipe call for?
            weight = tl.sum(tl.where(tl.arange(0, BLOCK_SIZE_SEQ) == v_idx, attn_weights, 0.0))
            
            # Add this ingredient to our final dish
            output += weight * value
    # 5. storing the results 
    
    o_ptrs = Output_ptr + query_idx * d_model + dim_offsets
    tl.store(o_ptrs, output, mask=dim_mask)
```
Now let's break it :-

First Question - <span style="font-size: 1.2em; font-weight: bold;"> Why Only one `program_id` ?</span>

But before I answer - 

<span style="font-size: 1.1em; font-weight: bold; color: #9ACD32;">Pop Quiz</span>: If we have 1000 queries to process, how many programs should we launch? 🤔

The answer wasn't obvious to me as well :-

```python
# We launch exactly seq_len programs - one per query
grid = (seq_len,)  # If seq_len = 1000, we get 1000 programs
program_id(0)  # Each sequence (in simple words each token) gets ID: 0, 1, 2, ..., 999
```

But Question is still 
<span style="color: #9ACD32; font-weight: bold;">Why not program_id(1) or 2D grid?</span> Because attention is fundamentally a row-wise operation:

*And when we say row-wise it means each query(in simple terms think each query as each word in sentence) needs to see which parts of the sentence they will pay attention too, and not bother about what other queries will be doing.*


Now in our kitchen and chef analogy think of it like a restaurant kitchen where:

•  Each program = One chef 👨‍🍳

•  Each query = One customer's order

•  The rule: Each chef handles ONE complete order from start to finish

In Last Blog we covered Memory Coalescing [here](https://sahibpreetsingh12.github.io/posts/the-first-rule-of-fast-triton-kernels-coalesce-your-memory-access/) but we will now check are we following that principle? 

```python
query_idx = tl.program_id(0)  # I'm chef #2, handling order #2

# Step 1: Load my customer's order (query vector)
dim_offsets = tl.arange(0, BLOCK_SIZE_DIM)  # [0,1,2,3,4,5,6,... BLOCK_SIZE_DIM]

dim_mask = dim_offsets < d_model # this is just a check of that we don't accidently cross our boundary - we're preventing the chef from accidentally using padding (empty slots) instead of real data dimensions.

q_ptrs = Q_ptr + query_idx * d_model + dim_offsets
#      = Q_ptr + 2 * 512 + [0,1,2,3,4,5,6,... BLOCK_SIZE_DIM]
#      = Q_ptr + [1024,1025,1026,1027,1028,1029,1030,1031]
```
1. And the Answer is <span style="color: #9ACD32; font-weight: bold;">yes</span> we are following since all the addresses are consecutive.

2. All 32 threads in a warp will be able access nearby memory locations

## Now the next piece is <span style="font-size: 1.3em; font-weight: bold; color: #ff6b35;">Nightmare</span> from GPU perspective :

```python
# Problem #1: Sequential key loading
for k_idx in range(seq_len):  # For EVERY key in sequence
    k_ptrs = K_ptr + k_idx * d_model + dim_offsets
    key = tl.load(k_ptrs, mask=dim_mask, other=0.0)  # Load one key
    score = tl.sum(query * key) * scale

# Problem #2: Sequential value loading  
for v_idx in range(seq_len):  # For EVERY value in sequence
    v_ptrs = V_ptr + v_idx * d_model + dim_offsets
    value = tl.load(v_ptrs, mask=dim_mask, other=0.0)  # Load one value
```

We learned in last blog problem with GPU is not the algorithm it's the time GPU sit IDLE waiting for data and the for loop above just does that (but we will ignore it for this blog) since the goal is to just understand how attention works.

The Restaurant Analogy:
- Chef #2 needs to check every ingredient in the pantry (all keys)

- Then make another trip to collect ingredients (all values)  

- Each trip is efficient (coalesced), but we make 2 × seq_len trips total!

Now to practically see how inefficient these numbers are let's take 

`seq_len=1024, d_model=512`

```python
Query loads:    1 × 512 = 512 elements
Key loads:      1024 × 512 = 524,288 elements  
Value loads:    1024 × 512 = 524,288 elements
Total:          1,048,888 elements loaded per query
```

Out Each program loads <span style="color: #9ACD32; font-weight: bold;">2000x</span> more data than it needs for the query itself! 

Our chef's  reading the entire cookbook for every single dish - technically it works, but it's wildly inefficient.

## Putting It All Together in a Python Wrapper

```python
def basic_attention(Q, K, V, scale=None):
    """
    Our simple attention implementation - one chef per customer!
    
    Args:
        Q: Query matrix [seq_len, d_model] - customer taste profiles
        K: Key matrix [seq_len, d_model] - spice jar labels  
        V: Value matrix [seq_len, d_model] - actual spices
        scale: Optional scaling factor (default: 1/√d_model)
    
    Returns:
        Output matrix [seq_len, d_model] - custom flavor blends
    """
    # Basic setup and validation
    seq_len, d_model = Q.shape
    assert K.shape == (seq_len, d_model), f"K shape mismatch: {K.shape}"
    assert V.shape == (seq_len, d_model), f"V shape mismatch: {V.shape}"
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "All tensors must be on GPU"
    
    # Default scaling (the √d_model part of the attention formula)
    if scale is None:
        scale = 1.0 / math.sqrt(d_model)
    
    # Allocate output - same size as queries
    output = torch.empty_like(Q, dtype=torch.float32)
    
    # Choose block sizes (must be powers of 2!)
    BLOCK_SIZE_DIM = triton.next_power_of_2(d_model)  # Fit all dimensions
    BLOCK_SIZE_SEQ = triton.next_power_of_2(seq_len)  # Fit all sequence positions
    
    # Launch one program per query (one chef per customer)
    grid = (seq_len,)  # 1D grid: [Program 0, Program 1, Program 2, ...]
    
    # Send our chefs to work!
    basic_attention_kernel[grid](
        Q, K, V, output,
        seq_len, d_model, scale,
        BLOCK_SIZE_SEQ=BLOCK_SIZE_SEQ,
        BLOCK_SIZE_DIM=BLOCK_SIZE_DIM
    )
    
    return output

# Helper function for easy testing
def create_test_data(seq_len=8, d_model=64, device='cuda'):
    """Create small test matrices for our attention kernel"""
    torch.manual_seed(42)  # Reproducible results
    Q = torch.randn(seq_len, d_model, device=device, dtype=torch.float32)
    K = torch.randn(seq_len, d_model, device=device, dtype=torch.float32) 
    V = torch.randn(seq_len, d_model, device=device, dtype=torch.float32)
    return Q, K, V


def test_attention_correctness():
    """
    The ultimate test: does our kernel produce the same results as PyTorch?
    """
    print("🧪 Testing our attention kernel...")
    
    # Start small - easier to debug if something goes wrong
    seq_len, d_model = 8, 64
    Q, K, V = create_test_data(seq_len, d_model)
    
    # Our attention kernel
    our_output = basic_attention(Q, K, V)
    
    # PyTorch's  implementation
    pytorch_output = torch.nn.functional.scaled_dot_product_attention(
        Q.unsqueeze(0).unsqueeze(0),  # Add batch and head dims
        K.unsqueeze(0).unsqueeze(0), 
        V.unsqueeze(0).unsqueeze(0)
    ).squeeze()  # Remove extra dims
    

    max_diff = torch.max(torch.abs(our_output - pytorch_output)).item()
    matches = torch.allclose(our_output, pytorch_output, rtol=1e-4, atol=1e-4)
    
    print(f"✅ Results match PyTorch: {matches}")
    print(f"📊 Maximum difference: {max_diff:.2e}")
    
    if matches:
        print("🎉 SUCCESS! Our kernel produces correct results!")
    else:
        print("❌ Something's wrong - time to debug...")
    
    return matches

# Test it out!
test_attention_correctness()
```
Perfect You will be able to run your first kernel.

Now Let's benchmark and see how **fast or slow** we are compared to [Pytorch](https://pytorch.org/).
and To keep the blog easy to read and length appropriate check the full code [here](https://github.com/sahibpreetsingh12/triton-learning/tree/main/triton-attention-series/01_basic_attention)

but results I got are 
<div align="center">
  <img src="{{ site.baseurl }}/assets/blog-3-simple-attention/co.png" alt="cooking-analogy" style="max-width: 100%; height: auto;">
</div>

Now <span style="font-size: 1.2em; font-weight: bold; color: #ff6b35;">Why the gulf keeps on increasing as we increase seq_len and dimension we use?</span>

Our kernel is doing two things wrong :

1.  **Sequential Memory Access**: We make 2,048 separate trips (1024 keys + 1024 values)

2. **No Sharing**: Every program loads the same K,V data independently. (We will do KV cache eventually)

In Simple Terms our chef is going back and forth in warehouse first to pickup each ingredient and then while making recipe reads complete cookbook to make our dish. Which causes these so much big delays.

## Key Takeways

- We took the abstract math `softmax(QK^T/√d)V` and turned it into working GPU code that you can run and understand

- Made sure we are applying Memory Coalescing principle.

- Identified the bottlenecks - The extreme slowdown isn't mysterious—it's those sequential loops that make our GPU wait instead of compute


The Performance Reality is Production kernels aren't just about correctness; they're about eliminating waste. Our kernel wastes memory bandwidth like a leaky faucet, but now we know exactly where the leaks are.

## What's Next

Next time, we'll fix our biggest bottleneck with <span style="color: #9ACD32; font-weight: bold;">vectorized loading</span>. Instead of the chef making 1,024 individual trips to the pantry, we'll teach them to load multiple spice jars at once: Keeping the mathematical correctness still the same.

```python
for k_idx in range(seq_len):  # 1024 separate trips
    key = load_one_key(k_idx)  # Load one key at a time
    process(key)
```

In next blog will dig into how
```python
keys_batch = load_multiple_keys(0, 32)  # Load 32 keys at once!  
process_batch(keys_batch)               # Process them together
```
Makes a hell lot of difference.