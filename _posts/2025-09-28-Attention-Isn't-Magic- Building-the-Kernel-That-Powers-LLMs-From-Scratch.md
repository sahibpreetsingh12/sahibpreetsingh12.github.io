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
import triton.language as tl

@triton.jit
def basic_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Output_ptr,
    seq_len, d_model, scale,
    BLOCK_SIZE_SEQ: tl.constexpr,
    BLOCK_SIZE_DIM: tl.constexpr
):
    # --- 1. GET OUR ASSIGNMENT ---
    # Each team of workers (program) is responsible for ONE order (query).
    # Which order is my team working on today?
    query_idx = tl.program_id(0)

    # --- 2. GET THE ORDER DETAILS (LOAD QUERY) ---
    # We need to grab the full instruction sheet for our order (the query vector).
    # This instruction sheet will stay on our workbench (SRAM) for the whole process.
    dim_offsets = tl.arange(0, BLOCK_SIZE_DIM)
    dim_mask = dim_offsets < d_model
    # Find the memory address for our specific order's instruction sheet.
    q_ptrs = Q_ptr + query_idx * d_model + dim_offsets
    # Load the instruction sheet.
    query = tl.load(q_ptrs, mask=dim_mask, other=0.0)

    # --- 3. COMPARE OUR ORDER TO EVERY ITEM IN THE WAREHOUSE (COMPUTE SCORES) ---
    # This is the slow, brute-force part of our job today.
    # To find out which items match our order, our worker must walk down EVERY aisle.
    scores = tl.full([BLOCK_SIZE_SEQ], value=-float('inf'), dtype=tl.float32)

    # This 'for' loop is our worker walking through the warehouse, one shelf at a time.
    for k_idx in range(seq_len):
        if k_idx < BLOCK_SIZE_SEQ:
            # Go to a shelf and load the label of an item (a key vector).
            k_ptrs = K_ptr + k_idx * d_model + dim_offsets
            key = tl.load(k_ptrs, mask=dim_mask, other=0.0)

            # Compare the item's label (key) to our order instructions (query).
            score = tl.sum(query * key) * scale

            # Write down the comparison score on our clipboard.
            scores = tl.where(tl.arange(0, BLOCK_SIZE_SEQ) == k_idx, score, scores)

    # --- 4. CREATE THE FINAL RECIPE (APPLY SOFTMAX) ---
    # Now that we've seen all the items, we convert our comparison scores
    # into a final recipe (the attention weights). e.g., "75% item A, 25% item B".
    seq_mask = tl.arange(0, BLOCK_SIZE_SEQ) < seq_len
    scores = tl.where(seq_mask, scores, -float('inf'))
    max_score = tl.max(scores, axis=0)
    attn_weights = tl.exp(scores - max_score)
    attn_weights = tl.where(seq_mask, attn_weights, 0.0)
    attn_weights /= tl.sum(attn_weights, axis=0)

    # --- 5. GATHER THE INGREDIENTS (COMPUTE OUTPUT) ---
    # This is our SECOND inefficient tour of the warehouse!
    # With recipe in hand, our worker must now go BACK to the shelves to get the actual items.
    output = tl.zeros([BLOCK_SIZE_DIM], dtype=tl.float32)

    # Another slow 'for' loop...
    for v_idx in range(seq_len):
        if v_idx < BLOCK_SIZE_SEQ:
            # Go back to a shelf and get the actual contents of the box (a value vector).
            v_ptrs = V_ptr + v_idx * d_model + dim_offsets
            value = tl.load(v_ptrs, mask=dim_mask, other=0.0)

            # Check our recipe for how much of this item to grab.
            weight = tl.sum(tl.where(tl.arange(0, BLOCK_SIZE_SEQ) == v_idx, attn_weights, 0.0))
            
            # Add the ingredient to our final package.
            output += weight * value

    # --- 6. SHIP THE FINAL PACKAGE (STORE RESULT) ---
    # Our final, blended product (the output vector) is ready.
    # Place it on the shipping dock (store it in output memory).
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

But Question was <span style="font-size: 1.1em; font-weight: bold">Why not program_id(1) or 2D grid?</span> Because attention is fundamentally a row-wise operation: