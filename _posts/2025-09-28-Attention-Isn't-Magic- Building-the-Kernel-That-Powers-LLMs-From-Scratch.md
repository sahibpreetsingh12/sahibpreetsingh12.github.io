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

### A 60-Second Refresher on Attention
Before we write the code, we need a quick mental model. The classic "Query, Key, Value" explanation always felt sterile to me. What does that *mean* when you're staring at memory pointers?
Let's go with basic analogy of quick receipe builder.

Imagine you want to create a new flavor (the Output).
-  <span style="color: #9ACD32; font-weight: bold;">Query</span>: You have a specific taste profile in mind. "I want something that's 80% spicy and 20% smoky." This is your Query vector.

-  <span style="color: #9ACD32; font-weight: bold;">Keys</span>: You have a shelf full of spice jars, each with a label describing its essence ("Cayenne Pepper," "Smoked Paprika," "Cinnamon"). These labels are the Key vectors.

-  <span style="color: #9ACD32; font-weight: bold;">Values</span>: Inside each jar is the actual spice itself. This is the Value vector.

Our kernel's(again saying it's Just like a FUNCTION in python) job is to perform these steps for each Query:
-  <span style="color: #9ACD32; font-weight: bold;">Step 1</span>: Find the Matches. It compares your desired taste profile (Query) to the label on every single jar (Key) to see how well they match. This generates your similarity scores.

-  <span style="color: #9ACD32; font-weight: bold;">Step 2</span>: Create the Recipe. It runs these scores through a softmax to create the final recipe: "Use 75% from the Cayenne jar, 23% from the Smoked Paprika jar, and 2% from the Cinnamon jar."

-  <span style="color: #9ACD32; font-weight: bold;">Step 3</span>: Mix the Ingredients. It then takes the actual spices (Values) from inside those jars according to the recipe and mixes them together to create your final, complex flavor (Output).

From the GPU's perspective, we're just building <span style="color: #9ACD32; font-weight: bold;">millions of tiny, unique recipes in parallel</span>. Our "slow" kernel today will do this one recipe at a time. Later, we'll learn how to do it much more efficiently.

If still not convinced with my basic analogy the best place to read about this is - [Jay Alammar](https://jalammar.github.io/illustrated-transformer/)