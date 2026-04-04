---
title: "Build GPT-2 from Scratch: Architecture, Tokenization, and Training"
date: 2026-03-28
tags: ["MachineLearning", "DeepNeuralNetwork", "Mathematics", "LLM"]
categories: ["posts"]
description: "A step-by-step GPT-2 walkthrough covering BPE tokenization, transformer blocks, masked self-attention, and training."
summary: "A first-principles guide to building GPT-2, from tokenization and embeddings to transformer training."
math: true
mermaid: true
ShowToc: true
TocOpen: true
draft: false
---
## Architecture
The overall structure of GPT-2 is a stack of Transformer decoder blocks:

$$
\text{Token Embedding} + \text{Positional Encoding} \rightarrow N \times \text{Transformer Block} \rightarrow \text{LayerNorm} \rightarrow \text{Linear Head}
$$

The core of each Transformer block is the **self-attention mechanism**, driven by three matrices: Query ($Q$), Key ($K$), and Value ($V$). The attention score is computed as:

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $d_k$ is the dimension of the key vectors, used to scale the dot product and prevent vanishing gradients.
## Tokenization and Raw Data
When processing text data, we need to convert it into numbers, because neural networks operate on vectors, not strings. Text is fundamentally a sequence of characters, each with a numeric encoding (e.g., ASCII or Unicode). However, mapping each character directly to a vector is too low-level and inefficient.

Instead, we map text into a high-dimensional vector space — this is called *Embedding*. The model is essentially a composition of functions that operate on these vectors. GPT-2 (small) uses an embedding dimension of 768.

A natural question is: what is the basic unit we feed into the Embedding layer — a character or a word? The answer is neither. We use *tokens*, produced by an algorithm called **BPE (Byte Pair Encoding)**. BPE works by iteratively merging the most frequent byte pairs in the corpus into a single token, striking a balance between vocabulary size and expressive power. Each token is assigned a unique integer ID, similar to a hashmap — `{token: id}` in Python.

In this section, we will use **tiktoken** (OpenAI's tokenizer) to handle tokenization. Install it with:

```bash
pip install tiktoken
# or
uv add tiktoken
```
## Masked Self-Attention
The attention mechanism is based on matrix dotproduct.
## Build GPT from the Beginning
## Training on an Unlabeled Dataset
## Fine-tuning for Classification
## Instruction Fine-tuning with Human Feedback
