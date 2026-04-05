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

GPT-2 is a **decoder-only Transformer** trained to predict the next token in a sequence.

Its high-level structure is:

$$
\text{Tokens} \rightarrow \text{Token Embedding} + \text{Position Embedding}
\rightarrow N \times \text{Transformer Block}
\rightarrow \text{LayerNorm}
\rightarrow \text{Linear Head}
$$

Each Transformer block contains:

1. masked self-attention
2. a feed-forward MLP
3. residual connections
4. layer normalization

If the hidden size is $d_{model}$ and the sequence length is $T$, then the hidden state has shape:

$$
X \in \mathbb{R}^{T \times d_{model}}
$$

For GPT-2 small:

- number of layers: `12`
- number of heads: `12`
- hidden size: `768`
- context window: `1024`

The key design choice is causal generation: token `t` can only attend to tokens `1..t`, never to future tokens.

---

## Tokenization and Raw Data

Neural networks do not operate directly on strings. We first convert text into **tokens**, then map tokens to integer IDs, and finally map IDs to vectors.

The basic unit is not a word and not a single character. GPT-2 uses **Byte Pair Encoding (BPE)**, which merges frequent byte patterns into reusable subword units.

That gives a practical compromise:

- character-level tokens are too long
- word-level vocabularies are too brittle
- subword tokens balance compression and flexibility

Using `tiktoken`:

```bash
uv add tiktoken
```

```python
import tiktoken

enc = tiktoken.get_encoding("gpt2")
ids = enc.encode("Transformers model conditional distributions over tokens.")
print(ids)
```

Once token IDs are produced, the embedding table converts them into vectors:

$$
E \in \mathbb{R}^{V \times d_{model}}
$$

where `V` is the vocabulary size. Looking up a token ID `i` returns row `E_i`.

GPT-2 also adds a learned positional embedding so the model can distinguish the order of tokens.

---

## Language Modeling Objective

GPT-2 is trained with the next-token prediction objective:

$$
P(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
$$

In words, the model learns a conditional probability distribution over the next token given the prefix.

Training minimizes cross-entropy:

$$
\mathcal{L} = - \sum_{t=1}^{T} \log P(x_t \mid x_{<t})
$$

This objective is simple, but it scales extremely well. The entire model is built around estimating that factorization efficiently.

---

## Masked Self-Attention

The central operation in GPT-2 is self-attention. From hidden states $X$, we compute:

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

Then attention is:

$$
\text{Attention}(Q,K,V) =
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

where $M$ is the **causal mask**. Entries corresponding to future positions are set to $-\infty$ before the softmax.

This ensures:

- token 5 can attend to 1,2,3,4,5
- token 5 cannot attend to 6,7,...

In PyTorch, a minimal version looks like:

```python
import math
import torch

def masked_attention(q, k, v):
    T = q.size(-2)
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.size(-1))
    mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v
```

This is the mechanism that lets every token build a context-aware representation of the prefix.

---

## Multi-Head Attention

A single attention map is often too limited. GPT-2 uses **multi-head attention**, which splits the hidden dimension across several heads:

$$
\text{MHA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W_O
$$

Each head can specialize differently. For example, one head may focus on:

- local syntax
- long-range dependencies
- delimiter structure
- repeated entities

The point is not that heads correspond cleanly to human-defined roles, but that multiple heads allow the model to represent several relational patterns in parallel.

---

## The Feed-Forward Block

After attention, each token independently passes through an MLP:

$$
\text{MLP}(x) = W_2 \phi(W_1 x + b_1) + b_2
$$

In GPT-2, this block expands the hidden dimension and then projects it back down. This lets the model perform nonlinear feature transformation after the context-mixing step.

You can think of the division of labor as:

- attention mixes information across positions
- the MLP transforms features at each position

Both are necessary.

---

## A Minimal Transformer Block

Putting the pieces together:

```python
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

Two implementation details matter:

- **residual connections** stabilize deep optimization
- **pre-layernorm** helps gradient flow in practice

Repeated stacking of this block is what gives GPT-2 its modeling capacity.

---

## Build GPT from the Beginning

A small educational GPT can be built in the following order:

1. write the tokenizer pipeline
2. create token and position embeddings
3. implement causal self-attention
4. implement the MLP and transformer block
5. stack multiple blocks
6. add the output projection to vocabulary logits
7. write the training loop

A minimal forward pass is:

```python
tok = token_embedding(idx)          # [B, T, d]
pos = position_embedding(pos_ids)   # [T, d]
x = tok + pos

for block in blocks:
    x = block(x)

x = final_ln(x)
logits = lm_head(x)                 # [B, T, vocab_size]
```

The logits are then compared with shifted targets using cross-entropy.

---

## Training on an Unlabeled Dataset

Training data for GPT-2 does not need class labels. Plain text is enough.

From a sequence:

```text
the cat sat on the mat
```

we create:

- input: `the cat sat on the`
- target: `cat sat on the mat`

More precisely, if `idx[:, :-1]` is the input sequence, then `idx[:, 1:]` is the target sequence.

In PyTorch:

```python
logits = model(x)                   # [B, T, V]
loss = torch.nn.functional.cross_entropy(
    logits[:, :-1].reshape(-1, logits.size(-1)),
    x[:, 1:].reshape(-1),
)
```

This teaches the model to compress statistical regularities from text into its parameters.

Important training considerations:

- large batch size improves throughput
- AdamW is commonly used
- learning-rate warmup helps early stability
- gradient clipping can prevent exploding updates
- context length strongly affects memory cost

Because attention has roughly quadratic cost in sequence length, longer contexts are much more expensive.

---

## Sampling and Inference

Once trained, GPT-2 generates text autoregressively:

1. feed a prompt
2. get logits for the next token
3. sample or choose the next token
4. append it to the prompt
5. repeat

Sampling is controlled by strategies such as:

- **temperature**: rescales logits
- **top-k**: keep only the `k` highest-probability tokens
- **top-p**: keep the smallest set whose cumulative probability exceeds `p`

These do not change the model itself; they change how we draw from its learned distribution.

---

## Fine-Tuning for Classification

A pretrained GPT model can also be adapted to downstream tasks such as sentiment classification, topic labeling, or spam detection.

There are two common strategies:

1. **Add a task-specific head**  
   Use the final hidden state to predict a label.
2. **Prompt the model generatively**  
   Reframe classification as text generation.

The first approach is more direct in a standard ML pipeline. The second is closer to modern instruction-style use.

The reason fine-tuning works is that pretraining already teaches the model:

- syntax
- semantics
- long-range dependencies
- broad world regularities in text

Downstream tasks then only need to reshape that knowledge.

---

## Instruction Fine-Tuning with Human Feedback

Base GPT-2 learns to continue text. It does **not** automatically learn to follow instructions helpfully.

Instruction tuning typically adds supervised examples like:

- prompt: "Summarize this article"
- response: a good summary

Human-feedback-based alignment adds another layer by preferring some responses over others.

Conceptually, the pipeline is:

1. pretrain on next-token prediction
2. supervised fine-tune on instruction-response pairs
3. apply preference optimization or RL-style alignment

This is the difference between a raw language model and an assistant-like system.

GPT-2 historically predates much of the alignment stack used in modern chat models, but understanding GPT-2 is still the cleanest way to understand the foundation.

---

## Final Intuition

If you strip away the scale, GPT-2 is conceptually elegant:

- tokenization turns text into discrete units
- embeddings turn units into vectors
- masked attention lets tokens read the prefix
- MLP blocks transform those contextual features
- training on next-token prediction teaches the model a distribution over text

That is why rebuilding GPT-2 from scratch is such a useful exercise. It exposes the core mechanics behind modern large language models without hiding them behind framework magic.
