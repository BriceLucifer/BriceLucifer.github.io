---
title: "AlphaFold Explained: Learning the Basics with PyTorch"
date: 2026-03-05
tags: ["MachineLearning", "DeepNeuralNetwork", "ProteinFolding", "PyTorch"]
categories: ["posts"]
description: "A beginner-friendly AlphaFold introduction that uses PyTorch, tensors, and attention to build intuition from first principles."
summary: "Learn the core ideas behind AlphaFold with a PyTorch-oriented walkthrough of tensors, attention, and protein structure modeling."
draft: false
math: true
mermaid: true
ShowToc: true
TocOpen: true
---

> This note is **not** a full reproduction of AlphaFold. The goal is to build the right mental model: what problem AlphaFold solves, what tensors it operates on, and why attention-based geometry reasoning makes sense for proteins.

## Setup

If you want to experiment locally, a lightweight Python environment is enough for the tensor-level ideas in this article.

```bash
brew install uv
mkdir alpha_fold && cd alpha_fold
uv init .
uv add torch numpy matplotlib jupyterlab
```

This is enough to prototype shapes, attention maps, distance matrices, and toy structure predictors in notebooks.

---

## What Problem Does AlphaFold Solve?

A protein is a sequence of amino acids:

$$
(a_1, a_2, \dots, a_n)
$$

but its function depends heavily on its **3D structure**, not only its 1D sequence.

The hard problem is:

> Given the amino-acid sequence, predict the final spatial arrangement of atoms.

This is difficult because:

- the number of possible conformations is enormous
- interactions are long-range
- local sequence neighbors are not the only important neighbors in 3D space

AlphaFold's key contribution is that it turns structure prediction into a learning problem over **representations of residues and residue pairs**, then repeatedly refines those representations using attention-style updates.

---

## Why PyTorch Is a Good Learning Tool

Before thinking about proteins, it helps to think in terms of tensors.

In PyTorch, a tensor is simply a multidimensional array with efficient numerical operations and automatic differentiation:

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.shape)  # torch.Size([2, 2])
```

AlphaFold is fundamentally a tensor program:

- sequence features become tensors
- pairwise residue relationships become tensors
- attention updates those tensors
- losses compare predicted geometry with known structures

Once you understand the shapes, the architecture becomes much less mysterious.

---

## The Core Representations

A useful simplification is to think of AlphaFold as operating on two main objects.

### 1. Sequence / residue representation

For a protein with `N` residues, we keep a feature vector for each residue:

$$
R \in \mathbb{R}^{N \times d_r}
$$

This stores per-residue information such as:

- amino-acid identity
- evolutionary context
- learned hidden features

### 2. Pair representation

We also keep a feature vector for each residue pair `(i, j)`:

$$
P \in \mathbb{R}^{N \times N \times d_p}
$$

This is the crucial idea. Protein geometry is relational, so we need a representation for questions like:

- are residues `i` and `j` likely to be close?
- what orientation might they have?
- does evidence suggest a contact or constraint?

In many ways, the pair tensor acts like a learned geometric memory.

---

## Why Pairwise Reasoning Matters

A protein chain is local in sequence but not local in space.

Residues that are far apart in index may become neighbors after folding. So a model that only looks at nearby tokens in 1D will miss important interactions.

This is where AlphaFold departs from simpler sequence models. It explicitly models **residue-residue interactions** rather than hoping they emerge implicitly.

If we define a crude distance matrix from coordinates:

$$
D_{ij} = \|x_i - x_j\|_2
$$

then the folding problem is deeply related to recovering a geometrically consistent version of this pairwise structure.

---

## Evolutionary Information and MSA Intuition

One major source of signal in protein prediction is the **multiple sequence alignment (MSA)**.

The intuition is:

- related proteins evolve over time
- positions that mutate together often reflect structural dependence
- co-evolution gives clues about spatial constraints

You do not need to implement a full MSA pipeline to understand the idea. Conceptually, the MSA gives the model a matrix like:

$$
M \in \mathbb{R}^{S \times N \times d_m}
$$

where:

- `S` = number of aligned sequences
- `N` = protein length
- `d_m` = feature dimension

AlphaFold uses attention to move information:

- across positions inside a sequence
- across related sequences at the same position
- from MSA features into pair features

This lets the network infer which residues likely constrain each other.

---

## Attention, but for Structure

Attention is useful because it lets every element selectively gather information from other elements.

The standard scaled dot-product attention is:

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

In NLP, tokens attend to other tokens. In protein modeling, residues or MSA positions attend to other residues or aligned positions.

A minimal PyTorch example:

```python
import math
import torch

def attention(x):
    q = x @ Wq
    k = x @ Wk
    v = x @ Wv
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.size(-1))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v
```

The important idea is not the code itself. It is that attention gives the model a mechanism for learning:

- long-range dependencies
- context-dependent interactions
- relational patterns that are hard to encode by hand

That is exactly what protein folding needs.

---

## A Toy Residue-Pair Pipeline in PyTorch

A simple educational prototype is:

1. embed the amino-acid sequence
2. build pair features by combining residue features
3. refine pair features with attention or MLP blocks
4. predict distances or contacts

For example:

```python
import torch
import torch.nn as nn

N = 128
d = 64

seq = torch.randint(0, 20, (N,))
embed = nn.Embedding(20, d)
r = embed(seq)                      # [N, d]

ri = r.unsqueeze(1).expand(N, N, d)
rj = r.unsqueeze(0).expand(N, N, d)
pair = torch.cat([ri, rj], dim=-1)  # [N, N, 2d]

pair_mlp = nn.Sequential(
    nn.Linear(2 * d, d),
    nn.ReLU(),
    nn.Linear(d, d),
)

pair_features = pair_mlp(pair)      # [N, N, d]
dist_head = nn.Linear(d, 1)
pred_dist = dist_head(pair_features).squeeze(-1)  # [N, N]
```

This is still far from AlphaFold, but it already captures a central idea:

> structure prediction benefits from explicitly learning a feature for each residue pair

---

## Geometry Is More Than Distances

Distances alone are not enough. A valid protein structure must satisfy consistent geometry:

- bond lengths
- bond angles
- torsion angles
- rigid-body constraints

That is why modern structure predictors go beyond contact maps. They learn richer geometric signals and refine them iteratively until the output becomes physically plausible.

One intuition is to think of the model as repeatedly asking:

- what should residue `i` know about residue `j`?
- what geometric relation is plausible?
- is the current global structure self-consistent?

---

## From Hidden States to Coordinates

A useful abstraction is:

$$
(\text{sequence features}, \text{pair features}) \rightarrow \text{geometric reasoning} \rightarrow \text{3D coordinates}
$$

The final output can be represented as coordinates:

$$
X \in \mathbb{R}^{N \times 3}
$$

for backbone atoms or residue-level positions, depending on the level of simplification.

During training, predicted structures are compared with target structures using geometry-aware losses. In a toy setting, you might start with a distance loss such as:

$$
\mathcal{L}_{dist} = \frac{1}{N^2}\sum_{i,j}\left(\hat{D}_{ij} - D_{ij}\right)^2
$$

where $\hat{D}_{ij}$ is the predicted distance and $D_{ij}$ is the ground truth.

Real systems use richer objectives, but this simplified loss is enough to understand how a model can learn geometry from supervision.

---

## What Makes AlphaFold Different from a Plain Transformer?

A plain transformer over the amino-acid sequence is not enough because folding is not just sequence modeling.

AlphaFold-like systems are distinctive because they:

- maintain **pair representations**
- incorporate **evolutionary evidence**
- use **geometry-aware refinement**
- enforce stronger **structural consistency**

So the architecture is not merely "attention on proteins". It is an architecture designed specifically for **relational geometry**.

---

## A Good Learning Path

If you want to study AlphaFold seriously, the order below is practical:

1. understand PyTorch tensors and broadcasting
2. learn attention thoroughly
3. study pairwise feature construction
4. implement contact-map or distance-matrix prediction
5. add simple geometric losses
6. only then read the full AlphaFold architecture

This order matters. Without strong intuition for tensors and attention, the full model is easy to memorize but hard to really understand.

---

## Final Intuition

The most useful mental model is this:

> AlphaFold is a learned system for turning sequence information and evolutionary constraints into a consistent geometric representation of a protein.

PyTorch helps because it makes the machinery concrete:

- residue embeddings are tensors
- pair relationships are tensors
- attention updates those tensors
- losses push them toward valid structure

Once you see the problem through that lens, AlphaFold stops looking like magic and starts looking like a very sophisticated, geometry-aware deep learning system.
