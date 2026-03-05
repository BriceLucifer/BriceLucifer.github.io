---
title: "ALPHA FOLD explain"
date: 2026-03-05
tags: ["MachineLearning", "DeepNeuralNetWork"]
categories: ["posts"]
description: "Using Pytorch to learning ALPHA-FOLD"
summary: "Build ALPHA-FOLD from scratch"
---------------------
> For great start, we should use some good tool for manuage the python and virtual environment
We will be using *UV*. On your MacBook, we use a simple command line.
```bash
brew install uv
```
Then we make a new directory for contain our sourcecode.
```bash
mkdir alpha_fold
cd alpha_fold
```
We init our directory and add a few packages to make our own modal.
```bash
# in the same directory
uv init .
uv add numpy matplotlib torch torchvison
```
From now on, we can actually focus on the building process without worry python virtual environment

# Introduction
In this Introduction, I will give you some fundimental concept and keywords just to get used to machine learning framework and how we combine with mathmatical theory.

## Tensor
What is a Tensor?
This can be different based on different backgroud. In the field of AI, we use *Tensor* as a continous memory of the data. We use it to store the numerical data and compute with it.

Here in Pytorch framework, we can easily create one.
```python
a: Tensor = torch.Tensor([1, 2])
```
Tensor has different dimensions, with can be used as a features of the data, since the protein structure is a 3d structure.

Tensor operation obey the rule of matrix operation and vector operation
1. Indexing and Slicing
2. Operation
3. broadcasting
4. auto-derivatitive

## Machine Learning
Machine Learning is about learning the pattern

1. Similarity Estimation