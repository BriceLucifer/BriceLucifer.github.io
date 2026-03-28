---
title: "Build a GPT-2 from scratch"
date: 2026-03-29
tags: ["MachineLearning", "DeepNeuralNetwork", "Mathematics", "LLM"]
categories: ["posts"]
description: "How to build a gpt2"
summary: "by build a gpt2 to show all the detail of how to train a llm"
math: true
mermaid: true
ShowToc: true
TocOpen: true
draft: false
---
# Build a GPT-2 from scratch
## Architecture
The architecture can be found online, you only need to know how it looks like, that is enough, and the equation of Q,K,V (Query, Key, Value)
## dealing with raw data
When we dealing the data, most of them is text, but since we are build a modal, we have to make it as number. If you are CS guy, you got know the ASCII table. This is a table that you can conver number to text and other way. Same idea here, we need to make the text into vector space, we call it *Embedding*, the modal is basicly a function with another fucntion which the input is vectors. We use *Embedding* to convert the text to the vectors, so the modal can compute. The vector has dimension, can leat to 1000 and even more. Thanks to gpt2's parameter is not very large, we only need it for 768.
Here is a new question, what kind of format we put into the Embedding Layer, a char or a word? Thanks to data compression, there is a algorithm called *BPE(Byte Pair Encoding)*, we can make it tokenized. One token got one ID like a hashtable or hashmap, in python we call it dict. We use a tokenizer, so far we only need to know this is cool.
In this section, we are going to use **tiktoken**, so use your pip or uv to install this package.
## masked attention mechanism
The attention mechanism is based on matrix dotproduct.
## build gpt from the begining
## traing on a unlabled dataset
## some finetuning of classification
## listen to human by finetuning