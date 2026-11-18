+++
title = "About"
date = 2025-10-08T15:13:34+08:00
draft = false
url = "/about/"
description = "About Brice — Research Assistant at the University of Auckland working on generative video world models, plus math and systems."
keywords = ["AI", "Research", "Video Generation", "World Model", "Rust", "C/C++","Quant", "Systems", "University of Auckland", "Tencent Cloud", "Brice"]

# 👇 PaperMod 专用
showToc = false
hideMeta = true
mermaid = true
+++

<img class="about-avatar" src="/images/avatar-320.webp" alt="Brice" width="140" decoding="async" />

Hi, I’m **Brice** — a Research Assistant at the **University of Auckland**, working on **generative video world models**: video models that don’t stop at a plausible clip, but keep a persistent, controllable world running over time.

I’m also studying for a **Master of Artificial Intelligence** at UoA, with the goal of continuing to a PhD in this direction.

The questions I keep coming back to:

- How can a generative model keep objects, scenes, and events consistent over long horizons?
- How should memory work when important state moves out of view?
- How can action and interaction turn video generation into world simulation?

Underneath them sit two habits: going down to the **mathematics** (probability, stochastic processes, optimization), and **building things from scratch** until I understand why they work.

---

## 🔬 Research

**Research Assistant — University of Auckland**  
*Supervisor: [Xinyu Zhang](https://zhangxinyu-xyz.github.io/) · 21 Mar 2026 – Present*

Video generation and generative video world models, including directions related to **LiveWorld**. The aim is to move video models from short-form generation toward long-horizon simulation. I break that into five problems:

- **Long-horizon generation** — temporal consistency well beyond short clips
- **Static memory consistency** — scene layout, object identity, spatial structure
- **Dynamic memory consistency** — out-of-view events, state changes, hidden dynamics
- **Action control** — conditioning generation on actions, motion, or latent behaviour
- **Interactivity** — letting users or agents change the generated world

I read the field problem-first rather than model-first: for each paper, which of these problems does it actually solve? [**Awesome Interactive World Model**](https://github.com/EasonTuT/Awesome-Interactive-World-Model) organizes the literature along the same axes — plus post-training, physics, and evaluation — and is a good entry point. Contributions and pull requests are welcome.

---

## 🎓 Education

**Master of Artificial Intelligence (180 points)**  
*University of Auckland · Feb 2026 – Dec 2027 (expected)*

Course notes on this site: [COMPSCI 713 — AI Fundamentals](/posts/mai/compsci-713/) · [COMPSCI 714 — AI Architecture](/posts/mai/compsci-714/)

**B.Eng. in Electronic Information Engineering**  
*Harbin University of Science and Technology · Aug 2021 – Jul 2025 · GPA top 1% of cohort*

Systems and mathematics in equal measure: computer systems and embedded architectures, mathematical modelling and numerical optimization, high-performance and parallel computing, compiler theory and low-level optimization.

---

## 💼 Industry

**System Technology Intern — Tencent Cloud**  
*CSIG · Xingxinghai Lab · Jul 2024 – Sep 2024*

- Automated **cloud server deployment pipelines** for production environments
- Improved internal **system workflows** for efficiency and reliability
- Collected and analysed **operational metrics** to drive optimization
- Wrote **technical analysis reports**, and took part in discussions on **distributed-system reliability** and **kernel-adjacent performance tuning**

This is where I picked up the habits I now bring to research: automate it, make it reproducible, debug it carefully.

---

## 🧩 Projects

| Project | What | Keywords |
|---------|------|----------|
| [**MiniTorch**](https://github.com/BriceLucifer/MiniTorch/tree/main/MiniTorch) | A PyTorch-style deep learning framework built from scratch in NumPy: reverse-mode autodiff, differentiable tensor ops, NN modules, SGD/Adam, MNIST training, computation-graph visualization, and unit tests. | Autograd · DL Systems · NumPy |
| [**GPT-2 from Scratch**](/posts/mai/compsci-714/build_gpt2/) | BPE tokenization, masked self-attention, transformer blocks, and training — implemented and explained step by step. | LLM · Transformer · PyTorch |
| [**SO-100 / LeRobot Reproduction**](https://bricelucifer.github.io/Robot-Docs/so100.html) | Independently reproduced and documented an unfamiliar robotics workflow — practice in fast self-learning, setup, debugging, and documentation. | Robotics · Reproduction · Docs |
| [**CPU Whirl Shader**](/posts/cowhirl-shader-cpu/) | Shader-style procedural graphics recreated on the CPU in Rust, with raymarching-inspired math and video output. | Rust · Graphics |

Alongside these, I’m working through **measure-theoretic probability and stochastic processes** (Markov chains, SDEs, probabilistic graphical models) as the foundation under diffusion-based generative models.

---

## 🌲 Skill Tree

<div class="mermaid">
mindmap
  root((Brice))
    AI & Generative
      Video Generation
        Diffusion Models
        Temporal Modeling
        Long-Horizon Generation
      World Models
        Static Memory Consistency
        Dynamic Memory Consistency
        Action Control
        Interactive Ability
      LLM & Transformers
        Attention Mechanisms
        Generative Architectures
      Deep Learning Systems
        Autograd
        Backpropagation
        Optimizers
        Training Visualization
    Mathematics
      Probability Theory
        Measure Theory
        Stochastic Processes
        SDEs & Diffusion
      Optimization
        Gradient Methods
        Convex Analysis
        Numerical Methods
      Linear Algebra
        Matrix Decompositions
        Spectral Methods
    Systems
      Rust
        Async Runtimes
        Memory Safety
        Concurrency
      C / C++
        Low-level Perf
        Embedded
      Cloud & Infra
        Distributed Systems
        Pipeline Automation
    Algorithms
      Graph Theory
      Data Structures
      Compiler Theory
      Complexity Analysis
</div>

**Languages:** `C` · `C++` · `Rust` · `Zig` · `Go` · `Python` · `Java` · `JavaScript` · `Swift`

> Languages are **tools** — what matters is abstraction, correctness, and performance trade-offs.

---

## 🌐 Contact

- **GitHub** → [github.com/BriceLucifer](https://github.com/BriceLucifer)
- **X (Twitter)** → [@Bricelucifer](https://x.com/Bricelucifer)
- **Email** → [2376671337@qq.com](mailto:2376671337@qq.com)
- **Knowledge Base** → [atlas-32q.pages.dev](https://atlas-32q.pages.dev/)

---

<sub>This site is built with **Hugo + PaperMod** and documents an evolving research trajectory in **video generation and interactive world models**.</sub>
