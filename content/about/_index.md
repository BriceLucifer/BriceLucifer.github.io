+++
title = "About"
date = 2025-10-08T15:13:34+08:00
draft = false
url = "/about/"
description = "About Brice — performance engineering, ML compilers, and the hardware–software boundary; previously a research assistant on generative video world models at the University of Auckland. C/C++, Rust, Zig, Neovim, and rebuilding things from scratch."
keywords = ["Performance Engineering", "ML Compilers", "Hardware-Software Co-design", "Systems", "C/C++", "Rust", "Zig", "Neovim", "World Models", "University of Auckland", "Tencent Cloud", "Brice"]

# 👇 PaperMod 专用
showToc = false
hideMeta = true
+++

<img class="about-avatar" src="/images/avatar-320.webp" alt="Brice" width="140" decoding="async" />

Hi, I’m **Brice** — an engineer who spent a while in AI research and came back to where software meets hardware.

```lua
-- ~/.config/brice/init.lua
local brice = {
  focus    = "performance · ML compilers · hardware <-> software",
  building = "mlcc, an ML compiler from scratch (C++23)",
  study    = "Master of AI, UoA (2026 – 2027)",
  previous = "research assistant · generative video world models",
  editor   = "neovim 0.12 · 17 plugins · ~30 ms startup",
  langs    = { "c", "c++", "rust", "zig", "python", "go", "typescript" },
  motto    = "rebuild it until it makes sense",
}

return brice
```

These days I work on **performance** and the layer where **software meets hardware**: why a program is fast or slow, what the memory hierarchy is doing to it, and how a compiler turns a model into code a CPU or GPU can actually run well. I’m doing a **Master of Artificial Intelligence** at the University of Auckland alongside it. The rest of the time it looks much the same: I take something I use every day — autograd, a tokenizer, an inference engine, a coding agent, my own editor — and rebuild a version small enough to fit in my head.

---

## Now

**Performance, compilers, and the hardware–software boundary.** The questions I care about now sit between the model and the machine: how a tensor program becomes kernels, which ones to fuse, how data moves through caches and memory, and what a piece of hardware is actually good at.

The main project is **mlcc**, an ML compiler written from scratch in C++23 that follows a semester-long syllabus from elementwise programs to training an LLM on GPUs. Week one is a hash-consed UOp IR with strict dtype checking; the progress log starts at [From-scratch ML Compiler, Part 1](/posts/mlcc-01-uop-and-hash-consing/).

---

## Previously: Video World Models

**Research Assistant — University of Auckland**  
*with [Xinyu Zhang](https://zhangxinyu-xyz.github.io/) · Mar – Nov 2026*

Most video models are good at a few convincing seconds. I worked on what it takes for one to hold a *world* together — to keep a room the same room when the camera comes back, let things keep happening off-screen, and respond when someone acts inside it. That’s the direction behind work like **LiveWorld**, and I broke it into five problems:

| Problem | The question it asks |
|---------|----------------------|
| **Long-horizon generation** | Does it stay coherent long after the first few seconds? |
| **Static memory** | When we look again, are the layout, objects, and geometry still right? |
| **Dynamic memory** | Did what moved out of view keep changing the way it should? |
| **Action control** | Can generation be steered by actions, motion, or latent controls? |
| **Interactivity** | Can a user or agent change the world — and see the consequences? |

I read papers problem-first: for each one, *which of these does it actually move forward?* [**Awesome Interactive World Model**](https://github.com/EasonTuT/Awesome-Interactive-World-Model) organizes the literature the same way — plus post-training, physics, and evaluation — and it’s where I’d point anyone starting out. Contributions and pull requests are welcome.

Underneath it was the math I still come back to: **measure-theoretic probability, stochastic processes, and SDEs** — the ground diffusion models stand on.

---

## Projects

> I don’t quite trust that I understand something until I’ve written a small version of it.

### Learning machines, from first principles

- [**MiniTorch**](https://github.com/BriceLucifer/MiniTorch/tree/main/MiniTorch) · `Python` — a PyTorch-shaped framework in plain NumPy: reverse-mode autodiff, NN modules, SGD/Adam, MNIST training, computation-graph visualization, and tests. My answer to *“what does `.backward()` actually do?”*
- [**micro_grad**](https://github.com/BriceLucifer/micro_grad) · `Rust` — Karpathy’s micrograd rebuilt in Rust, where a computation graph has to make peace with the borrow checker.
- [**uma**](https://github.com/BriceLucifer/uma) · `Python` — the same idea again, this time on top of Apple’s MLX.
- [**tensor-in-zig**](https://github.com/BriceLucifer/tensor-in-zig) · `Zig` — a tensor library in Zig, with an MLP that learns to read MNIST digits.
- [**min_bpe**](https://github.com/BriceLucifer/min_bpe) · `C++` — a byte-pair-encoding tokenizer, my C++ take on minbpe.
- [**MiniVLLM**](https://github.com/BriceLucifer/MiniVLLM) · `Python` — a small vLLM-style inference engine on PyTorch and Apple’s MPS backend.
- [**GPT2-back2front**](https://github.com/BriceLucifer/GPT2-back2front) · `Python` — a GPT-2 trained, fine-tuned, and served to a web frontend, end to end. The walkthrough is [Build GPT-2 from Scratch](/posts/mai/compsci-714/build_gpt2/).

### Systems & tools

- [**minikv**](https://github.com/BriceLucifer/minikv) · `C++` — a lightweight distributed object store for artifacts, datasets, and data-heavy workflows.
- [**minicoder**](https://github.com/BriceLucifer/minicoder) · `TypeScript` — a small but complete coding agent.
- [**codecrafters-claude-code-rust**](https://github.com/BriceLucifer/codecrafters-claude-code-rust) · `Rust` — a Claude Code-style command-line agent, built through CodeCrafters.
- [**zreal**](https://github.com/BriceLucifer/zreal) · `Zig` — a game engine, because Zig makes you want to own every byte.
- [**shader**](https://github.com/BriceLucifer/shader) · `Rust` — shader-style procedural graphics on the CPU, no GPU allowed; written up in [A CPU-based Whirl Shader](/posts/cowhirl-shader-cpu/).

### Making it visible

- [**atlas**](https://github.com/BriceLucifer/atlas) · `TypeScript` — my knowledge base, an Obsidian-inspired wiki.
- [**alphafold2-visualization**](https://github.com/BriceLucifer/alphafold2-visualization) · `TypeScript` — AlphaFold 2, made interactive.
- [**rnn-transformer-viz**](https://github.com/BriceLucifer/rnn-transformer-viz) · `TypeScript` — RNNs and Transformers you can watch think.
- [**SO-100 / LeRobot reproduction**](https://bricelucifer.github.io/Robot-Docs/so100.html) — a robotics workflow reproduced and documented from zero, as practice in learning something unfamiliar fast.

---

## Tools & Setup

- **Editor** — Neovim 0.12, 17 plugins on the built-in `vim.pack`, ~30 ms startup
- **Theme** — Kanagawa: Wave after dark, Lotus by day, following the system
- **Terminal** — Ghostty · Maple Mono NF CN · Starship
- **Shell** — fish · tmux
- **CLI** — fzf · ripgrep · zoxide · bat · btop · uv
- **Languages** — Rust · C · C++ · Zig · Go · Python · TypeScript

The editor is the thing I’ve tuned the most. [**nvim.setting**](https://github.com/BriceLucifer/nvim.setting) is seventeen single-purpose plugins and eight Lua files on Neovim’s built-in `vim.pack` — no distro, no plugin bundles. The screen has no chin: status lives in a small capsule at each window’s top-right, `:` opens the command line in the middle of the screen, and notifications arrive as little cards and fade away. The start screen (a slowly turning galaxy), the notification cards, the number column, and the input boxes are hand-written Lua rather than plugins. The terminal half lives in [**ghostty_setup**](https://github.com/BriceLucifer/ghostty_setup), and my CV is typeset in Typst with [**coral-cv**](https://github.com/BriceLucifer/coral-cv).

Languages are tools. What I actually care about is **abstraction, correctness, and performance trade-offs**.

---

## Experience

**Research Assistant — University of Auckland** · *Mar – Nov 2026*  
Video generation and generative video world models, with Xinyu Zhang (see [Previously](#previously-video-world-models)).

**System Technology Intern — Tencent Cloud** · *CSIG, Xingxinghai Lab · Jul – Sep 2024*  
Production cloud infrastructure: I automated server deployment pipelines, tightened internal workflows, collected and analysed operational metrics, wrote technical analysis reports, and sat in on distributed-reliability and kernel-adjacent performance work. It’s where *automate it, make it reproducible, debug it carefully* turned into habit.

---

## Education

**Master of Artificial Intelligence — University of Auckland** · *Feb 2026 – Dec 2027 (expected)*  
Course notes on this site: [COMPSCI 713 — AI Fundamentals](/posts/mai/compsci-713/) · [COMPSCI 714 — AI Architecture](/posts/mai/compsci-714/)

**B.Eng. in Electronic Information Engineering — Harbin University of Science and Technology** · *Aug 2021 – Jul 2025 · GPA top 1% of cohort*  
Systems and mathematics in equal measure: embedded architectures, mathematical modelling and numerical optimization, parallel computing, and compilers down to low-level optimization.

---

## Contact

- **GitHub** → [github.com/BriceLucifer](https://github.com/BriceLucifer)
- **X** → [@Bricelucifer](https://x.com/Bricelucifer)
- **Email** → [2376671337@qq.com](mailto:2376671337@qq.com)

<sub>Built with **Hugo + PaperMod**.</sub>
