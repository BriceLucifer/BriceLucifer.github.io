---
title: "从零写 GEMM（二）：分块为什么一开始没用"
date: 2026-11-26T11:00:00+13:00
tags: ["Performance", "Hardware", "Memory"]
series: ["GEMM from Scratch"]
keywords: ["GEMM", "Cache Blocking", "Tiling", "Register Blocking", "Roofline", "Arithmetic Intensity", "NEON", "Apple Silicon", "M4 Pro", "TLB"]
description: "Cache blocking made the i-k-j matrix multiply slower on an M4 Pro. Measuring peak FLOPS, memory bandwidth and the inner loop in L1 shows why: the bottleneck was loading and storing C, not memory. Register blocking reaches 104 GFLOPS, collapses on large matrices, and only then does cache blocking pay off."
summary: "教科书说矩阵乘法的下一步是分块。我照做了，结果变慢了。这篇先测出 M4 Pro 单核的计算峰值和内存带宽，再把内层循环放进 L1 单独测，找到真正的瓶颈：不是内存，而是每次乘加都要读写一次 C。寄存器分块把它推到 104 GFLOPS，但矩阵一大就崩到 16；直到这时，缓存分块才第一次真正派上用场。"
---

{{< series >}}

> **这篇你会搞清楚**
> 1. 怎么测出一台机器的两个天花板：计算峰值和内存带宽，以及"计算强度"把它们连起来的方式；
> 2. 为什么给 ikj 加缓存分块反而变慢了，真正的瓶颈在哪；
> 3. 寄存器分块是什么，为什么它一下子快了 3 倍；
> 4. 为什么寄存器分块在大矩阵上会崩，而缓存分块正好在这时候救场。

[第一篇](/posts/matmul-loop-order-and-layout/)结束时，最快的是 ikj 循环顺序：单线程约 **33 GFLOPS**。几乎所有讲矩阵乘法优化的资料都说，下一步是**分块**（blocking，也叫 tiling）：把矩阵切成小块，让数据留在缓存里被反复使用。

我照着做了，结果**变慢了**。这篇就从这个反常的结果出发。

测试环境和第一篇相同：Apple M4 Pro，clang 23，`-O3 -mcpu=apple-m4`，`float32`，行优先存储。所有结果都和 Accelerate 的 `cblas_sgemm` 逐元素比对过。完整代码在文末。

## 一、先量出两个天花板

优化之前，先要知道"最快能有多快"。一台机器有两个天花板：**每秒能做多少次浮点运算**，和**每秒能从内存搬多少字节**。

### 计算峰值：单核 125 GFLOPS

M4 的 NEON 寄存器是 128 位，一条 `fmla.4s` 同时做 4 次乘加，也就是 8 次浮点运算。要测峰值，就让 CPU 同时推进很多条**互不依赖**的 `fmla` 链：

```c
for (long it = 0; it < iters; it++)
    for (int i = 0; i < 16; i++) a[i] = vfmaq_f32(a[i], b, c);   // 16 条独立的链
```

| 测法 | 结果 |
|---|---|
| 16 条独立链（吞吐） | **125 GFLOPS**，即每纳秒约 15.6 条 `fmla.4s` |
| 1 条依赖链（延迟） | 每条 0.8 ns，只有 10 GFLOPS |

两个数字一除：$0.8 \times 15.6 \approx 12.5$。也就是说，**至少要有 12～13 条互不依赖的乘加链同时在跑，才能吃满一个核心**。记住这个数，它会决定第三篇里微内核的形状。（这也是[第一篇](/posts/matmul-loop-order-and-layout/)里 `ijk_col` 只有 5 GFLOPS 的原因：它只有一条加法链。）

### 内存带宽：单核 130 GB/s，全部核心约 229 GB/s

用类似 STREAM 的 triad（`a[i] = b[i] + 3*c[i]`，三个 256 MB 的数组）测带宽：

| 线程数 | 带宽 |
|---|---|
| 1 | 130 GB/s |
| 4 | 224 GB/s |
| 10 | 229 GB/s |
| 14 | 217 GB/s |

一个核心就能吃掉一半以上的带宽，4 个核心之后就饱和了。

### 计算强度：把两个天花板连起来

一段代码每从内存读写 1 个字节，做多少次浮点运算，叫它的**计算强度**（arithmetic intensity，单位 flop/字节）。它能跑到的速度上限是：

$$
\text{性能} \le \min\left(\text{计算峰值},\ \text{计算强度} \times \text{带宽}\right)
$$

这就是 roofline 模型。两个天花板的交点叫**拐点**：

- 单核：$125 / 130 \approx 1$ flop/字节
- 10 个性能核：$1250 / 229 \approx 5.5$ flop/字节

计算强度低于拐点，就是**受内存限制**，再怎么优化计算也没用；高于拐点，就是**受计算限制**。分块的作用正是**提高计算强度**：同一份数据在缓存里被用很多次，从内存搬的字节就少了。

但这里有个前提：**你得先真的受内存限制**。

## 二、实验一：给 ikj 加缓存分块

ikj 的问题看起来很明显：对每一行 i，整个 B 都要从头到尾扫一遍。n = 4096 时 B 有 64 MB，远大于 16 MB 的 L2。分块的思路是：把 B 切成 KB×NB 的小块，让一个小块被所有行用完再换下一块：

```c
for (int jj = 0; jj < n; jj += NB)
    for (int kk = 0; kk < n; kk += KB)          // B 的一个 KB x NB 小块
        for (int i = i0; i < i1; i++)            // 被每一行 i 复用
            for (int k = kk; k < kk + KB; k++) {
                float a = A[i * n + k];
                for (int j = jj; j < jj + NB; j++) C[i * n + j] += a * B[k * n + j];
            }
```

结果（KB = 256，NB = 512）：

| n | 线程 | ikj | ikj + 缓存分块 |
|---|---|---|---|
| 1024 | 1 | 34.4 | 29.2 |
| 2048 | 1 | 33.7 | 28.4 |
| 4096 | 1 | 32.4 | 27.4 |
| 2048 | 10 | 296 | 244 |
| 4096 | 10 | 262 | 231 |

（单位 GFLOPS。）**每一种情况都变慢了**，单线程和多线程都一样。换块大小也没用（n = 2048，单线程）：

| | NB = 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| KB = 64 | 21.6 | 26.7 | 26.4 | 29.2 | 32.7 |
| KB = 256 | 21.7 | 26.7 | 27.1 | 29.4 | 32.9 |
| KB = 512 | 22.0 | 27.4 | 27.6 | 29.7 | 33.0 |

KB 怎么取都差不多；NB 越大越快，最快的 NB = 2048 其实就是**不分块**。分块唯一的效果，是把内层循环切短、多了循环开销。

## 三、真正的瓶颈：每次乘加都要读写一次 C

既然分块减少了内存访问却没有变快，那瓶颈就不在内存。验证办法很直接：把 ikj 的内层循环 `C[j] += a * B[j]` 放到两个只有 4 KB 的数组上反复跑，所有数据永远在 L1 里，**完全没有内存访问**：

```c
for (int j = 0; j < n; j++) c[j] += a * b[j];   // b、c 各 4 KB，常驻 L1
```

结果：**40.6 GFLOPS**。这就是这个内层循环本身的天花板，和矩阵多大、缓存多大都无关。ikj 实测 34，已经达到它自己天花板的 **83%**；单核峰值是 125，它连三分之一都不到。

原因在于每一条 `fmla.4s` 都要伴随三次访存：读 4 个 B、读 4 个 C、写回 4 个 C。CPU 的读写端口先被占满了，乘加单元反而在空转：

<svg viewBox="0 0 680 136" role="img" aria-label="每条 fmla.4s 伴随几次访存指令（越短越好）" style="width:100%;height:auto;display:block;margin:1.5rem auto" xmlns="http://www.w3.org/2000/svg"><style>.mm-t{fill:var(--primary);font-family:var(--font-mono);font-size:12px}.mm-s{fill:var(--secondary);font-family:var(--font-mono);font-size:11px}.mm-h{fill:var(--primary);font-family:var(--font-ui);font-size:13px;font-weight:600}.mm-cell{stroke:var(--math-rule);stroke-width:1;fill:transparent}.mm-good{fill:#40a8a0;fill-opacity:.35;stroke:#40a8a0}.mm-bad{fill:#e5864b;fill-opacity:.35;stroke:#e5864b}.mm-one{fill:var(--math-accent);fill-opacity:.35;stroke:var(--math-accent)}.mm-reg{fill:var(--math-accent);fill-opacity:.18;stroke:var(--math-accent)}.mm-p0{fill:#40a8a0;fill-opacity:.38;stroke:#40a8a0}.mm-p1{fill:#9d7cd8;fill-opacity:.38;stroke:#9d7cd8}.mm-arrow{stroke:var(--primary);stroke-width:1.6;fill:none;marker-end:url(#gm-ah)}.mm-axis{fill:none;stroke:var(--math-rule);stroke-width:1}.mm-dash{fill:none;stroke:var(--secondary);stroke-width:1;stroke-dasharray:4 3}.mm-bar{fill:var(--math-accent);fill-opacity:.75}.mm-bar2{fill:#40a8a0;fill-opacity:.75}.mm-bar3{fill:#e5864b;fill-opacity:.75}.mm-bar4{fill:var(--secondary);fill-opacity:.45}.mm-box{fill:var(--math-accent);fill-opacity:.05;stroke:var(--math-rule);stroke-width:1.2}</style><text class="mm-h" x="16" y="18">每条 fmla.4s 伴随几次访存指令（越短越好）</text><text class="mm-t" x="162" y="45" text-anchor="end">ikj 内层循环</text><rect class="mm-bar3" x="170" y="32" width="403.1" height="18" rx="2"/><text class="mm-t" x="579.1" y="45">3</text><text class="mm-t" x="162" y="71" text-anchor="end">4×16 寄存器块</text><rect class="mm-bar" x="170" y="58" width="67.2" height="18" rx="2"/><text class="mm-t" x="243.2" y="71">0.5</text><text class="mm-t" x="162" y="97" text-anchor="end">8×12 微内核（第三篇）</text><rect class="mm-bar2" x="170" y="84" width="28.2" height="18" rx="2"/><text class="mm-t" x="204.2" y="97">0.21</text><path class="mm-axis" d="M170 28 V106"/><text class="mm-s" x="16" y="124">ikj：读 B、读 C、写 C；4×16：每个 k 读 4 个 B 向量 + 4 个 A 值，做 16 条 fmla；8×12：读 5 个向量，做 24 条</text></svg>

所以要变快，必须**减少每次乘加对应的访存次数**。缓存分块做不到这一点，它只能决定数据来自哪一级缓存；访存指令的条数一条都没少。

> **教训**：先找到瓶颈，再选优化手段。分块是解决"内存带宽不够"的工具；如果瓶颈不在这里，它只会增加开销。

## 四、实验二：寄存器分块

ikj 每次都把 C 读进来、加一下、再写回去，因为它一次只处理 C 的一个元素。如果一次处理 C 的一个**小块**，并且让这个小块在整个 k 循环里一直待在**寄存器**里，C 的读写就只在最开始和最后各发生一次。

具体做法：取 C 的 4 行 × 16 列，对每个 k，读 A 的一小列（4 个数）和 B 的一小行（16 个数），做一次**外积**，累加到 4×16 的块上：

<svg viewBox="0 0 620 230" role="img" aria-label="4×16 寄存器块的外积更新" style="width:100%;height:auto;display:block;margin:1.5rem auto" xmlns="http://www.w3.org/2000/svg"><style>.mm-t{fill:var(--primary);font-family:var(--font-mono);font-size:12px}.mm-s{fill:var(--secondary);font-family:var(--font-mono);font-size:11px}.mm-h{fill:var(--primary);font-family:var(--font-ui);font-size:13px;font-weight:600}.mm-cell{stroke:var(--math-rule);stroke-width:1;fill:transparent}.mm-good{fill:#40a8a0;fill-opacity:.35;stroke:#40a8a0}.mm-bad{fill:#e5864b;fill-opacity:.35;stroke:#e5864b}.mm-one{fill:var(--math-accent);fill-opacity:.35;stroke:var(--math-accent)}.mm-reg{fill:var(--math-accent);fill-opacity:.18;stroke:var(--math-accent)}.mm-p0{fill:#40a8a0;fill-opacity:.38;stroke:#40a8a0}.mm-p1{fill:#9d7cd8;fill-opacity:.38;stroke:#9d7cd8}.mm-arrow{stroke:var(--primary);stroke-width:1.6;fill:none;marker-end:url(#gm-ah)}.mm-axis{fill:none;stroke:var(--math-rule);stroke-width:1}.mm-dash{fill:none;stroke:var(--secondary);stroke-width:1;stroke-dasharray:4 3}.mm-bar{fill:var(--math-accent);fill-opacity:.75}.mm-bar2{fill:#40a8a0;fill-opacity:.75}.mm-bar3{fill:#e5864b;fill-opacity:.75}.mm-bar4{fill:var(--secondary);fill-opacity:.45}.mm-box{fill:var(--math-accent);fill-opacity:.05;stroke:var(--math-rule);stroke-width:1.2}</style><text class="mm-h" x="16" y="20">寄存器分块：每个 k 做一次 4×16 的外积</text><rect class="mm-good" x="110" y="40" width="22" height="22"/><rect class="mm-good" x="132" y="40" width="22" height="22"/><rect class="mm-good" x="154" y="40" width="22" height="22"/><rect class="mm-good" x="176" y="40" width="22" height="22"/><rect class="mm-good" x="198" y="40" width="22" height="22"/><rect class="mm-good" x="220" y="40" width="22" height="22"/><rect class="mm-good" x="242" y="40" width="22" height="22"/><rect class="mm-good" x="264" y="40" width="22" height="22"/><rect class="mm-good" x="286" y="40" width="22" height="22"/><rect class="mm-good" x="308" y="40" width="22" height="22"/><rect class="mm-good" x="330" y="40" width="22" height="22"/><rect class="mm-good" x="352" y="40" width="22" height="22"/><rect class="mm-good" x="374" y="40" width="22" height="22"/><rect class="mm-good" x="396" y="40" width="22" height="22"/><rect class="mm-good" x="418" y="40" width="22" height="22"/><rect class="mm-good" x="440" y="40" width="22" height="22"/><text class="mm-s" x="110" y="32">B[k][j .. j+15]：4 次向量读取（4 个 float32x4）</text><rect class="mm-one" x="40" y="96" width="22" height="22"/><rect class="mm-one" x="40" y="118" width="22" height="22"/><rect class="mm-one" x="40" y="140" width="22" height="22"/><rect class="mm-one" x="40" y="162" width="22" height="22"/><text class="mm-s" x="16" y="88">A[i..i+3][k]</text><text class="mm-s" x="16" y="200">4 次标量读取</text><rect class="mm-reg" x="110" y="96" width="22" height="22"/><rect class="mm-reg" x="132" y="96" width="22" height="22"/><rect class="mm-reg" x="154" y="96" width="22" height="22"/><rect class="mm-reg" x="176" y="96" width="22" height="22"/><rect class="mm-reg" x="198" y="96" width="22" height="22"/><rect class="mm-reg" x="220" y="96" width="22" height="22"/><rect class="mm-reg" x="242" y="96" width="22" height="22"/><rect class="mm-reg" x="264" y="96" width="22" height="22"/><rect class="mm-reg" x="286" y="96" width="22" height="22"/><rect class="mm-reg" x="308" y="96" width="22" height="22"/><rect class="mm-reg" x="330" y="96" width="22" height="22"/><rect class="mm-reg" x="352" y="96" width="22" height="22"/><rect class="mm-reg" x="374" y="96" width="22" height="22"/><rect class="mm-reg" x="396" y="96" width="22" height="22"/><rect class="mm-reg" x="418" y="96" width="22" height="22"/><rect class="mm-reg" x="440" y="96" width="22" height="22"/><rect class="mm-reg" x="110" y="118" width="22" height="22"/><rect class="mm-reg" x="132" y="118" width="22" height="22"/><rect class="mm-reg" x="154" y="118" width="22" height="22"/><rect class="mm-reg" x="176" y="118" width="22" height="22"/><rect class="mm-reg" x="198" y="118" width="22" height="22"/><rect class="mm-reg" x="220" y="118" width="22" height="22"/><rect class="mm-reg" x="242" y="118" width="22" height="22"/><rect class="mm-reg" x="264" y="118" width="22" height="22"/><rect class="mm-reg" x="286" y="118" width="22" height="22"/><rect class="mm-reg" x="308" y="118" width="22" height="22"/><rect class="mm-reg" x="330" y="118" width="22" height="22"/><rect class="mm-reg" x="352" y="118" width="22" height="22"/><rect class="mm-reg" x="374" y="118" width="22" height="22"/><rect class="mm-reg" x="396" y="118" width="22" height="22"/><rect class="mm-reg" x="418" y="118" width="22" height="22"/><rect class="mm-reg" x="440" y="118" width="22" height="22"/><rect class="mm-reg" x="110" y="140" width="22" height="22"/><rect class="mm-reg" x="132" y="140" width="22" height="22"/><rect class="mm-reg" x="154" y="140" width="22" height="22"/><rect class="mm-reg" x="176" y="140" width="22" height="22"/><rect class="mm-reg" x="198" y="140" width="22" height="22"/><rect class="mm-reg" x="220" y="140" width="22" height="22"/><rect class="mm-reg" x="242" y="140" width="22" height="22"/><rect class="mm-reg" x="264" y="140" width="22" height="22"/><rect class="mm-reg" x="286" y="140" width="22" height="22"/><rect class="mm-reg" x="308" y="140" width="22" height="22"/><rect class="mm-reg" x="330" y="140" width="22" height="22"/><rect class="mm-reg" x="352" y="140" width="22" height="22"/><rect class="mm-reg" x="374" y="140" width="22" height="22"/><rect class="mm-reg" x="396" y="140" width="22" height="22"/><rect class="mm-reg" x="418" y="140" width="22" height="22"/><rect class="mm-reg" x="440" y="140" width="22" height="22"/><rect class="mm-reg" x="110" y="162" width="22" height="22"/><rect class="mm-reg" x="132" y="162" width="22" height="22"/><rect class="mm-reg" x="154" y="162" width="22" height="22"/><rect class="mm-reg" x="176" y="162" width="22" height="22"/><rect class="mm-reg" x="198" y="162" width="22" height="22"/><rect class="mm-reg" x="220" y="162" width="22" height="22"/><rect class="mm-reg" x="242" y="162" width="22" height="22"/><rect class="mm-reg" x="264" y="162" width="22" height="22"/><rect class="mm-reg" x="286" y="162" width="22" height="22"/><rect class="mm-reg" x="308" y="162" width="22" height="22"/><rect class="mm-reg" x="330" y="162" width="22" height="22"/><rect class="mm-reg" x="352" y="162" width="22" height="22"/><rect class="mm-reg" x="374" y="162" width="22" height="22"/><rect class="mm-reg" x="396" y="162" width="22" height="22"/><rect class="mm-reg" x="418" y="162" width="22" height="22"/><rect class="mm-reg" x="440" y="162" width="22" height="22"/><path class="mm-axis" d="M198 96 V184" style="stroke:var(--math-accent);stroke-width:1.6"/><path class="mm-axis" d="M286 96 V184" style="stroke:var(--math-accent);stroke-width:1.6"/><path class="mm-axis" d="M374 96 V184" style="stroke:var(--math-accent);stroke-width:1.6"/><text class="mm-s" x="110" y="220">C 的 4×16 小块 = 16 个 NEON 寄存器，整个 k 循环都不离开寄存器</text><text class="mm-t" x="478" y="134">每个 k：</text><text class="mm-t" x="478" y="152">8 次读取</text><text class="mm-t" x="478" y="170">16 条 fmla.4s</text></svg>

$$
C_{[i:i+4],\,[j:j+16]} \mathrel{+}= \sum_k A_{[i:i+4],\,k} \; B_{k,\,[j:j+16]}
$$

```c
// C[i..i+4)[j..j+16) += A[i..i+4)[k0..k1) * B[k0..k1)[j..j+16)
static inline void tile_4x16(int n, const float *A, const float *B, float *C, int i, int j, int k0, int k1) {
    float32x4_t c[4][4];                                    // 16 个累加寄存器
    for (int r = 0; r < 4; r++)
        for (int v = 0; v < 4; v++) c[r][v] = vld1q_f32(&C[(i + r) * n + j + 4 * v]);
    for (int k = k0; k < k1; k++) {
        const float *b = &B[k * n + j];
        float32x4_t b0 = vld1q_f32(b), b1 = vld1q_f32(b + 4), b2 = vld1q_f32(b + 8), b3 = vld1q_f32(b + 12);
        for (int r = 0; r < 4; r++) {
            float a = A[(i + r) * n + k];                   // 一个 A 值喂 16 次乘加
            c[r][0] = vfmaq_n_f32(c[r][0], b0, a);
            c[r][1] = vfmaq_n_f32(c[r][1], b1, a);
            c[r][2] = vfmaq_n_f32(c[r][2], b2, a);
            c[r][3] = vfmaq_n_f32(c[r][3], b3, a);
        }
    }
    for (int r = 0; r < 4; r++)                              // 整个 k 循环结束才写回
        for (int v = 0; v < 4; v++) vst1q_f32(&C[(i + r) * n + j + 4 * v], c[r][v]);
}
```

几个关键点：

- **复用**：每个 B 向量被 4 行各用一次，每个 A 值被 16 列各用一次。每个 k 有 8 次读取、16 条 `fmla.4s`，每条乘加只伴随 0.5 次访存，ikj 是 3 次。
- **独立的链**：16 个累加器互不依赖，超过了第一节算出的"至少 12.5 条"。
- **寄存器预算**：16 个累加器 + 4 个 B 向量 = 20 个，NEON 一共 32 个，放得下。

结果（单线程）：

| n | ikj | 4×16 寄存器块 |
|---|---|---|
| 1024 | 34.4 | **103.8** |

**一步跳到峰值的 83%，是 ikj 的 3 倍。**

## 五、可是矩阵一大就崩了

| n | ikj | 4×16 寄存器块 |
|---|---|---|
| 1024 | 34.4 | 103.8 |
| 2048 | 33.7 | 56.8 |
| 4096 | 32.4 | **16.4** |

n = 4096 时，它比 ikj 还慢一半。问题出在对 B 的访问方式：

1. **每个 4×16 小块都要把 B 的整个 k 方向扫一遍**。对 A 的每 4 行，整个 B（n = 4096 时 64 MB）都要重新读一次，而 L2 只有 16 MB，所以每次都要从 SLC 或内存里重新搬。
2. **而且是跳着读**：每个 k 只读 B 一行里的 16 个 float（64 字节），下一个 k 就跳到下一行。一条 128 字节的缓存行只用了一半。
3. **TLB**：n = 4096 时 B 的一行正好 16 KB，而 macOS 在 Apple silicon 上的内存页也正好是 16 KB（`sysctl hw.pagesize`）。所以沿 k 每走一步就换一个内存页，一个小块要碰 4096 个页。TLB（缓存"虚拟页到物理页"映射的小表）容量有限，页数一多就会频繁失效，每次访存都要先多查一次页表。

前两条让它变成受内存限制，第三条让每次访存还更贵。这时它第一次真正需要缓存分块。

## 六、实验三：现在缓存分块终于有用了

把 B 切成 KC×NC 的块（KC = 256，NC = 512，即 512 KB），让一个块留在 L2 里，被**所有行**的 4×16 小块用完再换下一块：

```c
for (int jc = 0; jc < n; jc += NC)
    for (int pc = 0; pc < n; pc += KC)               // B 的一个 KC x NC 块：512 KB，留在 L2
        for (int i = 0; i < n; i += 4)                // 被每 4 行复用
            for (int j = jc; j < jc + NC; j += 16)
                tile_4x16(n, A, B, C, i, j, pc, pc + KC);
```

| n | 4×16 寄存器块 | 寄存器块 + 缓存分块 |
|---|---|---|
| 1024 | 103.8 | 99.3 |
| 2048 | 56.8 | **93.9** |
| 4096 | 16.4 | **77.7** |

<svg viewBox="0 0 680 162" role="img" aria-label="n = 4096，单线程，GFLOPS" style="width:100%;height:auto;display:block;margin:1.5rem auto" xmlns="http://www.w3.org/2000/svg"><style>.mm-t{fill:var(--primary);font-family:var(--font-mono);font-size:12px}.mm-s{fill:var(--secondary);font-family:var(--font-mono);font-size:11px}.mm-h{fill:var(--primary);font-family:var(--font-ui);font-size:13px;font-weight:600}.mm-cell{stroke:var(--math-rule);stroke-width:1;fill:transparent}.mm-good{fill:#40a8a0;fill-opacity:.35;stroke:#40a8a0}.mm-bad{fill:#e5864b;fill-opacity:.35;stroke:#e5864b}.mm-one{fill:var(--math-accent);fill-opacity:.35;stroke:var(--math-accent)}.mm-reg{fill:var(--math-accent);fill-opacity:.18;stroke:var(--math-accent)}.mm-p0{fill:#40a8a0;fill-opacity:.38;stroke:#40a8a0}.mm-p1{fill:#9d7cd8;fill-opacity:.38;stroke:#9d7cd8}.mm-arrow{stroke:var(--primary);stroke-width:1.6;fill:none;marker-end:url(#gm-ah)}.mm-axis{fill:none;stroke:var(--math-rule);stroke-width:1}.mm-dash{fill:none;stroke:var(--secondary);stroke-width:1;stroke-dasharray:4 3}.mm-bar{fill:var(--math-accent);fill-opacity:.75}.mm-bar2{fill:#40a8a0;fill-opacity:.75}.mm-bar3{fill:#e5864b;fill-opacity:.75}.mm-bar4{fill:var(--secondary);fill-opacity:.45}.mm-box{fill:var(--math-accent);fill-opacity:.05;stroke:var(--math-rule);stroke-width:1.2}</style><text class="mm-h" x="16" y="18">n = 4096，单线程，GFLOPS</text><text class="mm-t" x="162" y="45" text-anchor="end">ikj</text><rect class="mm-bar4" x="170" y="32" width="107.2" height="18" rx="2"/><text class="mm-t" x="283.2" y="45">32.4</text><text class="mm-t" x="162" y="71" text-anchor="end">ikj + 缓存分块</text><rect class="mm-bar3" x="170" y="58" width="90.6" height="18" rx="2"/><text class="mm-t" x="266.6" y="71">27.4</text><text class="mm-t" x="162" y="97" text-anchor="end">4×16 寄存器块</text><rect class="mm-bar3" x="170" y="84" width="54.2" height="18" rx="2"/><text class="mm-t" x="230.2" y="97">16.4</text><text class="mm-t" x="162" y="123" text-anchor="end">寄存器块 + 缓存分块</text><rect class="mm-bar2" x="170" y="110" width="257.0" height="18" rx="2"/><text class="mm-t" x="433.0" y="123">77.7</text><path class="mm-axis" d="M170 28 V132"/><path class="mm-dash" d="M583.5 24 V134"/><text class="mm-s" x="579.5" y="22" text-anchor="end">单核 NEON 峰值 125</text><text class="mm-s" x="16" y="150">n = 1024 时 4×16 寄存器块能到 104 GFLOPS；矩阵一大就掉到 16，缓存分块把它拉回 78</text></svg>

n = 4096 时快了 **4.7 倍**。块大小在相当大的范围内都差不多（n = 4096）：KC 在 64～512、NC 在 256～2048 之间，结果都在 72～78 GFLOPS。只要 B 块能留在 L2 里就行，不需要调得很精确。

小矩阵（n = 1024）时，加了缓存分块反而略慢一点。因为这时所有数据本来就在缓存里，分块只增加了循环开销。**又是同一个道理：分块只在受内存限制时才有用。**

## 七、分块的顺序很重要

把三个实验连起来看：

1. **ikj 的瓶颈是每次乘加的 C 读写**，计算强度在 L1 这一层就已经很低。缓存分块不解决这个问题，所以没用。
2. **寄存器分块**把每次乘加的访存从 3 次降到 0.5 次，计算强度提高了 6 倍，速度跳到 104。
3. 计算一快，**内存第一次成了瓶颈**：大矩阵上 B 被反复从远处搬，还撞上了 TLB。
4. 这时**缓存分块**登场，让 B 留在 L2 里被复用，从 16 拉回到 78。

所以分块不是一个能随手加上的优化。它要求你先把计算强度提上去，让内存真正成为瓶颈。

## 还差什么

n = 4096 时是 78 GFLOPS，比小矩阵的 104 还低，离单核峰值 125 也还有距离。剩下的问题在于访问模式：B 的每个 k 仍然只读一行里的 64 字节，A 同时在 4 行上跳着读。[第三篇](/posts/gemm-03-packing-microkernel/)用**打包**把这些分散的访问变成完全顺序的读取，再换一个更大的 8×12 微内核，并用上所有核心。

## 思考题

1. 用第一节的 roofline 公式估算：ikj 在 10 个线程时的计算强度要达到多少，才会受内存限制？对照实测的 296 GFLOPS，它受内存限制了吗？
2. 4×16 的寄存器块用了 20 个寄存器。换成 8×8（16 个累加器）或 4×24（24 个累加器），每次乘加对应的访存次数各是多少？哪个更好？
3. 把 n = 4096 换成 n = 4000（每行不再正好是 16 KB），只有寄存器分块的版本会变快吗？为什么？
4. 第六节说 B 块"只要能留在 L2 里就行"。M4 Pro 的 L2 由一组性能核共享。如果 10 个线程各自用一个 B 块，KC×NC 该怎么选？

## 完整代码

编译和运行：

```bash
clang -O3 -mcpu=apple-m4 -o peak peak.c && ./peak
clang -O3 -mcpu=apple-m4 -o bw bw.c && ./bw
clang -O3 -mcpu=apple-m4 -o axpy axpy.c && ./axpy
clang -O3 -mcpu=apple-m4 -DACCELERATE_NEW_LAPACK -o blk blk.c -framework Accelerate
VECLIB_MAXIMUM_THREADS=1 ./blk 2048 1          # ikj vs ikj + 缓存分块，单线程
./blk 4096 10                                   # 10 线程
VECLIB_MAXIMUM_THREADS=1 ./blk 2048 1 sweep    # 块大小扫描
clang -O3 -mcpu=apple-m4 -DACCELERATE_NEW_LAPACK -o rb rb.c -framework Accelerate
VECLIB_MAXIMUM_THREADS=1 ./rb 4096             # 寄存器块，有无缓存分块
VECLIB_MAXIMUM_THREADS=1 ./rb 4096 128 512     # 指定 KC、NC
```

`peak.c`：单核 NEON 峰值和单条依赖链的延迟。

```c
// Peak single-core FP32 throughput: many independent fmla chains so latency is hidden.
#include <arm_neon.h>
#include <stdio.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(void) {
    float32x4_t a[16], b = vdupq_n_f32(1.0000001f), c = vdupq_n_f32(0.9999999f);
    for (int i = 0; i < 16; i++) a[i] = vdupq_n_f32((float)i);
    long iters = 200000000;
    double best = 0;
    for (int rep = 0; rep < 3; rep++) {
        double t0 = now();
        for (long it = 0; it < iters; it++) {
            for (int i = 0; i < 16; i++) a[i] = vfmaq_f32(a[i], b, c);   // 16 independent chains
            __asm__ volatile("" : "+w"(a[0]), "+w"(a[1]), "+w"(a[2]), "+w"(a[3]), "+w"(a[4]), "+w"(a[5]),
                             "+w"(a[6]), "+w"(a[7]), "+w"(a[8]), "+w"(a[9]), "+w"(a[10]), "+w"(a[11]),
                             "+w"(a[12]), "+w"(a[13]), "+w"(a[14]), "+w"(a[15]));
        }
        double dt = now() - t0, g = iters * 16.0 * 4 * 2 / dt / 1e9;
        if (g > best) best = g;
    }
    float s = 0; for (int i = 0; i < 16; i++) s += vaddvq_f32(a[i]);
    printf("peak NEON fp32, 1 core: %.1f GFLOPS (checksum %g)\n", best, s);
    // latency: one dependent chain
    float32x4_t x = vdupq_n_f32(1.0f);
    double t0 = now();
    for (long it = 0; it < iters; it++) { x = vfmaq_f32(x, b, c); __asm__ volatile("" : "+w"(x)); }
    double dt = now() - t0;
    printf("one dependent chain: %.2f ns per fmla (%.1f GFLOPS)  (checksum %g)\n", dt / iters * 1e9, iters * 8.0 / dt / 1e9, vaddvq_f32(x));
    return 0;
}
```

`bw.c`：内存带宽（triad）。

```c
// Memory bandwidth (STREAM-like triad a = b + s*c) on 1 core and on all cores, 3 x 256 MB arrays.
#include <dispatch/dispatch.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(void) {
    size_t n = 64u << 20;  // 64M floats = 256 MB per array
    float *a = malloc(n * 4), *b = malloc(n * 4), *c = malloc(n * 4);
    for (size_t i = 0; i < n; i++) a[i] = 0, b[i] = 1, c[i] = 2;
    for (int threads = 1; threads <= 14; threads = threads == 1 ? 4 : threads + (threads < 10 ? 6 : 4)) {
        double best = 1e9;
        for (int r = 0; r < 4; r++) {
            double t0 = now();
            size_t chunk = n / threads;
            dispatch_apply(threads, DISPATCH_APPLY_AUTO, ^(size_t t) {
                float *pa = a + t * chunk, *pb = b + t * chunk, *pc = c + t * chunk;
                for (size_t i = 0; i < chunk; i++) pa[i] = pb[i] + 3.0f * pc[i];
            });
            double dt = now() - t0;
            if (dt < best) best = dt;
        }
        printf("triad %2d threads: %6.1f GB/s\n", threads, 3.0 * n * 4 / best / 1e9);
    }
    printf("(check %g)\n", a[n / 2]);
}
```

`axpy.c`：ikj 内层循环在 L1 里的天花板。

```c
// The i-k-j inner loop C[j] += a * B[j] on arrays that fit in L1: its ceiling without any memory traffic.
#include <stdio.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
__attribute__((noinline)) static void axpy(int n, float a, const float *restrict b, float *restrict c) {
    for (int j = 0; j < n; j++) c[j] += a * b[j];
}
int main(void) {
    enum { N = 1024 };                 // 2 x 4 KB: comfortably inside the 128 KB L1
    static float b[N], c[N];
    for (int i = 0; i < N; i++) b[i] = 1e-3f * i, c[i] = 0;
    long reps = 4000000;
    double t0 = now();
    for (long r = 0; r < reps; r++) axpy(N, 1e-7f, b, c);
    double dt = now() - t0;
    printf("axpy in L1: %.1f GFLOPS (%.2f ns per 4-wide fmla)  check %g\n", 2.0 * N * reps / dt / 1e9, dt / (reps * N / 4.0) * 1e9, c[7]);
}
```

`blk.c`：ikj 与 ikj + 缓存分块。

```c
// Part 2: cache blocking on top of the i-k-j loop. C = A * B, n x n, row-major float.
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

// Rows [i0, i1) of C += A * B, i-k-j order (C must be zeroed).
static void ikj_rows(int n, const float *A, const float *B, float *C, int i0, int i1) {
    for (int i = i0; i < i1; i++)
        for (int k = 0; k < n; k++) {
            float a = A[i * n + k];
            for (int j = 0; j < n; j++) C[i * n + j] += a * B[k * n + j];
        }
}

// Rows [i0, i1), blocked: a KB x NB block of B is reused by every row i before moving on.
static void ikj_blocked_rows(int n, const float *A, const float *B, float *C, int i0, int i1, int KB, int NB) {
    for (int jj = 0; jj < n; jj += NB)
        for (int kk = 0; kk < n; kk += KB)
            for (int i = i0; i < i1; i++)
                for (int k = kk; k < kk + KB; k++) {
                    float a = A[i * n + k];
                    for (int j = jj; j < jj + NB; j++) C[i * n + j] += a * B[k * n + j];
                }
}

static int KB = 256, NB = 512, THREADS = 1;

static void run(int which, int n, const float *A, const float *B, float *C) {
    memset(C, 0, sizeof(float) * n * n);
    int rows = (n + THREADS - 1) / THREADS;
    dispatch_apply(THREADS, DISPATCH_APPLY_AUTO, ^(size_t t) {
        int i0 = t * rows, i1 = i0 + rows > n ? n : i0 + rows;
        if (which == 0) ikj_rows(n, A, B, C, i0, i1);
        else ikj_blocked_rows(n, A, B, C, i0, i1, KB, NB);
    });
}

int main(int argc, char **argv) {
    int n = atoi(argv[1]);
    THREADS = argc > 2 ? atoi(argv[2]) : 1;
    const char *mode = argc > 3 ? argv[3] : "both";
    float *A = malloc(4ul * n * n), *B = malloc(4ul * n * n), *C = malloc(4ul * n * n), *R = malloc(4ul * n * n);
    srand(1);
    for (long i = 0; i < (long)n * n; i++) A[i] = rand() / (float)RAND_MAX - .5f, B[i] = rand() / (float)RAND_MAX - .5f;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, A, n, B, n, 0, R, n);

    if (!strcmp(mode, "sweep")) {             // block-size sweep, blocked version only
        int kbs[] = {64, 128, 256, 512}, nbs[] = {128, 256, 512, 1024, 2048};
        for (int a = 0; a < 4; a++) {
            printf("KB=%-4d", kbs[a]);
            for (int b = 0; b < 5; b++) {
                KB = kbs[a]; NB = nbs[b];
                if (NB > n || KB > n) { printf("    -   "); continue; }
                double t0 = now(); run(1, n, A, B, C); double dt = now() - t0;
                printf(" %7.1f", 2.0 * n * n * n / dt / 1e9);
                fflush(stdout);
            }
            printf("   (GFLOPS for NB = 128 256 512 1024 2048)\n");
        }
        return 0;
    }
    for (int w = 0; w < 2; w++) {
        double best = 1e30;
        for (int r = 0; r < (n <= 1024 ? 3 : 1); r++) {
            double t0 = now(); run(w, n, A, B, C); double dt = now() - t0;
            if (dt < best) best = dt;
        }
        float err = 0;
        for (long i = 0; i < (long)n * n; i++) err = fmaxf(err, fabsf(C[i] - R[i]));
        printf("n=%-5d threads=%-2d %-12s %9.1f ms %7.1f GFLOPS  err=%.0e\n", n, THREADS,
               w ? "ikj_blocked" : "ikj", best * 1e3, 2.0 * n * n * n / best / 1e9, err);
    }
}
```

`rb.c`：4×16 寄存器分块，以及寄存器分块 + 缓存分块。

```c
// Part 2b: register blocking (a 4 x 16 tile of C lives in 16 NEON registers), then + cache blocking.
// Assumes n is a multiple of 16 (and of KC, NC for the blocked version).
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

// C[i..i+4)[j..j+16) += A[i..i+4)[k0..k1) * B[k0..k1)[j..j+16)
static inline void tile_4x16(int n, const float *A, const float *B, float *C, int i, int j, int k0, int k1) {
    float32x4_t c[4][4];
    for (int r = 0; r < 4; r++)
        for (int v = 0; v < 4; v++) c[r][v] = vld1q_f32(&C[(i + r) * n + j + 4 * v]);
    for (int k = k0; k < k1; k++) {
        const float *b = &B[k * n + j];
        float32x4_t b0 = vld1q_f32(b), b1 = vld1q_f32(b + 4), b2 = vld1q_f32(b + 8), b3 = vld1q_f32(b + 12);
        for (int r = 0; r < 4; r++) {
            float a = A[(i + r) * n + k];          // one A value feeds 16 FMAs
            c[r][0] = vfmaq_n_f32(c[r][0], b0, a);
            c[r][1] = vfmaq_n_f32(c[r][1], b1, a);
            c[r][2] = vfmaq_n_f32(c[r][2], b2, a);
            c[r][3] = vfmaq_n_f32(c[r][3], b3, a);
        }
    }
    for (int r = 0; r < 4; r++)
        for (int v = 0; v < 4; v++) vst1q_f32(&C[(i + r) * n + j + 4 * v], c[r][v]);
}

// Register blocking only: every 4 x 16 tile runs over the whole k range.
static void rb(int n, const float *A, const float *B, float *C) {
    memset(C, 0, 4ul * n * n);
    for (int i = 0; i < n; i += 4)
        for (int j = 0; j < n; j += 16) tile_4x16(n, A, B, C, i, j, 0, n);
}

// Register blocking + cache blocking: a KC x NC block of B (KC*NC*4 bytes) is reused by all rows.
static int KC = 256, NC = 512;
static void rb_blocked(int n, const float *A, const float *B, float *C) {
    memset(C, 0, 4ul * n * n);
    for (int jc = 0; jc < n; jc += NC)
        for (int pc = 0; pc < n; pc += KC)
            for (int i = 0; i < n; i += 4)
                for (int j = jc; j < jc + NC; j += 16) tile_4x16(n, A, B, C, i, j, pc, pc + KC);
}

int main(int argc, char **argv) {
    int n = atoi(argv[1]);
    if (argc > 3) KC = atoi(argv[2]), NC = atoi(argv[3]);
    float *A = malloc(4ul * n * n), *B = malloc(4ul * n * n), *C = malloc(4ul * n * n), *R = malloc(4ul * n * n);
    srand(1);
    for (long i = 0; i < (long)n * n; i++) A[i] = rand() / (float)RAND_MAX - .5f, B[i] = rand() / (float)RAND_MAX - .5f;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, A, n, B, n, 0, R, n);
    struct { const char *name; void (*f)(int, const float *, const float *, float *); } v[] = {{"rb_4x16", rb}, {"rb_4x16+cache", rb_blocked}};
    for (int w = 0; w < 2; w++) {
        if (argc > 3 && w == 0) continue;
        double best = 1e30;
        for (int r = 0; r < (n <= 1024 ? 3 : 1); r++) { double t0 = now(); v[w].f(n, A, B, C); double dt = now() - t0; if (dt < best) best = dt; }
        float err = 0;
        for (long i = 0; i < (long)n * n; i++) err = fmaxf(err, fabsf(C[i] - R[i]));
        printf("n=%-5d %-15s KC=%-4d NC=%-5d %9.1f ms %7.1f GFLOPS  err=%.0e\n", n, v[w].name, KC, NC, best * 1e3, 2.0 * n * n * n / best / 1e9, err);
    }
}
```
