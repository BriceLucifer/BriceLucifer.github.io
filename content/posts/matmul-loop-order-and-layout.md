---
title: "从零写 GEMM（一）：循环顺序和存储顺序，为什么能差 70 倍"
date: 2026-11-26
tags: ["Performance", "Hardware", "Memory"]
series: ["GEMM from Scratch"]
keywords: ["Matrix Multiplication", "GEMM", "Row-major", "Column-major", "Cache Line", "Loop Order", "SIMD", "NEON", "Apple Silicon", "M4 Pro", "C"]
description: "Six loop orders of the same triple loop, row-major vs column-major B, measured on an M4 Pro: the innermost index decides the strides, stride decides cache-line use (128-byte lines on Apple silicon) and vectorization, and the spread is 0.45 to 33 GFLOPS."
summary: "同一个三重循环，只换 i、j、k 的嵌套顺序，或者把 B 改成按列存，在 M4 Pro 上就能从 0.45 GFLOPS 跑到 33 GFLOPS。这篇从外层到内层讲清楚：为什么只有最内层的下标决定访问步长，步长又怎样决定缓存行利用率和能不能向量化；以及为什么把 B 按列存之后，读取明明连续了，却只快了 2 倍。"
---

{{< series >}}

> **这篇你会搞清楚**
> 1. 二维矩阵在一维内存里是怎么排的，行优先和列优先差在哪；
> 2. 为什么三重循环里**只有最内层的下标**决定访问步长；
> 3. 步长怎样同时影响缓存行利用率和向量化；
> 4. 为什么"把 B 按列存"之后读取连续了，速度却只翻了一倍。

常听到一个说法："矩阵按列存，矩阵乘法更快"。这句话**不完全对**。真正起作用的是：**存储顺序要和最内层循环的访问顺序对上**。按列存只是让它们对上的一种办法，换一个循环顺序同样可以做到，而且在实测里效果更好。

这篇用同一个三重循环的 9 个版本，在 M4 Pro 上实测，把这件事从内存布局一路讲到汇编。

**测试环境**：Apple M4 Pro，单线程，`float32`，clang 23，`-O3 -mcpu=apple-m4`。每个版本跑多次取最快，并和 Accelerate 的结果逐元素对比。完整代码在文末。

## 一、矩阵在内存里是一条线

内存是一维的。二维矩阵存进去，必须先决定"哪个方向相邻"：

<svg viewBox="0 0 700 240" role="img" aria-label="行优先与列优先在内存中的排列" style="width:100%;height:auto;display:block;margin:1.5rem auto" xmlns="http://www.w3.org/2000/svg"><style>.mm-t{fill:var(--primary);font-family:var(--font-mono);font-size:12px}.mm-s{fill:var(--secondary);font-family:var(--font-mono);font-size:11px}.mm-h{fill:var(--primary);font-family:var(--font-ui);font-size:13px;font-weight:600}.mm-grid{fill:none;stroke:var(--math-rule);stroke-width:1}.mm-cell{stroke:var(--math-rule);stroke-width:1;fill:transparent}.mm-good{fill:#40a8a0;fill-opacity:.35;stroke:#40a8a0}.mm-bad{fill:#e5864b;fill-opacity:.35;stroke:#e5864b}.mm-one{fill:var(--math-accent);fill-opacity:.35;stroke:var(--math-accent)}.mm-waste{fill:var(--secondary);fill-opacity:.12;stroke:var(--math-rule)}.mm-arrow{stroke:var(--primary);stroke-width:1.6;fill:none;marker-end:url(#mm-ah)}.mm-bar{fill:var(--math-accent);fill-opacity:.75}.mm-bar2{fill:#40a8a0;fill-opacity:.75}.mm-bar3{fill:#e5864b;fill-opacity:.75}.mm-axis{fill:none;stroke:var(--math-rule);stroke-width:1}</style><text class="mm-h" x="30" y="30">4×4 矩阵 A</text><rect x="30" y="50" width="36" height="36" style="fill:#9d7cd8;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="48.0" y="72.0" text-anchor="middle">00</text><rect x="66" y="50" width="36" height="36" style="fill:#9d7cd8;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="84.0" y="72.0" text-anchor="middle">01</text><rect x="102" y="50" width="36" height="36" style="fill:#9d7cd8;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="120.0" y="72.0" text-anchor="middle">02</text><rect x="138" y="50" width="36" height="36" style="fill:#9d7cd8;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="156.0" y="72.0" text-anchor="middle">03</text><rect x="30" y="86" width="36" height="36" style="fill:#5b8def;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="48.0" y="108.0" text-anchor="middle">10</text><rect x="66" y="86" width="36" height="36" style="fill:#5b8def;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="84.0" y="108.0" text-anchor="middle">11</text><rect x="102" y="86" width="36" height="36" style="fill:#5b8def;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="120.0" y="108.0" text-anchor="middle">12</text><rect x="138" y="86" width="36" height="36" style="fill:#5b8def;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="156.0" y="108.0" text-anchor="middle">13</text><rect x="30" y="122" width="36" height="36" style="fill:#40a8a0;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="48.0" y="144.0" text-anchor="middle">20</text><rect x="66" y="122" width="36" height="36" style="fill:#40a8a0;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="84.0" y="144.0" text-anchor="middle">21</text><rect x="102" y="122" width="36" height="36" style="fill:#40a8a0;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="120.0" y="144.0" text-anchor="middle">22</text><rect x="138" y="122" width="36" height="36" style="fill:#40a8a0;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="156.0" y="144.0" text-anchor="middle">23</text><rect x="30" y="158" width="36" height="36" style="fill:#e5864b;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="48.0" y="180.0" text-anchor="middle">30</text><rect x="66" y="158" width="36" height="36" style="fill:#e5864b;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="84.0" y="180.0" text-anchor="middle">31</text><rect x="102" y="158" width="36" height="36" style="fill:#e5864b;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="120.0" y="180.0" text-anchor="middle">32</text><rect x="138" y="158" width="36" height="36" style="fill:#e5864b;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="156.0" y="180.0" text-anchor="middle">33</text><text class="mm-s" x="30" y="216">格子里是 (行, 列)，颜色 = 行号</text><text class="mm-h" x="230" y="42">行优先 row-major（C、NumPy 默认）</text><rect x="230" y="52" width="28" height="28" style="fill:#9d7cd8;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="244.0" y="70.0" text-anchor="middle">00</text><rect x="258" y="52" width="28" height="28" style="fill:#9d7cd8;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="272.0" y="70.0" text-anchor="middle">01</text><rect x="286" y="52" width="28" height="28" style="fill:#9d7cd8;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="300.0" y="70.0" text-anchor="middle">02</text><rect x="314" y="52" width="28" height="28" style="fill:#9d7cd8;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="328.0" y="70.0" text-anchor="middle">03</text><rect x="342" y="52" width="28" height="28" style="fill:#5b8def;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="356.0" y="70.0" text-anchor="middle">10</text><rect x="370" y="52" width="28" height="28" style="fill:#5b8def;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="384.0" y="70.0" text-anchor="middle">11</text><rect x="398" y="52" width="28" height="28" style="fill:#5b8def;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="412.0" y="70.0" text-anchor="middle">12</text><rect x="426" y="52" width="28" height="28" style="fill:#5b8def;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="440.0" y="70.0" text-anchor="middle">13</text><rect x="454" y="52" width="28" height="28" style="fill:#40a8a0;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="468.0" y="70.0" text-anchor="middle">20</text><rect x="482" y="52" width="28" height="28" style="fill:#40a8a0;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="496.0" y="70.0" text-anchor="middle">21</text><rect x="510" y="52" width="28" height="28" style="fill:#40a8a0;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="524.0" y="70.0" text-anchor="middle">22</text><rect x="538" y="52" width="28" height="28" style="fill:#40a8a0;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="552.0" y="70.0" text-anchor="middle">23</text><rect x="566" y="52" width="28" height="28" style="fill:#e5864b;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="580.0" y="70.0" text-anchor="middle">30</text><rect x="594" y="52" width="28" height="28" style="fill:#e5864b;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="608.0" y="70.0" text-anchor="middle">31</text><rect x="622" y="52" width="28" height="28" style="fill:#e5864b;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="636.0" y="70.0" text-anchor="middle">32</text><rect x="650" y="52" width="28" height="28" style="fill:#e5864b;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="664.0" y="70.0" text-anchor="middle">33</text><text class="mm-s" x="230" y="98">地址 →    A[i][j] 在第 i·n + j 个位置：同一行相邻</text><text class="mm-h" x="230" y="138">列优先 column-major（Fortran、BLAS 默认）</text><rect x="230" y="148" width="28" height="28" style="fill:#9d7cd8;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="244.0" y="166.0" text-anchor="middle">00</text><rect x="258" y="148" width="28" height="28" style="fill:#5b8def;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="272.0" y="166.0" text-anchor="middle">10</text><rect x="286" y="148" width="28" height="28" style="fill:#40a8a0;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="300.0" y="166.0" text-anchor="middle">20</text><rect x="314" y="148" width="28" height="28" style="fill:#e5864b;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="328.0" y="166.0" text-anchor="middle">30</text><rect x="342" y="148" width="28" height="28" style="fill:#9d7cd8;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="356.0" y="166.0" text-anchor="middle">01</text><rect x="370" y="148" width="28" height="28" style="fill:#5b8def;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="384.0" y="166.0" text-anchor="middle">11</text><rect x="398" y="148" width="28" height="28" style="fill:#40a8a0;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="412.0" y="166.0" text-anchor="middle">21</text><rect x="426" y="148" width="28" height="28" style="fill:#e5864b;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="440.0" y="166.0" text-anchor="middle">31</text><rect x="454" y="148" width="28" height="28" style="fill:#9d7cd8;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="468.0" y="166.0" text-anchor="middle">02</text><rect x="482" y="148" width="28" height="28" style="fill:#5b8def;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="496.0" y="166.0" text-anchor="middle">12</text><rect x="510" y="148" width="28" height="28" style="fill:#40a8a0;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="524.0" y="166.0" text-anchor="middle">22</text><rect x="538" y="148" width="28" height="28" style="fill:#e5864b;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="552.0" y="166.0" text-anchor="middle">32</text><rect x="566" y="148" width="28" height="28" style="fill:#9d7cd8;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="580.0" y="166.0" text-anchor="middle">03</text><rect x="594" y="148" width="28" height="28" style="fill:#5b8def;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="608.0" y="166.0" text-anchor="middle">13</text><rect x="622" y="148" width="28" height="28" style="fill:#40a8a0;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="636.0" y="166.0" text-anchor="middle">23</text><rect x="650" y="148" width="28" height="28" style="fill:#e5864b;fill-opacity:.28;stroke:var(--math-rule)"/><text class="mm-t" x="664.0" y="166.0" text-anchor="middle">33</text><text class="mm-s" x="230" y="194">地址 →    A[i][j] 在第 i + j·n 个位置：同一列相邻</text></svg>

- **行优先**（C、C++、NumPy 默认）：`A[i][j]` 在第 $i \cdot n + j$ 个位置，**同一行**的元素挨在一起。
- **列优先**（Fortran、MATLAB、BLAS 的传统约定）：`A[i][j]` 在第 $i + j \cdot n$ 个位置，**同一列**的元素挨在一起。

两种方式没有哪一种天生更快，区别只在于"沿哪个方向走是连续的"。

## 二、CPU 一次搬一整条缓存行

CPU 从内存读数据时，最小单位不是 4 字节的一个 float，而是**一整条缓存行**。x86 上通常是 64 字节；Apple silicon 上是 **128 字节**，也就是 32 个 float（`sysctl hw.cachelinesize` 返回 128）。所以沿着连续方向读，搬一次能用 32 个数；沿着跨步方向读，搬一次只用到 1 个：

<svg viewBox="0 0 680 224" role="img" aria-label="连续访问与跨步访问的缓存行利用率（Apple silicon：128 字节缓存行）" style="width:100%;height:auto;display:block;margin:1.5rem auto" xmlns="http://www.w3.org/2000/svg"><style>.mm-t{fill:var(--primary);font-family:var(--font-mono);font-size:12px}.mm-s{fill:var(--secondary);font-family:var(--font-mono);font-size:11px}.mm-h{fill:var(--primary);font-family:var(--font-ui);font-size:13px;font-weight:600}.mm-grid{fill:none;stroke:var(--math-rule);stroke-width:1}.mm-cell{stroke:var(--math-rule);stroke-width:1;fill:transparent}.mm-good{fill:#40a8a0;fill-opacity:.35;stroke:#40a8a0}.mm-bad{fill:#e5864b;fill-opacity:.35;stroke:#e5864b}.mm-one{fill:var(--math-accent);fill-opacity:.35;stroke:var(--math-accent)}.mm-waste{fill:var(--secondary);fill-opacity:.12;stroke:var(--math-rule)}.mm-arrow{stroke:var(--primary);stroke-width:1.6;fill:none;marker-end:url(#mm-ah)}.mm-bar{fill:var(--math-accent);fill-opacity:.75}.mm-bar2{fill:#40a8a0;fill-opacity:.75}.mm-bar3{fill:#e5864b;fill-opacity:.75}.mm-axis{fill:none;stroke:var(--math-rule);stroke-width:1}</style><text class="mm-h" x="20" y="34">步长 1：一条缓存行里的 32 个 float 全部用上</text><rect class="mm-good" x="20" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="30" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="40" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="50" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="60" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="70" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="80" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="90" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="100" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="110" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="120" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="130" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="140" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="150" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="160" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="170" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="180" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="190" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="200" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="210" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="220" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="230" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="240" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="250" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="260" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="270" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="280" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="290" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="300" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="310" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="320" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="330" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="340" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="350" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="360" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="370" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="380" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="390" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="400" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="410" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="420" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="430" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="440" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="450" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="460" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="470" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="480" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="490" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="500" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="510" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="520" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="530" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="540" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="550" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="560" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="570" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="580" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="590" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="600" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="610" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="620" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="630" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="640" y="46" width="8.5" height="26" rx="1.5"/><rect class="mm-good" x="650" y="46" width="8.5" height="26" rx="1.5"/><path class="mm-axis" d="M20 80 v6 h318 v-6"/><text class="mm-s" x="180" y="100" text-anchor="middle">缓存行 0（128 字节 = 32 个 float）</text><path class="mm-axis" d="M340 80 v6 h318 v-6"/><text class="mm-s" x="500" y="100" text-anchor="middle">缓存行 1（128 字节 = 32 个 float）</text><text class="mm-h" x="20" y="126">步长 n：每条缓存行只用到 1 个 float，其余 124 字节白搬</text><rect class="mm-bad" x="20" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="30" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="40" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="50" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="60" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="70" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="80" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="90" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="100" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="110" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="120" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="130" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="140" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="150" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="160" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="170" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="180" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="190" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="200" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="210" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="220" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="230" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="240" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="250" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="260" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="270" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="280" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="290" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="300" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="310" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="320" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="330" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-bad" x="340" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="350" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="360" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="370" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="380" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="390" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="400" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="410" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="420" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="430" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="440" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="450" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="460" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="470" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="480" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="490" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="500" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="510" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="520" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="530" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="540" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="550" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="560" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="570" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="580" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="590" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="600" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="610" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="620" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="630" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="640" y="138" width="8.5" height="26" rx="1.5"/><rect class="mm-waste" x="650" y="138" width="8.5" height="26" rx="1.5"/><path class="mm-axis" d="M20 172 v6 h318 v-6"/><text class="mm-s" x="180" y="192" text-anchor="middle">缓存行 0（128 字节 = 32 个 float）</text><path class="mm-axis" d="M340 172 v6 h318 v-6"/><text class="mm-s" x="500" y="192" text-anchor="middle">缓存行 1（128 字节 = 32 个 float）</text><text class="mm-s" x="20" y="212">n = 1024 时，“下一行同一列”相隔 4 KB：每读一个数就要搬一整条新的缓存行</text></svg>

跨步访问还会带来第二个问题：**没法向量化**。NEON 的一条指令一次处理 4 个相邻的 float，前提是这 4 个数在内存里是连续的。

## 三、从外层到内层：只有最内层的下标决定步长

最朴素的矩阵乘法 $C = AB$，即 $C_{ij} = \sum_k A_{ik} B_{kj}$：

```c
for (int i = 0; i < n; i++)          // 外层：选 C 的第 i 行
    for (int j = 0; j < n; j++) {    // 中层：选 C 的第 j 列，确定要算哪个 C[i][j]
        float s = 0;
        for (int k = 0; k < n; k++)  // 内层：沿 k 做点积
            s += A[i * n + k] * B[k * n + j];
        C[i * n + j] = s;
    }
```

三个下标可以任意交换嵌套顺序，一共 $3! = 6$ 种，算出来的结果完全一样（只是浮点加法顺序可能不同）。但它们的速度天差地别，原因是：

> **一个数组的访问步长，只取决于最内层循环的下标出现在它的哪个维度上。**
> 下标在最后一维（列号）→ 步长 1；在第一维（行号）→ 步长 n；不出现 → 步长 0，整个内层循环只用一个数。

以行优先存储为例，6 种顺序按最内层下标分成三类：

| 最内层 | 循环顺序 | `A[i][k]` | `B[k][j]` | `C[i][j]` |
|---|---|---|---|---|
| **k** | ijk、jik | 1 | **n** | 0 |
| **j** | ikj、kij | 0 | 1 | 1 |
| **i** | jki、kji | **n** | 0 | **n** |

<svg viewBox="0 0 712 210" role="img" aria-label="三种最内层循环下 A、B、C 的访问方式" style="width:100%;height:auto;display:block;margin:1.5rem auto" xmlns="http://www.w3.org/2000/svg"><style>.mm-t{fill:var(--primary);font-family:var(--font-mono);font-size:12px}.mm-s{fill:var(--secondary);font-family:var(--font-mono);font-size:11px}.mm-h{fill:var(--primary);font-family:var(--font-ui);font-size:13px;font-weight:600}.mm-grid{fill:none;stroke:var(--math-rule);stroke-width:1}.mm-cell{stroke:var(--math-rule);stroke-width:1;fill:transparent}.mm-good{fill:#40a8a0;fill-opacity:.35;stroke:#40a8a0}.mm-bad{fill:#e5864b;fill-opacity:.35;stroke:#e5864b}.mm-one{fill:var(--math-accent);fill-opacity:.35;stroke:var(--math-accent)}.mm-waste{fill:var(--secondary);fill-opacity:.12;stroke:var(--math-rule)}.mm-arrow{stroke:var(--primary);stroke-width:1.6;fill:none;marker-end:url(#mm-ah)}.mm-bar{fill:var(--math-accent);fill-opacity:.75}.mm-bar2{fill:#40a8a0;fill-opacity:.75}.mm-bar3{fill:#e5864b;fill-opacity:.75}.mm-axis{fill:none;stroke:var(--math-rule);stroke-width:1}</style><defs><marker id="mm-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--primary)"/></marker></defs><text class="mm-h" x="16" y="24">最内层是 k：ijk、jik</text><text class="mm-s" x="16" y="42">A 步长 1 · B 步长 n · C 不动</text><text class="mm-t" x="58.0" y="64" text-anchor="middle">A</text><rect class="mm-cell" x="16" y="70" width="14" height="14"/><rect class="mm-cell" x="30" y="70" width="14" height="14"/><rect class="mm-cell" x="44" y="70" width="14" height="14"/><rect class="mm-cell" x="58" y="70" width="14" height="14"/><rect class="mm-cell" x="72" y="70" width="14" height="14"/><rect class="mm-cell" x="86" y="70" width="14" height="14"/><rect class="mm-cell" x="16" y="84" width="14" height="14"/><rect class="mm-cell" x="30" y="84" width="14" height="14"/><rect class="mm-cell" x="44" y="84" width="14" height="14"/><rect class="mm-cell" x="58" y="84" width="14" height="14"/><rect class="mm-cell" x="72" y="84" width="14" height="14"/><rect class="mm-cell" x="86" y="84" width="14" height="14"/><rect class="mm-good" x="16" y="98" width="14" height="14"/><rect class="mm-good" x="30" y="98" width="14" height="14"/><rect class="mm-good" x="44" y="98" width="14" height="14"/><rect class="mm-good" x="58" y="98" width="14" height="14"/><rect class="mm-good" x="72" y="98" width="14" height="14"/><rect class="mm-good" x="86" y="98" width="14" height="14"/><rect class="mm-cell" x="16" y="112" width="14" height="14"/><rect class="mm-cell" x="30" y="112" width="14" height="14"/><rect class="mm-cell" x="44" y="112" width="14" height="14"/><rect class="mm-cell" x="58" y="112" width="14" height="14"/><rect class="mm-cell" x="72" y="112" width="14" height="14"/><rect class="mm-cell" x="86" y="112" width="14" height="14"/><rect class="mm-cell" x="16" y="126" width="14" height="14"/><rect class="mm-cell" x="30" y="126" width="14" height="14"/><rect class="mm-cell" x="44" y="126" width="14" height="14"/><rect class="mm-cell" x="58" y="126" width="14" height="14"/><rect class="mm-cell" x="72" y="126" width="14" height="14"/><rect class="mm-cell" x="86" y="126" width="14" height="14"/><rect class="mm-cell" x="16" y="140" width="14" height="14"/><rect class="mm-cell" x="30" y="140" width="14" height="14"/><rect class="mm-cell" x="44" y="140" width="14" height="14"/><rect class="mm-cell" x="58" y="140" width="14" height="14"/><rect class="mm-cell" x="72" y="140" width="14" height="14"/><rect class="mm-cell" x="86" y="140" width="14" height="14"/><path class="mm-arrow" d="M19 105.0 H97"/><text class="mm-t" x="132.0" y="64" text-anchor="middle">B</text><rect class="mm-cell" x="90" y="70" width="14" height="14"/><rect class="mm-cell" x="104" y="70" width="14" height="14"/><rect class="mm-cell" x="118" y="70" width="14" height="14"/><rect class="mm-bad" x="132" y="70" width="14" height="14"/><rect class="mm-cell" x="146" y="70" width="14" height="14"/><rect class="mm-cell" x="160" y="70" width="14" height="14"/><rect class="mm-cell" x="90" y="84" width="14" height="14"/><rect class="mm-cell" x="104" y="84" width="14" height="14"/><rect class="mm-cell" x="118" y="84" width="14" height="14"/><rect class="mm-bad" x="132" y="84" width="14" height="14"/><rect class="mm-cell" x="146" y="84" width="14" height="14"/><rect class="mm-cell" x="160" y="84" width="14" height="14"/><rect class="mm-cell" x="90" y="98" width="14" height="14"/><rect class="mm-cell" x="104" y="98" width="14" height="14"/><rect class="mm-cell" x="118" y="98" width="14" height="14"/><rect class="mm-bad" x="132" y="98" width="14" height="14"/><rect class="mm-cell" x="146" y="98" width="14" height="14"/><rect class="mm-cell" x="160" y="98" width="14" height="14"/><rect class="mm-cell" x="90" y="112" width="14" height="14"/><rect class="mm-cell" x="104" y="112" width="14" height="14"/><rect class="mm-cell" x="118" y="112" width="14" height="14"/><rect class="mm-bad" x="132" y="112" width="14" height="14"/><rect class="mm-cell" x="146" y="112" width="14" height="14"/><rect class="mm-cell" x="160" y="112" width="14" height="14"/><rect class="mm-cell" x="90" y="126" width="14" height="14"/><rect class="mm-cell" x="104" y="126" width="14" height="14"/><rect class="mm-cell" x="118" y="126" width="14" height="14"/><rect class="mm-bad" x="132" y="126" width="14" height="14"/><rect class="mm-cell" x="146" y="126" width="14" height="14"/><rect class="mm-cell" x="160" y="126" width="14" height="14"/><rect class="mm-cell" x="90" y="140" width="14" height="14"/><rect class="mm-cell" x="104" y="140" width="14" height="14"/><rect class="mm-cell" x="118" y="140" width="14" height="14"/><rect class="mm-bad" x="132" y="140" width="14" height="14"/><rect class="mm-cell" x="146" y="140" width="14" height="14"/><rect class="mm-cell" x="160" y="140" width="14" height="14"/><path class="mm-arrow" d="M139.0 73 V151"/><text class="mm-t" x="206.0" y="64" text-anchor="middle">C</text><rect class="mm-cell" x="164" y="70" width="14" height="14"/><rect class="mm-cell" x="178" y="70" width="14" height="14"/><rect class="mm-cell" x="192" y="70" width="14" height="14"/><rect class="mm-cell" x="206" y="70" width="14" height="14"/><rect class="mm-cell" x="220" y="70" width="14" height="14"/><rect class="mm-cell" x="234" y="70" width="14" height="14"/><rect class="mm-cell" x="164" y="84" width="14" height="14"/><rect class="mm-cell" x="178" y="84" width="14" height="14"/><rect class="mm-cell" x="192" y="84" width="14" height="14"/><rect class="mm-cell" x="206" y="84" width="14" height="14"/><rect class="mm-cell" x="220" y="84" width="14" height="14"/><rect class="mm-cell" x="234" y="84" width="14" height="14"/><rect class="mm-cell" x="164" y="98" width="14" height="14"/><rect class="mm-cell" x="178" y="98" width="14" height="14"/><rect class="mm-cell" x="192" y="98" width="14" height="14"/><rect class="mm-one" x="206" y="98" width="14" height="14"/><rect class="mm-cell" x="220" y="98" width="14" height="14"/><rect class="mm-cell" x="234" y="98" width="14" height="14"/><rect class="mm-cell" x="164" y="112" width="14" height="14"/><rect class="mm-cell" x="178" y="112" width="14" height="14"/><rect class="mm-cell" x="192" y="112" width="14" height="14"/><rect class="mm-cell" x="206" y="112" width="14" height="14"/><rect class="mm-cell" x="220" y="112" width="14" height="14"/><rect class="mm-cell" x="234" y="112" width="14" height="14"/><rect class="mm-cell" x="164" y="126" width="14" height="14"/><rect class="mm-cell" x="178" y="126" width="14" height="14"/><rect class="mm-cell" x="192" y="126" width="14" height="14"/><rect class="mm-cell" x="206" y="126" width="14" height="14"/><rect class="mm-cell" x="220" y="126" width="14" height="14"/><rect class="mm-cell" x="234" y="126" width="14" height="14"/><rect class="mm-cell" x="164" y="140" width="14" height="14"/><rect class="mm-cell" x="178" y="140" width="14" height="14"/><rect class="mm-cell" x="192" y="140" width="14" height="14"/><rect class="mm-cell" x="206" y="140" width="14" height="14"/><rect class="mm-cell" x="220" y="140" width="14" height="14"/><rect class="mm-cell" x="234" y="140" width="14" height="14"/><text class="mm-h" x="248" y="24">最内层是 j：ikj、kij</text><text class="mm-s" x="248" y="42">A 不动 · B 步长 1 · C 步长 1</text><text class="mm-t" x="290.0" y="64" text-anchor="middle">A</text><rect class="mm-cell" x="248" y="70" width="14" height="14"/><rect class="mm-cell" x="262" y="70" width="14" height="14"/><rect class="mm-cell" x="276" y="70" width="14" height="14"/><rect class="mm-cell" x="290" y="70" width="14" height="14"/><rect class="mm-cell" x="304" y="70" width="14" height="14"/><rect class="mm-cell" x="318" y="70" width="14" height="14"/><rect class="mm-cell" x="248" y="84" width="14" height="14"/><rect class="mm-cell" x="262" y="84" width="14" height="14"/><rect class="mm-cell" x="276" y="84" width="14" height="14"/><rect class="mm-cell" x="290" y="84" width="14" height="14"/><rect class="mm-cell" x="304" y="84" width="14" height="14"/><rect class="mm-cell" x="318" y="84" width="14" height="14"/><rect class="mm-cell" x="248" y="98" width="14" height="14"/><rect class="mm-one" x="262" y="98" width="14" height="14"/><rect class="mm-cell" x="276" y="98" width="14" height="14"/><rect class="mm-cell" x="290" y="98" width="14" height="14"/><rect class="mm-cell" x="304" y="98" width="14" height="14"/><rect class="mm-cell" x="318" y="98" width="14" height="14"/><rect class="mm-cell" x="248" y="112" width="14" height="14"/><rect class="mm-cell" x="262" y="112" width="14" height="14"/><rect class="mm-cell" x="276" y="112" width="14" height="14"/><rect class="mm-cell" x="290" y="112" width="14" height="14"/><rect class="mm-cell" x="304" y="112" width="14" height="14"/><rect class="mm-cell" x="318" y="112" width="14" height="14"/><rect class="mm-cell" x="248" y="126" width="14" height="14"/><rect class="mm-cell" x="262" y="126" width="14" height="14"/><rect class="mm-cell" x="276" y="126" width="14" height="14"/><rect class="mm-cell" x="290" y="126" width="14" height="14"/><rect class="mm-cell" x="304" y="126" width="14" height="14"/><rect class="mm-cell" x="318" y="126" width="14" height="14"/><rect class="mm-cell" x="248" y="140" width="14" height="14"/><rect class="mm-cell" x="262" y="140" width="14" height="14"/><rect class="mm-cell" x="276" y="140" width="14" height="14"/><rect class="mm-cell" x="290" y="140" width="14" height="14"/><rect class="mm-cell" x="304" y="140" width="14" height="14"/><rect class="mm-cell" x="318" y="140" width="14" height="14"/><text class="mm-t" x="364.0" y="64" text-anchor="middle">B</text><rect class="mm-cell" x="322" y="70" width="14" height="14"/><rect class="mm-cell" x="336" y="70" width="14" height="14"/><rect class="mm-cell" x="350" y="70" width="14" height="14"/><rect class="mm-cell" x="364" y="70" width="14" height="14"/><rect class="mm-cell" x="378" y="70" width="14" height="14"/><rect class="mm-cell" x="392" y="70" width="14" height="14"/><rect class="mm-good" x="322" y="84" width="14" height="14"/><rect class="mm-good" x="336" y="84" width="14" height="14"/><rect class="mm-good" x="350" y="84" width="14" height="14"/><rect class="mm-good" x="364" y="84" width="14" height="14"/><rect class="mm-good" x="378" y="84" width="14" height="14"/><rect class="mm-good" x="392" y="84" width="14" height="14"/><rect class="mm-cell" x="322" y="98" width="14" height="14"/><rect class="mm-cell" x="336" y="98" width="14" height="14"/><rect class="mm-cell" x="350" y="98" width="14" height="14"/><rect class="mm-cell" x="364" y="98" width="14" height="14"/><rect class="mm-cell" x="378" y="98" width="14" height="14"/><rect class="mm-cell" x="392" y="98" width="14" height="14"/><rect class="mm-cell" x="322" y="112" width="14" height="14"/><rect class="mm-cell" x="336" y="112" width="14" height="14"/><rect class="mm-cell" x="350" y="112" width="14" height="14"/><rect class="mm-cell" x="364" y="112" width="14" height="14"/><rect class="mm-cell" x="378" y="112" width="14" height="14"/><rect class="mm-cell" x="392" y="112" width="14" height="14"/><rect class="mm-cell" x="322" y="126" width="14" height="14"/><rect class="mm-cell" x="336" y="126" width="14" height="14"/><rect class="mm-cell" x="350" y="126" width="14" height="14"/><rect class="mm-cell" x="364" y="126" width="14" height="14"/><rect class="mm-cell" x="378" y="126" width="14" height="14"/><rect class="mm-cell" x="392" y="126" width="14" height="14"/><rect class="mm-cell" x="322" y="140" width="14" height="14"/><rect class="mm-cell" x="336" y="140" width="14" height="14"/><rect class="mm-cell" x="350" y="140" width="14" height="14"/><rect class="mm-cell" x="364" y="140" width="14" height="14"/><rect class="mm-cell" x="378" y="140" width="14" height="14"/><rect class="mm-cell" x="392" y="140" width="14" height="14"/><path class="mm-arrow" d="M325 91.0 H403"/><text class="mm-t" x="438.0" y="64" text-anchor="middle">C</text><rect class="mm-cell" x="396" y="70" width="14" height="14"/><rect class="mm-cell" x="410" y="70" width="14" height="14"/><rect class="mm-cell" x="424" y="70" width="14" height="14"/><rect class="mm-cell" x="438" y="70" width="14" height="14"/><rect class="mm-cell" x="452" y="70" width="14" height="14"/><rect class="mm-cell" x="466" y="70" width="14" height="14"/><rect class="mm-cell" x="396" y="84" width="14" height="14"/><rect class="mm-cell" x="410" y="84" width="14" height="14"/><rect class="mm-cell" x="424" y="84" width="14" height="14"/><rect class="mm-cell" x="438" y="84" width="14" height="14"/><rect class="mm-cell" x="452" y="84" width="14" height="14"/><rect class="mm-cell" x="466" y="84" width="14" height="14"/><rect class="mm-good" x="396" y="98" width="14" height="14"/><rect class="mm-good" x="410" y="98" width="14" height="14"/><rect class="mm-good" x="424" y="98" width="14" height="14"/><rect class="mm-good" x="438" y="98" width="14" height="14"/><rect class="mm-good" x="452" y="98" width="14" height="14"/><rect class="mm-good" x="466" y="98" width="14" height="14"/><rect class="mm-cell" x="396" y="112" width="14" height="14"/><rect class="mm-cell" x="410" y="112" width="14" height="14"/><rect class="mm-cell" x="424" y="112" width="14" height="14"/><rect class="mm-cell" x="438" y="112" width="14" height="14"/><rect class="mm-cell" x="452" y="112" width="14" height="14"/><rect class="mm-cell" x="466" y="112" width="14" height="14"/><rect class="mm-cell" x="396" y="126" width="14" height="14"/><rect class="mm-cell" x="410" y="126" width="14" height="14"/><rect class="mm-cell" x="424" y="126" width="14" height="14"/><rect class="mm-cell" x="438" y="126" width="14" height="14"/><rect class="mm-cell" x="452" y="126" width="14" height="14"/><rect class="mm-cell" x="466" y="126" width="14" height="14"/><rect class="mm-cell" x="396" y="140" width="14" height="14"/><rect class="mm-cell" x="410" y="140" width="14" height="14"/><rect class="mm-cell" x="424" y="140" width="14" height="14"/><rect class="mm-cell" x="438" y="140" width="14" height="14"/><rect class="mm-cell" x="452" y="140" width="14" height="14"/><rect class="mm-cell" x="466" y="140" width="14" height="14"/><path class="mm-arrow" d="M399 105.0 H477"/><text class="mm-h" x="480" y="24">最内层是 i：jki、kji</text><text class="mm-s" x="480" y="42">A 步长 n · B 不动 · C 步长 n</text><text class="mm-t" x="522.0" y="64" text-anchor="middle">A</text><rect class="mm-cell" x="480" y="70" width="14" height="14"/><rect class="mm-bad" x="494" y="70" width="14" height="14"/><rect class="mm-cell" x="508" y="70" width="14" height="14"/><rect class="mm-cell" x="522" y="70" width="14" height="14"/><rect class="mm-cell" x="536" y="70" width="14" height="14"/><rect class="mm-cell" x="550" y="70" width="14" height="14"/><rect class="mm-cell" x="480" y="84" width="14" height="14"/><rect class="mm-bad" x="494" y="84" width="14" height="14"/><rect class="mm-cell" x="508" y="84" width="14" height="14"/><rect class="mm-cell" x="522" y="84" width="14" height="14"/><rect class="mm-cell" x="536" y="84" width="14" height="14"/><rect class="mm-cell" x="550" y="84" width="14" height="14"/><rect class="mm-cell" x="480" y="98" width="14" height="14"/><rect class="mm-bad" x="494" y="98" width="14" height="14"/><rect class="mm-cell" x="508" y="98" width="14" height="14"/><rect class="mm-cell" x="522" y="98" width="14" height="14"/><rect class="mm-cell" x="536" y="98" width="14" height="14"/><rect class="mm-cell" x="550" y="98" width="14" height="14"/><rect class="mm-cell" x="480" y="112" width="14" height="14"/><rect class="mm-bad" x="494" y="112" width="14" height="14"/><rect class="mm-cell" x="508" y="112" width="14" height="14"/><rect class="mm-cell" x="522" y="112" width="14" height="14"/><rect class="mm-cell" x="536" y="112" width="14" height="14"/><rect class="mm-cell" x="550" y="112" width="14" height="14"/><rect class="mm-cell" x="480" y="126" width="14" height="14"/><rect class="mm-bad" x="494" y="126" width="14" height="14"/><rect class="mm-cell" x="508" y="126" width="14" height="14"/><rect class="mm-cell" x="522" y="126" width="14" height="14"/><rect class="mm-cell" x="536" y="126" width="14" height="14"/><rect class="mm-cell" x="550" y="126" width="14" height="14"/><rect class="mm-cell" x="480" y="140" width="14" height="14"/><rect class="mm-bad" x="494" y="140" width="14" height="14"/><rect class="mm-cell" x="508" y="140" width="14" height="14"/><rect class="mm-cell" x="522" y="140" width="14" height="14"/><rect class="mm-cell" x="536" y="140" width="14" height="14"/><rect class="mm-cell" x="550" y="140" width="14" height="14"/><path class="mm-arrow" d="M501.0 73 V151"/><text class="mm-t" x="596.0" y="64" text-anchor="middle">B</text><rect class="mm-cell" x="554" y="70" width="14" height="14"/><rect class="mm-cell" x="568" y="70" width="14" height="14"/><rect class="mm-cell" x="582" y="70" width="14" height="14"/><rect class="mm-cell" x="596" y="70" width="14" height="14"/><rect class="mm-cell" x="610" y="70" width="14" height="14"/><rect class="mm-cell" x="624" y="70" width="14" height="14"/><rect class="mm-cell" x="554" y="84" width="14" height="14"/><rect class="mm-cell" x="568" y="84" width="14" height="14"/><rect class="mm-cell" x="582" y="84" width="14" height="14"/><rect class="mm-one" x="596" y="84" width="14" height="14"/><rect class="mm-cell" x="610" y="84" width="14" height="14"/><rect class="mm-cell" x="624" y="84" width="14" height="14"/><rect class="mm-cell" x="554" y="98" width="14" height="14"/><rect class="mm-cell" x="568" y="98" width="14" height="14"/><rect class="mm-cell" x="582" y="98" width="14" height="14"/><rect class="mm-cell" x="596" y="98" width="14" height="14"/><rect class="mm-cell" x="610" y="98" width="14" height="14"/><rect class="mm-cell" x="624" y="98" width="14" height="14"/><rect class="mm-cell" x="554" y="112" width="14" height="14"/><rect class="mm-cell" x="568" y="112" width="14" height="14"/><rect class="mm-cell" x="582" y="112" width="14" height="14"/><rect class="mm-cell" x="596" y="112" width="14" height="14"/><rect class="mm-cell" x="610" y="112" width="14" height="14"/><rect class="mm-cell" x="624" y="112" width="14" height="14"/><rect class="mm-cell" x="554" y="126" width="14" height="14"/><rect class="mm-cell" x="568" y="126" width="14" height="14"/><rect class="mm-cell" x="582" y="126" width="14" height="14"/><rect class="mm-cell" x="596" y="126" width="14" height="14"/><rect class="mm-cell" x="610" y="126" width="14" height="14"/><rect class="mm-cell" x="624" y="126" width="14" height="14"/><rect class="mm-cell" x="554" y="140" width="14" height="14"/><rect class="mm-cell" x="568" y="140" width="14" height="14"/><rect class="mm-cell" x="582" y="140" width="14" height="14"/><rect class="mm-cell" x="596" y="140" width="14" height="14"/><rect class="mm-cell" x="610" y="140" width="14" height="14"/><rect class="mm-cell" x="624" y="140" width="14" height="14"/><text class="mm-t" x="670.0" y="64" text-anchor="middle">C</text><rect class="mm-cell" x="628" y="70" width="14" height="14"/><rect class="mm-cell" x="642" y="70" width="14" height="14"/><rect class="mm-cell" x="656" y="70" width="14" height="14"/><rect class="mm-bad" x="670" y="70" width="14" height="14"/><rect class="mm-cell" x="684" y="70" width="14" height="14"/><rect class="mm-cell" x="698" y="70" width="14" height="14"/><rect class="mm-cell" x="628" y="84" width="14" height="14"/><rect class="mm-cell" x="642" y="84" width="14" height="14"/><rect class="mm-cell" x="656" y="84" width="14" height="14"/><rect class="mm-bad" x="670" y="84" width="14" height="14"/><rect class="mm-cell" x="684" y="84" width="14" height="14"/><rect class="mm-cell" x="698" y="84" width="14" height="14"/><rect class="mm-cell" x="628" y="98" width="14" height="14"/><rect class="mm-cell" x="642" y="98" width="14" height="14"/><rect class="mm-cell" x="656" y="98" width="14" height="14"/><rect class="mm-bad" x="670" y="98" width="14" height="14"/><rect class="mm-cell" x="684" y="98" width="14" height="14"/><rect class="mm-cell" x="698" y="98" width="14" height="14"/><rect class="mm-cell" x="628" y="112" width="14" height="14"/><rect class="mm-cell" x="642" y="112" width="14" height="14"/><rect class="mm-cell" x="656" y="112" width="14" height="14"/><rect class="mm-bad" x="670" y="112" width="14" height="14"/><rect class="mm-cell" x="684" y="112" width="14" height="14"/><rect class="mm-cell" x="698" y="112" width="14" height="14"/><rect class="mm-cell" x="628" y="126" width="14" height="14"/><rect class="mm-cell" x="642" y="126" width="14" height="14"/><rect class="mm-cell" x="656" y="126" width="14" height="14"/><rect class="mm-bad" x="670" y="126" width="14" height="14"/><rect class="mm-cell" x="684" y="126" width="14" height="14"/><rect class="mm-cell" x="698" y="126" width="14" height="14"/><rect class="mm-cell" x="628" y="140" width="14" height="14"/><rect class="mm-cell" x="642" y="140" width="14" height="14"/><rect class="mm-cell" x="656" y="140" width="14" height="14"/><rect class="mm-bad" x="670" y="140" width="14" height="14"/><rect class="mm-cell" x="684" y="140" width="14" height="14"/><rect class="mm-cell" x="698" y="140" width="14" height="14"/><path class="mm-arrow" d="M677.0 73 V151"/><rect class="mm-good" x="16" y="186" width="14" height="14"/><text class="mm-s" x="36" y="197">连续访问（步长 1）</text><rect class="mm-bad" x="170" y="186" width="14" height="14"/><text class="mm-s" x="190" y="197">跨步访问（步长 n）</text><rect class="mm-one" x="324" y="186" width="14" height="14"/><text class="mm-s" x="344" y="197">整个内层循环只用这一个数（放在寄存器里）</text></svg>

- **最内层是 j**：B 和 C 都沿行走，全部连续；A 的那个数在整个内层循环里不变，放在寄存器里。这是行优先存储下的最佳情况。
- **最内层是 k**：A 连续，但 B 每次要跳 n 个数，等于每读一个 B 就搬一整条新的缓存行。
- **最内层是 i**：A 和 C **两个**数组都在跨步。C 还要每次读出来、加上、再写回去，最差。

外层的两个下标不影响步长，但会影响**复用**。比如 ikj 和 kij 的内层完全一样，区别在中层：ikj 先固定 i，C 的第 i 行（n=1024 时是 4 KB）会被连续用 n 次，一直留在 L1 缓存里；kij 先固定 k，每一轮都要把整个 C（4 MB）扫一遍。实测 ikj 也确实快了约 20%。

对应的 ikj 写法：

```c
memset(C, 0, sizeof(float) * n * n);
for (int i = 0; i < n; i++)
    for (int k = 0; k < n; k++) {
        float a = A[i * n + k];                  // 内层不变，进寄存器
        for (int j = 0; j < n; j++)
            C[i * n + j] += a * B[k * n + j];    // B、C 都是步长 1
    }
```

## 四、"把 B 按列存"是另一种解法

回到 ijk。它慢在 B 被跨步读取；如果把 B 改成按列存（相当于提前转置一次，代价是 $O(n^2)$，和 $O(n^3)$ 的乘法比可以忽略），`B[k][j]` 就变成 `Bc[j * n + k]`，内层的 k 也变成了步长 1：

```c
// Bc[j * n + k] == B[k][j]：B 按列存
for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
        float s = 0;
        for (int k = 0; k < n; k++)
            s += A[i * n + k] * Bc[j * n + k];   // A、Bc 都是步长 1
        C[i * n + j] = s;
    }
```

这就是"按列存更快"这个说法的来源：**不是列优先本身快，而是它让 ijk 的内层循环在两个操作数上都变成了连续访问**。

## 五、实测

<svg viewBox="0 0 680 264" role="img" aria-label="各版本矩阵乘法的 GFLOPS" style="width:100%;height:auto;display:block;margin:1.5rem auto" xmlns="http://www.w3.org/2000/svg"><style>.mm-t{fill:var(--primary);font-family:var(--font-mono);font-size:12px}.mm-s{fill:var(--secondary);font-family:var(--font-mono);font-size:11px}.mm-h{fill:var(--primary);font-family:var(--font-ui);font-size:13px;font-weight:600}.mm-grid{fill:none;stroke:var(--math-rule);stroke-width:1}.mm-cell{stroke:var(--math-rule);stroke-width:1;fill:transparent}.mm-good{fill:#40a8a0;fill-opacity:.35;stroke:#40a8a0}.mm-bad{fill:#e5864b;fill-opacity:.35;stroke:#e5864b}.mm-one{fill:var(--math-accent);fill-opacity:.35;stroke:var(--math-accent)}.mm-waste{fill:var(--secondary);fill-opacity:.12;stroke:var(--math-rule)}.mm-arrow{stroke:var(--primary);stroke-width:1.6;fill:none;marker-end:url(#mm-ah)}.mm-bar{fill:var(--math-accent);fill-opacity:.75}.mm-bar2{fill:#40a8a0;fill-opacity:.75}.mm-bar3{fill:#e5864b;fill-opacity:.75}.mm-axis{fill:none;stroke:var(--math-rule);stroke-width:1}</style><text class="mm-h" x="16" y="18">n = 1024，单线程，GFLOPS（越长越快）</text><text class="mm-t" x="112" y="43" text-anchor="end">ikj_row</text><rect class="mm-bar2" x="120" y="30" width="452.3" height="18" rx="2"/><text class="mm-t" x="578.3" y="43">32.98</text><text class="mm-t" x="112" y="69" text-anchor="end">ijk_col_x8</text><rect class="mm-bar" x="120" y="56" width="387.3" height="18" rx="2"/><text class="mm-t" x="513.3" y="69">28.24</text><text class="mm-t" x="112" y="95" text-anchor="end">kij_row</text><rect class="mm-bar2" x="120" y="82" width="375.2" height="18" rx="2"/><text class="mm-t" x="501.2" y="95">27.36</text><text class="mm-t" x="112" y="121" text-anchor="end">ijk_col</text><rect class="mm-bar" x="120" y="108" width="67.6" height="18" rx="2"/><text class="mm-t" x="193.6" y="121">4.93</text><text class="mm-t" x="112" y="147" text-anchor="end">ijk_row</text><rect class="mm-bar3" x="120" y="134" width="35.0" height="18" rx="2"/><text class="mm-t" x="161.0" y="147">2.55</text><text class="mm-t" x="112" y="173" text-anchor="end">jik_row</text><rect class="mm-bar3" x="120" y="160" width="33.2" height="18" rx="2"/><text class="mm-t" x="159.2" y="173">2.42</text><text class="mm-t" x="112" y="199" text-anchor="end">kji_row</text><rect class="mm-bar3" x="120" y="186" width="6.3" height="18" rx="2"/><text class="mm-t" x="132.3" y="199">0.46</text><text class="mm-t" x="112" y="225" text-anchor="end">jki_row</text><rect class="mm-bar3" x="120" y="212" width="6.2" height="18" rx="2"/><text class="mm-t" x="132.2" y="225">0.45</text><path class="mm-axis" d="M120 26 V236"/><text class="mm-s" x="16" y="254">Accelerate（cblas_sgemm）约 1700 GFLOPS：画进来是这张图宽度的 50 倍</text></svg>

| 版本 | 最内层 | n = 256 | n = 512 | n = 1024 |
|---|---|---|---|---|
| `ikj_row` | j | 33.3 | 34.7 | **33.0** |
| `kij_row` | j | 28.8 | 28.7 | 27.4 |
| `ijk_col_x8` | k（B 按列存，8 个累加器） | 41.3 | 33.7 | 28.2 |
| `ijk_col` | k（B 按列存） | 7.3 | 6.0 | 4.9 |
| `ijk_row` | k | 3.0 | 3.1 | 2.6 |
| `jik_row` | k | 3.5 | 3.0 | 2.4 |
| `kji_row` | i | 0.68 | 0.61 | 0.46 |
| `jki_row` | i | 0.64 | 0.60 | **0.45** |
| Accelerate | — | 1766 | 1814 | 1669 |

（单位 GFLOPS，即 $2n^3 / t$。）

三类循环顺序的排名和上一节的步长分析完全一致：**最内层是 j 的约 30，是 k 的约 3，是 i 的约 0.5**，最好和最差相差 **73 倍**。而且这种差距在 n=256 时就已经存在了，那时三个矩阵加起来只有 768 KB，基本都在缓存里。所以问题不只是"去内存里取数慢"，还有缓存行浪费，以及能不能向量化。

## 六、为什么按列存了，ijk_col 却只快了 2 倍

把 B 按列存之后，ijk 的两个读取都连续了，但只从 2.6 涨到 4.9 GFLOPS，远不如 ikj 的 33。看汇编就明白了。`ijk_col` 的内层循环里，乘法其实**已经向量化了**（`fmul.4s`，一次 4 个），但加法变成了这样：

```text
fadd  s0, s0, s1
fadd  s0, s0, s18
fadd  s0, s0, s17
fadd  s0, s0, s5
...
```

全是**标量**加法，而且都加到同一个寄存器 `s0` 上。原因是 `s += …` 这个点积是一个**归约**：C 语言规定浮点加法要按代码顺序执行，而浮点加法不满足结合律，所以编译器不能擅自把它拆成 4 路并行再合并。结果每一次加法都要等上一次算完，形成一条串行的依赖链，内存读得再快也没用。

解决办法是**自己拆成 8 个互不依赖的累加器**：

```c
float s[8] = {0};
for (int k = 0; k < n; k += 8)
    for (int l = 0; l < 8; l++)
        s[l] += A[i * n + k + l] * Bc[j * n + k + l];
C[i * n + j] = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
```

这一次编译器生成的是 `fmla.4s`（向量乘加），两组各 4 路，彼此没有依赖。速度从 4.9 跳到 28 GFLOPS，追上了 ikj。代价是加法顺序变了，和 Accelerate 结果的最大误差从 $5 \times 10^{-6}$ 变成 $2 \times 10^{-5}$，仍然在 float 的正常范围内。（这个例子要求 n 是 8 的倍数。）

ikj 天然没有这个问题：它的内层是 `C[j] += a * B[j]`，每个 j 写的是不同的位置，**根本不存在归约**，编译器可以直接向量化。这也是为什么它不需要任何技巧就是最快的那一个。

## 七、离硬件上限还有多远

最快的 33 GFLOPS，只有 Accelerate 的 **2%**。差距来自两件这篇还没碰的事：

1. **分块（blocking / tiling）**：把矩阵切成能放进缓存、甚至寄存器的小块，让每个读进来的数被反复用很多次。我们的三重循环里，每个数从缓存里读一次，只做一次乘加。
2. **专用硬件**：M4 有专门做矩阵运算的 SME 单元（`sysctl hw.optional.arm.FEAT_SME` 返回 1）。Accelerate 单线程跑出约 1700 GFLOPS，是单核 NEON 实测峰值（125 GFLOPS）的 13 倍以上，只可能来自它。

接下来两篇从这里继续：[第二篇](/posts/gemm-02-blocking/)讲分块，以及为什么分块一开始反而让程序变慢；[第三篇](/posts/gemm-03-packing-microkernel/)用打包、8×12 微内核和多线程，把手写的 NEON 版本推到单核峰值的 97%、多核约 1 TFLOPS。

## 总结

1. **只有最内层的下标决定步长**：它在哪一维，那个数组就沿哪个方向走。外层只影响复用。
2. **步长 1 有两重收益**：一条缓存行能用满（M4 上是 32 个 float），而且能向量化。
3. **存储顺序和循环顺序是同一个问题的两个旋钮**：要么改循环顺序（ikj），要么改存储顺序（B 按列存），目标都是让最内层连续。"按列存更快"只在它让内层变连续时才成立。
4. **连续了还要能并行**：归约形式的内层循环会被浮点加法链卡住，需要多个累加器，或者换成没有归约的循环顺序。

## 思考题

1. 如果 A、B、C 都改成**列优先**存储，6 种循环顺序里哪几种最快？先用"最内层下标在哪一维"推一推，再改代码验证。
2. `ijk_col_x8` 用了 8 个累加器。单核一条 `fmla.4s` 的延迟约 0.8 ns，而峰值吞吐约 15.6 条/ns（见[第二篇](/posts/gemm-02-blocking/)的峰值测量），至少需要几条独立的累加链才能跑满？改成 4 个、16 个累加器会怎样？
3. 把编译选项加上 `-ffast-math`，`ijk_col` 会变快吗？为什么？它和手写 8 个累加器有什么区别？
4. ikj 比 kij 快约 20%，我们的解释是 C 的一行能留在 L1 里。在 n = 256（所有矩阵都在缓存里）时，这个差距应该变大还是变小？测一下。

## 完整代码

```bash
clang -O3 -mcpu=apple-m4 -DACCELERATE_NEW_LAPACK -o mm matmul.c -framework Accelerate
VECLIB_MAXIMUM_THREADS=1 ./mm 1024          # 全部版本
VECLIB_MAXIMUM_THREADS=1 ./mm 1024 ikj_row  # 单个版本
```

```c
// C = A * B for N x N float matrices; GFLOPS = 2 N^3 / t.
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

// 1. textbook i-j-k, everything row-major: B is walked down a column (stride n).
static void ijk_row(int n, const float *A, const float *B, float *C) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int k = 0; k < n; k++) s += A[i * n + k] * B[k * n + j];
            C[i * n + j] = s;
        }
}

// 2. same loops, B stored column-major (Bc[j*n + k] = B[k][j]): both reads contiguous.
static void ijk_col(int n, const float *A, const float *Bc, float *C) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int k = 0; k < n; k++) s += A[i * n + k] * Bc[j * n + k];
            C[i * n + j] = s;
        }
}

// 3. as 2, with 8 independent partial sums so the dot product can use SIMD lanes.
static void ijk_col_x8(int n, const float *A, const float *Bc, float *C) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float s[8] = {0};
            for (int k = 0; k < n; k += 8)
                for (int l = 0; l < 8; l++) s[l] += A[i * n + k + l] * Bc[j * n + k + l];
            C[i * n + j] = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
        }
}

// 4. i-k-j loop order, row-major B: the inner loop streams a row of B and a row of C.
static void ikj_row(int n, const float *A, const float *B, float *C) {
    memset(C, 0, sizeof(float) * n * n);
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            float a = A[i * n + k];
            for (int j = 0; j < n; j++) C[i * n + j] += a * B[k * n + j];
        }
}


// The other four loop orders (all row-major). Only the innermost index decides the strides:
//   inner k (ijk, jik): A 1, B n, C fixed      inner j (ikj, kij): A fixed, B 1, C 1
//   inner i (jki, kji): A n, B fixed, C n
static void jik_row(int n, const float *A, const float *B, float *C) {
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) {
            float s = 0;
            for (int k = 0; k < n; k++) s += A[i * n + k] * B[k * n + j];
            C[i * n + j] = s;
        }
}
static void kij_row(int n, const float *A, const float *B, float *C) {
    memset(C, 0, sizeof(float) * n * n);
    for (int k = 0; k < n; k++)
        for (int i = 0; i < n; i++) {
            float a = A[i * n + k];
            for (int j = 0; j < n; j++) C[i * n + j] += a * B[k * n + j];
        }
}
static void jki_row(int n, const float *A, const float *B, float *C) {
    memset(C, 0, sizeof(float) * n * n);
    for (int j = 0; j < n; j++)
        for (int k = 0; k < n; k++) {
            float b = B[k * n + j];
            for (int i = 0; i < n; i++) C[i * n + j] += A[i * n + k] * b;
        }
}
static void kji_row(int n, const float *A, const float *B, float *C) {
    memset(C, 0, sizeof(float) * n * n);
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++) {
            float b = B[k * n + j];
            for (int i = 0; i < n; i++) C[i * n + j] += A[i * n + k] * b;
        }
}

static void blas(int n, const float *A, const float *B, float *C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, A, n, B, n, 0.0f, C, n);
}

typedef void (*mm_fn)(int, const float *, const float *, float *);

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 1024;
    const char *which = argc > 2 ? argv[2] : "all";
    float *A = malloc(sizeof(float) * n * n), *B = malloc(sizeof(float) * n * n);
    float *Bc = malloc(sizeof(float) * n * n), *C = malloc(sizeof(float) * n * n), *R = malloc(sizeof(float) * n * n);
    srand(1);
    for (int i = 0; i < n * n; i++) A[i] = rand() / (float)RAND_MAX - 0.5f, B[i] = rand() / (float)RAND_MAX - 0.5f;
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++) Bc[j * n + k] = B[k * n + j];
    blas(n, A, B, R);

    struct { const char *name; mm_fn f; int col; } v[] = {
        {"ijk_row", ijk_row, 0}, {"jik_row", jik_row, 0}, {"ikj_row", ikj_row, 0},
        {"kij_row", kij_row, 0}, {"jki_row", jki_row, 0}, {"kji_row", kji_row, 0},
        {"ijk_col", ijk_col, 1}, {"ijk_col_x8", ijk_col_x8, 1}, {"accelerate", blas, 0},
    };
    for (unsigned t = 0; t < sizeof v / sizeof *v; t++) {
        if (strcmp(which, "all") && strcmp(which, v[t].name)) continue;
        double best = 1e30;
        int reps = n <= 512 ? 5 : (n <= 1024 ? 3 : 1);
        for (int r = 0; r < reps; r++) {
            double t0 = now();
            v[t].f(n, A, v[t].col ? Bc : B, C);
            double dt = now() - t0;
            if (dt < best) best = dt;
        }
        float err = 0;
        for (int i = 0; i < n * n; i++) err = fmaxf(err, fabsf(C[i] - R[i]));
        printf("n=%-5d %-11s %9.2f ms  %7.2f GFLOPS  max|err|=%.1e\n", n, v[t].name, best * 1e3,
               2.0 * n * n * n / best / 1e9, err);
        fflush(stdout);
    }
    return 0;
}
```
