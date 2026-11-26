---
title: "从零写 GEMM（三）：打包、8×12 微内核和多线程"
date: 2026-11-26T12:00:00+13:00
tags: ["Performance", "Hardware", "Memory"]
series: ["GEMM from Scratch"]
keywords: ["GEMM", "SGEMM", "BLIS", "GotoBLAS", "Packing", "Micro-kernel", "NEON", "SME", "Apple Silicon", "M4 Pro", "Grand Central Dispatch", "Multithreading"]
description: "A GotoBLAS/BLIS-style SGEMM on an M4 Pro: packing, a five-level loop nest sized to L1/L2, and an 8x12 NEON micro-kernel reach 121.6 GFLOPS on one core (97% of measured peak) and about 1 TFLOPS on all cores; Accelerate gets 1.6 to 3 TFLOPS from the SME matrix unit."
summary: "第二篇停在 78 GFLOPS：寄存器分块加缓存分块有效，但 B 和 A 仍在被跳着读。这篇按 GotoBLAS/BLIS 的结构，把矩阵打包成微内核读取的顺序，用五层循环把每块数据放进合适的缓存，再写一个 8×12 的 NEON 微内核：单核 121.6 GFLOPS，达到实测峰值的 97%。多线程第一版只扩展到 67%，找出两个损失后到了约 1 TFLOPS。最后看看 Accelerate 为什么还能再快 3 倍。"
---

{{< series >}}

> **这篇你会搞清楚**
> 1. "打包"在做什么，为什么要多花一次拷贝；
> 2. GotoBLAS/BLIS 的五层循环：每一层分别把哪块数据放进哪一级缓存；
> 3. 微内核的形状（8×12）是怎样由寄存器数量、延迟和访存比例推出来的；
> 4. 多线程为什么第一版只扩展到 67%，以及怎么修；
> 5. Accelerate 比我们快 3 倍的那部分来自哪里。

[第二篇](/posts/gemm-02-blocking/)的结论是：先用寄存器分块把计算强度提上去，再用缓存分块让 B 留在 L2 里。n = 4096 时到了 **78 GFLOPS**，但离小矩阵时的 104、离单核峰值 125 都还有距离。

测试环境同前两篇：Apple M4 Pro，clang 23，`-O3 -mcpu=apple-m4`，`float32`，行优先存储，结果与 Accelerate 逐元素比对。完整代码在文末，可以直接编译运行。

## 一、还剩什么问题：访问太分散

第二篇的 4×16 小块里，对每个 k：

- B：读一行里的 16 个 float（64 字节），下一个 k 跳到下一行，相隔 n 个 float；
- A：从 4 个相隔 n 的行里各读 1 个 float。

即使 B 块已经在 L2 里，这些读取仍然非常分散：一条 128 字节的缓存行只用到一半，硬件预取器很难猜到下一步要读哪里，n = 4096 时每一步还要换一个 16 KB 的内存页。

## 二、打包：按微内核读取的顺序重排

GotoBLAS 的做法是：在计算之前，先把要用到的 A 块和 B 块**拷贝**到一块连续的缓冲区里，排成微内核读取的顺序。这一步叫**打包**（packing）。

<svg viewBox="0 0 720 148" role="img" aria-label="打包前后 B 的内存布局" style="width:100%;height:auto;display:block;margin:1.5rem auto" xmlns="http://www.w3.org/2000/svg"><style>.mm-t{fill:var(--primary);font-family:var(--font-mono);font-size:12px}.mm-s{fill:var(--secondary);font-family:var(--font-mono);font-size:11px}.mm-h{fill:var(--primary);font-family:var(--font-ui);font-size:13px;font-weight:600}.mm-cell{stroke:var(--math-rule);stroke-width:1;fill:transparent}.mm-good{fill:#40a8a0;fill-opacity:.35;stroke:#40a8a0}.mm-bad{fill:#e5864b;fill-opacity:.35;stroke:#e5864b}.mm-one{fill:var(--math-accent);fill-opacity:.35;stroke:var(--math-accent)}.mm-reg{fill:var(--math-accent);fill-opacity:.18;stroke:var(--math-accent)}.mm-p0{fill:#40a8a0;fill-opacity:.38;stroke:#40a8a0}.mm-p1{fill:#9d7cd8;fill-opacity:.38;stroke:#9d7cd8}.mm-arrow{stroke:var(--primary);stroke-width:1.6;fill:none;marker-end:url(#gm-ah)}.mm-axis{fill:none;stroke:var(--math-rule);stroke-width:1}.mm-dash{fill:none;stroke:var(--secondary);stroke-width:1;stroke-dasharray:4 3}.mm-bar{fill:var(--math-accent);fill-opacity:.75}.mm-bar2{fill:#40a8a0;fill-opacity:.75}.mm-bar3{fill:#e5864b;fill-opacity:.75}.mm-bar4{fill:var(--secondary);fill-opacity:.45}.mm-box{fill:var(--math-accent);fill-opacity:.05;stroke:var(--math-rule);stroke-width:1.2}</style><defs><marker id="gm-ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--primary)"/></marker></defs><text class="mm-h" x="16" y="20">打包 B：把 KC×NC 的块切成 12 列宽的微面板，按“微内核读取的顺序”重排</text><rect class="mm-p0" x="20" y="44" width="13" height="13"/><text class="mm-s" x="22" y="54" style="font-size:8px">k0</text><rect class="mm-p0" x="33" y="44" width="13" height="13"/><rect class="mm-p0" x="46" y="44" width="13" height="13"/><rect class="mm-p0" x="59" y="44" width="13" height="13"/><rect class="mm-p0" x="72" y="44" width="13" height="13"/><rect class="mm-p0" x="85" y="44" width="13" height="13"/><rect class="mm-p0" x="98" y="44" width="13" height="13"/><rect class="mm-p0" x="111" y="44" width="13" height="13"/><rect class="mm-p0" x="124" y="44" width="13" height="13"/><rect class="mm-p0" x="137" y="44" width="13" height="13"/><rect class="mm-p0" x="150" y="44" width="13" height="13"/><rect class="mm-p0" x="163" y="44" width="13" height="13"/><rect class="mm-p1" x="176" y="44" width="13" height="13"/><text class="mm-s" x="178" y="54" style="font-size:8px">k0</text><rect class="mm-p1" x="189" y="44" width="13" height="13"/><rect class="mm-p1" x="202" y="44" width="13" height="13"/><rect class="mm-p1" x="215" y="44" width="13" height="13"/><rect class="mm-p1" x="228" y="44" width="13" height="13"/><rect class="mm-p1" x="241" y="44" width="13" height="13"/><rect class="mm-p1" x="254" y="44" width="13" height="13"/><rect class="mm-p1" x="267" y="44" width="13" height="13"/><rect class="mm-p1" x="280" y="44" width="13" height="13"/><rect class="mm-p1" x="293" y="44" width="13" height="13"/><rect class="mm-p1" x="306" y="44" width="13" height="13"/><rect class="mm-p1" x="319" y="44" width="13" height="13"/><rect class="mm-p0" x="20" y="57" width="13" height="13"/><text class="mm-s" x="22" y="67" style="font-size:8px">k1</text><rect class="mm-p0" x="33" y="57" width="13" height="13"/><rect class="mm-p0" x="46" y="57" width="13" height="13"/><rect class="mm-p0" x="59" y="57" width="13" height="13"/><rect class="mm-p0" x="72" y="57" width="13" height="13"/><rect class="mm-p0" x="85" y="57" width="13" height="13"/><rect class="mm-p0" x="98" y="57" width="13" height="13"/><rect class="mm-p0" x="111" y="57" width="13" height="13"/><rect class="mm-p0" x="124" y="57" width="13" height="13"/><rect class="mm-p0" x="137" y="57" width="13" height="13"/><rect class="mm-p0" x="150" y="57" width="13" height="13"/><rect class="mm-p0" x="163" y="57" width="13" height="13"/><rect class="mm-p1" x="176" y="57" width="13" height="13"/><text class="mm-s" x="178" y="67" style="font-size:8px">k1</text><rect class="mm-p1" x="189" y="57" width="13" height="13"/><rect class="mm-p1" x="202" y="57" width="13" height="13"/><rect class="mm-p1" x="215" y="57" width="13" height="13"/><rect class="mm-p1" x="228" y="57" width="13" height="13"/><rect class="mm-p1" x="241" y="57" width="13" height="13"/><rect class="mm-p1" x="254" y="57" width="13" height="13"/><rect class="mm-p1" x="267" y="57" width="13" height="13"/><rect class="mm-p1" x="280" y="57" width="13" height="13"/><rect class="mm-p1" x="293" y="57" width="13" height="13"/><rect class="mm-p1" x="306" y="57" width="13" height="13"/><rect class="mm-p1" x="319" y="57" width="13" height="13"/><rect class="mm-p0" x="20" y="70" width="13" height="13"/><text class="mm-s" x="22" y="80" style="font-size:8px">k2</text><rect class="mm-p0" x="33" y="70" width="13" height="13"/><rect class="mm-p0" x="46" y="70" width="13" height="13"/><rect class="mm-p0" x="59" y="70" width="13" height="13"/><rect class="mm-p0" x="72" y="70" width="13" height="13"/><rect class="mm-p0" x="85" y="70" width="13" height="13"/><rect class="mm-p0" x="98" y="70" width="13" height="13"/><rect class="mm-p0" x="111" y="70" width="13" height="13"/><rect class="mm-p0" x="124" y="70" width="13" height="13"/><rect class="mm-p0" x="137" y="70" width="13" height="13"/><rect class="mm-p0" x="150" y="70" width="13" height="13"/><rect class="mm-p0" x="163" y="70" width="13" height="13"/><rect class="mm-p1" x="176" y="70" width="13" height="13"/><text class="mm-s" x="178" y="80" style="font-size:8px">k2</text><rect class="mm-p1" x="189" y="70" width="13" height="13"/><rect class="mm-p1" x="202" y="70" width="13" height="13"/><rect class="mm-p1" x="215" y="70" width="13" height="13"/><rect class="mm-p1" x="228" y="70" width="13" height="13"/><rect class="mm-p1" x="241" y="70" width="13" height="13"/><rect class="mm-p1" x="254" y="70" width="13" height="13"/><rect class="mm-p1" x="267" y="70" width="13" height="13"/><rect class="mm-p1" x="280" y="70" width="13" height="13"/><rect class="mm-p1" x="293" y="70" width="13" height="13"/><rect class="mm-p1" x="306" y="70" width="13" height="13"/><rect class="mm-p1" x="319" y="70" width="13" height="13"/><rect class="mm-p0" x="20" y="83" width="13" height="13"/><text class="mm-s" x="22" y="93" style="font-size:8px">k3</text><rect class="mm-p0" x="33" y="83" width="13" height="13"/><rect class="mm-p0" x="46" y="83" width="13" height="13"/><rect class="mm-p0" x="59" y="83" width="13" height="13"/><rect class="mm-p0" x="72" y="83" width="13" height="13"/><rect class="mm-p0" x="85" y="83" width="13" height="13"/><rect class="mm-p0" x="98" y="83" width="13" height="13"/><rect class="mm-p0" x="111" y="83" width="13" height="13"/><rect class="mm-p0" x="124" y="83" width="13" height="13"/><rect class="mm-p0" x="137" y="83" width="13" height="13"/><rect class="mm-p0" x="150" y="83" width="13" height="13"/><rect class="mm-p0" x="163" y="83" width="13" height="13"/><rect class="mm-p1" x="176" y="83" width="13" height="13"/><text class="mm-s" x="178" y="93" style="font-size:8px">k3</text><rect class="mm-p1" x="189" y="83" width="13" height="13"/><rect class="mm-p1" x="202" y="83" width="13" height="13"/><rect class="mm-p1" x="215" y="83" width="13" height="13"/><rect class="mm-p1" x="228" y="83" width="13" height="13"/><rect class="mm-p1" x="241" y="83" width="13" height="13"/><rect class="mm-p1" x="254" y="83" width="13" height="13"/><rect class="mm-p1" x="267" y="83" width="13" height="13"/><rect class="mm-p1" x="280" y="83" width="13" height="13"/><rect class="mm-p1" x="293" y="83" width="13" height="13"/><rect class="mm-p1" x="306" y="83" width="13" height="13"/><rect class="mm-p1" x="319" y="83" width="13" height="13"/><text class="mm-s" x="20" y="112">原始 B（行优先）：同一个微面板的相邻两行相隔 n 个 float</text><path class="mm-arrow" d="M344 70 h40"/><rect class="mm-p0" x="394.0" y="57" width="3.0" height="26"/><rect class="mm-p0" x="397.6" y="57" width="3.0" height="26"/><rect class="mm-p0" x="401.2" y="57" width="3.0" height="26"/><rect class="mm-p0" x="404.8" y="57" width="3.0" height="26"/><rect class="mm-p0" x="408.4" y="57" width="3.0" height="26"/><rect class="mm-p0" x="412.0" y="57" width="3.0" height="26"/><rect class="mm-p0" x="415.6" y="57" width="3.0" height="26"/><rect class="mm-p0" x="419.2" y="57" width="3.0" height="26"/><rect class="mm-p0" x="422.8" y="57" width="3.0" height="26"/><rect class="mm-p0" x="426.4" y="57" width="3.0" height="26"/><rect class="mm-p0" x="430.0" y="57" width="3.0" height="26"/><rect class="mm-p0" x="433.6" y="57" width="3.0" height="26"/><rect class="mm-p0" x="437.2" y="57" width="3.0" height="26"/><rect class="mm-p0" x="440.8" y="57" width="3.0" height="26"/><rect class="mm-p0" x="444.4" y="57" width="3.0" height="26"/><rect class="mm-p0" x="448.0" y="57" width="3.0" height="26"/><rect class="mm-p0" x="451.6" y="57" width="3.0" height="26"/><rect class="mm-p0" x="455.2" y="57" width="3.0" height="26"/><rect class="mm-p0" x="458.8" y="57" width="3.0" height="26"/><rect class="mm-p0" x="462.4" y="57" width="3.0" height="26"/><rect class="mm-p0" x="466.0" y="57" width="3.0" height="26"/><rect class="mm-p0" x="469.6" y="57" width="3.0" height="26"/><rect class="mm-p0" x="473.2" y="57" width="3.0" height="26"/><rect class="mm-p0" x="476.8" y="57" width="3.0" height="26"/><rect class="mm-p0" x="480.4" y="57" width="3.0" height="26"/><rect class="mm-p0" x="484.0" y="57" width="3.0" height="26"/><rect class="mm-p0" x="487.6" y="57" width="3.0" height="26"/><rect class="mm-p0" x="491.2" y="57" width="3.0" height="26"/><rect class="mm-p0" x="494.8" y="57" width="3.0" height="26"/><rect class="mm-p0" x="498.4" y="57" width="3.0" height="26"/><rect class="mm-p0" x="502.0" y="57" width="3.0" height="26"/><rect class="mm-p0" x="505.6" y="57" width="3.0" height="26"/><rect class="mm-p0" x="509.2" y="57" width="3.0" height="26"/><rect class="mm-p0" x="512.8" y="57" width="3.0" height="26"/><rect class="mm-p0" x="516.4" y="57" width="3.0" height="26"/><rect class="mm-p0" x="520.0" y="57" width="3.0" height="26"/><rect class="mm-p0" x="523.6" y="57" width="3.0" height="26"/><rect class="mm-p0" x="527.2" y="57" width="3.0" height="26"/><rect class="mm-p0" x="530.8" y="57" width="3.0" height="26"/><rect class="mm-p0" x="534.4" y="57" width="3.0" height="26"/><rect class="mm-p0" x="538.0" y="57" width="3.0" height="26"/><rect class="mm-p0" x="541.6" y="57" width="3.0" height="26"/><rect class="mm-p0" x="545.2" y="57" width="3.0" height="26"/><rect class="mm-p0" x="548.8" y="57" width="3.0" height="26"/><rect class="mm-p0" x="552.4" y="57" width="3.0" height="26"/><rect class="mm-p0" x="556.0" y="57" width="3.0" height="26"/><rect class="mm-p0" x="559.6" y="57" width="3.0" height="26"/><rect class="mm-p0" x="563.2" y="57" width="3.0" height="26"/><rect class="mm-p1" x="566.8" y="57" width="3.0" height="26"/><rect class="mm-p1" x="570.4" y="57" width="3.0" height="26"/><rect class="mm-p1" x="574.0" y="57" width="3.0" height="26"/><rect class="mm-p1" x="577.6" y="57" width="3.0" height="26"/><rect class="mm-p1" x="581.2" y="57" width="3.0" height="26"/><rect class="mm-p1" x="584.8" y="57" width="3.0" height="26"/><rect class="mm-p1" x="588.4" y="57" width="3.0" height="26"/><rect class="mm-p1" x="592.0" y="57" width="3.0" height="26"/><rect class="mm-p1" x="595.6" y="57" width="3.0" height="26"/><rect class="mm-p1" x="599.2" y="57" width="3.0" height="26"/><rect class="mm-p1" x="602.8" y="57" width="3.0" height="26"/><rect class="mm-p1" x="606.4" y="57" width="3.0" height="26"/><rect class="mm-p1" x="610.0" y="57" width="3.0" height="26"/><rect class="mm-p1" x="613.6" y="57" width="3.0" height="26"/><rect class="mm-p1" x="617.2" y="57" width="3.0" height="26"/><rect class="mm-p1" x="620.8" y="57" width="3.0" height="26"/><rect class="mm-p1" x="624.4" y="57" width="3.0" height="26"/><rect class="mm-p1" x="628.0" y="57" width="3.0" height="26"/><rect class="mm-p1" x="631.6" y="57" width="3.0" height="26"/><rect class="mm-p1" x="635.2" y="57" width="3.0" height="26"/><rect class="mm-p1" x="638.8" y="57" width="3.0" height="26"/><rect class="mm-p1" x="642.4" y="57" width="3.0" height="26"/><rect class="mm-p1" x="646.0" y="57" width="3.0" height="26"/><rect class="mm-p1" x="649.6" y="57" width="3.0" height="26"/><rect class="mm-p1" x="653.2" y="57" width="3.0" height="26"/><rect class="mm-p1" x="656.8" y="57" width="3.0" height="26"/><rect class="mm-p1" x="660.4" y="57" width="3.0" height="26"/><rect class="mm-p1" x="664.0" y="57" width="3.0" height="26"/><rect class="mm-p1" x="667.6" y="57" width="3.0" height="26"/><rect class="mm-p1" x="671.2" y="57" width="3.0" height="26"/><rect class="mm-p1" x="674.8" y="57" width="3.0" height="26"/><rect class="mm-p1" x="678.4" y="57" width="3.0" height="26"/><rect class="mm-p1" x="682.0" y="57" width="3.0" height="26"/><rect class="mm-p1" x="685.6" y="57" width="3.0" height="26"/><rect class="mm-p1" x="689.2" y="57" width="3.0" height="26"/><rect class="mm-p1" x="692.8" y="57" width="3.0" height="26"/><rect class="mm-p1" x="696.4" y="57" width="3.0" height="26"/><rect class="mm-p1" x="700.0" y="57" width="3.0" height="26"/><rect class="mm-p1" x="703.6" y="57" width="3.0" height="26"/><rect class="mm-p1" x="707.2" y="57" width="3.0" height="26"/><rect class="mm-p1" x="710.8" y="57" width="3.0" height="26"/><rect class="mm-p1" x="714.4" y="57" width="3.0" height="26"/><rect class="mm-p1" x="718.0" y="57" width="3.0" height="26"/><rect class="mm-p1" x="721.6" y="57" width="3.0" height="26"/><rect class="mm-p1" x="725.2" y="57" width="3.0" height="26"/><rect class="mm-p1" x="728.8" y="57" width="3.0" height="26"/><rect class="mm-p1" x="732.4" y="57" width="3.0" height="26"/><rect class="mm-p1" x="736.0" y="57" width="3.0" height="26"/><text class="mm-s" x="396.0" y="53" style="font-size:8px">k0</text><text class="mm-s" x="439.2" y="53" style="font-size:8px">k1</text><text class="mm-s" x="482.4" y="53" style="font-size:8px">k2</text><text class="mm-s" x="525.6" y="53" style="font-size:8px">k3</text><text class="mm-s" x="568.8" y="53" style="font-size:8px">k0</text><text class="mm-s" x="612.0" y="53" style="font-size:8px">k1</text><text class="mm-s" x="655.2" y="53" style="font-size:8px">k2</text><text class="mm-s" x="698.4" y="53" style="font-size:8px">k3</text><text class="mm-s" x="394" y="112">打包后：微面板 0 的 k0, k1, … 首尾相接，再接微面板 1</text><text class="mm-s" x="16" y="136">微内核每一步 k 只读接下来的 12 个 float：完全顺序，预取器和 TLB 都轻松。A 同理，按 8 行一组打包。</text></svg>

B 被切成 12 列宽的**微面板**：一个微面板里 k = 0 的 12 个数、k = 1 的 12 个数……首尾相接。微内核每一步 k 只读接下来的 12 个 float，完全顺序。A 同理，切成 8 行一组，每一步 k 读接下来的 8 个 float。

```c
// Pack B[pc:pc+kc, jc:jc+nc] into micro-panels of NR columns: Bp[panel][k][0..NR), zero-padded.
static void pack_B(int kc, int nc, const float *B, int ldb, float *Bp) {
    for (int j = 0; j < nc; j += NR) {
        int w = nc - j < NR ? nc - j : NR;
        for (int k = 0; k < kc; k++) {
            const float *src = &B[k * ldb + j];
            int x = 0;
            for (; x < w; x++) Bp[x] = src[x];
            for (; x < NR; x++) Bp[x] = 0;
            Bp += NR;
        }
    }
}
```

```c
// Pack A[ic:ic+mc, pc:pc+kc] into micro-panels of MR rows: Ap[panel][k][0..MR), zero-padded.
static void pack_A(int mc, int kc, const float *A, int lda, float *Ap) {
    for (int i = 0; i < mc; i += MR) {
        int h = mc - i < MR ? mc - i : MR;
        for (int k = 0; k < kc; k++) {
            int r = 0;
            for (; r < h; r++) Ap[r] = A[(i + r) * lda + k];
            for (; r < MR; r++) Ap[r] = 0;
            Ap += MR;
        }
    }
}
```

不够 12 列或 8 行的边角用 **0 填充**。这样微内核永远处理完整的 8×12，不需要为边界写特殊版本，n 也不必是 8 或 12 的倍数。

拷贝本身有成本，但很划算：一个打包好的 B 块会被所有行反复使用，一个打包好的 A 块会被 B 块里所有的微面板使用。拷贝是 $O(\text{块大小})$，复用是 $O(\text{块大小} \times \text{另一维})$。

## 三、五层循环：每一层管一块数据住在哪

打包之后，整个算法是五层循环，每一层决定一块数据驻留在哪一级存储：

<svg viewBox="0 0 700 204" role="img" aria-label="BLIS 风格的五层循环与缓存层级" style="width:100%;height:auto;display:block;margin:1.5rem auto" xmlns="http://www.w3.org/2000/svg"><style>.mm-t{fill:var(--primary);font-family:var(--font-mono);font-size:12px}.mm-s{fill:var(--secondary);font-family:var(--font-mono);font-size:11px}.mm-h{fill:var(--primary);font-family:var(--font-ui);font-size:13px;font-weight:600}.mm-cell{stroke:var(--math-rule);stroke-width:1;fill:transparent}.mm-good{fill:#40a8a0;fill-opacity:.35;stroke:#40a8a0}.mm-bad{fill:#e5864b;fill-opacity:.35;stroke:#e5864b}.mm-one{fill:var(--math-accent);fill-opacity:.35;stroke:var(--math-accent)}.mm-reg{fill:var(--math-accent);fill-opacity:.18;stroke:var(--math-accent)}.mm-p0{fill:#40a8a0;fill-opacity:.38;stroke:#40a8a0}.mm-p1{fill:#9d7cd8;fill-opacity:.38;stroke:#9d7cd8}.mm-arrow{stroke:var(--primary);stroke-width:1.6;fill:none;marker-end:url(#gm-ah)}.mm-axis{fill:none;stroke:var(--math-rule);stroke-width:1}.mm-dash{fill:none;stroke:var(--secondary);stroke-width:1;stroke-dasharray:4 3}.mm-bar{fill:var(--math-accent);fill-opacity:.75}.mm-bar2{fill:#40a8a0;fill-opacity:.75}.mm-bar3{fill:#e5864b;fill-opacity:.75}.mm-bar4{fill:var(--secondary);fill-opacity:.45}.mm-box{fill:var(--math-accent);fill-opacity:.05;stroke:var(--math-rule);stroke-width:1.2}</style><text class="mm-h" x="16" y="20">五层循环：每一层决定一块数据住在哪一级存储</text><rect class="mm-box" x="16" y="34" width="660" height="158" rx="6"/><text class="mm-t" x="26" y="53">for jc：每次 NC = 3072 列</text><text class="mm-s" x="666" y="53" text-anchor="end">B 面板 KC×NC（3 MB）→ L2 / SLC</text><rect class="mm-box" x="34" y="64" width="624" height="128" rx="6"/><text class="mm-t" x="44" y="83">for pc：每次 KC = 256 个 k</text><text class="mm-s" x="648" y="83" text-anchor="end">打包 B（所有线程共享）</text><rect class="mm-box" x="52" y="94" width="588" height="98" rx="6"/><text class="mm-t" x="62" y="113">for ic：每次 MC = 128 行（多线程并行）</text><text class="mm-s" x="630" y="113" text-anchor="end">打包 A 块 MC×KC（128 KB）→ L2</text><rect class="mm-box" x="70" y="124" width="552" height="68" rx="6"/><text class="mm-t" x="80" y="143">for jr：每次 NR = 12 列</text><text class="mm-s" x="612" y="143" text-anchor="end">B 微面板 KC×12（12 KB）→ L1</text><rect class="mm-box" x="88" y="154" width="516" height="38" rx="6"/><text class="mm-t" x="98" y="173">for ir：每次 MR = 8 行</text><text class="mm-s" x="594" y="173" text-anchor="end">8×12 微内核：C 小块在 24 个寄存器里</text></svg>

| 层 | 块 | 大小（float32） | 驻留位置 |
|---|---|---|---|
| jc | B 面板 KC×NC = 256×3072 | 3 MB | L2（16 MB）/ SLC |
| ic | A 块 MC×KC = 128×256 | 128 KB | L2 |
| jr | B 微面板 KC×NR = 256×12 | 12 KB | L1（128 KB） |
| ir | A 微面板 KC×MR = 256×8 | 8 KB | L1 |
| 微内核 | C 小块 MR×NR = 8×12 | 96 个 float | 24 个寄存器 |

最内层的微内核每次都读同一个 B 微面板（12 KB，在 L1 里），换着读不同的 A 微面板。这些大小都是从缓存容量倒推出来的：L1 要同时装下一个 B 微面板和几个 A 微面板，L2 要装下 A 块和正在用的那部分 B 面板。

## 四、8×12 微内核

绝大部分计算都发生在微内核里，它的形状由三个约束决定：

1. **寄存器**：NEON 有 32 个 128 位寄存器。8×12 的 C 小块需要 $8 \times 12 / 4 = 24$ 个累加器，加上 3 个 B 向量、2 个 A 向量，一共 29 个，刚好放得下。8×16 就需要 32 个累加器，放不下。
2. **延迟**：[第二篇](/posts/gemm-02-blocking/)测出至少要 12.5 条独立的乘加链。24 个累加器互不依赖，绰绰有余。
3. **访存比例**：每一步 k 读 2 个 A 向量和 3 个 B 向量（共 5 次读取），做 24 条 `fmla.4s`，每条乘加只伴随约 0.21 次访存。第二篇的 4×16 是 0.5，ikj 是 3。

```c
// 8 x 12 micro-kernel: acc += Ap (8 x kc) * Bp (kc x 12). 24 accumulators + 3 B + 2 A = 29 of 32 registers.
// Per k: 5 vector loads, 24 fmla (192 flops).
static void kernel_8x12(int kc, const float *Ap, const float *Bp, float *C, int ldc, int h, int w) {
    float32x4_t c[8][3];
    for (int r = 0; r < 8; r++) c[r][0] = c[r][1] = c[r][2] = vdupq_n_f32(0);
    for (int k = 0; k < kc; k++) {
        float32x4_t a0 = vld1q_f32(Ap), a1 = vld1q_f32(Ap + 4);
        float32x4_t b0 = vld1q_f32(Bp), b1 = vld1q_f32(Bp + 4), b2 = vld1q_f32(Bp + 8);
#define ROW(r, a, l)                                \
        c[r][0] = vfmaq_laneq_f32(c[r][0], b0, a, l); \
        c[r][1] = vfmaq_laneq_f32(c[r][1], b1, a, l); \
        c[r][2] = vfmaq_laneq_f32(c[r][2], b2, a, l);
        ROW(0, a0, 0) ROW(1, a0, 1) ROW(2, a0, 2) ROW(3, a0, 3)
        ROW(4, a1, 0) ROW(5, a1, 1) ROW(6, a1, 2) ROW(7, a1, 3)
#undef ROW
        Ap += MR;
        Bp += NR;
    }
    if (h == MR && w == NR) {                       // full tile: C += acc straight from registers
        for (int r = 0; r < 8; r++) {
            float *cr = &C[r * ldc];
            vst1q_f32(cr, vaddq_f32(vld1q_f32(cr), c[r][0]));
            vst1q_f32(cr + 4, vaddq_f32(vld1q_f32(cr + 4), c[r][1]));
            vst1q_f32(cr + 8, vaddq_f32(vld1q_f32(cr + 8), c[r][2]));
        }
    } else {                                        // edge tile: spill, then add the valid h x w part
        float t[8][12];
        for (int r = 0; r < 8; r++) vst1q_f32(t[r], c[r][0]), vst1q_f32(t[r] + 4, c[r][1]), vst1q_f32(t[r] + 8, c[r][2]);
        for (int r = 0; r < h; r++)
            for (int x = 0; x < w; x++) C[r * ldc + x] += t[r][x];
    }
}
```

`vfmaq_laneq_f32(c, b, a, l)` 对应 `fmla` 的"按元素"形式：用 a 的第 l 个元素乘以整个 b 向量。所以 8 个 A 值只需要两次向量读取，不用逐个广播。

结果（单线程）：

| n | 打包 + 8×12 微内核 | 占单核峰值 125 的比例 |
|---|---|---|
| 1000 | 121.5 | 97% |
| 2048 | 121.6 | 97% |
| 4096 | 121.6 | **97%** |

和第二篇不同，**矩阵大小对它几乎没有影响**：从 1000 到 4096 都是 121.6，n = 1000（不是 8 或 12 的倍数）也一样快。这就是打包加五层循环的意义：不管原始矩阵多大，微内核看到的永远是装在 L1 里、顺序排列的小块。

## 五、多线程：第一版只扩展到 67%

ic 这一层的 M 块之间互不依赖，可以并行。第一版用 GCD 的 `dispatch_apply` 起 T 个任务，每个任务按固定步长领取 M 块：

```c
static void sgemm(int m, int n, int k, const float *A, const float *B, float *C, int threads) {
    memset(C, 0, sizeof(float) * m * n);
    float *Bp = aligned_alloc(64, sizeof(float) * KC * (NC + NR));
    int mblocks = (m + MC - 1) / MC;
    float *Apool = aligned_alloc(64, sizeof(float) * (size_t)threads * (MC + MR) * KC);
    for (int jc = 0; jc < n; jc += NC) {
        int nc = n - jc < NC ? n - jc : NC;
        for (int pc = 0; pc < k; pc += KC) {
            int kc = k - pc < KC ? k - pc : KC;
            pack_B(kc, nc, &B[pc * n + jc], n, Bp);
            // M-blocks are independent: spread them over the threads (strided, so each thread
            // keeps one A buffer). dispatch_apply returns only when all are done.
            dispatch_apply(threads, DISPATCH_APPLY_AUTO, ^(size_t t) {
                float *Ap = Apool + t * (MC + MR) * KC;
                for (int b = t; b < mblocks; b += threads) {
                    int ic = b * MC, mc = m - ic < MC ? m - ic : MC;
                    pack_A(mc, kc, &A[ic * k + pc], k, Ap);
                    for (int jr = 0; jr < nc; jr += NR)
                        for (int ir = 0; ir < mc; ir += MR)
                            kernel_8x12(kc, &Ap[ir * kc], &Bp[jr * kc], &C[(ic + ir) * n + jc + jr], n,
                                        mc - ir < MR ? mc - ir : MR, nc - jr < NR ? nc - jr : NR);
                }
            });
        }
    }
    free(Bp);
    free(Apool);
}
```

| 线程数 | n = 4096 | 理想值（线程数 × 121.6） | 效率 |
|---|---|---|---|
| 4 | 442 | 486 | 91% |
| 10 | 812 | 1216 | **67%** |
| 14 | 900 | — | — |

10 个线程只扩展到 67%。有两个原因：

1. **打包 B 是串行的**。每个 (jc, pc) 块开始时，主线程独自打包 B（3 MB），其他线程都在等。按 Amdahl 定律，串行部分越占比，加线程的收益越小。
2. **负载不均**。n = 4096、MC = 128 时只有 32 个 M 块，分给 10 个线程要跑 4 轮，最后一轮只有 2 个线程在干活，理论上最多 $32 / 40 = 80\%$。

第二版把两处都改掉：**打包 B 也拆给所有线程**，每个线程负责一段连续的微面板；**每个 M 块单独作为一个任务**，交给 GCD 动态调度，谁空闲谁领，每个 M 块有自己的 A 缓冲区：

```c
static void sgemm(int m, int n, int k, const float *A, const float *B, float *C, int threads) {
    memset(C, 0, sizeof(float) * m * n);
    int mblocks = (m + MC - 1) / MC;
    float *Bp = aligned_alloc(64, sizeof(float) * KC * (NC + NR));
    // one A buffer per M-block, so GCD may hand blocks to whichever thread is free
    float *Apool = aligned_alloc(64, sizeof(float) * (size_t)mblocks * (MC + MR) * KC);
    for (int jc = 0; jc < n; jc += NC) {
        int nc = n - jc < NC ? n - jc : NC;
        for (int pc = 0; pc < k; pc += KC) {
            int kc = k - pc < KC ? k - pc : KC;
            // pack B in parallel too: each task packs a contiguous run of NR-wide micro-panels
            int panels = (nc + NR - 1) / NR, per = (panels + threads - 1) / threads;
            dispatch_apply(threads, DISPATCH_APPLY_AUTO, ^(size_t t) {
                int p0 = t * per, p1 = p0 + per > panels ? panels : p0 + per;
                if (p0 < p1) {
                    int j0 = p0 * NR, j1 = p1 * NR > nc ? nc : p1 * NR;
                    pack_B(kc, j1 - j0, &B[pc * n + jc + j0], n, Bp + (size_t)p0 * NR * kc);
                }
            });
            // one task per M-block, scheduled dynamically (threads == 1: plain serial loop)
            void (^block)(size_t) = ^(size_t b) {
                float *Ap = Apool + b * (MC + MR) * KC;
                int ic = b * MC, mc = m - ic < MC ? m - ic : MC;
                pack_A(mc, kc, &A[ic * k + pc], k, Ap);
                for (int jr = 0; jr < nc; jr += NR)
                    for (int ir = 0; ir < mc; ir += MR)
                        kernel_8x12(kc, &Ap[ir * kc], &Bp[jr * kc], &C[(ic + ir) * n + jc + jr], n,
                                    mc - ir < MR ? mc - ir : MR, nc - jr < NR ? nc - jr : NR);
            };
            if (threads == 1)
                for (int b = 0; b < mblocks; b++) block(b);
            else
                dispatch_apply(mblocks, DISPATCH_APPLY_AUTO, block);
        }
    }
    free(Bp);
    free(Apool);
}
```

| n | 第一版（14 线程） | 第二版（全部核心） |
|---|---|---|
| 2048 | — | 847 |
| 4096 | 900 | **1055** |

n = 4096 时到了 **1055 GFLOPS**，大约是 10 个性能核理想值的 87%。n = 2048 只有 16 个 M 块，分给 14 个核心仍然不均匀，所以低一些。剩下的损失还有：4 个能效核比性能核慢得多、所有核心共享 229 GB/s 的带宽，以及每个 (jc, pc) 块之间的同步。

## 六、Accelerate 为什么还能再快 3 倍

| | 单线程 | 全部核心 |
|---|---|---|
| 我们的 NEON 版本 | 121.6 | 1055 |
| Accelerate `cblas_sgemm`（n ≥ 1024） | 1600～1700 | 2800～3300 |

Accelerate **单线程**就有约 1.6 TFLOPS，是单核 NEON 实测峰值（125）的 13 倍。这不可能来自 NEON。M4 支持 Arm 的 SME（Scalable Matrix Extension）：

```bash
sysctl hw.optional.arm.FEAT_SME hw.optional.arm.FEAT_SME2 hw.optional.arm.SME_F32F32
# 都返回 1
```

SME 有一组专门的矩阵寄存器（ZA），一条指令就能完成两个向量的外积并累加到一整块矩阵上。这正是第二篇里寄存器分块在做的事，只不过是由硬件一次完成。13 倍的差距只能用这个矩阵单元来解释：Accelerate 的单线程调用把计算交给了它，所以 `VECLIB_MAXIMUM_THREADS=1` 也挡不住它。

所以，比较公平的说法是：**这个 NEON 实现已经跑到了 NEON 的 97%**；要再往上，就得换硬件路径，直接写 SME，而那是另一个系列了。

## 七、回顾整个系列

<svg viewBox="0 0 700 278" role="img" aria-label="GEMM 系列各阶段的性能" style="width:100%;height:auto;display:block;margin:1.5rem auto" xmlns="http://www.w3.org/2000/svg"><style>.mm-t{fill:var(--primary);font-family:var(--font-mono);font-size:12px}.mm-s{fill:var(--secondary);font-family:var(--font-mono);font-size:11px}.mm-h{fill:var(--primary);font-family:var(--font-ui);font-size:13px;font-weight:600}.mm-cell{stroke:var(--math-rule);stroke-width:1;fill:transparent}.mm-good{fill:#40a8a0;fill-opacity:.35;stroke:#40a8a0}.mm-bad{fill:#e5864b;fill-opacity:.35;stroke:#e5864b}.mm-one{fill:var(--math-accent);fill-opacity:.35;stroke:var(--math-accent)}.mm-reg{fill:var(--math-accent);fill-opacity:.18;stroke:var(--math-accent)}.mm-p0{fill:#40a8a0;fill-opacity:.38;stroke:#40a8a0}.mm-p1{fill:#9d7cd8;fill-opacity:.38;stroke:#9d7cd8}.mm-arrow{stroke:var(--primary);stroke-width:1.6;fill:none;marker-end:url(#gm-ah)}.mm-axis{fill:none;stroke:var(--math-rule);stroke-width:1}.mm-dash{fill:none;stroke:var(--secondary);stroke-width:1;stroke-dasharray:4 3}.mm-bar{fill:var(--math-accent);fill-opacity:.75}.mm-bar2{fill:#40a8a0;fill-opacity:.75}.mm-bar3{fill:#e5864b;fill-opacity:.75}.mm-bar4{fill:var(--secondary);fill-opacity:.45}.mm-box{fill:var(--math-accent);fill-opacity:.05;stroke:var(--math-rule);stroke-width:1.2}</style><text class="mm-h" x="16" y="18">整个系列：GFLOPS（对数坐标）</text><path class="mm-dash" d="M190.0 28 V238" style="stroke-opacity:.5"/><text class="mm-s" x="190.0" y="252" text-anchor="middle">0.1</text><path class="mm-dash" d="M278.0 28 V238" style="stroke-opacity:.5"/><text class="mm-s" x="278.0" y="252" text-anchor="middle">1</text><path class="mm-dash" d="M366.0 28 V238" style="stroke-opacity:.5"/><text class="mm-s" x="366.0" y="252" text-anchor="middle">10</text><path class="mm-dash" d="M454.0 28 V238" style="stroke-opacity:.5"/><text class="mm-s" x="454.0" y="252" text-anchor="middle">100</text><path class="mm-dash" d="M542.0 28 V238" style="stroke-opacity:.5"/><text class="mm-s" x="542.0" y="252" text-anchor="middle">1000</text><path class="mm-dash" d="M630.0 28 V238" style="stroke-opacity:.5"/><text class="mm-s" x="630.0" y="252" text-anchor="middle">10000</text><text class="mm-t" x="182" y="47" text-anchor="end">jki（最差循环顺序）</text><rect class="mm-bar3" x="190" y="34" width="57.5" height="18" rx="2"/><text class="mm-t" x="253.5" y="47">0.45</text><text class="mm-t" x="182" y="73" text-anchor="end">ijk（教科书写法）</text><rect class="mm-bar3" x="190" y="60" width="123.8" height="18" rx="2"/><text class="mm-t" x="319.8" y="73">2.55</text><text class="mm-t" x="182" y="99" text-anchor="end">ikj（循环换序）</text><rect class="mm-bar4" x="190" y="86" width="221.6" height="18" rx="2"/><text class="mm-t" x="417.6" y="99">32.98</text><text class="mm-t" x="182" y="125" text-anchor="end">寄存器块 + 缓存分块</text><rect class="mm-bar" x="190" y="112" width="254.4" height="18" rx="2"/><text class="mm-t" x="450.4" y="125">77.7</text><text class="mm-t" x="182" y="151" text-anchor="end">打包 + 8×12 微内核</text><rect class="mm-bar2" x="190" y="138" width="271.5" height="18" rx="2"/><text class="mm-t" x="467.5" y="151">121.6</text><text class="mm-t" x="182" y="177" text-anchor="end">同上，全部核心</text><rect class="mm-bar2" x="190" y="164" width="354.0" height="18" rx="2"/><text class="mm-t" x="550.0" y="177">1055</text><text class="mm-t" x="182" y="203" text-anchor="end">Accelerate，单线程</text><rect class="mm-bar4" x="190" y="190" width="370.0" height="18" rx="2"/><text class="mm-t" x="566.0" y="203">1600</text><text class="mm-t" x="182" y="229" text-anchor="end">Accelerate，全部核心</text><rect class="mm-bar4" x="190" y="216" width="394.0" height="18" rx="2"/><text class="mm-t" x="590.0" y="229">3000</text><text class="mm-s" x="16" y="268">第一篇 → 第二篇 → 第三篇；n = 1024 / 4096，M4 Pro。Accelerate 用的是 SME 矩阵单元</text></svg>

| 阶段 | 做了什么 | GFLOPS | 瓶颈 |
|---|---|---|---|
| jki | 最差的循环顺序 | 0.45 | A、C 都跨步，无法向量化 |
| ijk | 教科书写法 | 2.6 | B 跨步，点积是一条加法链 |
| ikj（第一篇） | 换循环顺序，内层连续 | 33 | 每次乘加都读写一次 C |
| 寄存器块（第二篇） | C 小块放进寄存器 | 104（n=1024） | 大矩阵时 B 从内存反复搬 |
| + 缓存分块 | B 块留在 L2 | 78（n=4096） | 访问分散：缓存行、预取、TLB |
| 打包 + 8×12（本篇） | 顺序读取 + 更大的微内核 | 121.6 | 已到 NEON 峰值的 97% |
| 多线程 | 并行打包，动态调度 | 1055 | 能效核、带宽、同步 |
| Accelerate | SME 矩阵单元 | 1600～3300 | 不同的硬件 |

从 0.45 到 1055，相差约 2300 倍，算的是同一个 $C = AB$。每一步都在回答同一个问题：**此刻 CPU 在等什么？** 等加法链、等 C 的读写、等内存、等 TLB、等其他线程，每次找到它，就换一种手段把它去掉。

## 思考题

1. 把微内核改成 12×8（12 行 × 8 列）：需要多少个累加器？每一步 k 读几个向量？和 8×12 相比，哪个访存比例更好？
2. KC 从 256 改成 512，B 微面板和 A 微面板分别有多大？还能都放进 L1 吗？实测验证。
3. 第二版里每个 M 块有自己的 A 缓冲区，一共要多少内存？如果 m 非常大，你会怎么改？
4. 打包 B 的串行部分在第一版里大约占多少时间？用"打包一个 B 面板的时间 ÷ 一个 (jc, pc) 块的总时间"估算，再用 Amdahl 定律算一下 10 个线程的理论上限，和实测的 67% 对得上吗？
5. 要把这个实现推广到 $C = \alpha AB + \beta C$、转置的 A 或 B，需要改哪几个函数？

## 完整代码

```bash
clang -O3 -mcpu=apple-m4 -DACCELERATE_NEW_LAPACK -o gemm gemm.c -framework Accelerate
VECLIB_MAXIMUM_THREADS=1 ./gemm 4096 1    # 单线程，对比单线程的 Accelerate
./gemm 4096 14                            # 全部核心
./gemm 1000 1                             # 非 8/12 倍数的尺寸，检查边界处理
```

```c
// Part 3: GotoBLAS/BLIS-style SGEMM on Apple silicon with NEON.
//   C (m x n) = A (m x k) * B (k x n), all row-major float.
// Loop nest (outer -> inner):
//   jc: NC columns of B/C      -> the packed B panel (KC x NC) lives in L2
//   pc: KC slice of k          -> pack B[pc:pc+KC, jc:jc+NC] once, shared by all threads
//   ic: MC rows of A/C (parallel) -> pack A[ic:ic+MC, pc:pc+KC]: each thread its own, ~L2
//   jr: NR = 12 columns         -> a KC x 12 micro-panel of B (12 KB) stays in L1
//   ir: MR = 8 rows             -> 8 x 12 micro-kernel, C tile in 24 NEON registers
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { MR = 8, NR = 12, KC = 256, MC = 128, NC = 3072 };

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

// Pack B[pc:pc+kc, jc:jc+nc] into micro-panels of NR columns: Bp[panel][k][0..NR), zero-padded.
static void pack_B(int kc, int nc, const float *B, int ldb, float *Bp) {
    for (int j = 0; j < nc; j += NR) {
        int w = nc - j < NR ? nc - j : NR;
        for (int k = 0; k < kc; k++) {
            const float *src = &B[k * ldb + j];
            int x = 0;
            for (; x < w; x++) Bp[x] = src[x];
            for (; x < NR; x++) Bp[x] = 0;
            Bp += NR;
        }
    }
}

// Pack A[ic:ic+mc, pc:pc+kc] into micro-panels of MR rows: Ap[panel][k][0..MR), zero-padded.
static void pack_A(int mc, int kc, const float *A, int lda, float *Ap) {
    for (int i = 0; i < mc; i += MR) {
        int h = mc - i < MR ? mc - i : MR;
        for (int k = 0; k < kc; k++) {
            int r = 0;
            for (; r < h; r++) Ap[r] = A[(i + r) * lda + k];
            for (; r < MR; r++) Ap[r] = 0;
            Ap += MR;
        }
    }
}

// 8 x 12 micro-kernel: acc += Ap (8 x kc) * Bp (kc x 12). 24 accumulators + 3 B + 2 A = 29 of 32 registers.
// Per k: 5 vector loads, 24 fmla (192 flops).
static void kernel_8x12(int kc, const float *Ap, const float *Bp, float *C, int ldc, int h, int w) {
    float32x4_t c[8][3];
    for (int r = 0; r < 8; r++) c[r][0] = c[r][1] = c[r][2] = vdupq_n_f32(0);
    for (int k = 0; k < kc; k++) {
        float32x4_t a0 = vld1q_f32(Ap), a1 = vld1q_f32(Ap + 4);
        float32x4_t b0 = vld1q_f32(Bp), b1 = vld1q_f32(Bp + 4), b2 = vld1q_f32(Bp + 8);
#define ROW(r, a, l)                                \
        c[r][0] = vfmaq_laneq_f32(c[r][0], b0, a, l); \
        c[r][1] = vfmaq_laneq_f32(c[r][1], b1, a, l); \
        c[r][2] = vfmaq_laneq_f32(c[r][2], b2, a, l);
        ROW(0, a0, 0) ROW(1, a0, 1) ROW(2, a0, 2) ROW(3, a0, 3)
        ROW(4, a1, 0) ROW(5, a1, 1) ROW(6, a1, 2) ROW(7, a1, 3)
#undef ROW
        Ap += MR;
        Bp += NR;
    }
    if (h == MR && w == NR) {                       // full tile: C += acc straight from registers
        for (int r = 0; r < 8; r++) {
            float *cr = &C[r * ldc];
            vst1q_f32(cr, vaddq_f32(vld1q_f32(cr), c[r][0]));
            vst1q_f32(cr + 4, vaddq_f32(vld1q_f32(cr + 4), c[r][1]));
            vst1q_f32(cr + 8, vaddq_f32(vld1q_f32(cr + 8), c[r][2]));
        }
    } else {                                        // edge tile: spill, then add the valid h x w part
        float t[8][12];
        for (int r = 0; r < 8; r++) vst1q_f32(t[r], c[r][0]), vst1q_f32(t[r] + 4, c[r][1]), vst1q_f32(t[r] + 8, c[r][2]);
        for (int r = 0; r < h; r++)
            for (int x = 0; x < w; x++) C[r * ldc + x] += t[r][x];
    }
}

static void sgemm(int m, int n, int k, const float *A, const float *B, float *C, int threads) {
    memset(C, 0, sizeof(float) * m * n);
    int mblocks = (m + MC - 1) / MC;
    float *Bp = aligned_alloc(64, sizeof(float) * KC * (NC + NR));
    // one A buffer per M-block, so GCD may hand blocks to whichever thread is free
    float *Apool = aligned_alloc(64, sizeof(float) * (size_t)mblocks * (MC + MR) * KC);
    for (int jc = 0; jc < n; jc += NC) {
        int nc = n - jc < NC ? n - jc : NC;
        for (int pc = 0; pc < k; pc += KC) {
            int kc = k - pc < KC ? k - pc : KC;
            // pack B in parallel too: each task packs a contiguous run of NR-wide micro-panels
            int panels = (nc + NR - 1) / NR, per = (panels + threads - 1) / threads;
            dispatch_apply(threads, DISPATCH_APPLY_AUTO, ^(size_t t) {
                int p0 = t * per, p1 = p0 + per > panels ? panels : p0 + per;
                if (p0 < p1) {
                    int j0 = p0 * NR, j1 = p1 * NR > nc ? nc : p1 * NR;
                    pack_B(kc, j1 - j0, &B[pc * n + jc + j0], n, Bp + (size_t)p0 * NR * kc);
                }
            });
            // one task per M-block, scheduled dynamically (threads == 1: plain serial loop)
            void (^block)(size_t) = ^(size_t b) {
                float *Ap = Apool + b * (MC + MR) * KC;
                int ic = b * MC, mc = m - ic < MC ? m - ic : MC;
                pack_A(mc, kc, &A[ic * k + pc], k, Ap);
                for (int jr = 0; jr < nc; jr += NR)
                    for (int ir = 0; ir < mc; ir += MR)
                        kernel_8x12(kc, &Ap[ir * kc], &Bp[jr * kc], &C[(ic + ir) * n + jc + jr], n,
                                    mc - ir < MR ? mc - ir : MR, nc - jr < NR ? nc - jr : NR);
            };
            if (threads == 1)
                for (int b = 0; b < mblocks; b++) block(b);
            else
                dispatch_apply(mblocks, DISPATCH_APPLY_AUTO, block);
        }
    }
    free(Bp);
    free(Apool);
}

int main(int argc, char **argv) {
    int n = atoi(argv[1]), threads = argc > 2 ? atoi(argv[2]) : 1;
    float *A = malloc(4ul * n * n), *B = malloc(4ul * n * n), *C = malloc(4ul * n * n), *R = malloc(4ul * n * n);
    srand(1);
    for (long i = 0; i < (long)n * n; i++) A[i] = rand() / (float)RAND_MAX - .5f, B[i] = rand() / (float)RAND_MAX - .5f;
    double best = 1e30;
    for (int r = 0; r < 3; r++) { double t0 = now(); cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, A, n, B, n, 0, R, n); double dt = now() - t0; if (dt < best) best = dt; }
    double blas = 2.0 * n * n * n / best / 1e9;
    best = 1e30;
    for (int r = 0; r < (n <= 2048 ? 3 : 2); r++) { double t0 = now(); sgemm(n, n, n, A, B, C, threads); double dt = now() - t0; if (dt < best) best = dt; }
    float err = 0;
    for (long i = 0; i < (long)n * n; i++) err = fmaxf(err, fabsf(C[i] - R[i]));
    printf("n=%-5d threads=%-2d  mine %8.1f GFLOPS   accelerate %8.1f GFLOPS   max|err|=%.1e\n", n, threads, 2.0 * n * n * n / best / 1e9, blas, err);
}
```
