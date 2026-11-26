---
title: "CPU Whirl Shader（二）：删掉三分之一的三角函数，为什么一点没变快"
date: 2026-11-26
tags: ["Performance", "Hardware", "Rust"]
keywords: ["SIMD", "NEON", "std::simd", "Apple Silicon", "M4 Pro", "ILP", "Latency", "Profiling", "Rust", "Shader"]
series: ["CPU Whirl Shader"]
description: "Profiling a CPU fragment shader on an M4 Pro: removing a third of the libm calls changed nothing, interleaving four pixels gave 1.6x with identical output, and std::simd with a hand-written vector sin/cos reached 6.7x — because the loop was latency-bound, not throughput-bound."
summary: "上一篇在 CPU 上跑的 whirl shader，1280×720 一帧要 0.69 秒。这篇在 M4 Pro 上测量、猜错、再测量：删掉三分之一的三角函数毫无收益，同样的调用交错 4 个像素就快了 1.6 倍，std::simd 加手写的向量 sin/cos 最终快了 6.7 倍。原因只有一个：这个循环受的是延迟限制，不是吞吐限制。"
---

{{< series >}}

[上一篇](/posts/cowhirl-shader-cpu/)把一个 fragment shader 搬到了 CPU 上：每个像素跑 179 次迭代，Rayon 把像素分给所有核心。能跑，但不快：在 M4 Pro（10 个性能核 + 4 个能效核）上，1280×720 一帧要 **686 ms**。

这篇的目标很简单：**画面不变，让它变快，并且说清楚为什么快**。中间我猜错了两次，错的部分也一起记下来。

代码在 [BriceLucifer/shader](https://github.com/BriceLucifer/shader) 的 `perf` 分支。

## 先搭测量台

优化之前先要能可靠地测量。我给程序加了一个 `bench` 模式：只渲染、不写文件，并且把结果和原版逐字节比较：

```bash
cargo run --release -- bench <variant> [frames]
RAYON_NUM_THREADS=1 cargo run --release -- bench <variant> 1   # 单核
```

原版 shader 原封不动地留在 `src/reference.rs` 里，作为速度和正确性的基准。每个优化版本都是一个新的 variant，可以随时切换对比。

## 数一数：每次迭代 12 次三角函数

shader 的循环体（简化后）：

```rust
for _ in 1..180 {
    p = dir * z;  p.z -= t;
    // 旋转：3 次 cos
    let (a, b, c) = (cos(q), cos(q + 11.0), cos(q + 33.0));      // q = 0.1 * p.z
    p.xy = (p.x * a - p.y * b, p.x * c + p.y * a);
    // 形变：3 次 sin + 3 次 cos
    v = cos(p + sin(p.yzx / 0.3));
    v = max(v, v.zxy * 0.1);
    d = length(v) / 6.0;
    z += d;
    // 颜色：3 次 cos
    o += (cos(p.z + vec3(0, 1, 2)) + 1.0) / (d + 1e-3);
}
```

一帧是 $1280 \times 720 \times 179 \approx 1.65 \times 10^8$ 次迭代，每次 12 次 sin/cos，一帧就是**约 20 亿次三角函数**。

用 macOS 自带的 `sample` 采样单线程运行，结果和直觉一致：

| 函数 | 采样占比 |
|---|---|
| `sinf` | 61% |
| `cosf`（含调用桩） | 28% |
| shader 本身 | 8% |

**三角函数占了大约 90% 的时间。** 那就减少三角函数。

## 优化一：12 次减到 8 次，结果没有变快

两个代数技巧：

1. **提出循环不变量**：光线方向 `normalize(fc*2 - r.xyy)` 在循环里不变，算一次就够。
2. **和角公式**：`cos(q)`、`cos(q+11)`、`cos(q+33)` 只差常数，用 $\cos(q+k) = \cos q \cos k - \sin q \sin k$，一次 `sin_cos(q)` 加几次乘加就能全部得到。颜色项 `cos(p.z + {0,1,2})` 同理。

每次迭代从 12 次 libm 调用降到 8 次（3 次 sin、3 次 cos、2 次 `__sincosf_stret`），画面几乎一致。测一下单线程：

| 实现 | 单线程 / 帧 |
|---|---|
| 原版 | 7329 ms |
| 代数变换 | 7088 ms |

**快了 3%，在误差范围内。** 调用少了三分之一，时间却几乎没动。

### 错误猜测：大参数让 `sinf` 变慢？

采样里有个怪现象：每次迭代只有 3 次 `sin`、9 次 `cos`，`sinf` 的时间却是 `cosf` 的两倍多。而 `sin(p.y / 0.3)` 的参数会随光线前进变大，我猜 libm 对大参数要走更慢的范围约简路径；如果真是这样，我删掉的那几次参数很小的 `cos` 本来就很便宜。

写个微基准验证（`examples/sin_cost.rs`）：

| 参数范围 | `sinf` | `cosf` |
|---|---|---|
| $\lvert x \rvert < 1$ | 3.2 ns | 2.1 ns |
| $\lvert x \rvert < 100$ | 2.4 ns | 2.8 ns |
| $\lvert x \rvert < 10^5$ | 1.5 ns | 2.8 ns |

**耗时和参数大小无关**，都在 2 ns 左右，而且这个 shader 里 sin 的参数全都小于 100。猜错了。采样里 `sinf` 偏高，更可能是 Apple libm 里两者共用了同一段代码。

### 真正的原因：延迟，不是吞吐

换个角度算：12 次调用、每次约 2 ns，如果 CPU 能把它们排满，一次迭代应该在 25 ns 左右。实测单线程是 $7329\,\text{ms} / 1.65 \times 10^8 \approx 44$ ns。

差的那部分是**等待**。每次迭代里有一条串行的依赖链：

```text
z ─► p.z ─► cos(q) ─► 旋转 ─► sin(p.y/.3) ─► cos(p.x + …) ─► max ─► sqrt ─► d ─► z (下一次迭代)
            └► cos(q+11), cos(q+33)   ← 不在链上
                                            颜色: cos(p.z+1), cos(p.z+2) ← 不在链上
```

下一次迭代要等 `z` 更新，`z` 要等这条链走完。乱序执行的核心本来就会把链外的计算塞进等待的空隙里，所以**我删掉的四次调用恰好都在链外**，它们本来就是"免费"的。

验证这个说法：**调用一次不少，只是在一个循环里同时推进 4 个像素**：

```rust
for _ in 1..180 {
    for l in 0..4 {   // 4 条互不相关的依赖链
        /* 和上面完全相同的标量代码，作用在 pixel[l] 上 */
    }
}
```

| 实现 | 单线程 / 帧 | 加速 |
|---|---|---|
| 代数变换 | 7088 ms | 1× |
| **同样的调用，4 像素交错** | **4463 ms** | **1.59×** |

**计算量一点没少，输出逐位相同，快了 59%**。CPU 之前不是忙不过来，而是在等。

## 优化二：`std::simd`，一次算 16 个像素

既然瓶颈是"可以同时推进的独立工作太少"，SIMD 就是最直接的答案：一条指令处理多个像素，同时天然提供多条独立的依赖链。

问题是 `f32::sin` 只是标量的 libm 调用，没有向量版本，所以得自己写。做法是经典的两步：

1. **范围约简**：$j = \operatorname{round}(x \cdot 2/\pi)$，$r = x - j \cdot \pi/2$。$\pi/2$ 拆成三段常数（Cody–Waite），前两段的乘法是精确的，$r \in [-\pi/4, \pi/4]$。
2. **多项式**：在这个区间上，Cephes 的 minimax 多项式对 sin 和 cos 都能达到 float 精度；再按象限 $j \bmod 4$ 选择 sin 或 cos，并决定符号。

```rust
pub fn sin_cos<const N: usize>(x: Simd<f32, N>) -> (Simd<f32, N>, Simd<f32, N>) {
    let j = (x * splat(FRAC_2_PI)).round();
    let r = x - j * splat(1.570_312_5) - j * splat(4.837_513e-4) - j * splat(7.549_79e-8);
    let r2 = r * r;
    let s = r + r * r2 * (splat(-1.666_665_5e-1) + r2 * (splat(8.332_161e-3) + r2 * splat(-1.951_529_6e-4)));
    let c = splat(1.0) - splat(0.5) * r2
        + r2 * r2 * (splat(4.166_664_6e-2) + r2 * (splat(-1.388_731_6e-3) + r2 * splat(2.443_315_7e-5)));
    // 象限 q = j mod 4:  sin = [s, c, -s, -c][q],  cos = [c, -s, -c, s][q]
    let q = j.cast::<i32>();
    let swap    = (q & splat(1)).simd_ne(splat(0));
    let sin_neg = (q & splat(2)).simd_ne(splat(0));
    let cos_neg = ((q + splat(1)) & splat(2)).simd_ne(splat(0));
    let (sv, cv) = (swap.select(c, s), swap.select(s, c));
    (sin_neg.select(-sv, sv), cos_neg.select(-cv, cv))
}
```

整个 shader 改成对 `Simd<f32, N>` 运算，lane 数 N 做成泛型参数，于是可以直接比较不同宽度：

| 实现 | 单线程 / 帧 | 每像素·迭代 | 相对标量 |
|---|---|---|---|
| 标量 | 7088 ms | 43 ns | 1× |
| 标量 ×4 交错 | 4463 ms | 27 ns | 1.6× |
| SIMD 4 路 | 1957 ms | 11.9 ns | 3.6× |
| SIMD 8 路 | 1241 ms | 7.5 ns | 5.7× |
| **SIMD 16 路** | **1068 ms** | **6.5 ns** | **6.6×** |

M4 的 NEON 寄存器是 128 位，也就是 4 个 float。**那为什么 8 路、16 路还会更快？** 因为超过 4 路之后，`Simd<f32, 16>` 实际上是 4 个寄存器。多出来的不是更宽的向量，而是 4 条可以交错执行的独立依赖链，和"标量 ×4 交错"是同一个道理，只不过这次每条链本身也是向量。另外，内联的多项式没有函数调用的开销，编译器能把整段计算一起排指令。

## 多线程：最终 6.7 倍

最后把 14 个核心都用上（每行一个 Rayon 任务）：

| 实现 | 14 线程 / 帧 | 像素/秒 |
|---|---|---|
| 原版 | 686 ms | 1.34 M |
| 标量 ×4 交错 | 441 ms | 2.09 M |
| SIMD 8 路 | 126 ms | 7.34 M |
| **SIMD 16 路** | **102 ms** | **9.02 M** |

从单线程到 14 线程扩展了 10.5 倍（10 个性能核加 4 个较慢的能效核）。**一帧从 686 ms 降到 102 ms，快了 6.7 倍**；180 帧的动画从 2 分钟缩短到 18 秒。

## 画面真的没变吗

把 SIMD 16 路和原版在 $t = 1$ 这一帧上逐字节比较：

- **99.68%** 的字节完全相同；
- 0.31% 相差 1–2（8 位颜色的最后一位）；
- 92 万个像素里只有 **37 个**相差超过 2，最大差 159。

这 37 个像素很特别。下图左边是原版，中间是 SIMD 版，右边是差值放大 16 倍：

![原版、SIMD 版、差值 ×16 的局部放大](/images/shader/diff-zoom.webp)

差异都是单个像素，其中有 14 个是明显的"萤火虫"亮点，两张图里都有。它们出现在 $d$ 接近 0 的位置，累加项 $1/(d + 10^{-3})$ 会把最后一位的舍入误差放大几百倍。只用标量 libm、改了运算顺序的代数变换版本也有 35 个这样的像素，而且全部落在这 37 个里面，所以这是迭代本身对舍入敏感，不是向量 sin/cos 的精度问题。

## 学到的

1. **数调用次数，不等于估时间。** 删掉三分之一的三角函数毫无收益，因为它们不在关键路径上。
2. **先问循环是被延迟卡住，还是被吞吐卡住。** 一个简单的判断：同样的工作交错几份，如果变快了，就是延迟限制。
3. **SIMD 的收益有一部分来自 ILP。** 硬件只有 4 路宽，16 路的抽象仍然更快，因为它同时提供了多条独立的依赖链。
4. **先建测量台，再动手。** 每个版本都能一键测速、一键和原版对比，猜错的时候才能马上发现。

下一步想把同一个 shader 移到 GPU 上，用 roofline 模型看看它离硬件上限还有多远。
