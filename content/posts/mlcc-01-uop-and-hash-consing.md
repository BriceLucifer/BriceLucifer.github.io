---
title: "从零写一个 ML 编译器（一）：UOp、类型检查与 Hash Consing"
date: 2026-11-26
tags: ["Compiler", "C++", "ML Systems", "IR", "MLCC"]
description: "Progress log for MLCC, an ML compiler written from scratch in C++23: the UOp IR, strict dtype checking, a hash-consed graph, and fixing an exponential-time dtype check with memoization."
summary: "MLCC 是我用 C++23 从零写的 ML 编译器，目标是从 elementwise 程序一路走到在 GPU 上训练 LLM。这篇记录第一阶段：UOp 这个唯一的 IR 节点、严格的 dtype 推导、hash consing 的图，以及一个测试全绿却藏着 2ⁿ 复杂度的坑是怎么用记忆化修掉的。"
---

我在跟着一份 *Compilers for Machine Learning* 的课程大纲，从零写一个机器学习编译器，叫 **MLCC**。大纲只给规格不给实现：没有 starter code，语言自选，每周在上一周的基础上继续加功能，最终要能训练一个真正的 LLM。

大纲里有句话我很喜欢：

> This course aggressively builds on the previous week, and is an exercise in slop management. If you let any slop in early, it will compound and you will not finish the class.

所以第一周我刻意写得很慢，在 IR 和不变量上花了大部分时间。这篇是第一阶段的进度记录：**做完了什么、为什么这么设计、还差什么**。

## 路线图

| 周 | 主题 | 里程碑 | 状态 |
|---|---|---|---|
| 1 | UOp、elementwise、rewrite、常量折叠 | 编译并化简 elementwise 程序 | **进行中** |
| 2 | Renderer、runtime、memory | 生成并运行 C 代码 | |
| 3–4 | 循环、movement ops、rangeify | 把任意模型编译成 C（慢但正确） | |
| 5–6 | Kernel fusion、autodiff、memory hierarchy | 训练 MNIST，CPU 上的 SOTA 级 GEMM | |
| 7–8 | GPU、tensor core | 和 torch 相当的 CUDA/HIP 代码 | |
| 9–10 | 真实模型 | 训练 LLM | |

目前 Week 1 完成了 IR、类型推导和图的驻留（interning），rewrite engine 还只有一个空壳。

## 一切都是 UOp

整个编译器只有一种节点：

```cpp
struct UOp {
    Op op;                    // 做什么
    std::vector<UOpRef> src;  // 输入节点
    Arg arg;                  // 额外参数：常量值、CAST 的目标类型……
    DType dtype() const;
};

using UOpRef = std::shared_ptr<const UOp>;
```

没有 `AddNode`、`ConstNode` 这样的类层次。一个节点就是 `(op, src, arg)` 三元组，语义完全由 `op` 决定。这和大纲里约定的文本交换格式是一一对应的，比如一个 256×256 的 GEMM：

```text
%0 = buffer 65536 : dtype=f32 slot=0
%1 = buffer 65536 : dtype=f32 slot=1
%2 = reshape %0, (256, 256, 1)
%3 = reshape %1, (256, 1, 256)
%4 = mul %2, %3
%5 = reduce %4 : op=add pop=1
```

GEMM 不是一个原语，而是 reshape + 广播乘法 + reduce 拼出来的。后面几周的 movement ops 和 rangeify 都建立在这个统一的表示上。

`arg` 用 `std::variant` 表示，目前只有标量和 dtype：

```cpp
using Arg = std::variant<
    std::monostate,  // 大多数 op 没有参数，比如 ADD、NEG
    std::int32_t, float, double, bool,
    DType            // CAST 的目标类型
>;
```

`UOpRef` 是 `shared_ptr<const UOp>`：**节点一旦创建就不可变**。rewrite 不会原地修改节点，而是构造新节点。这一条是后面 hash consing 能成立的前提。

Week 1 的 op 一共 16 个：

| 类别 | Op |
|---|---|
| 常量与函数 | `CONST` `PARAM` `CALL` |
| 一元 | `CAST` `NEG` `RECIP` `EXP2` `LOG2` |
| 二元 | `ADD` `MUL` `MAX` `FLOORDIV` `FLOORMOD` |
| 比较 | `CMPLT` `CMPNE` |
| 三元 | `WHERE` |

没有 `SUB`、`DIV`、`EXP`：$a - b = a + (-b)$，$a / b = a \cdot \frac{1}{b}$，$e^x = 2^{x \log_2 e}$。op 越少，rewrite 规则和后端需要覆盖的情况就越少。

## dtype：严格，不做隐式提升

`dtype` 是一个递归性质，由 `op` 和输入的 dtype 推出来，第一次用到时再计算（后文会讲为什么要缓存）。我选择了最严格的规则：**二元运算两边类型必须相同，不做任何隐式类型提升**，需要时显式插入 `CAST`。

```cpp
case Op::ADD: case Op::MUL: /* ... */ {
    if (src.size() != 2)
        throw std::invalid_argument(prefix + "requires two sources");
    const DType lhs = src[0]->dtype();
    const DType rhs = src[1]->dtype();
    if (lhs != rhs)
        throw std::invalid_argument(prefix + "source dtypes must match; use an explicit CAST");
    // ...
}
```

`i32 + f32` 应该得到什么？C 的答案、NumPy 的答案和 PyTorch 的答案各不相同。与其在 IR 里内置某一套提升规则，不如让前端显式插入 `CAST`，IR 只负责检查。

几条具体规则：

- `EXP2`、`LOG2`、`RECIP` 只接受浮点；对 `i32` 求倒数必须先 `CAST`。
- `CMPLT`、`CMPNE` 的结果永远是 `BOOL`。
- `WHERE` 的条件必须是 `BOOL`，两个分支类型相同。
- `BOOL` 上的二元运算复用算术 op：`MUL` 是 AND，`MAX` 是 OR，`CMPNE` 是 XOR。这样不需要单独的逻辑运算 op。

两个整数 op 的语义需要特别写清楚：`FLOORDIV` 是**向下取整**，`FLOORMOD` 的结果和除数同号：

$$
\operatorname{FLOORDIV}(-7, 3) = \left\lfloor -\tfrac{7}{3} \right\rfloor = -3,
\qquad
\operatorname{FLOORMOD}(-7, 3) = -7 - 3 \cdot (-3) = 2
$$

C++ 的 `/` 是向零取整（`-7 / 3 == -2`），`%` 的符号跟被除数（`-7 % 3 == -1`）。等到写常量折叠和 C renderer 的时候，不能直接用 `/` 和 `%`。

## Hash Consing：结构相同的节点只存在一份

`Graph` 负责创建所有节点。创建前先查表，如果 `(op, src, arg)` 完全相同的节点已经存在，就直接返回它：

```cpp
UOpRef Graph::make(Op op, std::vector<UOpRef> src, Arg arg) {
    // ...检查每个 src 都非空，并且属于这个 Graph
    NodeKey key{op, src, arg};
    if (auto found = nodes_.find(key); found != nodes_.end())
        return found->second;               // 已存在：复用

    auto node = std::make_shared<const UOp>(op, src, arg);
    node->dtype();                          // 进图之前先做类型检查
    nodes_.emplace(key, node);
    return node;
}
```

这样做有三个直接的好处：

1. **免费的公共子表达式消除。** `a*b + a*b` 里的两个 `a*b` 是同一个节点。
2. **结构相等等于指针相等。** 判断两棵子树是否相同只需要比较指针，$O(1)$。
3. **rewrite 可以按节点做 memo。** 同一个子图不会被处理两次。

第 2 点也让 `NodeKey` 的比较变得很简单：因为 `src` 里的节点已经驻留过，比较 `src` 只要逐个比较指针，不需要递归比较子树。哈希也一样，把 `op`、`arg`、`src` 的指针依次组合起来：

$$
h \leftarrow h \oplus \left(v + \texttt{0x9e3779b9} + (h \ll 6) + (h \gg 2)\right)
$$

`Graph::make` 还会拒绝不属于当前图的输入节点。如果允许混用两个图的节点，"结构相等等于指针相等"这条不变量就会被悄悄破坏。

### 浮点常量按位比较

最容易出错的是 `CONST` 节点里的浮点数。如果用 `==` 比较 `arg`：

- `0.0 == -0.0` 为真，两个常量会被合并成一个节点。但 $1/0.0 = +\infty$，$1/(-0.0) = -\infty$，合并之后常量折叠的结果就错了。
- `NaN == NaN` 为假，同一个 NaN 常量永远查不到自己。`unordered_map` 要求比较是一个等价关系，`==` 对浮点数并不满足。

所以 `arg_equal` 和 `arg_hash` 对 `float`、`double` 都按 bit pattern 处理：

```cpp
if constexpr (std::is_same_v<T, float>) {
    return std::bit_cast<std::uint32_t>(x) == std::bit_cast<std::uint32_t>(y);
}
```

这样 `0.0` 和 `-0.0` 是两个不同的节点，同一个 NaN payload 是同一个节点，相等和哈希也始终一致。这几条都有对应的测试。

## 工程：一开始就把检查开满

既然大纲强调"slop 会复利增长"，我在第一周就把工具链配好了：

- **C++23**，禁止降级，关闭编译器扩展。
- **CMake Presets**：`debug` 和 `release` 两个预设，`cmake --workflow --preset debug` 一条命令完成配置、编译和测试。
- **ASan + UBSan 在 Release 下也开着。** 测试跑的是带优化的代码，优化后才暴露的未定义行为也能被抓到。
- **每个公开头文件单独编译一次**（`mlcc.headers` 测试），保证任何头文件都能单独 include。
- **GoogleTest 自动发现**：`tests/test_*.cpp` 自动成为测试目标。

目前 6 个测试文件、21 个测试全部通过。覆盖了 dtype 推导、非法输入的拒绝、`arg` 的相等与哈希、浮点特殊值，以及节点复用。

## 测试全绿，但有一个 2ⁿ 的坑

写这篇文章的时候，我回头看 `dtype()`，发现了一个测试没有覆盖到的问题。

`dtype()` 每次调用都会递归检查整棵子树，而且没有缓存结果。树形结构下这没问题，但 hash consing 把树变成了 **DAG**，共享的子节点会被重复遍历。最坏的情况是每一层都复用上一层：

```cpp
auto x = g.make(Op::CONST, {}, std::int32_t{1});
for (int i = 0; i < n; ++i)
    x = g.make(Op::ADD, {x, x}, {});   // x_i = x_{i-1} + x_{i-1}
```

图里只有 $n+1$ 个节点，但对 $x_n$ 求 dtype 的调用次数满足

$$
T(n) = 2\,T(n-1) + 1 \quad\Longrightarrow\quad T(n) = 2^{n+1} - 1
$$

而 `Graph::make` 每创建一个节点都会调用一次 `dtype()`。实测（`-O2`，不开 sanitizer）：

| n | 节点数 | 建图耗时 |
|---|---|---|
| 16 | 17 | 5.5 ms |
| 20 | 21 | 50 ms |
| 24 | 25 | 512 ms |

25 个节点要半秒，$n$ 每加 2，时间大约翻 4 倍。真实模型的计算图共享程度远比这个高。

### 修复：记忆化

最直接的想法是在构造节点时就算好 dtype 存起来，但这会改变现有的语义：测试里有一批**单独构造**的非法 `UOp`（比如只有一个输入的 `ADD`），约定是构造时不报错，调用 `dtype()` 时才抛异常，而且非法的子节点也要在父节点求 dtype 时报出来。

所以我选了记忆化：`dtype()` 第一次成功后把结果缓存起来，失败不缓存，下次调用照样抛异常。

```cpp
DType UOp::dtype() const {
    if (!dtype_cache_)
        dtype_cache_ = infer_dtype();  // 只有成功时才会写入
    return *dtype_cache_;
}
```

原来的推导逻辑原样搬进 `infer_dtype()`，它对子节点调用的 `src[i]->dtype()` 在第一次之后都直接命中缓存。每个节点只推导一次，整张图的代价是 $O(V + E)$。缓存是 `mutable std::optional<DType>`，所以这一步不是线程安全的，好在目前建图是单线程的，头文件注释里也写明了。

修复之后：

| n | 节点数 | 建图耗时 |
|---|---|---|
| 24 | 25 | 35 µs |
| 1,000 | 1,001 | 0.56 ms |
| 10,000 | 10,001 | 5.0 ms |
| 100,000 | 100,001 | 30 ms |

还加了两个回归测试：分别用单独构造的 `UOp` 和 `Graph::make` 建一条 64 层的共享链。没有记忆化的话，这需要 $2^{64}$ 次调用，永远跑不完。同时给所有单元测试加了 10 秒的超时，以后复杂度再退化会直接表现为测试失败，而不是 CI 卡住。

还有一处小的开销没有动：`Graph::make` 校验"输入属于本图"时，会为每个输入重新构造一个 `NodeKey`，这会复制它的 `src` 向量。可以改成在 `UOp` 里记录所属的图，或者另外维护一个指针集合。

## 下一步

Week 1 剩下的部分：

- **Rewrite engine。** 自底向上的贪心重写：模式 → 替换，跑到不动点为止。hash consing 让"这个子图处理过没有"的判断非常便宜。
- **常量折叠。** 用 rewrite engine 把只含常量的程序"跑"出结果，这也是 Week 1 的验收方式。
- **代数化简。** `x + 0 → x`、`x * 1 → x` 这类规则。要注意浮点上 `x * 0 → 0` **不成立**：`NaN * 0`、`inf * 0` 都是 NaN，`-3.0 * 0` 是 `-0.0`。
- **`PARAM` 和 `CALL`。** 目前这两个 op 会抛出 "not implemented"，需要先设计 `ParamArg` 和 `CallArg`。

然后进入 Week 2：toposort、把 UOp 图渲染成 C 函数、实现 runtime，让编译出的代码真正跑起来。

## Wrap-up

第一阶段的代码不多：头文件和实现加起来大约 440 行，测试大约 430 行。但几个最重要的决定都在这里定下来了：

1. **只有一种节点**：`(op, src, arg)`，不可变；
2. **类型规则严格**：不做隐式提升，需要转换时显式 `CAST`；
3. **hash consing**：结构相等等于指针相等，浮点常量按位比较；
4. **检查从第一天就开满**：ASan、UBSan、头文件独立编译、每个不变量都有测试。

还有一个教训：**测试全绿不代表复杂度是对的。** hash consing 把树变成 DAG 之后，所有递归遍历都要考虑共享，后面写 rewrite engine 时也一样。下一篇写 rewrite engine 和常量折叠。
