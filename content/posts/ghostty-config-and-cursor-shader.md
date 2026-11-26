---
title: "Ghostty 配置笔记：从字体到一个 Neovide 风格的 Cursor Shader"
date: 2026-11-24
tags: ["Tooling", "Graphics"]
keywords: ["Ghostty", "Terminal", "Shader", "GLSL", "macOS", "Setup"]
math: true
description: "My Ghostty config, section by section, plus a custom GLSL cursor shader that reproduces Neovide's spring-animated cursor inside the terminal."
summary: "我的 Ghostty 配置逐段拆解：字体、主题、窗口、Quick Terminal、快捷键，以及一个自己写的 cursor shader，在终端里还原 Neovide 那种四个角各自弹簧跟随的光标动画。"
---

终端是我每天待得最久的地方，所以它值得被认真配置一次。我现在用的是 [Ghostty](https://ghostty.org)：GPU 渲染、原生 macOS 窗口、配置文件是纯文本，而且支持 **custom shader**——这一点让我把 Neovide 最让人上瘾的光标动画搬进了普通终端。

这篇分两部分：前半部分是配置逐段拆解，后半部分讲这个 cursor shader 是怎么写的。

<video controls autoplay loop muted playsinline width="100%" poster="/images/ghostty/cursor-trail-poster.webp">
  <source src="/images/ghostty/cursor-trail.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

*光标在 Neovim 里跳转时的效果：长距离跳转会拉出一道拖尾，打字时的短距离移动几乎是瞬间完成。*

## 配置文件在哪

Ghostty 读 `~/.config/ghostty/config`，格式是一行一个 `key = value`。改完按 `Cmd + Shift + ,` 重新加载，不需要重启。

```text
~/.config/ghostty/
├── config
└── shaders/
    └── cursor_neovide.glsl
```

## 字体：Maple Mono NF CN

```ini
font-family = Maple Mono NF CN
font-size = 17
font-feature = +calt
font-thicken = false
adjust-cell-height = 2
```

[Maple Mono](https://github.com/subframe7536/maple-font) 的 **NF CN** 版本一次解决三件事：英文是等宽编程字体，内置 [Nerd Font](https://www.nerdfonts.com) 图标（Neovim 里的文件图标、Git 分支符号都靠它），中文字形和英文严格 2:1 对齐。中英混排的注释不会歪。

- **官网 & 在线预览**：[font.subf.dev](https://font.subf.dev)，可以直接在网页上试各种连字和字重
- **GitHub**：[subframe7536/maple-font](https://github.com/subframe7536/maple-font)
- **下载**：[Releases](https://github.com/subframe7536/maple-font/releases) 页面里找 `MapleMono-NF-CN` 开头的压缩包

macOS 上用 Homebrew 一行装好：

```bash
brew install --cask font-maple-mono-nf-cn
```

装完在 Ghostty 里用 `ghostty +list-fonts | grep -i maple` 确认字体名，`font-family` 要和这里列出来的名字完全一致。

- `+calt` 打开 contextual alternates，`->`、`!=` 这类连字会生效。
- `adjust-cell-height = 2` 把行高稍微拉开一点，代码不那么挤。Neovide 里我用 `linespace = 2` 对应同一个值，两边看起来一致。

## 主题：和 Neovim 用同一套

```ini
theme = Catppuccin Mocha
```

[Catppuccin](https://catppuccin.com) Mocha 是 Ghostty 内置的主题，不用额外下载。我的 Neovim 用的也是 Catppuccin Mocha（[catppuccin/nvim](https://github.com/catppuccin/nvim)），而且开了 `transparent_background`。所以 **Neovim 里看到的底色其实就是 Ghostty 的底色**。终端和编辑器不会出现两种接近但不一样的深蓝，切进切出没有色差。

## 窗口

```ini
background-opacity = 1.0
background-blur = false
macos-titlebar-style = transparent
window-padding-x = 10
window-padding-y = 8
window-padding-balance = true
window-save-state = always
window-theme = auto
```

- 不透明、不模糊。半透明背景看久了眼睛累，代码对比度也会下降。
- `macos-titlebar-style = transparent`：标题栏和内容区同色，窗口看起来是一整块。
- `window-padding-balance = true`：格子对不齐窗口时，多出来的空白在四边平均分，而不是全堆在右下角。
- `window-save-state = always`：退出再打开，窗口、tab、分屏都恢复原样。

## 光标和鼠标

```ini
cursor-style = block
cursor-style-blink = true
cursor-color = cell-foreground
cursor-text = cell-background
cursor-opacity = 0.9
adjust-cursor-thickness = 2

mouse-hide-while-typing = true
copy-on-select = clipboard
focus-follows-mouse = true
```

`cursor-color = cell-foreground` 是我很喜欢的一个小设置：光标颜色取它所在格子的文字颜色，光标下的字用背景色画。效果就是光标停在关键字上是紫色，停在字符串上是绿色，**永远和代码高亮协调**，换主题也不用改。后面的 shader 会直接利用这一点。

`copy-on-select = clipboard` 选中即复制到系统剪贴板；`focus-follows-mouse` 让分屏之间鼠标移过去就聚焦。

## Quick Terminal：Quake 风格下拉终端

```ini
quick-terminal-position = top
quick-terminal-screen = mouse
quick-terminal-autohide = true
quick-terminal-animation-duration = 0.15

keybind = global:ctrl+grave_accent=toggle_quick_terminal
```

在任何 App 里按 `` Ctrl + ` ``，终端从屏幕顶部滑下来；鼠标在哪块屏幕就在哪块出现；点别的地方自动收起。跑个一次性命令（`brew upgrade`、查个端口）特别顺手，不用切窗口。

> `global:` 前缀的快捷键需要在「系统设置 → 隐私与安全性 → 辅助功能」里给 Ghostty 授权。

## Shell 集成与安全

```ini
command = /opt/homebrew/bin/fish
shell-integration = detect
shell-integration-features = no-cursor,sudo
link-url = true

clipboard-paste-protection = true
clipboard-paste-bracketed-safe = true
```

- Shell 是 fish。
- `no-cursor`：不让 shell integration 在提示符处把光标改成竖线。我要它一直是 block，这样 cursor shader 的效果在哪儿都一样。
- `sudo`：`sudo` 时保留 Ghostty 的 terminfo，`sudo vim` 之类不会显示错乱。
- 粘贴保护：粘贴的内容里如果有换行这类可能直接执行的东西，会先弹窗确认。

## 快捷键：Alt 给终端，Ctrl 给 Neovim

```ini
# 标签页
keybind = cmd+t=new_tab
keybind = cmd+shift+h=previous_tab
keybind = cmd+shift+l=next_tab
keybind = cmd+w=close_surface

# 分屏
keybind = cmd+d=new_split:right
keybind = cmd+shift+d=new_split:down
keybind = alt+h=goto_split:left
keybind = alt+l=goto_split:right
keybind = alt+k=goto_split:top
keybind = alt+j=goto_split:bottom
keybind = cmd+shift+e=equalize_splits
keybind = cmd+shift+f=toggle_split_zoom

# 字号与重载
keybind = cmd+plus=increase_font_size:1
keybind = cmd+minus=decrease_font_size:1
keybind = cmd+zero=reset_font_size
keybind = cmd+shift+comma=reload_config
```

这里有一个和 Neovim 配置配套的约定：

| 层级 | 修饰键 | 动作 |
|------|--------|------|
| Ghostty 分屏 | `Alt + h/j/k/l` | 在终端的 split 之间移动 |
| Neovim 窗口 | `Ctrl + h/j/k/l` | 在 nvim 的 window 之间移动 |
| Ghostty 标签页 | `Cmd + Shift + h/l` | 上一个 / 下一个 tab |

方向键全是 `hjkl`，用修饰键区分层级，两层互不抢键。`macos-option-as-alt = true` 让 Option 键作为 Alt 发出去，否则按 `Option + h` 只会打出一个 `˙`。

## Scrollback

```ini
scrollback-limit-bytes = unlimited
```

回滚不设上限，内存按需分配；需要释放时 `clear` 或 `Cmd + K` 会连同回滚一起清掉。编译日志、测试输出再长也能翻回去。

---

## Cursor Shader：把 Neovide 的光标搬进终端

用过 [Neovide](https://neovide.dev) 的人大概都对它的光标动画印象深刻：光标不是"瞬移"到新位置，而是**飞**过去，起飞时被拉长，落地时弹回方块。这个动画对追踪光标位置非常有帮助：`G`、`gd`、搜索跳转之后，眼睛不用在屏幕上找光标在哪。

Ghostty 支持 Shadertoy 风格的 GLSL post-processing shader，并且给 shader 暴露了光标相关的 uniform：

| Uniform | 含义 |
|---------|------|
| `iCurrentCursor` | 当前光标矩形（`xy` 左上角，`zw` 宽高） |
| `iPreviousCursor` | 上一次的光标矩形 |
| `iTimeCursorChange` | 光标最后一次移动的时间 |
| `iFocus` / `iCursorVisible` | 窗口是否聚焦、光标是否可见 |

有这些就够了。启用方式两行：

```ini
custom-shader = /Users/<you>/.config/ghostty/shaders/cursor_neovide.glsl
custom-shader-animation = true
```

### 模型：四个角，四根弹簧

Neovide 的光标动画核心思路是：**光标矩形的四个角各自是一个独立的弹簧**。

1. 光标从 `prev` 移动到 `cur` 时，四个角分别从旧位置向新位置运动。
2. 面朝移动方向的角（比如往右下跳时的右下角）动画时间短，几乎立刻到位；背向的角动画时间长，拖在后面。
3. 四个角围成的四边形就是被拉长的"飞行中"的光标。所有角都到位后，它又变回一个格子大小的方块。

每个角的位置用**临界阻尼弹簧**（critically damped spring）插值：没有过冲，不会抖，也不会闪。剩余距离随时间的变化是：

$$
f(t) = e^{-u}(1 + u), \quad u = \frac{4t}{L}
$$

其中 $L$ 是这个角的动画时长。$f$ 从 1 平滑地降到 0，在 $t = 2L$ 时已经小于 0.3%，所以 shader 在 `2 × length` 之后就直接退出、不再画任何东西。

每个角的动画时长由它和移动方向的"对齐程度"决定。设移动向量是 $d$，用曼哈顿距离归一化：

```glsl
float invManhattan = 0.5 / (abs(d.x) + abs(d.y));
float nTR = 0.5 + (d.x + d.y) * invManhattan;   // 右上角对齐度
float nTL = 0.5 + (d.y - d.x) * invManhattan;   // 左上角对齐度
vec4 alignment = vec4(nTL, nTR, 1.0 - nTL, 1.0 - nTR);  // TL, TR, BR, BL

vec4 animLen = mix(vec4(baseLength),
                   vec4(baseLength * (1.0 - TRAIL_SIZE)),
                   alignment);
```

对齐度为 1 的角动画时长是 `baseLength × (1 − TRAIL_SIZE)`；`TRAIL_SIZE = 1.0` 时就是 0，**瞬间到位**。对齐度为 0 的角用完整的 `baseLength`，拖得最长。

下面是同一次跳转（从第 5 行的 `ring_init` 跳到第 23 行的 `tail`）在 0 / 30 / 60 / 110 ms 时的样子：

![Cursor trail at 0, 30, 60 and 110 ms after a jump](/images/ghostty/cursor-trail-strip.webp)

*0 ms 时旧光标位置还在；30 ms 时目标位置的角已经到位，后面的角刚开始追；60 ms 拖尾明显缩短；110 ms 基本收拢。*

### 打字时不要拖尾

如果每次按键光标右移一格都拉出拖尾，打字时会非常吵。Neovide 对此的处理是：**同一行、两格以内**的移动算 "short hop"，用一个短得多的动画时长：

```glsl
bool shortHop =
    abs(d.x) <= 2.001 * iCurrentCursor.z &&
    abs(d.y) <= 0.01  * iCurrentCursor.w;

float baseLength = shortHop
    ? min(ANIMATION_LENGTH, SHORT_ANIMATION_LENGTH)
    : ANIMATION_LENGTH;
```

我的参数比 Neovide 默认慢一点，让滑动更容易被看见：

| 参数 | 我的值 | Neovide 默认 |
|------|--------|--------------|
| `ANIMATION_LENGTH` | 0.220 s | 0.150 s |
| `SHORT_ANIMATION_LENGTH` | 0.060 s | 0.040 s |
| `TRAIL_SIZE` | 1.0 | 1.0 |

### 颜色：不引入任何新颜色

很多 cursor trail shader 会硬编码一个颜色，换主题就不搭了。我的做法是**直接从 Ghostty 已经画好的光标上取色**：

```glsl
vec2 samplePos = vec2(
    iCurrentCursor.x + min(curHalf.x, 1.5),
    iCurrentCursor.y - min(curHalf.y, 1.5)
);
vec3 cursorColor = texture(iChannel0, samplePos / iResolution.xy).rgb;
```

采样点在光标左上角往里 1.5 像素，这个位置字形几乎不会有笔画，所以拿到的是纯光标色。配合前面的 `cursor-color = cell-foreground`，**拖尾颜色 = 光标颜色 = 目标位置文字的高亮色**，整个链条自动跟随主题。

### 性能：大部分像素第一行就返回

Post-processing shader 对**每一帧的每一个像素**都会执行。一个 4K 屏幕就是 800 多万次调用，所以必须让绝大多数像素尽早退出。我的 shader 按代价从低到高设了几道关卡：

1. **只看 uniform 的判断**：窗口没聚焦、光标不可见、光标没动、动画已经结束，所有像素直接返回原图。光标静止时整个 shader 就是一次纹理采样。
2. **包围盒**：只有落在新旧光标共同包围盒里的像素才继续。
3. **光标本体**：Ghostty 已经在目标位置画好了光标和光标下的字，这块不动。
4. 剩下的少量像素才去计算四边形的 signed distance，并做 1 像素的抗锯齿。

```glsl
float inside = quadSDF(fragCoord, p0, p1, p2, p3);
if (inside <= 0.0) {
    return;
}
fragColor.rgb = mix(fragColor.rgb, cursorColor, min(inside, 1.0));
```

`quadSDF` 对凸四边形的四条边分别算有向距离，取最小值：为正就在内部，数值是到最近边的像素距离，直接当抗锯齿的 alpha 用。

### 完整 Shader

<details>
<summary><code>cursor_neovide.glsl</code>（点击展开）</summary>

```glsl
// Ghostty cursor — Neovide-style animated cursor
//
// The cursor itself appears to fly from the old position to the new
// one: a solid shape in the cursor's own colour whose four corners
// each follow a critically damped spring. Corners facing the
// destination arrive first, the far corners trail behind, so the
// cursor stretches on take-off and snaps back to a block on landing.

// Slower than Neovide's defaults so the slide is easier to see.
const float ANIMATION_LENGTH       = 0.220;
const float SHORT_ANIMATION_LENGTH = 0.060;
const float TRAIL_SIZE             = 1.0;

// The springs are within 0.3% of rest after 2x the animation length.
const float SETTLE                 = 2.0;

// Signed distance to a convex clockwise (y-up) quad; positive inside.
float quadSDF(vec2 p, vec2 p0, vec2 p1, vec2 p2, vec2 p3) {
    vec2 e0 = p1 - p0;
    vec2 e1 = p2 - p1;
    vec2 e2 = p3 - p2;
    vec2 e3 = p0 - p3;

    float d0 = (e0.y * (p.x - p0.x) - e0.x * (p.y - p0.y)) * inversesqrt(max(dot(e0, e0), 1e-6));
    float d1 = (e1.y * (p.x - p1.x) - e1.x * (p.y - p1.y)) * inversesqrt(max(dot(e1, e1), 1e-6));
    float d2 = (e2.y * (p.x - p2.x) - e2.x * (p.y - p2.y)) * inversesqrt(max(dot(e2, e2), 1e-6));
    float d3 = (e3.y * (p.x - p3.x) - e3.x * (p.y - p3.y)) * inversesqrt(max(dot(e3, e3), 1e-6));

    return min(min(d0, d1), min(d2, d3));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // Unavoidable: Ghostty replaces the framebuffer with our output.
    fragColor = texture(iChannel0, fragCoord / iResolution.xy);

    // ---- uniform-only gates ----
    if (iFocus == 0 || iCursorVisible == 0) {
        return;
    }

    vec2 curHalf  = iCurrentCursor.zw * 0.5;
    vec2 prevHalf = iPreviousCursor.zw * 0.5;

    // Cursor centers (origin bottom-left, y up; .xy is top-left).
    vec2 cur  = vec2(iCurrentCursor.x  + curHalf.x,  iCurrentCursor.y  - curHalf.y);
    vec2 prev = vec2(iPreviousCursor.x + prevHalf.x, iPreviousCursor.y - prevHalf.y);
    vec2 d    = cur - prev;

    if (dot(d, d) < 0.01) {
        return;
    }

    // Short hop (typing): <= 2 cells horizontally, same row.
    bool shortHop =
        abs(d.x) <= 2.001 * iCurrentCursor.z &&
        abs(d.y) <= 0.01  * iCurrentCursor.w;

    float baseLength =
        shortHop
        ? min(ANIMATION_LENGTH, SHORT_ANIMATION_LENGTH)
        : ANIMATION_LENGTH;

    float elapsed = iTime - iTimeCursorChange;
    if (elapsed > baseLength * SETTLE) {
        return;
    }

    // ---- bounding box of old + new cursor ----
    vec2 lo = min(cur - curHalf, prev - prevHalf) - 1.0;
    vec2 hi = max(cur + curHalf, prev + prevHalf) + 1.0;
    if (fragCoord.x < lo.x || fragCoord.y < lo.y ||
        fragCoord.x > hi.x || fragCoord.y > hi.y) {
        return;
    }

    // The native cursor is already drawn at its destination; leave it
    // (and the glyph under it) untouched.
    vec2 inCursor = abs(fragCoord - cur);
    if (inCursor.x < curHalf.x && inCursor.y < curHalf.y) {
        return;
    }

    // ---- corner springs (Neovide) ----
    // Corner order: TL, TR, BR, BL. Corners aligned with the travel
    // direction get a shorter animation, by TRAIL_SIZE.
    float invManhattan = 0.5 / (abs(d.x) + abs(d.y));
    float nTR = 0.5 + (d.x + d.y) * invManhattan;
    float nTL = 0.5 + (d.y - d.x) * invManhattan;
    vec4 alignment = vec4(nTL, nTR, 1.0 - nTL, 1.0 - nTR);

    vec4 animLen = mix(
        vec4(baseLength),
        vec4(baseLength * (1.0 - TRAIL_SIZE)),
        alignment
    );

    // Critically damped spring, 1 -> 0, no overshoot: e^(-u) (1 + u),
    // with omega = 4 / length as in Neovide.
    vec4 u = (4.0 * elapsed) / max(animLen, vec4(1e-3));
    vec4 f = exp(-u) * (1.0 + u);

    vec2 a = vec2(-1.0,  1.0);   // TL
    vec2 b = vec2( 1.0,  1.0);   // TR
    vec2 c = vec2( 1.0, -1.0);   // BR
    vec2 e = vec2(-1.0, -1.0);   // BL

    vec2 p0 = mix(cur + a * curHalf, prev + a * prevHalf, f.x);
    vec2 p1 = mix(cur + b * curHalf, prev + b * prevHalf, f.y);
    vec2 p2 = mix(cur + c * curHalf, prev + c * prevHalf, f.z);
    vec2 p3 = mix(cur + e * curHalf, prev + e * prevHalf, f.w);

    float inside = quadSDF(fragCoord, p0, p1, p2, p3);
    if (inside <= 0.0) {
        return;
    }

    // ---- colour: the cursor's own ----
    // Sampled just inside the top-left corner of the drawn cursor, where
    // a glyph almost never has ink; works for block and bar cursors.
    vec2 samplePos = vec2(
        iCurrentCursor.x + min(curHalf.x, 1.5),
        iCurrentCursor.y - min(curHalf.y, 1.5)
    );
    vec3 cursorColor = texture(iChannel0, samplePos / iResolution.xy).rgb;

    // One pixel of antialiasing on the slanted edges.
    fragColor.rgb = mix(fragColor.rgb, cursorColor, min(inside, 1.0));
}
```

</details>

### 关于文中的动图

上面的视频和分镜图不是录屏，而是我把这个 shader 用 numpy 逐行移植了一遍，在一帧真实的 Neovim 画面上按 60 fps 逐帧渲染出来的：数学和 GLSL 完全一致，只是从 GPU 换到了 CPU。顺带也验证了一遍公式：四个角的轨迹、short hop 的判定、`2 × length` 之后完全收拢，都和在 Ghostty 里实际看到的一样。

## Wrap-up

这套配置的思路可以总结成三点：

1. **终端和编辑器是一个整体**：同一套字体、同一个主题、同一个行高，快捷键按层级分修饰键。
2. **颜色不硬编码**：光标取文字色，拖尾取光标色，换主题全部跟着走。
3. **动画要有用，而且要便宜**：拖尾帮眼睛追踪光标，打字时自动收敛；静止时 shader 的开销接近于零。

Neovim 那一侧的配置（键位设计、"没有下巴"的界面）写在另一篇：[My Neovim Setup](/posts/neovim-keymaps-and-ui/)。
