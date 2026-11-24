---
title: "My Neovim Setup：从 Keymap 到一个「没有下巴」的 UI"
date: 2026-11-24
tags: ["Neovim", "Lua", "Editor", "UI Design", "Setup"]
description: "A tour of my Neovim 0.12 config — 17 single-purpose plugins in 8 Lua files — covering keymap design, native LSP, and a chin-less floating UI with a hand-written dashboard."
summary: "我的 Neovim 0.12 配置：17 个插件、8 个 Lua 文件、约 30 ms 启动。从 leader 键怎么分组、Ctrl 和 Alt 怎么分工，到去掉状态栏和命令行之后，状态胶囊、浮动命令行、通知卡片和会转的星系启动页是怎么设计的。"
---

这是我的 Neovim 配置的一次完整 walkthrough。配置本身开源在 [BriceLucifer/nvim.setting](https://github.com/BriceLucifer/nvim.setting)，这里主要讲 **why**：键位为什么这么分、界面为什么长这样。

先放一张"全家福"：

![Neovim with two splits, floating status capsules and inline diagnostics](/images/neovim/code.webp)

*左右分屏；右上角是状态胶囊，没有状态栏、没有命令行那一行、没有标签栏；第 23 行的未使用变量只在光标所在行显示诊断文字；行号右边的绿色细条是 Git 改动。*

## 设计原则

整份配置围绕几条规则：

- **Neovim 0.12+，能用内置就用内置。** 插件管理用内置的 `vim.pack`，LSP 用原生 `vim.lsp`，不用 lazy.nvim，也不用 lspconfig。
- **17 个插件，每个只做一件事。** 没有 LazyVim、NvChad 这类打包型发行版。出了问题知道该找谁。
- **8 个 Lua 文件，按功能分模块**，每个模块单独 `pcall` 保护。
- **没有下巴（no chin）。** 屏幕上下都留给代码，状态信息全部浮动显示。
- **快。** 启动约 30 ms。

```text
init.lua                入口：按顺序加载下面 8 个模块
lua/config/
├── options.lua         选项和自动命令
├── plugins.lua         插件清单
├── ui.lua              配色、命令行、通知卡片、行号列、胶囊、缩进线、滚动、按键提示
├── dashboard.lua       启动页（菜单和星系画面）
├── tools.lua           oil、FZF-Lua、GrugFar、gitsigns、Flash
├── code.lua            treesitter、文本对象、补全、LSP
├── keymaps.lua         快捷键和自定义命令
└── neovide.lua         Neovide 专用
```

`init.lua` 里最值得一提的是**模块隔离**：某个插件升级后改了接口，只会让它所在的模块报错，其他模块照常加载，错误在启动后以通知的形式弹出来，而不是整个编辑器打不开。

```lua
for _, name in ipairs(modules) do
  local ok, err = pcall(require, "config." .. name)
  if not ok then
    vim.schedule(function()
      vim.notify("config." .. name .. " 加载失败：\n" .. err, vim.log.levels.ERROR)
    end)
  end
end
```

---

## Part 1：Keymap 设计

### Leader = Space，按字母分组

`<leader>` 是空格。所有 leader 键按**第一个字母 = 一个功能域**来组织，记一个字母就能找到一整组：

| 前缀 | 功能域 | 例子 |
|------|--------|------|
| `f` | **F**ind：搜索 | `ff` 文件、`fg` 全文、`fw` 光标下的词、`fR` 搜索替换 |
| `s` | **S**plit：窗口 | `sv` 左右分、`sh` 上下分、`sz` 放大 / 恢复 |
| `b` | **B**uffer | `bd` 关闭但保留布局、`bo` 关闭其他 |
| `t` | **T**ab：工作区 | `tn` 新建、`tc` 关闭、`to` 只留当前 |
| `g` | **G**it | `gp` 预览、`gs` 暂存、`gr` 还原、`gb` blame |
| `l` | **L**SP | `lf` 格式化、`ld` 诊断、`ls` 符号、`lt` 开关 |
| `a` | **A**rgument | `an` / `ap` 与下一个 / 上一个参数交换 |
| `e` | **E**xplorer | 浮窗文件管理 |
| `w` / `q` | Write / Quit | 保存 / 退出 |
| `m` / `n` | Messages / Notifications | 翻看历史 |

记不住的时候按下空格停 0.2 秒，which-key 会在右下角列出所有分组：

![which-key popup listing leader groups](/images/neovim/whichkey.webp)

每个 keymap 的 `desc` 都是**英文 + 中文**，比如 `"Find files (查找文件)"`。英文对齐 Vim 社区的说法，中文是给自己扫一眼就懂的。

### 冲突怎么解决

键位设计里最花心思的是处理冲突，这里有几个具体的取舍：

- **`]c` / `[c` 给了 Git hunk**（和 Vim 原生 diff 模式的含义一致），所以 treesitter 的"下一个类"只能换成 `]k` / `[k`。在 diff 模式下 `]c` 仍然走原生跳转：

  ```lua
  local function hunk(key, direction)
    return function()
      if vim.wo.diff then
        vim.cmd.normal({ key, bang = true })
      else
        require("gitsigns").nav_hunk(direction)
      end
    end
  end
  ```

- **`<leader>fr` 是最近文件**，所以搜索替换用大写的 `<leader>fR`。
- **`s` 给了 Flash 跳转**，`S` 是按语法结构选择；但可视模式下 `S` 要留给 nvim-surround（选中后按 `S"` 加引号），所以可视模式里按语法结构选择改用 `R`。
- **调整窗口大小不用 `Ctrl + 方向键`**：macOS 用它切换桌面，nvim 根本收不到。改用 `<leader>sH/J/K/L`，而且语义是"分隔线往哪个方向移动"，与光标在哪个窗口无关，比 `:resize +5` 直观得多。

### Ctrl 给 Neovim，Alt 给终端

```lua
-- Ctrl 切换 Neovim 窗口，Alt 留给终端。
map("n", "<C-h>", "<C-w>h", { desc = "Window left (左侧窗口)" })
map("n", "<C-j>", "<C-w>j", { desc = "Window down (下方窗口)" })
map("n", "<C-k>", "<C-w>k", { desc = "Window up (上方窗口)" })
map("n", "<C-l>", "<C-w>l", { desc = "Window right (右侧窗口)" })
```

Ghostty 那边分屏切换是 `Alt + hjkl`（见 [Ghostty 配置笔记](/posts/ghostty-config-and-cursor-shader/)）。同样是 `hjkl`，用修饰键区分是在终端层还是编辑器层移动，两层互不抢键。

### 一些小但每天都用的映射

```lua
-- 搜索结果居中并展开折叠
map("n", "n", "nzzzv")
map("n", "N", "Nzzzv")

-- 缩进后保留选区，可以连续按 > > >
map("x", "<", "<gv")
map("x", ">", ">gv")

-- 覆盖选区时不把被覆盖的内容放进寄存器
map("x", "<leader>p", [["_dP]])

-- 上下移动选中的行并重新缩进；到文件头尾不报 E16
map("x", "J", ":<C-u>silent! '<,'>move '>+1<CR>gv=gv")
map("x", "K", ":<C-u>silent! '<,'>move '<-2<CR>gv=gv")
```

还有一个 `<leader>bd`：**关闭 buffer 但保留窗口布局**。原生 `:bd` 会把显示它的窗口也一起关掉，分屏布局就乱了。我的实现是先把每个显示它的窗口切到 alternate buffer（或者任意一个别的 listed buffer），再删掉它；有未保存的修改时用 `confirm()` 问"保存 / 放弃 / 取消"。

### Treesitter 文本对象

这是我觉得 Vim 语法最"值钱"的部分：`d` / `c` / `y` / `v` 加上语法树级别的对象。

| 按键 | 对象 | 例子 |
|------|------|------|
| `af` / `if` | 函数 | `daf` 删除整个函数 |
| `ac` / `ic` | 类 / 结构体 | `yac` 复制整个 struct |
| `aa` / `ia` | 参数 | `cia` 改写当前参数 |
| `al` / `il` | 循环 | `vil` 选中循环体 |
| `ai` / `ii` | 条件语句 | `dai` 删掉整个 if |

配合 `]f` / `[f` 在函数之间跳、`<leader>an` / `<leader>ap` 交换参数位置，重构参数顺序只需要两个键。实现上就是一个循环：

```lua
for _, obj in ipairs(objects) do
  local key, name, desc = obj[1], obj[2], obj[3]
  map({ "x", "o" }, "a" .. key, function()
    ts_select.select_textobject("@" .. name .. ".outer", "textobjects")
  end, { desc = "Around " .. desc })
  map({ "x", "o" }, "i" .. key, function()
    ts_select.select_textobject("@" .. name .. ".inner", "textobjects")
  end, { desc = "Inside " .. desc })
end
```

### LSP：原生 `vim.lsp`，按键随服务挂载

LSP 完全用 Neovim 原生的 `vim.lsp.config` / `vim.lsp.enable`。每个语言服务一项，`cmd` 用绝对路径，可执行文件不存在就自动跳过：

```lua
clangd = {
  enabled = true,
  cmd = {
    "/opt/homebrew/opt/llvm/bin/clangd",
    "--background-index",
    "--clang-tidy",
    "--completion-style=detailed",
    "--header-insertion=never",
  },
  filetypes = { "c", "cpp", "objc", "objcpp", "cuda" },
  root_markers = { ".clangd", "compile_commands.json", "compile_flags.txt", ".git" },
},
```

目前配了 clangd、zls、ocamllsp、sourcekit-lsp、gopls、rust-analyzer、haskell-language-server 和 tinymist（Typst）。

LSP 相关的按键**只在 `LspAttach` 时绑定到当前 buffer**，最后一个服务离开（`LspDetach`）时再按同一张表原样删掉。所以 `K`、`gd` 在没有语言服务的文件里还是 Vim 原本的意思。原生的默认键（`grn` 重命名、`gra` code action、`grr` 引用、`]d` 下一个诊断）直接沿用，不重新发明。

---

## Part 2：UI 设计

### 没有下巴

```lua
opt.laststatus = 0   -- 不要状态栏
opt.cmdheight = 0    -- 不要命令行那一行
opt.showtabline = 0  -- 不要标签栏
opt.showmode = false
opt.showcmd = false
opt.ruler = false
```

去掉这三样之后，屏幕的上下边缘全部是代码。代价是原来这些地方显示的信息需要新的去处，下面这几个组件就是答案。

一个小细节：`laststatus = 0` 时，上下分屏之间 Neovim 仍然会画一行 statusline。我把它画成一条横线，当分隔线用：

```lua
opt.statusline = "%#WinSeparator#%{repeat('─', winwidth(0))}"
```

### 状态胶囊（incline.nvim）

每个窗口右上角一个浮动的胶囊，替代状态栏和标签栏：

![Floating status capsule](/images/neovim/completion.webp)

*Insert 模式下的胶囊：`INS` 模式色块、错误和警告数量、Git 改动行数、分支、语言服务名、文件名（未保存时带 `●`）。下面是 blink.cmp 的补全菜单。*

当前窗口从左到右：

1. **模式色块**（`NOR` `INS` `VIS` `CMD`…），颜色随模式变；
2. **正在发生的事**：录制宏 `@a`、选区大小、搜索位置 `2/6`，只在发生时出现；
3. **诊断数量**、**Git 改动**、**分支**、**语言服务**；
4. 最右边永远是**图标 + 文件名**。两个打开的文件同名时，前面自动带上级目录。

非当前窗口的胶囊只有图标和文件名，颜色压暗。胶囊**永远不会盖住代码**：它能用的宽度是"窗口宽度减去窗口第一行文字的宽度"，空间不够时从右往左依次省略语言服务、分支、Git、诊断，只保留模式和文件名。

> 这里有个性能坑：压缩过的 JSON 一行可能有几 MB，每次光标移动都去量整行的宽度会卡。所以只量窗口宽度那么长的一段。

### 浮动命令行

`cmdheight = 0` 之后，命令行用 Neovim 0.12 实验性的 **ui2** 加上 [tiny-cmdline.nvim](https://github.com/rachartier/tiny-cmdline.nvim)，在屏幕中上方浮出一个圆角小窗：

![Floating cmdline with inccommand preview](/images/neovim/cmdline.webp)

*输入 `:%s/r->len/r->count` 时，`inccommand = "split"` 在缓冲区里实时高亮替换结果，底部的预览窗口列出所有受影响的行。*

只有"保存吗？是 / 否 / 取消"这类确认框留在底部，和它的问题文字待在一起。ui2 毕竟还是实验性的，所以配置里留了一个开关 `use_ui2 = false`，随时可以退回传统命令行。

### 通知卡片与消息

`vim.notify` 被我**手写的通知卡片**替换：右上角的圆角卡片，排在胶囊下面，多条往下叠，几秒后消失；警告和错误的边框带颜色。语言服务的加载进度是一张**原地更新**的卡片，同一件事只占一张，结束后消失。

普通消息（`:echo` 之类）出现在右下角，几秒后消失。`<leader>n` 和 `<leader>m` 分别翻看通知和消息的历史。

### 行号列与诊断

- **行号列是手写的 `statuscolumn`**：折叠标记 + 行号 + Git 细条。`signcolumn = "no"`，省掉一整列。
- Git 修改统一画成一根 `▎` 细条，颜色区分新增 / 修改 / 删除。
- **只有当前窗口显示相对行号**，其他窗口显示普通行号；当前行只点亮行号，代码区不铺横条（`cursorlineopt = "number"`）。
- **诊断只在光标所在行显示文字**，其他行只有波浪线：

```lua
vim.diagnostic.config({
  severity_sort = true,
  underline = true,
  signs = false,
  virtual_text = { current_line = true, prefix = "●", spacing = 2 },
})
```

满屏的 virtual text 会把代码挤得没法读，这样屏幕始终是干净的，需要细节时光标移过去就有。

### 统一的浮窗语言

所有浮窗共用一套视觉规则：**无底色，只有圆角边框，标题用主题的蓝色**。

```lua
opt.winborder = "rounded"

NormalFloat = { fg = C.text, bg = "NONE" },
FloatBorder = { fg = C.surface1, bg = "NONE" },
FloatTitle  = { fg = C.blue, bg = "NONE", bold = true },
```

搜索用 [fzf-lua](https://github.com/ibhagwan/fzf-lua)，以内置的 `telescope` profile 为底（输入框在下、结果在上、预览在右），但颜色换成和其他浮窗一致，不压暗背后的编辑区：

![fzf-lua file picker with preview](/images/neovim/picker.webp)

*`<leader>ff` 在配置目录里找文件。文件名在前、目录在后变暗（`path.filename_first`）；右侧预览里中英混排的注释靠 [Maple Mono NF CN](https://github.com/subframe7536/maple-font) 严格对齐。*

> 字体是 [Maple Mono NF CN](https://font.subf.dev)：等宽、内置 Nerd Font 图标、中英文 2:1 对齐，这份配置里所有的图标（文件类型、Git 分支、诊断）都依赖它。安装：`brew install --cask font-maple-mono-nf-cn`，或者从 [Releases](https://github.com/subframe7536/maple-font/releases) 下载。终端字体不是 Nerd Font 的话，图标会显示成方块。

文件管理用 [oil.nvim](https://github.com/stevearc/oil.nvim)：目录就是一个可以编辑的 buffer，改名直接改字、删除直接 `dd`、`:w` 才真正生效，删除的文件进废纸篓。`-` 原地打开，`<leader>e` 浮窗打开：

![oil.nvim floating explorer](/images/neovim/oil.webp)

### 透明背景

配色是 Catppuccin Mocha，开了 `transparent_background = true`。编辑区直接显示终端的背景色，而 Ghostty 用的也是 Catppuccin Mocha，所以终端和编辑器之间没有色差。自定义的高亮组（胶囊的 `Bar*`、启动页的 `Dash*`）颜色全部从主题色板里取，换 flavour 时一起换。

### 启动页：一个会转的星系

不带文件打开 nvim 时是一个**手写的启动页**，没有用 alpha / dashboard 插件：

![Dashboard with a braille spiral galaxy](/images/neovim/dashboard.webp)

画面是一个倾斜的旋涡星系，用**盲文点阵字符**画成：每个格子是 2×4 个点，亮度越高点越密。金色的核心、由冰白到蓝再到紫的两条旋臂、粉色的恒星诞生区，周围是星空和淡淡的星云。颜色是一条条从 Catppuccin 色板插值出来的渐变。

它会以每秒 6 帧的速度慢慢转动，为了不拖慢启动和不浪费 CPU，做了几件事：

- 两条旋臂是对称的，**转半圈画面就重复**，所以只需要 120 帧。
- 每一帧只在第一次出现时计算一次，之后存进缓存文件 `~/.cache/nvim/dashboard-frames.txt`，后面直接读。
- 离开启动页、正在输入命令、终端失去焦点时画面停住；启动页关掉后定时器销毁。
- 窗口小于 96×26 时只显示菜单。

菜单下面那行 `17 plugins · 33 ms · v0.12.6` 是实时的：插件数、启动耗时（`init.lua` 第一行记下 `vim.uv.hrtime()`）和 Neovim 版本。

---

## 性能：先画第一屏，再做别的

30 ms 的启动时间主要靠三件事：

**1. `vim.loader.enable()`**：缓存编译后的 Lua 字节码。

**2. 一个手写的 `later()`**：不影响第一屏的东西（补全、Git 标记、缩进线）推迟到 `VimEnter` 之后再加载。用起来没有区别，但打开 nvim 的那一下更快：

```lua
local queue, started = {}, false
function M.later(fn)
  if started then fn() else queue[#queue + 1] = fn end
end
vim.api.nvim_create_autocmd("VimEnter", {
  once = true,
  callback = function()
    vim.schedule(function()
      started = true
      for _, fn in ipairs(queue) do
        local ok, err = pcall(fn)
        if not ok then vim.notify("延迟加载出错：\n" .. err, vim.log.levels.ERROR) end
      end
    end)
  end,
})
```

**3. 按需初始化**：fzf-lua 在第一次按下搜索键时才 `setup()`；treesitter 解析器只在打开对应文件时加载；**超过 1 MB 的文件不启动 treesitter**，折叠也改成手动，打开大日志、压缩过的 JSON 不会卡。

## 关于截图

文中的截图不是对着屏幕截的。我写了一个小脚本，以 `nvim --embed` 启动**这份真实配置**，作为 UI client 接收 Neovim 的 `redraw` 事件（grid、高亮、光标），再用 Maple Mono NF CN 把整个 grid 画成图片。画面内容和在 Ghostty 里看到的一模一样，只是少了窗口边框和光标动画。光标动画见 [Ghostty 那篇](/posts/ghostty-config-and-cursor-shader/)。

## Wrap-up

回头看，这份配置最核心的几个决定是：

1. **键位按功能域分组**，冲突逐个显式解决，而不是靠记忆硬背；
2. **能用内置就用内置**：`vim.pack`、`vim.lsp`、ui2、`statuscolumn`；
3. **去掉下巴**，信息浮动显示、只在需要时出现；
4. **所有浮窗一套视觉语言**，颜色全部取自主题；
5. **第一屏优先**，其余的都往后放。

完整配置和全部快捷键表在 [GitHub](https://github.com/BriceLucifer/nvim.setting) 上，`git clone` 到 `~/.config/nvim`、打开 nvim 等插件装完就能用。
