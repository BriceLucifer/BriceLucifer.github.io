---
title: "My macOS Defaults: Dock, Finder 和 Keyboard 调教笔记"
date: 2026-11-24
tags: ["Tooling"]
keywords: ["macOS", "Productivity", "Terminal", "Setup"]
description: "A handful of `defaults write` commands I run on every new Mac to make the Dock instant, Finder honest, and key repeat fast — with one-line rollbacks for each."
summary: "每次拿到新 Mac 我都会先跑这几行 defaults write：让 Dock 瞬间弹出、Finder 显示隐藏文件和完整路径、键盘重复速度拉满。每一项都附带 back to default 的命令。"
---

每次装新机或者重装系统，我都会先做同一件事：打开 Terminal，跑几行 `defaults write`。这些设置都很小，但每天用几百次，累积起来体感差别非常明显。

这篇是我的 personal checklist，分成三块：**Dock**、**Finder**、**Keyboard**。每一块都附上 rollback 命令——`defaults` 改的是 plist 偏好文件，想恢复的时候 `defaults delete` 掉对应的 key 就会回到系统默认值，完全无副作用。

## Dock：让它极速闪现

我习惯把 Dock 设成 auto-hide，省下屏幕底部那一条空间。但默认的 auto-hide 有两个问题：鼠标移到边缘后要**等一下**才开始弹出，弹出的**动画也慢**。两个参数分别对付这两点：

```bash
# 彻底消除鼠标滑到边缘时 Dock 的“等待时间”（delay = 0）
defaults write com.apple.dock autohide-delay -float 0

# 把 Dock 弹出的动画时间缩短到近乎没有（0.1 秒）
defaults write com.apple.dock autohide-time-modifier -float 0.1

# 重启 Dock 让设置立刻生效（屏幕会闪一下，正常现象，别慌）
killall Dock
```

- `autohide-delay`：hover 到触发之间的延迟。设成 `0` 就是指哪打哪。
- `autohide-time-modifier`：动画时长。设成 `0` 会完全没有动画，我觉得有点突兀，`0.1` 保留了一点点过渡，刚好。

### Back to default

不喜欢了，一行恢复：

```bash
defaults delete com.apple.dock autohide-delay && \
defaults delete com.apple.dock autohide-time-modifier && \
killall Dock
```

## Finder：显示隐藏文件 + 完整路径

作为开发者，`.zshrc`、`.git`、`.env` 这些 dotfiles 天天要碰。Finder 默认把它们藏起来，标题栏也只显示当前文件夹名，不知道自己在哪一层。

```bash
# 显示隐藏文件（临时切换也可以用快捷键 Cmd + Shift + .）
defaults write com.apple.finder AppleShowAllFiles -bool true

# 在 Finder 标题栏显示完整的 POSIX 路径
defaults write com.apple.finder _FXShowPosixPathInTitle -bool true

killall Finder
```

Tip：如果只是偶尔需要看隐藏文件，直接 `Cmd + Shift + .` 就够了，不用改 defaults。我是常年都要看，所以干脆设成永久。

### Back to default

```bash
defaults delete com.apple.finder AppleShowAllFiles
defaults delete com.apple.finder _FXShowPosixPathInTitle
killall Finder
```

## Keyboard：Key Repeat 拉满

用 Vim / Helix 这类 modal editor，或者只是在终端里按住 `←` 删东西，key repeat 的速度直接决定手感。

**My way**：先走 GUI。打开 **系统设置 → 键盘**，把 “Key repeat rate（键重复速率）” 拖到最快，“Delay until repeat（重复前延迟）” 拖到最短。这个改完立刻生效，不需要注销。

然后再用命令行补两刀：

```bash
# 关闭“长按弹出重音字符菜单”（é è ê 那个），改为长按连续重复
defaults write NSGlobalDomain ApplePressAndHoldEnabled -bool false

# 重复速率与初始延迟（单位是 15ms 一格，数字越小越快）
defaults write NSGlobalDomain KeyRepeat -int 2
defaults write NSGlobalDomain InitialKeyRepeat -int 15
```

几点说明：

- **`ApplePressAndHoldEnabled`**：macOS 默认长按字母会弹出 accent menu，对写代码的人来说基本是干扰。关掉之后长按 `j` 就是一路往下走。这个需要**重启对应的 App** 才生效。
- **`KeyRepeat = 2`** 约等于每 30ms 重复一次，**`InitialKeyRepeat = 15`** 约等于 225ms 后开始重复——这其实就是 GUI 滑块拉到头的值，写成命令主要是方便在新机器上一键复现。
- 如果想**超越 GUI 的极限**，可以试 `KeyRepeat 1` / `InitialKeyRepeat 10`，但会快到有点难控制，建议先从上面的值开始。命令行改的这两个值一般要**注销重新登录**才完全生效。

### Back to default

```bash
defaults delete NSGlobalDomain ApplePressAndHoldEnabled
defaults delete NSGlobalDomain KeyRepeat
defaults delete NSGlobalDomain InitialKeyRepeat
```

## Wrap-up

总结一下这套 setup 的思路：

| 区域 | 改了什么 | 生效方式 |
|------|----------|----------|
| Dock | 去掉 hover delay，动画缩到 0.1s | `killall Dock` |
| Finder | 显示隐藏文件，标题栏显示完整路径 | `killall Finder` |
| Keyboard | 关闭长按重音菜单，key repeat 拉满 | 重启 App / 重新登录 |

都是很小的改动，但这就是 *defaults* 的魅力：几行命令，把系统调成顺手的样子；不喜欢了，`defaults delete` 一下就回到原点。建议把这些命令丢进自己的 dotfiles repo，下次换机器直接一键跑完。
