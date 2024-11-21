---
title: Helix 尝鲜
date: 2024-11-21 21:00:00
categories: 
- Editor
toc: true
tags:
- Helix
---

## 主题设置
由于本人很喜欢doom设计类游戏 所以使用了doom dark theme 
### 默认设置主题
Unix:
1. 在.config文件夹下 创建helix文件夹
2. 创建config.toml，然后写入以下内容
```toml
theme = "doom_acario_dark"

[editor]
line-number = "relative"
mouse = false

[editor.cursor-shape]
insert = "bar"
normal = "block"
select = "underline"

[editor.file-picker]
hidden = false
```
3. 重新启动helix即可
