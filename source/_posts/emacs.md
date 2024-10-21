---
title: Emacs 修炼手册✨(Master Emacs)
---



## 快捷键

C -> `ctrl`
M -> `meta/alt` 少部分是`esc`
> - C-n   **nextline**
> - C-p   **previous line**
> - C-f   **forword**
> - C-b   **backforword**
> - C-k   **kill (从光标位置到末尾全部删除)**
> - C-a   **a (a是字母表的开始) start of line**
> - C-e   **end of line**
> - M-<   **去往编辑最开始的位置**
> - M->   **去往编辑结束的位置**
> - C-v   **向下翻一屏**
> - M-v   **向上翻一屏**

### 必备helper

C-h 		  **help tutorial**
C-h k 	       **help Keybind**
C-h f 		**help function**

### 给外观做点改变

- 图形化配置:
  - 菜单栏 menu-bar-mode
  - 工具栏 tool-bar-mode
  - 滚动条 scroll-bar-mode

- 文件配置 ~/.emacs

```lisp
(menu-bar-mode -1)
(tool-bar-mode -1)
(scroll-bar-mode -1)
```

### 认识配置文件

1. ~/.emacs

    > 单一配置文件

2. ~/.emacs.d

    > 更符合工程化

3. ~/.config/emacs/init.el

    > 仅仅适用与>=27的版本

### 第一行配置代码

```lisp
(setq inhibit-startup-screen t)
```

#### 软件源

- 网路问题需要配置

```lisp
(setq package-archives
      '( ("melpa" . "https://mirrors.tuna.tsinghua.edu.cn/elpa/melpa/")
        ("gnu" . "https://mirrors.tuna.tsinghua.edu.cn/elpa/gnu/")
        ("org" . "https://mirrors.tuna.tsinghua.edu.cn/elpa/org/") 
        )
)
```

### 安装第一个扩展

```lisp
;;个别时候会出现签名检验失败
(setq package-check-signature nil) 

;; 初始化软件包管理器
(require 'package)
(unless (bound-and-true-p package--initialized)
    (package-initialize))

;; 刷新软件源索引
(unless package-archive-contents
    (package-refresh-contents))

;; 第一个扩展插件:`use-package`,用来批量统一管理软件包
(unless (package-installed-p 'use-package)
    (package-refresh-contents)
    (package-install 'use-package))

```

#### 使用use-package管理扩展

- 最简洁的形式

```lisp
(use-package restart-emacs)
```

- 常用配置

```lisp
(use-package SOME-PACKAGE-NAME
             :ensure t ; 是否一定要确保已安装
             :defer t ; 是否要延迟加载,很多时候可以加速Emacs的启动速度
             :init (setq ...) ; 初始化配置
             :config (...) ; 初始化后的基本配置参数
             :bind (...) ; 快捷按键绑定
             :hook (...) ; hook的绑定
             )
```

- 建议添加的配置

```lisp
;; `use-package-always-ensure' 避免每个软件包都需要加 ":ensure t" 
;; `use-package-always-defer' 避免每个软件包都需要加 ":defer t" 
(setq use-package-always-ensure t
      use-package-always-defer t
      use-package-enable-imenu-support t
      use-package-expand-minimally t)

```

#### 配置主题

```lisp
(use-package gruvbox-theme
             :init (load-theme 'gruvbox-dark-soft t))
```

好看的*gruvbox-theme*
顺便配置一个好看的*Mode-line*

> 需要提前下载好 所以需要M-x package-install gruvbox

```lisp
(use-package smart-mode-line
  :init
  (setq sml/no-confirm-load-theme t
        sml/theme 'respectful)
  (sml/setup))
```

## 总结我的基础配置101

```lisp
;; my config ~/.emacs

(menu-bar-mode -1)
(tool-bar-mode -1)
(scroll-bar-mode -1)
;; close startup menu 
(setq inhibit-startup-screen t)
;; set up mirror for plugins
(setq package-archives
     '( 
        ("melpa" . "https://mirrors.tuna.tsinghua.edu.cn/elpa/melpa/")
        ("gnu" . "https://mirrors.tuna.tsinghua.edu.cn/elpa/gnu/")
        ("org" . "https://mirrors.tuna.tsinghua.edu.cn/elpa/org/") 
     )
)

;; refresh mirrors
;;个别时候会出现签名检验失败
(setq package-check-signature nil) 

;; 初始化软件包管理器
(require 'package)
(unless (bound-and-true-p package--initialized)
    (package-initialize))

;; 刷新软件源索引
(unless package-archive-contents
    (package-refresh-contents))

;; 第一个扩展插件:use-package,用来批量统一管理软件包
(unless (package-installed-p 'use-package)
    (package-refresh-contents)
    (package-install 'use-package))
(setq use-package-always-ensure t
      use-package-always-defer t
      use-package-enable-imenu-support t
      use-package-expand-minimally t)

(use-package restart-emacs)
(use-package gruvbox-theme
   :init (load-theme 'gruvbox-dark-soft t))
```
