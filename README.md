# BriceLucifer.github.io

Personal website and technical notebook of **Brice** — an engineer working on performance, compilers, and the hardware–software boundary.

Live at: **[https://brice6.pages.dev/](https://brice6.pages.dev/)**

---

## Focus Areas

- Performance engineering: profiling, memory hierarchies, and making programs fast
- ML compilers: IR design, kernel fusion, code generation (see mlcc)
- The hardware–software boundary: CPU/GPU architecture and system-model co-design
- Systems programming: C / C++ / Rust / Zig
- Previously: video generation and interactive world models (research assistant, 2026)

---

## Site Structure

| Section | Purpose |
|---------|---------|
| **Blog** (`/posts/`) | Write-ups on AI, math, systems, and reflections. |
| **Master of AI** (`/posts/mai/`) | UoA course notes (COMPSCI 713 / 714) and research-assistant guides. |
| **Archive / Search / Tags** | Chronological index, full-text search, and topic index. |
| **About** (`/about/`) | Background, research focus, experience, and projects. |

---

## Tech Stack

- **Static site generator**: Hugo
- **Theme**: PaperMod
- **Content**: Markdown + KaTeX (rendered at build time)
- **Fonts** (self-hosted, `static/fonts/`): Source Serif 4 for article text, Noto Serif SC for Chinese (loaded only on pages with Chinese), JetBrains Mono for code and metadata, system sans for UI
- **Colors**: Catppuccin Latte / Mocha for the site and code highlighting (`assets/css/extended/theme-catppuccin.css`)
- **Math**: KaTeX 0.18.4 CSS + fonts self-hosted (`assets/katex/`, `static/katex/`) — must match the KaTeX bundled in Hugo; re-vendor when upgrading Hugo
- **Hosting**: Cloudflare Pages

---

## Local Development

```bash
git submodule update --init --recursive   # fetch PaperMod
hugo server -D                            # preview at http://localhost:1313
hugo --gc                                 # production build into public/
```

Requires Hugo extended **≥ 0.146** (PaperMod requirement). Prefer WebP for images under `static/images/`.

---

## Philosophy

This is not a traditional blog. It is a long-term **thinking archive** — tracking how ideas evolve through study, experimentation, and implementation, with emphasis on understanding from first principles.

---

© Brice · All rights reserved
