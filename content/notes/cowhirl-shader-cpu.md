---
title: "Notes: A Simple CPU-based Whirl Shader Experiment"
date: 2026-01-01
tags: ["shader", "graphics", "rust", "raymarching", "experiment"]
categories: ["notes"]
---------------------

## Context

This note records a small experiment I wrote recently to explore **shader-like procedural graphics**, implemented **entirely on the CPU** using Rust.

Code Link: https://github.com/BriceLucifer/shader    

The goal was **not performance**, but to understand:

* how fragment-shader style math maps to plain code
* how time, space, and iteration interact visually
* how much of “shader thinking” is independent of GPU APIs

The final result is a short rendered video:

👉 Output video:
<video controls autoplay loop muted playsinline width="100%">
  <source
    src="https://raw.githubusercontent.com/BriceLucifer/shader/main/out.mp4"
    type="video/mp4">
  Your browser does not support the video tag.
</video>
---

## High-level Idea

The program emulates a **fragment shader loop**:

* each pixel corresponds to a ray / sample direction
* a procedural function iteratively transforms a point in space
* color is accumulated along a pseudo ray-marching path
* time (`t`) is used to animate rotation and deformation

Instead of running on the GPU, everything is computed on the CPU and written out as **PPM frames**, which are later combined into a video using `ffmpeg`.

---

## Coordinate Setup

Each pixel `(x, y)` is mapped into a centered coordinate system:

* normalized to `[-1, 1]`
* aspect-ratio corrected
* embedded into a pseudo-3D vector

```rust
let fx = (x as f32 / w as f32) * 2.0 - 1.0;
let fy = (y as f32 / h as f32) * 2.0 - 1.0;
let fc = Vec3::new(fx * aspect, fy, 1.0);
```

This mirrors how fragment coordinates (`fragCoord`) are usually handled in shaders.

---

## The Whirl Shader Loop

The core logic lives in `whirl_shader`, which repeatedly:

1. projects the fragment direction into space
2. applies a time-dependent swirl on the XY plane
3. applies trigonometric deformation
4. estimates a step distance
5. accumulates color along the path

Conceptually, this behaves like a **very rough ray-marching loop**:

```rust
while i < 180.0 {
    p = (fc * 2.0 - r).normalize() * z;
    p.z -= t;

    // swirl rotation
    let a = (p.z * 0.1).cos();
    let b = (p.z * 0.1 + 11.0).cos();
    let c = (p.z * 0.1 + 33.0).cos();

    p.x = p.x * a - p.y * b;
    p.y = p.x * c + p.y * a;

    v = (p + (p.yzx() / 0.3).sin()).cos();
    v = v.max(v.zxy() * 0.1);

    d = v.length() / 6.0;
    z += d;

    o = o + color(p.z) / (d + 1e-3);
}
```

There is no strict signed-distance function here — it is closer to **procedural exploration** than geometric correctness.

---

## Color Accumulation and Tone Mapping

Color is accumulated incrementally based on the depth (`p.z`) and iteration distance.

After the loop, a simple tone mapping is applied:

```rust
Vec3::new(
    (o.x / 5000.0).tanh(),
    (o.y / 5000.0).tanh(),
    (o.z / 5000.0).tanh(),
)
```

This compresses high dynamic range values into `[0,1]` smoothly, without abrupt clipping.

A simple gamma correction is then applied before converting to `u8`.

---

## Parallel Rendering

Each frame is rendered using **Rayon**:

* pixels are independent
* parallelism is embarrassingly parallel
* easy speedup without changing logic

```rust
buf.par_chunks_mut(3).enumerate().for_each(|(idx, pix)| {
    ...
});
```

This reinforces how naturally shader workloads map to data-parallel execution.

---

## Frame Output Pipeline

The rendering pipeline is deliberately simple:

1. Render frames as `PPM (P6)` images
2. Store them in `frames/`
3. Use `ffmpeg` to assemble a video

```bash
cargo run --release

ffmpeg -framerate 60 \
  -i frames/frame_%03d.ppm \
  -c:v libx264 -preset slow -crf 16 \
  -pix_fmt yuv420p \
  out.mp4
```

This keeps the experiment focused on **math and structure**, not tooling.

---

## Observations

* Shader-style math is largely **API-independent**
* Many visual effects emerge purely from iteration + trigonometry
* CPU implementations are slow, but extremely transparent for learning
* Writing this in Rust made vector operations and ownership explicit

---

## Next Steps (Ideas)

* Move the same logic to a real GPU fragment shader
* Explore signed-distance–based ray marching
* Experiment with fewer iterations and smarter step estimation
* Compare CPU vs GPU mental models directly

---

This note is intentionally informal and exploratory.
It serves as a record of understanding rather than a polished tutorial.
