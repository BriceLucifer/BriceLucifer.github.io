---
title: "The Cloud Server Guide"
date: 2026-03-28
tags: ["MachineLearning", "DeepNeuralNetwork", "Research", "VideoGen"]
categories: ["posts"]
description: "A step-by-step guide for doing research on UOA cloud servers"
summary: "A practical guide for setting up environments, training models, and managing GPU jobs on UOA's compute cluster"
math: true
mermaid: true
ShowToc: true
TocOpen: true
draft: false
---

# UOA Cloud Server: Setup and Usage Guide

This guide walks through everything you need to start running deep learning workloads on the UOA compute cluster — from initial login to submitting GPU jobs and monitoring training.

## 0. First-Time Setup

When you first get access to your machine, point your data and home directories at the large `/data/<UPI>/` partition so that datasets, checkpoints, and virtual environments don't fill up the small system disk.

```bash
export TMPDIR=/data/<UPI>/
export HOME=/data/<UPI>/
```

> Tip: add these lines to your `~/.bashrc` or `~/.zshrc` so they persist across sessions.

## 1. SSH Login

```bash
ssh <UPI>0@foscsmlprd01.its.auckland.ac.nz
# foscsmlprd02 and foscsmlprd03 are also available
```

Replace `<UPI>` with your University of Auckland UPI (e.g., `jdoe001`).

## 2. Python Environment with `uv`

We use [`uv`](https://github.com/astral-sh/uv) for fast, reproducible Python environment management.

### Install `uv`

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

### Set Up an Environment

**If the project has a `pyproject.toml`:**

```bash
uv sync
```

`uv` creates the virtual environment and installs all dependencies in one step.

**If the project only has a `requirements.txt`:**

```bash
uv venv                              # create the virtual environment
uv pip install -r requirements.txt   # install dependencies
```

## 3. Submitting GPU Jobs with SLURM

To run training on a GPU, submit a shell script to the SLURM scheduler — don't run heavy workloads directly on the login node.

### Example Job Script

```bash
#!/bin/bash

# ======== SLURM Job Configuration ========
#SBATCH --job-name=<job_name>             # [REQUIRED] Name shown in the job queue (e.g., "AAAI-training")
#SBATCH --time=00:30:00                   # [REQUIRED] Wall-clock limit (HH:MM:SS)
#SBATCH --open-mode=append                # Append to log files instead of overwriting
#SBATCH --output=<output_file>.log        # [RECOMMENDED] Standard output log
#SBATCH --error=<error_file>.log          # [RECOMMENDED] Standard error log
#SBATCH --gres=gpu:1                      # [REQUIRED] Number of GPUs (>1 needs Slack channel approval)

# ======== Job Execution ========

# Navigate to your project directory
cd /data/<your_username>/<project_path>

# Activate the virtual environment
source venv/bin/activate

# Run the training script
python3 train.py
```

### Common SLURM Commands

| Action          | Command                |
| --------------- | ---------------------- |
| Submit a job    | `sbatch xxx.sh`        |
| Check your jobs | `squeue -u $USER`      |
| Cancel a job    | `scancel <job_id>`     |

## 4. Monitoring Your Workflow

**Watch your job queue in real time:**

```bash
watch squeue -u $USER
```

**Tail a log file to follow output or errors live:**

```bash
tail -f logs/xxx.txt
```

This is the fastest way to debug a running job without waiting for it to finish.

## 5. Downloading Output Files

For terminal users who want to pull results back to a local machine:

```bash
scp <UPI>0@foscsmlprd03.its.auckland.ac.nz:/data/<UPI>/LongLive/videos/interactive/rank0-0-0_lora.mp4 ~/Nexus/ai_build/new.mp4
```

The pattern is `scp <user>@<host>:<remote_path> <local_path>`.