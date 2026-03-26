---
title: "Deep Learning in a Mathematical Way"
date: 2026-03-08
tags: ["MachineLearning", "DeepNeuralNetwork", "Mathematics", "LLM"]
categories: ["posts"]
description: "A rigorous mathematical journey from Linear Regression to Multimodal AI"
summary: "Building intuition for deep learning through formal mathematics — from scalars to transformers"
math: true
mermaid: true
ShowToc: true
TocOpen: true
draft: false
---
# Deep Learning in a Mathematical Way

<div class="mermaid">
mindmap
  root((Deep Learning))
    Probability and Stats
      Gaussian Distribution
      Expectation and Variance
      Entropy
      KL Divergence
      Bayes Theorem
    Fundamentals
      Linear Regression
      Loss Functions
        SE MSE
        Cross-Entropy
      Derivatives
        Gradient
        Chain Rule
        Gradient Descent
    Linear Algebra
      Vectors
        Dot Product
        Norm
        Cosine Similarity
      Matrices
        Matrix Multiply
        Transpose
        Broadcast
    Neural Networks
      Single Neuron
      MLP
        Forward Pass
        Backpropagation
      Activations
        ReLU Sigmoid Tanh
        Softmax GELU
      Normalisation
        Batch Norm
        Layer Norm
      Pathologies
        Vanishing Gradient
        Exploding Gradient
        Zigzag
      ResNet Skip Connection
      Training
        Dropout
        Adam
        L2 Regularisation
    Architectures
      CNN
        2D Convolution
        Pooling
        Feature Maps
      RNN
        Vanilla RNN
        LSTM
        GRU
      Attention
        Scaled Dot-Product
        Multi-Head
        Positional Encoding
        Transformer Block
      Encoder-Decoder
        BERT MLM
        GPT vs BERT
      ViT
    Transfer Learning
      Pre-train to Fine-tune
      LoRA
    Generative AI
      VAE
        ELBO
        Reparameterisation
      GAN
        Generator
        Discriminator
      Diffusion DDPM
        Forward Noise
        Reverse Denoise
      Autoregressive
        Temperature
        Top-p Sampling
      RLHF
        Reward Model
        PPO
        DPO
    LLM
      Architecture
        Tokenisation BPE
        Causal Attention
        Unembedding
      KV Cache
      MoE
      Perplexity PPL
      Scaling Laws
      Prompting
        k-shot
        Chain-of-Thought
        Self-Consistency
      RAG
    Multimodal AI
      CLIP
        InfoNCE Loss
      VLM
        Visual Encoder
        Projection
      Cross-Modal Attention
      Text-to-Image
        CFG
        Latent Diffusion
</div>

## Probability & Statistics Foundations

### Gaussian Distribution

The most important distribution in deep learning. A random variable $X \sim \mathcal{N}(\mu, \sigma^2)$ has density:

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

- $\mu$: mean (centre of the bell curve)
- $\sigma^2$: variance (spread); $\sigma$ is the standard deviation

**Multivariate Gaussian** $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$, $\Sigma \in \mathbb{R}^{d\times d}$ covariance matrix:

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\!\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

### Expectation & Variance

$$\mathbb{E}[X] = \int x\, p(x)\, dx, \qquad \mathbb{E}[f(X)] = \int f(x)\, p(x)\, dx$$

$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

### Entropy & KL Divergence

**Entropy** — measures uncertainty of distribution $p$:

$$H(p) = -\sum_x p(x)\log p(x) \qquad (\text{discrete}), \quad H(p) = -\int p(x)\log p(x)\,dx \quad (\text{continuous})$$

**KL Divergence** — measures how much $q$ differs from $p$ (not a distance — asymmetric):

$$D_{\text{KL}}(p \| q) = \sum_x p(x)\log\frac{p(x)}{q(x)} \geq 0, \quad = 0 \iff p = q$$

Cross-entropy loss is directly related: $\mathcal{L}_{\text{CE}} = H(p) + D_{\text{KL}}(p\|q)$ where $p$ is the true label distribution.

### Bayes' Theorem

$$P(\theta \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \theta)\; P(\theta)}{P(\mathcal{D})}$$

- $P(\theta)$: **prior** — belief about parameters before seeing data
- $P(\mathcal{D}\mid\theta)$: **likelihood** — how well $\theta$ explains data
- $P(\theta\mid\mathcal{D})$: **posterior** — updated belief after data
- $P(\mathcal{D})$: **evidence** — normalisation constant

MLE (Maximum Likelihood Estimation) maximises $P(\mathcal{D}\mid\theta)$; MAP adds a prior regulariser.

---

## Some ML essential concept in Methematical way
### Linear Regression(LR)

$$f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b = \sum_{j=1}^{d} w_j x_j + b$$

In matrix form over $n$ samples: $\hat{\mathbf{y}} = X\mathbf{w} + b\mathbf{1},\quad X \in \mathbb{R}^{n \times d}$

### Square Error(SE)

$$\text{SE}(\hat{y}, y) = (\hat{y} - y)^2$$

### Mean Square Error(MSE)

$$\mathcal{L}_{\text{MSE}}(\theta) = \frac{1}{n} \sum_{i=1}^{n} \left( f_\theta(\mathbf{x}^{(i)}) - y^{(i)} \right)^2$$

### Cross-Entropy Loss (CE)

Used for classification. For a true label $y \in \{1,\ldots,K\}$ and predicted probability vector $\hat{\mathbf{p}} = \text{softmax}(\mathbf{z})$:

$$\mathcal{L}_{\text{CE}} = -\sum_{k=1}^{K} y_k \log \hat{p}_k$$

For binary classification ($K=2$):

$$\mathcal{L}_{\text{BCE}} = -\left[y \log \hat{p} + (1-y)\log(1-\hat{p})\right]$$

Over $n$ samples (multiclass):

$$\mathcal{L}_{\text{CE}}(\theta) = -\frac{1}{n}\sum_{i=1}^{n} \log \hat{p}_{y^{(i)}}^{(i)}$$

### Derivatives

**Derivative** — instantaneous rate of change:

$$f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**Gradient** — multivariate generalisation $f : \mathbb{R}^d \to \mathbb{R}$:

$$\nabla_\theta \mathcal{L} = \begin{bmatrix} \partial \mathcal{L}/\partial \theta_1 \\ \vdots \\ \partial \mathcal{L}/\partial \theta_d \end{bmatrix} \in \mathbb{R}^d$$

**Chain Rule** — for $z = g(f(x))$:

$$\frac{dz}{dx} = \frac{dz}{df} \cdot \frac{df}{dx}$$

For deep compositions $z = h(g(f(\mathbf{x})))$:

$$\frac{\partial z}{\partial \mathbf{x}} = \frac{\partial z}{\partial g} \cdot \frac{\partial g}{\partial f} \cdot \frac{\partial f}{\partial \mathbf{x}}$$

**Gradient Descent** — minimise $\mathcal{L}(\theta)$ by stepping opposite the gradient:

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}(\theta_t)$$

### Four Step Process For Machine Learning
1. Collect the data
2. Define the model's structure
3. Define the loss function
4. Minimize the loss

### Vector is all you need

A vector $\mathbf{x} \in \mathbb{R}^d$ encodes a data point with $d$ features.

Operations:
- **Add:** $(\mathbf{u} + \mathbf{v})_i = u_i + v_i$
- **Dot product:** $\mathbf{u} \cdot \mathbf{v} = \mathbf{u}^\top \mathbf{v} = \sum_i u_i v_i$
- **$\ell_2$ Norm:** $\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$
- **Cosine similarity:** $\cos(\mathbf{u},\mathbf{v}) = \dfrac{\mathbf{u}^\top \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$

### Matrix

A matrix $W \in \mathbb{R}^{m \times n}$ represents a linear transformation $\mathbb{R}^n \to \mathbb{R}^m$.

Operations:
- **Add:** $(A+B)_{ij} = A_{ij}+B_{ij}$
- **Mul (matrix product):** $(AB)_{ij} = \sum_k A_{ik}B_{kj}$
- **Broadcast:** scalar/vector ops extend across batch dimensions
- **Dot product / inner product:** $\mathbf{u}^\top \mathbf{v} = \sum_i u_i v_i$
- **Transpose:** $(A^\top)_{ij} = A_{ji}$

### Neural Network

**Single neuron:**

$$a = \phi(\mathbf{w}^\top \mathbf{x} + b)$$

**MLP — layer $\ell$ forward pass:**

$$\mathbf{z}^{(\ell)} = W^{(\ell)}\mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)}, \qquad \mathbf{a}^{(\ell)} = \phi\!\left(\mathbf{z}^{(\ell)}\right)$$

**Backpropagation** (chain rule applied layer by layer):

$$\boldsymbol{\delta}^{(\ell)} = \left(W^{(\ell+1)\top}\boldsymbol{\delta}^{(\ell+1)}\right) \odot \phi'\!\left(\mathbf{z}^{(\ell)}\right)$$

$$\frac{\partial \mathcal{L}}{\partial W^{(\ell)}} = \boldsymbol{\delta}^{(\ell)}\mathbf{a}^{(\ell-1)\top}$$

Activation:

**ReLU:**
$$\text{ReLU}(x) = \max(0,x), \qquad \text{ReLU}'(x) = \mathbf{1}[x>0]$$

**Sigmoid:**
$$\sigma(x) = \frac{1}{1+e^{-x}}, \qquad \sigma'(x) = \sigma(x)(1-\sigma(x))$$

**Tanh:**
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}, \qquad \tanh'(x) = 1 - \tanh^2(x)$$

**Softmax:**
$$\text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

**GELU** (Gaussian Error Linear Unit) — used in BERT, GPT, and all modern Transformers:

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]$$

where $\Phi(x)$ is the standard Gaussian CDF. Practical approximation:

$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}}\,(x + 0.044715\,x^3)\right)\right)$$

Unlike ReLU, GELU is smooth and non-zero for $x < 0$ with small probability, which helps gradient flow.

### Training detail

- **Parameters** $\theta = \{W^{(\ell)}, \mathbf{b}^{(\ell)}\}$: learned by gradient descent.
- **Hyperparameters**: learning rate $\eta$, batch size $B$, layers $L$, hidden dim $d$ — set before training.

**$\ell_2$ Regularisation** (weight decay):

$$\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2}\|\theta\|_2^2$$

**Dropout** — randomly zero out neurons during training with probability $p$:

$$\tilde{\mathbf{a}} = \mathbf{m} \odot \mathbf{a} \cdot \frac{1}{1-p}, \quad \mathbf{m}_i \sim \text{Bernoulli}(1-p)$$

At inference, no dropout is applied (weights already scaled by $1/(1-p)$ at train time — **inverted dropout**).

**Batch Normalisation (BN)** — normalise activations across the batch dimension, then scale and shift:

$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}, \qquad y_i = \gamma\hat{x}_i + \beta$$

where $\mu_\mathcal{B}, \sigma_\mathcal{B}^2$ are batch mean/variance; $\gamma,\beta$ are learned parameters.

**Layer Normalisation (LN)** — same formula but normalise across the *feature* dimension (not batch). Preferred in Transformers because it is batch-size independent:

$$\hat{\mathbf{x}} = \frac{\mathbf{x} - \mu_\mathbf{x}}{\sqrt{\sigma_\mathbf{x}^2 + \epsilon}} \cdot \boldsymbol{\gamma} + \boldsymbol{\beta}, \quad \mu_\mathbf{x} = \frac{1}{d}\sum_j x_j$$

**Adam optimiser** — bias-corrected moment estimates:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$

$$\theta_{t+1} = \theta_t - \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}, \quad \hat{m}_t = \frac{m_t}{1-\beta_1^t},\; \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

### Vanishing & Exploding Gradients

In a deep network with $L$ layers, the gradient flowing back to layer $\ell$ is a product of Jacobians:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(\ell)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(L)}} \prod_{k=\ell}^{L-1} \frac{\partial \mathbf{a}^{(k+1)}}{\partial \mathbf{a}^{(k)}}$$

Each factor $\approx W^{(k)} \cdot \text{diag}(\phi'(\mathbf{z}^{(k)}))$. If the spectral norm $\|W\| < 1$ repeatedly, gradients **vanish** exponentially; if $\|W\| > 1$, they **explode**.

- **Vanishing**: gradients $\to 0$, early layers learn nothing. Caused by sigmoid/tanh saturation ($\sigma' \leq 0.25$) + many layers.
- **Exploding**: parameter updates become huge, training diverges.

**Fixes**: ReLU (gradient = 1 in positive region), residual connections, LayerNorm, gradient clipping:

$$g_t \leftarrow g_t \cdot \min\!\left(1,\; \frac{\tau}{\|g_t\|}\right), \quad \tau \text{ is the clip threshold}$$

### Zigzag in Gradient Descent

Vanilla SGD with a fixed learning rate $\eta$ oscillates (**zigzags**) when the loss surface has different curvatures along different directions — it overshoots in high-curvature directions and undershoots in low-curvature ones:

$$\theta_{t+1}^{(i)} = \theta_t^{(i)} - \eta\, \frac{\partial \mathcal{L}}{\partial \theta^{(i)}}$$

If $\eta$ is large enough to make progress along the flat direction, it overshoots along the steep direction → zigzag trajectory.

**Momentum** damps oscillations by accumulating a velocity vector:

$$\mathbf{v}_{t+1} = \gamma \mathbf{v}_t + \eta \nabla_\theta \mathcal{L}, \qquad \theta_{t+1} = \theta_t - \mathbf{v}_{t+1}$$

The exponential moving average of gradients cancels out oscillating components while reinforcing consistent directions. Adam further applies **per-parameter** adaptive learning rates via $\hat{v}_t$ (second moment), which is why it almost always converges faster than SGD on non-convex landscapes.

### ResNet — Residual Connections

Add a **skip connection** that bypasses one or more layers, letting gradients flow directly to earlier layers:

$$\mathbf{y} = \mathcal{F}(\mathbf{x},\,\{W_i\}) + \mathbf{x}$$

- $\mathcal{F}(\mathbf{x})$: the residual to learn (e.g. two conv layers)
- $\mathbf{x}$: identity shortcut

**Why it works:** the gradient through the skip path is exactly $1$ — no matter how deep, the chain rule always has a direct path with gradient $\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + I$. This prevents vanishing gradients in very deep networks (ResNet-152, Transformers).

### CNN

2D convolution — kernel $K \in \mathbb{R}^{k\times k}$ slides over image $X \in \mathbb{R}^{H\times W}$:

$$(X * K)_{i,j} = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} K_{m,n} \cdot X_{i+m,\,j+n}$$

Each layer applies $C_{\text{out}}$ kernels → feature map $\in \mathbb{R}^{C_{\text{out}}\times H'\times W'}$.

**Pooling** — downsample spatial dimensions to reduce computation and add translation invariance:

$$\text{MaxPool}(X)_{i,j} = \max_{(m,n)\in\text{window}} X_{i\cdot s+m,\,j\cdot s+n}$$

$$\text{AvgPool}(X)_{i,j} = \frac{1}{k^2}\sum_{(m,n)\in\text{window}} X_{i\cdot s+m,\,j\cdot s+n}$$

where $k$ is the pool size and $s$ is the stride.

### RNN

**Vanilla RNN** — hidden state recurrence:

$$\mathbf{h}_t = \tanh(W_h \mathbf{h}_{t-1} + W_x \mathbf{x}_t + \mathbf{b})$$

**LSTM** — gated memory cell $(i,f,o$: input/forget/output gates$)$:

$$\mathbf{f}_t = \sigma(W_f[\mathbf{h}_{t-1};\mathbf{x}_t]+\mathbf{b}_f), \quad \mathbf{i}_t = \sigma(W_i[\mathbf{h}_{t-1};\mathbf{x}_t]+\mathbf{b}_i)$$

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tanh(W_c[\mathbf{h}_{t-1};\mathbf{x}_t]+\mathbf{b}_c)$$

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

**GRU** — simplified gating (reset $\mathbf{r}_t$, update $\mathbf{z}_t$):

$$\mathbf{z}_t = \sigma(W_z[\mathbf{h}_{t-1};\mathbf{x}_t]), \quad \mathbf{r}_t = \sigma(W_r[\mathbf{h}_{t-1};\mathbf{x}_t])$$

$$\mathbf{h}_t = (1-\mathbf{z}_t)\odot\mathbf{h}_{t-1} + \mathbf{z}_t\odot\tanh(W_h[\mathbf{r}_t\odot\mathbf{h}_{t-1};\mathbf{x}_t])$$

### Attention

**Scaled Dot-Product Attention** — $Q,K,V \in \mathbb{R}^{n\times d_k}$:

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

**Multi-Head Attention** — $h$ parallel heads, then project:

$$\text{head}_i = \text{Attention}(QW_i^Q,\,KW_i^K,\,VW_i^V)$$

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h)\,W^O$$

**Positional Encoding** (sinusoidal):

$$\text{PE}_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad \text{PE}_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

**Transformer block** (with residual + LayerNorm):

$$\mathbf{h}' = \text{LayerNorm}(\mathbf{h} + \text{MultiHead}(\mathbf{h},\mathbf{h},\mathbf{h}))$$

$$\mathbf{h}'' = \text{LayerNorm}(\mathbf{h}' + \text{FFN}(\mathbf{h}')), \quad \text{FFN}(\mathbf{x})=W_2\,\text{ReLU}(W_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2$$

**Language Modelling Objective** — maximise log-likelihood over token sequence:

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^T \log P_\theta(t_t \mid t_{\lt t})$$

**ViT** — split image into $N$ patches $\mathbf{p}_i \in \mathbb{R}^{P^2 C}$, linearly project to token embeddings, then apply Transformer encoder:

$$\mathbf{z}_0 = [\mathbf{x}_{\text{cls}};\, E\mathbf{p}_1;\,\ldots;\,E\mathbf{p}_N] + \mathbf{E}_{\text{pos}}$$

**Encoder-Decoder Transformer** (original seq2seq, e.g. T5, BART) — encoder processes source sequence bidirectionally; decoder generates target tokens autoregressively with **cross-attention** over encoder outputs:

$$\text{head}_i^{\text{cross}} = \text{Attention}(Q_{\text{dec}}\,W_i^Q,\; K_{\text{enc}}\,W_i^K,\; V_{\text{enc}}\,W_i^V)$$

Decoder block = (causal self-attention) → (cross-attention to encoder) → (FFN), each with residual + LayerNorm.

### BERT — Bidirectional Encoder

BERT uses the **encoder-only** Transformer and pre-trains on two objectives:

**1. Masked Language Model (MLM)** — randomly mask 15 % of tokens, predict them from full bidirectional context:

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P_\theta(t_i \mid \mathbf{t}_{\backslash \mathcal{M}})$$

- $\mathcal{M}$: set of masked positions
- $\mathbf{t}_{\backslash \mathcal{M}}$: all tokens except the masked ones

Because attention is **bidirectional** (no causal mask), every token can attend to every other token — unlike GPT which only sees past context.

**2. Next Sentence Prediction (NSP)** — binary classification: are sentence B follows sentence A?

$$\mathcal{L}_{\text{NSP}} = -\log P_\theta(\text{IsNext} \mid [\text{CLS}])$$

**GPT vs BERT comparison:**

| | GPT (decoder-only) | BERT (encoder-only) |
|---|---|---|
| Attention | Causal (left-to-right) | Bidirectional (full) |
| Pre-training | Next-token prediction | MLM + NSP |
| Strength | Generation | Understanding / Classification |
| Representation | $\mathbf{h}_i$ sees only $t_1,\ldots,t_i$ | $\mathbf{h}_i$ sees all tokens |

### Transfer Learning

Pre-train on large corpus $\mathcal{D}_{\text{pre}}$ to get $\theta^*$, then fine-tune on target task $\mathcal{D}_{\text{ft}}$:

$$\theta_{\text{ft}} = \arg\min_\theta \mathcal{L}_{\text{ft}}(\theta;\,\mathcal{D}_{\text{ft}}), \quad \theta \leftarrow \theta^* \text{ (initialised)}$$

**LoRA** — freeze $W_0$, inject low-rank update $\Delta W = BA$ ($B\in\mathbb{R}^{d\times r},\,A\in\mathbb{R}^{r\times k},\,r\ll\min(d,k)$):

$$W = W_0 + \Delta W = W_0 + BA$$

---

## Generative AI

Generative models learn the data distribution $p(\mathbf{x})$ and can **sample new data** from it.

The core distinction from discriminative models:

| | Discriminative | Generative |
|---|---|---|
| **Goal** | $p(y\|\mathbf{x})$ | $p(\mathbf{x})$ or $p(\mathbf{x},y)$ |
| **Output** | Label / decision | New data sample |
| **Examples** | Classifier, Regression | VAE, GAN, Diffusion, LLM |

### Variational Autoencoder (VAE)

Encode data $\mathbf{x}$ into a latent variable $\mathbf{z}$, then decode back.

**Encoder** — approximate posterior $q_\phi(\mathbf{z}|\mathbf{x}) \approx p(\mathbf{z}|\mathbf{x})$, parameterised as Gaussian:

$$q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mathbf{z};\,\boldsymbol{\mu}_\phi(\mathbf{x}),\,\text{diag}(\boldsymbol{\sigma}^2_\phi(\mathbf{x})))$$

**Decoder** — likelihood $p_\theta(\mathbf{x}|\mathbf{z})$.

**ELBO objective** (Evidence Lower Bound, maximise):

$$\mathcal{L}_{\text{VAE}}(\theta,\phi) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\!\left[\log p_\theta(\mathbf{x}|\mathbf{z})\right] - D_{\text{KL}}\!\left(q_\phi(\mathbf{z}|\mathbf{x})\;\|\;p(\mathbf{z})\right)$$

**KL divergence** between two Gaussians:

$$D_{\text{KL}}(\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\sigma}^2)\;\|\;\mathcal{N}(\mathbf{0},I)) = \frac{1}{2}\sum_j\!\left(\sigma_j^2 + \mu_j^2 - 1 - \log\sigma_j^2\right)$$

**Reparameterisation trick** — make sampling differentiable:

$$\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, I)$$

---

### Generative Adversarial Network (GAN)

Two networks compete: **Generator** $G_\theta$ tries to fool **Discriminator** $D_\phi$.

$$\min_\theta \max_\phi\; \mathbb{E}_{\mathbf{x}\sim p_{\text{data}}}\!\left[\log D_\phi(\mathbf{x})\right] + \mathbb{E}_{\mathbf{z}\sim p(\mathbf{z})}\!\left[\log(1 - D_\phi(G_\theta(\mathbf{z})))\right]$$

- $G_\theta(\mathbf{z})$: maps noise $\mathbf{z}\sim\mathcal{N}(\mathbf{0},I)$ to fake samples.
- $D_\phi(\mathbf{x})\in(0,1)$: probability that $\mathbf{x}$ is real.
- At equilibrium: $D_\phi(\mathbf{x}) = \tfrac{1}{2}$ everywhere — generator perfectly mimics data.

---

### Diffusion Models (DDPM)

Gradually add Gaussian noise over $T$ steps (**forward process**), then learn to reverse it (**reverse process**).

**Forward process** — fixed Markov chain:

$$q(\mathbf{x}_t|\mathbf{x}_{t-1}) = \mathcal{N}\!\left(\mathbf{x}_t;\,\sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\,\beta_t I\right)$$

**Closed-form sampling** at any step $t$ (let $\bar\alpha_t = \prod_{s=1}^t(1-\beta_s)$):

$$\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},I)$$

**Reverse process** — learned denoiser $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$:

$$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}\!\left(\mathbf{x}_{t-1};\,\boldsymbol{\mu}_\theta(\mathbf{x}_t,t),\,\tilde\beta_t I\right)$$

$$\boldsymbol{\mu}_\theta(\mathbf{x}_t,t) = \frac{1}{\sqrt{\alpha_t}}\!\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)\right)$$

**Training objective** — predict the noise:

$$\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}}\!\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]$$

---

### Autoregressive Generation (LLM Decoding)

Given a prompt $(t_1,\ldots,t_k)$, generate token-by-token:

$$t_{k+i} \sim P_\theta(\cdot \mid t_1,\ldots,t_{k+i-1})$$

**Temperature scaling** — control sharpness of distribution:

$$P_\tau(t) = \frac{\exp(z_t / \tau)}{\sum_j \exp(z_j / \tau)}$$

- $\tau \to 0$: greedy (deterministic); $\tau = 1$: standard softmax; $\tau > 1$: more random.

**Top-$p$ (nucleus) sampling** — sample from smallest set $\mathcal{V}$ s.t.:

$$\sum_{t \in \mathcal{V}} P(t) \geq p$$

---

### Reinforcement Learning from Human Feedback (RLHF)

**Step 1 — Supervised Fine-Tuning (SFT):** fine-tune LLM on human demonstrations.

**Step 2 — Reward Model:** train $r_\phi(\mathbf{x}, \mathbf{y})$ from preference pairs $(y_w \succ y_l)$:

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x,y_w,y_l)}\!\left[\log\sigma\!\left(r_\phi(x,y_w) - r_\phi(x,y_l)\right)\right]$$

**Step 3 — PPO fine-tuning** — maximise reward while staying close to SFT policy $\pi_{\text{ref}}$:

$$\mathcal{L}_{\text{RLHF}}(\theta) = \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta}\!\left[r_\phi(x,y)\right] - \beta\,D_{\text{KL}}\!\left(\pi_\theta(\cdot|x)\;\|\;\pi_{\text{ref}}(\cdot|x)\right)$$

**DPO** (Direct Preference Optimisation) — skips reward model, optimise preferences directly:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x,y_w,y_l)}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

---

## Large Language Models (LLM)

### Architecture Overview

A decoder-only Transformer stacks $L$ blocks. Given token sequence $(t_1,\ldots,t_n)$:

1. **Tokenisation** — map text to integer IDs via vocabulary $\mathcal{V}$, $|\mathcal{V}|\sim 50\text{k}$–$200\text{k}$.
2. **Embedding** — $\mathbf{E} \in \mathbb{R}^{|\mathcal{V}|\times d}$, look up row $t_i$: $\mathbf{h}_i^{(0)} = \mathbf{E}_{t_i} + \mathbf{PE}_i$
3. **$L$ Transformer blocks** (causal masked attention + FFN).
4. **Unembedding** — project to logits and apply softmax: $\mathbf{l}_i = \mathbf{E}\,\mathbf{h}_i^{(L)} \in \mathbb{R}^{|\mathcal{V}|}$

**Causal (masked) self-attention** — token $i$ can only attend to positions $j \leq i$:

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V, \quad M_{ij} = \begin{cases}0 & j\leq i\\-\infty & j>i\end{cases}$$

---

### Tokenisation & Embedding

**Byte-Pair Encoding (BPE)** — iteratively merge the most frequent adjacent pair until vocabulary size $|\mathcal{V}|$ is reached.

**Token embedding** + **positional encoding** → input representation:

$$\mathbf{h}_i = \mathbf{E}_{t_i} + \mathbf{PE}_i \in \mathbb{R}^d$$

**RoPE** (Rotary Positional Embedding) — encode position by rotating query/key vectors:

$$\mathbf{q}_m = R_m \mathbf{q}, \quad \mathbf{k}_n = R_n \mathbf{k}, \quad \mathbf{q}_m^\top \mathbf{k}_n = \mathbf{q}^\top R_{n-m} \mathbf{k}$$

where $R_m$ is a block-diagonal rotation matrix at angle $m\theta_i$.

---

### Perplexity (PPL)

The standard intrinsic evaluation metric for language models — how "surprised" the model is by the test set:

$$\text{PPL}(T) = \exp\!\left(-\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(t_t \mid t_{\lt t})\right) = \exp(\mathcal{L})$$

- Lower PPL = model assigns higher probability to real text = better.
- PPL is simply the exponential of the cross-entropy loss $\mathcal{L}$.
- PPL $= k$ means the model is as uncertain as choosing uniformly among $k$ tokens.
- GPT-2 (117M): PPL ≈ 35 on WikiText-103; GPT-4 class models: PPL $\approx$ single digits.

**Bits-per-character (BPC)** — alternative unit used for character-level models:

$$\text{BPC} = \frac{\mathcal{L}}{\ln 2}$$

---

### Scaling Laws

Model performance scales predictably with compute $C$, data $D$, and parameters $N$ (Chinchilla):

$$\mathcal{L}(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + \mathcal{L}_\infty$$

Optimal allocation for a compute budget $C = 6ND$:

$$N_{\text{opt}} \propto C^{0.5}, \qquad D_{\text{opt}} \propto C^{0.5}$$

i.e. **tokens and parameters should scale equally**.

---

### In-Context Learning (ICL) & Prompting

LLMs can learn from examples in the prompt **without updating weights**.

**$k$-shot prompting** — prepend $k$ (input, output) examples:

$$P_\theta(y \mid x, (x_1,y_1),\ldots,(x_k,y_k))$$

**Chain-of-Thought (CoT)** — include reasoning steps $r$ before answer $a$:

$$P_\theta(r, a \mid x) = P_\theta(r \mid x) \cdot P_\theta(a \mid x, r)$$

**Self-consistency** — sample $M$ reasoning paths, take majority vote:

$$\hat{a} = \arg\max_a \sum_{m=1}^{M} \mathbf{1}[a_m = a]$$

---

### Retrieval-Augmented Generation (RAG)

Augment generation with retrieved documents $\mathcal{D}_r$ from an external knowledge base:

$$P_\theta(y \mid x) = \sum_{d \in \mathcal{D}_r} P_\theta(y \mid x, d)\, P_{\text{ret}}(d \mid x)$$

**Retrieval** — encode query and documents, fetch top-$k$ by cosine similarity:

$$\text{score}(x, d) = \frac{f(x)^\top g(d)}{\|f(x)\|\|g(d)\|}$$

---

### KV Cache

During autoregressive inference, keys and values for all past tokens are cached — avoid recomputing on each new token:

$$K_{\leq t} = [k_1,\ldots,k_t], \quad V_{\leq t} = [v_1,\ldots,v_t]$$

New token $t+1$ only computes $q_{t+1}$, then attends over cached $K_{\leq t}, V_{\leq t}$. Reduces per-step cost from $\mathcal{O}(t^2 d)$ to $\mathcal{O}(t d)$; memory grows linearly with sequence length.

---

### Mixture of Experts (MoE)

Replace the dense FFN in each Transformer block with $E$ expert FFNs. A learned **router** selects top-$k$ experts per token:

$$\text{MoE}(\mathbf{x}) = \sum_{i=1}^{k} g_i(\mathbf{x})\; \text{FFN}_i(\mathbf{x})$$

$$g_i(\mathbf{x}) = \text{softmax}\!\left(\text{TopK}\!\left(W_g\mathbf{x},\; k\right)\right)_i$$

- **Activated params** per token: $\sim k/E$ of total params → same inference cost as a smaller dense model.
- **Load balancing loss** encourages uniform expert utilisation: $\mathcal{L}_{\text{bal}} = E \sum_i f_i \cdot p_i$ where $f_i$ is fraction of tokens routed to expert $i$ and $p_i$ is mean router probability.

Used in **Mixtral**, **GPT-4**, **Switch Transformer**.

---

### Key LLM Concepts Summary

| Concept | What it does | Key formula / idea |
|---|---|---|
| **Tokenisation (BPE)** | Text → integer IDs | Merge most-frequent pairs |
| **Causal Attention** | Each token sees only past | Mask $M_{ij}=-\infty$ for $j>i$ |
| **Scaling Law** | Predict loss from $N,D,C$ | $\mathcal{L} \propto N^{-\alpha} + D^{-\beta}$ |
| **SFT** | Align model to instructions | Cross-entropy on demonstrations |
| **RLHF / DPO** | Align to human preferences | Reward signal or preference pairs |
| **CoT Prompting** | Elicit step-by-step reasoning | $P(r,a\|x) = P(r\|x)P(a\|x,r)$ |
| **RAG** | Ground generation in facts | Retrieve then generate |
| **LoRA** | Parameter-efficient fine-tuning | $W = W_0 + BA$, $r \ll d$ |

---

## Multimodal AI

Multimodal models process and generate **more than one modality** (text, image, audio, video) within a unified framework.

$$f_\theta : \mathcal{M}_1 \times \mathcal{M}_2 \times \cdots \to \mathcal{Y}$$

### Modality Encoding

Each modality is first encoded into a shared embedding space $\mathbb{R}^d$:

| Modality | Encoder | Output |
|---|---|---|
| Text | Tokeniser + Embedding | $\mathbf{h}_t \in \mathbb{R}^{L\times d}$ |
| Image | ViT / CNN patch encoder | $\mathbf{h}_v \in \mathbb{R}^{N\times d}$ |
| Audio | Spectrogram + Conv / Whisper | $\mathbf{h}_a \in \mathbb{R}^{S\times d}$ |
| Video | Frame-level ViT + temporal attention | $\mathbf{h}_f \in \mathbb{R}^{T\times N\times d}$ |

---

### CLIP — Contrastive Vision-Language Pre-training

Learn aligned image and text embeddings by maximising agreement between matched pairs.

Given a batch of $N$ (image $\mathbf{v}_i$, text $\mathbf{t}_i$) pairs, compute cosine similarities:

$$s_{ij} = \frac{f(\mathbf{v}_i)^\top g(\mathbf{t}_j)}{\|f(\mathbf{v}_i)\|\|g(\mathbf{t}_j)\|} \cdot \exp(\tau)$$

**Symmetric InfoNCE loss** (maximise diagonal, minimise off-diagonal):

$$\mathcal{L}_{\text{CLIP}} = -\frac{1}{2N}\sum_{i=1}^N \left[\log\frac{e^{s_{ii}}}{\sum_j e^{s_{ij}}} + \log\frac{e^{s_{ii}}}{\sum_j e^{s_{ji}}}\right]$$

At test time: zero-shot classification by picking the text prompt with highest cosine similarity to the image.

---

### Vision-Language Models (VLM)

Connect a **vision encoder** to an **LLM** via a projection layer.

**Architecture:**

$$\mathbf{h}_v = \text{VisualEncoder}(\mathbf{I}) \in \mathbb{R}^{N\times d_v}$$

$$\tilde{\mathbf{h}}_v = W_{\text{proj}}\,\mathbf{h}_v \in \mathbb{R}^{N\times d}, \quad W_{\text{proj}} \in \mathbb{R}^{d\times d_v}$$

Visual tokens $\tilde{\mathbf{h}}_v$ are prepended (or interleaved) with text tokens and fed into the LLM:

$$\text{input} = [\tilde{\mathbf{h}}_v;\, \mathbf{h}_{\text{text}}]$$

**LLaVA-style training** — two stages:
1. Pre-train projection only (freeze encoder + LLM): learn $W_{\text{proj}}$
2. Instruction fine-tuning: unfreeze LLM, train on (image, instruction, response) triplets

**Objective** — standard next-token prediction on response tokens only:

$$\mathcal{L} = -\sum_{t} \log P_\theta(y_t \mid \mathbf{I},\, \mathbf{x},\, y_{\lt t})$$

---

### Cross-Modal Attention

Allow one modality to attend over another. Text queries attend to visual keys/values:

$$\mathbf{h}'_{\text{text}} = \text{Attention}(Q_{\text{text}},\; K_{\text{vision}},\; V_{\text{vision}})$$

$$= \text{softmax}\!\left(\frac{Q_{\text{text}} K_{\text{vision}}^\top}{\sqrt{d_k}}\right) V_{\text{vision}}$$

Used in **Flamingo**, **Perceiver Resampler**, etc.

---

### Image Generation — Text-to-Image

Condition a diffusion model on a text embedding $\mathbf{c} = g_\phi(\text{prompt})$.

**Classifier-Free Guidance (CFG)** — blend conditional and unconditional score:

$$\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, \mathbf{c}) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \emptyset) + w\!\left(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \emptyset)\right)$$

- $w > 1$: stronger text conditioning (higher fidelity to prompt, less diversity).
- $w = 1$: standard conditional generation.
- $w = 0$: unconditional generation.

**Latent Diffusion (Stable Diffusion)** — run diffusion in a compressed latent space:

$$\mathbf{z} = \mathcal{E}(\mathbf{x}), \quad \hat{\mathbf{x}} = \mathcal{D}(\mathbf{z}), \quad \mathcal{L} = \mathbb{E}_{t,\mathbf{z}_0,\boldsymbol{\epsilon}}\!\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \mathbf{c})\|^2\right]$$

---

### Multimodal Summary

| Model / Concept | Modalities | Key idea |
|---|---|---|
| **CLIP** | Image + Text | Contrastive alignment, InfoNCE |
| **LLaVA / InternVL** | Image + Text | Visual tokens → LLM via projection |
| **Flamingo** | Image + Text | Cross-modal attention layers |
| **Stable Diffusion** | Text → Image | Latent diffusion + CFG |
| **Whisper** | Audio → Text | Spectrogram encoder + decoder |
| **GPT-4o** | Image/Audio/Text | Unified multimodal autoregressive LLM |$