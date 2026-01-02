---

title: "Step 3: MLE of Gaussian Parameters"
toc: true
date: 2026-01-02
tags: ["Normal Distribution", "Machine Learning", "Diffusion Model"]
categories: ["notes"]
---

I am going to start with numpy, we use numpy for data, it is a unique library of linear algebra. Here are some example of usage.
``` python
# 3.1.1 multivariable array
import numpy as np
x = np.array([1, 2, 3])

print(x.__class__)
print(x.shape)
print(x.ndim)
```

    <class 'numpy.ndarray'>
    (3,)
    1

``` python
W = np.array([[1, 2, 3],
              [4, 5, 6]])
print(W.ndim)
print(W.shape)
```

    2
    (2, 3)

``` python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
y = a.T @ b
print(y)

A = np.array([[1,2], [3,4]])
B = np.array([[4,5], [6,7]])
Y = np.dot(A, B)
print(Y)
```

    32
    [[16 19]
     [36 43]]

we use
$$ 
    \mathrm{Var}[x_i] = \mathbb{E}\left[(x_i - \mu_i)^2\right]
$$

$$ Cov[x_i,x_j] = \mathbb{E}\left[(x_i - \mu_i)(x_j- \mu_j)\right] $$

## Transpose
``` python
print(A)
# row become col, col become row
print(A.T)
print()

A = np.array([[1, 2, 3], [4, 5, 6]])
print(A)
print("--------------")
print(A.T)
```

    [[1 2]
     [3 4]]
    [[1 3]
     [2 4]]

    [[1 2 3]
     [4 5 6]]
    --------------
    [[1 4]
     [2 5]
     [3 6]]

## determinant (which is a square matrix (cols = rows))

$$
A = \begin{pmatrix}
    3 & 4   \\\\
    5 & 6
\end{pmatrix}
$$

$$
\lvert A\rvert = 3 \cdot 6 - 4 \cdot 5 = -2
$$

``` python
A = np.array([[3, 4],[5, 6]])
d = np.linalg.det(A)
print(d)
# because the internal implement of numpy we got little numerical error, close to -2
```

    -1.9999999999999971

## inverse matrix

$$
AA^{-1} = A^{-1}A = I
$$
For example:
we got
$$
    A = \begin{pmatrix}
    a_{11} & a_{12} \\\\ 
    a_{21} & a_{22}
    \end{pmatrix}
$$
then we make trans:

$$
    A^{-1} = \frac{1}{\lvert A \rvert}
    \begin{pmatrix} 
    a_{11} & -a_{12} \\\\
    -a_{21} & a_{22}
    \end{pmatrix}
$$
we can use `np.linalg.inv()` for inverse matrix

``` python
A = np.array([[3, 4], [5, 6]])
B = np.linalg.inv(A)
print(B)
print(A @ B)
```

    [[-3.   2. ]
     [ 2.5 -1.5]]
    [[ 1.0000000e+00 -8.8817842e-16]
     [ 0.0000000e+00  1.0000000e+00]]

``` python
# now we jump to multivariable normal distribution
def multivariable_normal(x, mu, cov):
    '''
    x is variable
    mu is E[X]
    cov = Var[x, y]
    '''
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    D = len(x)
    z = 1 / np.sqrt((2 * np.pi) ** D * det)
    y = z * np.exp((x-mu).T @ inv @ (x - mu) / -2.0)
    return y

print("test the function")
x = np.array([0,0])
mu = np.array([1, 2])
cov = np.array([[1, 0], [0, 1]])
y = multivariable_normal(x, mu, cov)
print(y)
```

    test the function
    0.013064233284684921

> we can see it is a single element, so we can call it returns a scalar

``` python
# we plot now
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure # pyright: ignore

X = np.array([[-2, -1, 0, 1, 2],
              [-2, -1, 0, 1, 2],
              [-2, -1, 0, 1, 2],
              [-2, -1, 0, 1, 2],
              [-2, -1, 0, 1, 2]])
Y = np.array([[-2, -2, -2, -2, -2],
              [-1, -1, -1, -1, -1],
              [0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1],
              [2, 2, 2, 2, 2]])

Z = X ** 2 + Y ** 2

# we try the new version of create 3d figure
fig : Figure = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z') 

plt.show()
```

![](/step3_files/figure-markdown_strict/cell-9-output-1.png)

``` python
xs = np.arange(-2, 2, 0.1)
ys = np.arange(-2, 2, 0.1)

X, Y = np.meshgrid(xs, ys)
Z = X ** 2 + Y ** 2

fig : Figure = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z') 

plt.show()
```

![](/step3_files/figure-markdown_strict/cell-10-output-1.png)

``` python
# 绘制等高线
ax = plt.axes()
ax.contour(X, Y, Z)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
```

![](/step3_files/figure-markdown_strict/cell-11-output-1.png)

``` python
# we use the formal variables
mu = np.array([0.5, -0.2])
cov = np.array([[2.0, 0.3], [0.3, 0.5]])
xs = ys = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(xs, ys)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x = np.array([X[i, j], Y[i, j]])
        Z[i, j] = multivariable_normal(x, mu, cov)

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.set_xlabel('x')
ax1.set_xlabel('y')
ax1.set_zlabel('z')
ax1.plot_surface(X, Y, Z, cmap='viridis')

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.contour(X, Y, Z)
plt.show()
```

![](/step3_files/figure-markdown_strict/cell-12-output-1.png)

in step1 we get single variable element $$\hat{\mu} \space and \space \hat{\Sigma}$$

$$
\hat{\mu}
= \frac{1}{N}\sum_{n=1}^{N} x^{(n)}
$$

$$
\hat{\Sigma}
= \frac{1}{N}\sum_{n=1}^{N}
\bigl(x^{(n)} - \hat{\mu}\bigr)
\bigl(x^{(n)} - \hat{\mu}\bigr)^{T}
$$

``` python
# 最大似然的实现

np.random.seed(0)

N = 10000
D = 2
xs = np.random.rand(N, D)

mu = np.sum(xs, axis=0)
mu /= N

cov = 0

for n in range(N):
    x = xs[n]
    z = x - mu
    z = z[:, np.newaxis]
    cov += z @ z.T

cov /= N
print(mu)
print(cov)
```

    [0.49443495 0.49726356]
    [[ 0.08476319 -0.00023128]
     [-0.00023128  0.08394656]]

-   样本均值 $\mu$ 在代码中可直接用 `np.mean(X, axis=0)` 计算。
-   协方差矩阵 $\Sigma$ 可用 `np.cov(X, rowvar=False)` 直接得到。

``` python
# we load the data
import os

path = os.path.join(os.path.dirname('.'), 'height_weight.txt')
xs = np.loadtxt(path)

print(xs.shape)
```

    (25000, 2)

since we got 25000 element with 2 feature which are height and weight
now we are going to split it

``` python
small_xs = xs[:500] # we select first 500 data for demo
plt.scatter(x = small_xs[:, 0], y = small_xs[:, 1])
plt.xlabel('Height(cm)')
plt.ylabel("Weight(kg)")
plt.show()
```

![](/step3_files/figure-markdown_strict/cell-15-output-1.png)

we use 2d figure to show this distribution

``` python
mu = np.mean(xs, axis=0)
cov = np.cov(xs, rowvar=False)

# we use the same way you can copy paste the code above
X, Y = np.meshgrid(np.arange(150, 195, 0.5), np.arange(45, 75, 0.5))
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x = np.array([X[i, j], Y[i, j]])
        Z[i, j] = multivariable_normal(x, mu, cov)

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.set_xlabel('x')
ax1.set_xlabel('y')
ax1.set_zlabel('z')
ax1.plot_surface(X, Y, Z, cmap='viridis')

ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(x = small_xs[:, 0], y = small_xs[:, 1])
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_xlim(156, 189)
ax2.set_ylim(36, 79)
ax2.contour(X, Y, Z)
plt.show()
```

![](/step3_files/figure-markdown_strict/cell-16-output-1.png)
