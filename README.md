# 北京大学2025秋季机器学习期末作业

## kissingnumber

## Kissing Number (接吻数) 项目数学解释与公式汇总

这份内容可以直接用于 PDF 报告的 **"Problem Formulation" (问题建模)** 和 **"Methodology" (方法论)** 章节。

### 1. 基础定义：从几何到代数

**几何定义：**
在 $n$ 维欧几里得空间 $\mathbb{R}^n$ 中，接吻数 $\tau_n$ 定义为能同时与一个单位中心球相切（touching）且互不重叠（non-overlapping）的单位球的最大数量。

**代数转化（球面编码）：**
假设中心球的半径为 1，球心在原点 $O$。周围的单位球如果与中心球相切，那么它们的球心 $x_1, x_2, \dots, x_N$ 必定位于半径为 $R=2$ 的球面上。
为了简化计算，我们通常将整个系统缩放 1/2，考虑中心球半径为 0.5，周围球半径为 0.5。或者更常见的是：**直接考虑周围球心的方向向量**。

等价问题：在 $n$ 维单位球面 $S^{n-1} = \{x \in \mathbb{R}^n : \|x\|_2 = 1\}$ 上，最多能放置多少个点 $x_1, \dots, x_N$，使得任意两点之间的欧几里得距离至少为 1？

**核心约束公式：**
对于任意两个不同的点 $x_i, x_j$ ($i \neq j$)：
1.  **模长约束（位于单位球面上）：**
    $$\|x_i\|_2 = 1$$
2.  **距离约束（不重叠）：**
    $$\|x_i - x_j\|_2 \ge 1$$

---

### 2. 向量与余弦相似度视角 (Machine Learning 建模基础)

在机器学习中，处理距离往往不如处理向量点积（内积）方便。我们可以利用余弦定律将距离约束转化为角度约束。

**推导：**
$$\|x_i - x_j\|^2 = (x_i - x_j) \cdot (x_i - x_j) = \|x_i\|^2 + \|x_j\|^2 - 2 x_i \cdot x_j$$
因为 $\|x_i\| = \|x_j\| = 1$，所以：
$$\|x_i - x_j\|^2 = 2 - 2 \cos(\theta_{ij})$$
其中 $\theta_{ij}$ 是向量 $x_i$ 和 $x_j$ 的夹角。

我们要满足 $\|x_i - x_j\| \ge 1$，即 $\|x_i - x_j\|^2 \ge 1$，代入得：
$$2 - 2 x_i \cdot x_j \ge 1 \implies x_i \cdot x_j \le 0.5$$

**最终公式 (ML 约束)：**
在 $n$ 维空间寻找 $N$ 个点，满足：
$$x_i^T x_j \le \frac{1}{2}, \quad \forall i \neq j$$
这意味任意两点的夹角 $\theta_{ij} \ge 60^\circ$。

**损失函数 (Loss Function)：**
为了用梯度下降求解，我们将上述硬约束（Hard Constraint）转化为软约束（Soft Constraint）的能量函数（Potential Energy）：
$$L(X) = \sum_{i \neq j} \text{ReLU}(x_i^T x_j - 0.5)$$
或者使用更平滑的平方形式：
$$L(X) = \sum_{i \neq j} \left( \max(0, x_i^T x_j - 0.5) \right)^2$$
*目标是最小化 $L(X)$。如果 $L(X)$ 能降到 0，说明存在这样的构型。*

---

### 3. 进阶视角：AlphaEvolve 论文中的离散化方法

在 AlphaEvolve 论文中，DeepMind 的团队采用了一种 **基于整数格点（Integer Lattice）** 的构造方法。

**核心引理 (Lemma 1)：**
AlphaEvolve 论文提出了一个非常巧妙的引理：

设 $C \subset \mathbb{R}^d$ 是一个不包含原点的点集（可以是整数点），如果它满足：
$$\min \{ \|u - v\| : u, v \in C, u \neq v \} \ge \max \{ \|u\| : u \in C \}$$
即：**集合中任意两点的最小距离 $\ge$ 集合中模长最大的向量长度。**

**构造结论：**
如果找到了这样的集合 $C$，那么集合 $C' = \{ \frac{2u}{\|u\|} : u \in C \}$ 就构成了一个合法的 Kissing Configuration。
也就是说，$C$ 的大小 $|C|$ 就是接吻数的一个 **下界 (Lower Bound)**。

**证明逻辑：**
对于任意 $x, y \in C$，根据条件有 $\|x - y\|^2 \ge \max(\|x\|^2, \|y\|^2)$。
展开左边：$\|x\|^2 + \|y\|^2 - 2\langle x, y \rangle \ge \max(\|x\|^2, \|y\|^2)$。
这蕴含了 $2\langle x, y \rangle \le \min(\|x\|^2, \|y\|^2) \le \|x\| \cdot \|y\|$。
进而推导出归一化后的点积 $\frac{x}{\|x\|} \cdot \frac{y}{\|y\|} \le 0.5$。

**应用：**
1.  **连续优化 (Continuous Optimization):** 使用 PyTorch 优化 $x_i \cdot x_j \le 0.5$。
2.  **离散搜索 (Discrete Search - AlphaEvolve):** 搜索整数点集 $C \subset \mathbb{Z}^n$，满足 $\min \|u-v\| \ge \max \|u\|$。这是论文能在 11 维找到 593 个点（打破记录）的关键数学工具 。

### 总结：报告中可用的公式清单

| 类别 | 公式描述 | 数学公式 |
| :--- | :--- | :--- |
| **基础约束** | 距离定义 | $\|x_i - x_j\|_2 \ge 1, \quad \|x_i\|_2 = 1$ |
| **ML模型** | 余弦相似度 | $\cos(\theta_{ij}) = x_i \cdot x_j \le 0.5$ |
| **ML模型** | 损失函数 | $Loss = \sum_{i \neq j} \max(0, x_i \cdot x_j - 0.5)^2$ |
| **进阶(论文)** | 构造引理 | $\min_{u \neq v} \|u - v\| \ge \max_u \|u\|$ |
| **已知结论** | 常见维度 | $d=2 \to 6, \quad d=3 \to 12, \quad d=8 \to 240$ |
