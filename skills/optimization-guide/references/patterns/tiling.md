# 分块优化（Tiling）

分块优化是最常用且最有效的优化技术之一，通过将大数据块分解为小块处理，提高缓存命中率和数据复用。

---

## 一、优化原理

### 1.1 为什么需要分块

```
问题：大数据块处理
┌─────────────────────────────────────┐
│ 大块数据无法全部放入片上缓存          │
│ → 频繁访问 HBM                       │
│ → 内存带宽成为瓶颈                    │
└─────────────────────────────────────┘

解决方案：分块处理
┌─────────────────────────────────────┐
│ 将大块分解为小块                      │
│ → 小块可放入 L1 Buffer               │
│ → 减少对 HBM 的访问次数               │
│ → 提高数据复用率                      │
└─────────────────────────────────────┘
```

### 1.2 分块带来的收益

| 收益 | 说明 |
|------|------|
| 缓存命中率提升 | 数据块可放入片上缓存 |
| 数据复用 | 同一块数据可多次使用 |
| 并行度提升 | 更细粒度的任务划分 |
| 内存带宽优化 | 减少 HBM 访问次数 |

---

## 二、分块策略

### 2.1 一维分块

**适用场景**：一维数据操作（如向量运算）

```python
@triton.jit
def vector_add_tiled(
    x_ptr, y_ptr, out_ptr, n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)
```

**参数选择**：
- BLOCK_SIZE: 128-1024
- 推荐：256 或 512

### 2.2 二维分块

**适用场景**：矩阵运算、二维卷积

```python
@triton.jit
def matmul_2d_tiled(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        
        a = tl.load(a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak)
        b = tl.load(b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn)
        
        acc += tl.dot(a, b)
    
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc, mask=c_mask)
```

**参数选择**：

| 场景 | BLOCK_M | BLOCK_N | BLOCK_K |
|------|---------|---------|---------|
| 小矩阵 | 64 | 64 | 32 |
| 中等矩阵 | 128 | 128 | 64 |
| 大矩阵 | 128 | 128 | 128 |

### 2.3 三维分块

**适用场景**：三维张量运算、Batch 操作

```python
@triton.jit
def batch_matmul_3d_tiled(
    a_ptr, b_ptr, c_ptr,
    B, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    a_base = a_ptr + pid_b * stride_ab
    b_base = b_ptr + pid_b * stride_bb
    c_base = c_ptr + pid_b * stride_cb
    
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        
        a = tl.load(a_base + rm[:, None] * stride_am + rk[None, :] * stride_ak)
        b = tl.load(b_base + rk[:, None] * stride_bk + rn[None, :] * stride_bn)
        
        acc += tl.dot(a, b)
    
    tl.store(c_base + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc)
```

---

## 三、分块大小选择

### 3.1 选择原则

```
分块大小选择考虑因素：
├── 片上缓存大小
│   └── 块大小 × 数据类型 × 块数量 ≤ L1 Buffer
│
├── 并行度
│   └── Grid 大小 = 总数据量 / 块大小
│   └── Grid ≥ AI Core 数量 × 2
│
├── 内存访问效率
│   └── 块大小应为 2 的幂次
│   └── 块大小应能被向量宽度整除
│
└── 硬件特性
    └── Cube Unit 要求块大小为 16 的倍数
```

### 3.2 Ascend 特定建议

| 架构 | 推荐 BLOCK_SIZE | 最大块大小 |
|------|----------------|-----------|
| 910B1 | 64-128 | 256 |
| 910B2 | 128-256 | 512 |
| 910B4 | 128-512 | 1024 |

### 3.3 自动调优

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=1),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=1),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=1),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=1),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_autotuned(...):
    ...
```

---

## 四、分块优化案例

### 4.1 案例：Softmax 分块优化

**问题**：Softmax 需要对整行进行 reduce 操作，数据量大时效率低

```python
# 原始实现：整行处理
@triton.jit
def softmax_naive(x_ptr, out_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # 加载整行
    offsets = tl.arange(0, n_cols)
    x = tl.load(x_ptr + row_start + offsets)
    
    # Softmax 计算
    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)
    sum_exp = tl.sum(x_exp, axis=0)
    out = x_exp / sum_exp
    
    tl.store(out_ptr + row_start + offsets, out)
```

**优化**：分块处理，支持超长行

```python
@triton.jit
def softmax_tiled(
    x_ptr, out_ptr, n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # 第一遍：计算最大值
    row_max = float('-inf')
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=float('-inf'))
        row_max = tl.maximum(row_max, tl.max(x, axis=0))
    
    # 第二遍：计算 exp 和 sum
    sum_exp = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=float('-inf'))
        x_exp = tl.exp(x - row_max)
        sum_exp += tl.sum(x_exp, axis=0)
    
    # 第三遍：计算输出
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=float('-inf'))
        x_exp = tl.exp(x - row_max)
        out = x_exp / sum_exp
        tl.store(out_ptr + row_start + offsets, out, mask=mask)
```

### 4.2 案例：LayerNorm 分块优化

```python
@triton.jit
def layernorm_tiled(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    n_rows, n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # 分块计算均值和方差
    mean = 0.0
    var = 0.0
    
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean /= n_cols
    
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        var += tl.sum((x - mean) ** 2, axis=0)
    var /= n_cols
    
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # 分块计算输出
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
        b = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        out = (x - mean) * rstd * w + b
        tl.store(out_ptr + row_start + offsets, out, mask=mask)
```

---

## 五、分块优化检查清单

- [ ] 块大小是否为 2 的幂次
- [ ] 块大小是否适合目标架构
- [ ] Grid 大小是否充分利用 AI Core
- [ ] 是否正确处理边界条件（mask）
- [ ] 是否避免重复加载同一数据
- [ ] 是否利用了数据复用

---

## 六、常见问题

### Q1: 块大小选择太大怎么办？

**问题**：块太大导致 Grid 过小，并行度不足

**解决**：减小块大小，增加 Grid

```python
# 问题
BLOCK_SIZE = 4096  # 太大
grid = (triton.cdiv(n, BLOCK_SIZE),)  # Grid 可能只有几个

# 解决
BLOCK_SIZE = 256  # 适中
grid = (triton.cdiv(n, BLOCK_SIZE),)  # Grid 充分
```

### Q2: 块大小选择太小怎么办？

**问题**：块太小导致内存访问效率低

**解决**：增大块大小

```python
# 问题
BLOCK_SIZE = 16  # 太小，内存效率低

# 解决
BLOCK_SIZE = 128  # 适中
```

### Q3: 如何确定最优块大小？

**方法1**：使用 autotune 自动搜索

```python
@triton.autotune(
    configs=[triton.Config({'BLOCK_SIZE': s}) for s in [64, 128, 256, 512]],
    key=['n'],
)
```

**方法2**：根据数据规模手动选择

| 数据规模 | 推荐块大小 |
|---------|-----------|
| < 10K | 64-128 |
| 10K - 1M | 128-256 |
| > 1M | 256-512 |
