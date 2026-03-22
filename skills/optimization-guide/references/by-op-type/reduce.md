# Reduce 类算子优化指南

Reduce 算子对张量的某些维度进行归约操作（如求和、最大值、均值等），是深度学习中常见的算子类型。

---

## 一、优化策略概览

| 策略 | 适用场景 | 预期提升 | 风险等级 |
|------|---------|---------|---------|
| 并行归约 | 所有 Reduce | 50-200% | 中 |
| 分块归约 | 大规模数据 | 30-100% | 中 |
| Warp 归约 | 小规模数据 | 20-50% | 中 |
| 内存访问优化 | 非连续访问 | 30-100% | 中 |

---

## 二、基础 Reduce Kernel

### 2.1 一维 Reduce（行归约）

```python
@triton.jit
def reduce_sum_1d_kernel(
    x_ptr, out_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # 分块归约
    row_sum = 0.0
    
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        row_sum += tl.sum(x, axis=0)
    
    tl.store(out_ptr + row_idx, row_sum)


def reduce_sum_1d(x: torch.Tensor, dim: int = -1):
    if dim == -1:
        dim = x.dim() - 1
    
    n_rows = x.shape[0] if dim == 1 else x.shape[1]
    n_cols = x.shape[1] if dim == 1 else x.shape[0]
    
    out = torch.empty(n_rows, device=x.device, dtype=torch.float32)
    
    BLOCK_SIZE = 1024
    grid = (n_rows,)
    
    reduce_sum_1d_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE)
    
    return out
```

### 2.2 二维 Reduce（全局归约）

```python
@triton.jit
def reduce_sum_2d_kernel(
    x_ptr, out_ptr,
    n_rows, n_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    
    # 每个 block 处理一部分行
    row_start = pid_m * BLOCK_M
    row_end = min(row_start + BLOCK_M, n_rows)
    
    partial_sum = tl.zeros([BLOCK_N], dtype=tl.float32)
    
    for row in range(row_start, row_end):
        for col_block in range(0, n_cols, BLOCK_N):
            col_offsets = col_block + tl.arange(0, BLOCK_N)
            mask = col_offsets < n_cols
            x = tl.load(x_ptr + row * n_cols + col_offsets, mask=mask, other=0.0)
            partial_sum += x
    
    # 存储部分和（需要二次归约）
    tl.store(out_ptr + pid_m * BLOCK_N + tl.arange(0, BLOCK_N), partial_sum)
```

---

## 三、常见 Reduce 算子

### 3.1 Sum

```python
@triton.jit
def sum_kernel(x_ptr, out_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    row_sum = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        row_sum += tl.sum(x, axis=0)
    
    tl.store(out_ptr + row_idx, row_sum)
```

### 3.2 Max/Min

```python
@triton.jit
def max_kernel(x_ptr, out_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    row_max = float('-inf')
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=float('-inf'))
        row_max = tl.maximum(row_max, tl.max(x, axis=0))
    
    tl.store(out_ptr + row_idx, row_max)


@triton.jit
def min_kernel(x_ptr, out_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    row_min = float('inf')
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=float('inf'))
        row_min = tl.minimum(row_min, tl.min(x, axis=0))
    
    tl.store(out_ptr + row_idx, row_min)
```

### 3.3 Mean

```python
@triton.jit
def mean_kernel(x_ptr, out_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    row_sum = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        row_sum += tl.sum(x, axis=0)
    
    tl.store(out_ptr + row_idx, row_sum / n_cols)
```

### 3.4 ArgMax/ArgMin

```python
@triton.jit
def argmax_kernel(x_ptr, out_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    max_val = float('-inf')
    max_idx = 0
    
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=float('-inf'))
        
        # 找到块内最大值和索引
        block_max = tl.max(x, axis=0)
        if block_max > max_val:
            max_val = block_max
            # 需要找到最大值的索引
            block_max_idx = tl.argmax(x, axis=0)
            max_idx = block_start + block_max_idx
    
    tl.store(out_ptr + row_idx, max_idx)
```

---

## 四、Softmax 优化

Softmax 是一个特殊的 Reduce 算子，需要多次归约。

```python
@triton.jit
def softmax_kernel(
    x_ptr, out_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # 第一遍：计算最大值（数值稳定性）
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


def softmax(x: torch.Tensor, dim: int = -1):
    if dim == -1:
        dim = x.dim() - 1
    
    n_rows = x.shape[0] if dim == 1 else x.shape[1]
    n_cols = x.shape[1] if dim == 1 else x.shape[0]
    
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    grid = (n_rows,)
    
    softmax_kernel[grid](x, out, n_rows, n_cols, BLOCK_SIZE)
    
    return out
```

---

## 五、LayerNorm 优化

```python
@triton.jit
def layernorm_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    n_rows, n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # 计算均值
    mean = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean /= n_cols
    
    # 计算方差
    var = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        var += tl.sum((x - mean) ** 2, axis=0)
    var /= n_cols
    
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # 计算输出
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

## 六、优化检查清单

- [ ] 是否使用分块归约处理大数据
- [ ] 是否正确处理数值稳定性（如 Softmax 的 max 技巧）
- [ ] 内存访问是否连续
- [ ] BLOCK_SIZE 是否合理
- [ ] Grid 是否充分利用 AI Core

---

## 七、常见问题

### Q1: Reduce 性能为什么不如预期？

可能原因：
1. 内存访问不连续
2. BLOCK_SIZE 选择不当
3. 存在原子操作瓶颈

### Q2: 如何处理超长行？

使用多次遍历，每次处理一部分数据
