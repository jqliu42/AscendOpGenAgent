# Elementwise 类算子优化指南

Elementwise 算子对输入张量的每个元素独立执行相同操作，是最常见的算子类型之一。

---

## 一、优化策略概览

| 策略 | 适用场景 | 预期提升 | 风险等级 |
|------|---------|---------|---------|
| 向量化 | 所有 Elementwise | 20-50% | 低 |
| 内存合并 | 非连续访问 | 30-100% | 中 |
| 算子融合 | 多个 Elementwise | 50-200% | 中 |
| BLOCK_SIZE 调优 | 所有 Elementwise | 10-30% | 低 |

---

## 二、基础 Elementwise Kernel

### 2.1 单输入 Elementwise

```python
@triton.jit
def unary_elementwise_kernel(
    x_ptr, out_ptr, n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Elementwise 操作（示例：ReLU）
    out = tl.maximum(x, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


def relu(x: torch.Tensor):
    n = x.numel()
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    unary_elementwise_kernel[grid](x, out, n, BLOCK_SIZE)
    
    return out
```

### 2.2 双输入 Elementwise

```python
@triton.jit
def binary_elementwise_kernel(
    a_ptr, b_ptr, out_ptr, n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    
    # Elementwise 操作（示例：Add）
    out = a + b
    
    tl.store(out_ptr + offsets, out, mask=mask)


def add(a: torch.Tensor, b: torch.Tensor):
    n = a.numel()
    out = torch.empty_like(a)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    binary_elementwise_kernel[grid](a, b, out, n, BLOCK_SIZE)
    
    return out
```

---

## 三、常见 Elementwise 算子

### 3.1 激活函数

```python
# ReLU
@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


# GELU
@triton.jit
def gelu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    sqrt2 = 1.41421356237
    out = 0.5 * x * (1.0 + tl.erf(x / sqrt2))
    
    tl.store(out_ptr + offsets, out, mask=mask)


# SiLU (Swish)
@triton.jit
def silu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # SiLU(x) = x * sigmoid(x)
    out = x * tl.sigmoid(x)
    
    tl.store(out_ptr + offsets, out, mask=mask)
```

### 3.2 数学运算

```python
# Exp
@triton.jit
def exp_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.exp(x)
    tl.store(out_ptr + offsets, out, mask=mask)


# Sqrt
@triton.jit
def sqrt_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.sqrt(x)
    tl.store(out_ptr + offsets, out, mask=mask)


# Rsqrt (1/sqrt(x))
@triton.jit
def rsqrt_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.rsqrt(x)  # Triton 内置 rsqrt
    tl.store(out_ptr + offsets, out, mask=mask)
```

---

## 四、算子融合

### 4.1 为什么融合

```
不融合：
x → ReLU → temp → Dropout → out
    (kernel 1)    (kernel 2)
    内存读写 2 次

融合后：
x → FusedReLU Dropout → out
         (kernel 1)
    内存读写 1 次
```

### 4.2 融合示例

```python
# 融合：Add + LayerNorm
@triton.jit
def add_layernorm_kernel(
    a_ptr, b_ptr, out_ptr, weight_ptr, bias_ptr,
    n_rows, n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    # Add
    a = tl.load(a_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    x = a + b
    
    # LayerNorm
    mean = tl.sum(x, axis=0) / n_cols
    var = tl.sum((x - mean) ** 2, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    
    w = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    b = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    out = (x - mean) * rstd * w + b
    tl.store(out_ptr + row_start + offsets, out, mask=mask)


# 融合：Silu + Mul (SiLU 的反向或 SwiGLU)
@triton.jit
def silu_mul_kernel(
    x_ptr, y_ptr, out_ptr, n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # SiLU(x) * y
    out = x * tl.sigmoid(x) * y
    
    tl.store(out_ptr + offsets, out, mask=mask)
```

---

## 五、参数调优

### 5.1 BLOCK_SIZE 选择

| 数据规模 | 推荐 BLOCK_SIZE |
|---------|----------------|
| < 100K | 256-512 |
| 100K - 1M | 512-1024 |
| > 1M | 1024-2048 |

### 5.2 自动调优

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['n'],
)
@triton.jit
def elementwise_autotuned(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    ...
```

---

## 六、优化检查清单

- [ ] 内存访问是否连续
- [ ] BLOCK_SIZE 是否合理
- [ ] 是否可以融合多个算子
- [ ] 是否正确处理边界（mask）
- [ ] Grid 是否充分利用 AI Core

---

## 七、常见问题

### Q1: Elementwise 性能为什么不如 PyTorch？

可能原因：
1. PyTorch 使用了向量化指令
2. BLOCK_SIZE 选择不当
3. 内存访问不连续

### Q2: 什么时候应该融合算子？

融合条件：
1. 算子之间存在数据依赖
2. 中间结果不需要保存
3. 融合后不会增加寄存器压力
