# MatMul 类算子优化指南

矩阵乘法是最核心的算子之一，本指南提供针对 MatMul 类算子的优化策略。

---

## 一、优化策略概览

| 策略 | 适用场景 | 预期提升 | 风险等级 |
|------|---------|---------|---------|
| 使用 tl.dot | 所有 MatMul | 100-500% | 中 |
| 分块大小调优 | 所有 MatMul | 10-50% | 低 |
| 内存访问优化 | 非连续输入 | 30-100% | 中 |
| 精度优化 | FP32 计算 | 10-30% | 低 |
| 流水线优化 | 大矩阵 | 20-80% | 中 |

---

## 二、核心优化：使用 tl.dot

### 2.1 为什么必须使用 tl.dot

```
tl.dot 是触发 Ascend Cube Unit 的唯一方式

Cube Unit 性能：
- 矩阵乘法专用硬件
- 性能是逐元素计算的 10-100 倍
- 必须使用 tl.dot 才能触发
```

### 2.2 基础 MatMul Kernel

```python
import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(
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


def matmul(a: torch.Tensor, b: torch.Tensor):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    
    return c
```

---

## 三、参数调优

### 3.1 BLOCK_SIZE 选择

| 矩阵规模 | BLOCK_M | BLOCK_N | BLOCK_K | 说明 |
|---------|---------|---------|---------|------|
| 小 (< 512) | 64 | 64 | 32 | 小块减少浪费 |
| 中 (512-2048) | 128 | 128 | 64 | 平衡选择 |
| 大 (> 2048) | 128 | 128 | 128 | 大块提高复用 |

### 3.2 架构特定建议

| 架构 | 推荐 BLOCK_SIZE | 说明 |
|------|----------------|------|
| 910B1 | 64-128 | 较小的块 |
| 910B2 | 128-256 | 中等块 |
| 910B4 | 128-512 | 可用更大的块 |

### 3.3 自动调优配置

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_autotuned(...):
    ...
```

---

## 四、内存访问优化

### 4.1 使用块指针

```python
@triton.jit
def matmul_block_ptr(
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
    
    # 使用 make_block_ptr 创建块指针
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        acc += tl.dot(a, b)
        
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
    
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, acc)
```

### 4.2 处理非连续输入

```python
def matmul_non_contiguous(a: torch.Tensor, b: torch.Tensor):
    # 确保输入是连续的
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    
    return matmul(a, b)
```

---

## 五、精度优化

### 5.1 混合精度计算

```python
@triton.jit
def matmul_mixed_precision(
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
    
    # 累加器使用 FP32，避免精度损失
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        
        a = tl.load(a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak)
        b = tl.load(b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn)
        
        # 输入可以是 BF16/FP16，累加使用 FP32
        acc += tl.dot(a, b, out_dtype=tl.float32)
    
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc, mask=c_mask)
```

### 5.2 精度选择建议

| 场景 | 输入精度 | 累加精度 | 输出精度 |
|------|---------|---------|---------|
| 训练前向 | BF16 | FP32 | BF16 |
| 训练反向 | FP32 | FP32 | FP32 |
| 推理 | BF16/FP16 | FP32 | BF16/FP16 |

---

## 六、Batch MatMul 优化

```python
@triton.jit
def batch_matmul_kernel(
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
    
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        
        a = tl.load(a_base + rm[:, None] * stride_am + rk[None, :] * stride_ak)
        b = tl.load(b_base + rk[:, None] * stride_bk + rn[None, :] * stride_bn)
        
        acc += tl.dot(a, b)
    
    c_base = c_ptr + pid_b * stride_cb
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_base + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc, mask=c_mask)


def batch_matmul(a: torch.Tensor, b: torch.Tensor):
    B, M, K = a.shape
    B, K, N = b.shape
    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    
    grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    batch_matmul_kernel[grid](
        a, b, c,
        B, M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    
    return c
```

---

## 七、优化检查清单

- [ ] 是否使用 tl.dot（必须）
- [ ] BLOCK_SIZE 是否为 16 的倍数
- [ ] 累加器是否使用 FP32
- [ ] 输入是否连续
- [ ] Grid 是否充分利用 AI Core
- [ ] 是否正确处理边界（mask）

---

## 八、常见问题

### Q1: 为什么性能还是不如 PyTorch？

可能原因：
1. PyTorch 使用了高度优化的 cuBLAS/aclBLAS 库
2. BLOCK_SIZE 选择不当
3. 输入数据不连续

### Q2: 如何处理非方阵？

调整 BLOCK_M、BLOCK_N、BLOCK_K 以适应矩阵形状

### Q3: 如何处理大矩阵？

增大 BLOCK_K，提高数据复用率
