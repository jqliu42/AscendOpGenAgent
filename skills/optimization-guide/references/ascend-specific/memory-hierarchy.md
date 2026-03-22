# Ascend 内存层次优化指南

理解 Ascend NPU 的内存层次结构，对于优化内存访问模式至关重要。

---

## 一、内存层次结构

### 1.1 内存层次概览

```
┌─────────────────────────────────────────────────────────────┐
│                    HBM (高带宽内存)                          │
│  容量: 32-64 GB                                             │
│  带宽: ~1.0-1.5 TB/s                                        │
│  延迟: 高 (~100s of cycles)                                 │
│  位置: 芯片外                                                │
└─────────────────────────────────────────────────────────────┘
                          ↑↓
┌─────────────────────────────────────────────────────────────┐
│                    L1 Buffer (片上缓存)                      │
│  容量: ~1-2 MB                                              │
│  带宽: 极高                                                  │
│  延迟: 低 (~10s of cycles)                                  │
│  位置: AI Core 内                                            │
└─────────────────────────────────────────────────────────────┘
                          ↑↓
┌─────────────────────────────────────────────────────────────┐
│                    L0 Buffer (寄存器级)                      │
│  容量: ~KB 级别                                              │
│  带宽: 最高                                                  │
│  延迟: 最低 (~1-2 cycles)                                   │
│  位置: 计算单元内                                            │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 各层次规格

| 层次 | 容量 | 带宽 | 延迟 | 用途 |
|------|------|------|------|------|
| HBM | 32-64 GB | 1.0-1.5 TB/s | 高 | 存储模型和数据 |
| L1 Buffer | 1-2 MB | 极高 | 低 | 存储分块数据 |
| L0 Buffer | KB 级 | 最高 | 最低 | 存储临时变量 |

---

## 二、内存访问优化原则

### 2.1 数据局部性

```
原则：尽量复用已加载的数据

时间局部性：同一数据多次使用
空间局部性：访问相邻数据

优化方法：
1. 分块处理，数据复用
2. 减少重复加载
3. 预取数据
```

### 2.2 内存带宽优化

```
原则：最大化内存带宽利用率

优化方法：
1. 连续访问（合并内存事务）
2. 对齐访问
3. 向量化加载
```

---

## 三、分块策略

### 3.1 分块大小选择

```python
# 分块大小需要考虑 L1 Buffer 容量

# 假设 L1 Buffer = 1 MB
# 每个元素 4 bytes (FP32)
# 可容纳元素数 = 1 MB / 4 bytes = 256K 元素

# MatMul 分块示例
# A block: [BLOCK_M, BLOCK_K]
# B block: [BLOCK_K, BLOCK_N]
# 总大小: BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N

# 例如 BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
# 大小: 128*64 + 64*128 = 16K 元素 = 64 KB
# 远小于 L1 Buffer，可行
```

### 3.2 分块代码示例

```python
@triton.jit
def tiled_matmul(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # 分块遍历 K 维度
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        
        # 加载块到 L1 Buffer
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])  # [BLOCK_M, BLOCK_K]
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])  # [BLOCK_K, BLOCK_N]
        
        # 计算（数据在 L1 Buffer 中）
        acc += tl.dot(a, b)
    
    # 存储结果
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)
```

---

## 四、内存访问模式

### 4.1 连续访问

```python
# ✓ 连续访问：高效
@triton.jit
def continuous_access(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 连续
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask)
    out = x * 2
    tl.store(out_ptr + offsets, out, mask=mask)

# ✗ 非连续访问：低效
@triton.jit
def strided_access(x_ptr, out_ptr, n, stride, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE * stride + tl.arange(0, BLOCK_SIZE) * stride  # 非连续
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask)
    out = x * 2
    tl.store(out_ptr + offsets, out, mask=mask)
```

### 4.2 对齐访问

```python
# 对齐访问：起始地址是 64 字节的倍数
# 对于 FP32，即 16 个元素的倍数

@triton.jit
def aligned_access(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # 确保 BLOCK_SIZE 是 16 的倍数
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask)
    out = x * 2
    tl.store(out_ptr + offsets, out, mask=mask)
```

---

## 五、数据复用

### 5.1 矩阵乘法中的数据复用

```
矩阵乘法 C = A @ B

A: [M, K] - 每行被复用 N 次
B: [K, N] - 每列被复用 M 次

分块策略：
- 将 A 的块加载到 L1 Buffer
- 将 B 的块加载到 L1 Buffer
- 在块内复用数据
```

### 5.2 代码示例

```python
@triton.jit
def matmul_with_reuse(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
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
        
        # 加载块（数据复用）
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])  # 复用 BLOCK_N 次
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])  # 复用 BLOCK_M 次
        
        acc += tl.dot(a, b)
    
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)
```

---

## 六、内存带宽利用率

### 6.1 计算内存带宽利用率

```python
def calculate_bandwidth_utilization(data_size_bytes, time_ms, theoretical_bandwidth_gbps):
    """
    计算内存带宽利用率
    
    Args:
        data_size_bytes: 数据传输量（字节）
        time_ms: 执行时间（毫秒）
        theoretical_bandwidth_gbps: 理论带宽（GB/s）
    
    Returns:
        利用率（百分比）
    """
    actual_bandwidth_gbps = data_size_bytes / (time_ms / 1000) / 1e9
    utilization = actual_bandwidth_gbps / theoretical_bandwidth_gbps * 100
    return utilization

# 示例
# 910B2 理论带宽 ~1.2 TB/s
# 数据量: 3 * 4096 * 4096 * 4 bytes = 192 MB
# 时间: 10 ms
# 实际带宽: 192 MB / 0.01 s = 19.2 GB/s
# 利用率: 19.2 / 1200 = 1.6%（低！）
```

### 6.2 提高内存带宽利用率

| 方法 | 说明 |
|------|------|
| 增大 BLOCK_SIZE | 减少内存事务数量 |
| 连续访问 | 合并内存事务 |
| 数据复用 | 减少重复加载 |

---

## 七、性能调优清单

### 7.1 内存访问检查

- [ ] 访问是否连续
- [ ] 访问是否对齐
- [ ] 是否有重复加载

### 7.2 分块检查

- [ ] 分块大小是否适合 L1 Buffer
- [ ] 是否充分利用数据复用
- [ ] Grid 是否充分利用 AI Core

### 7.3 带宽检查

- [ ] 内存带宽利用率是否 > 70%
- [ ] 是否存在内存瓶颈

---

## 八、常见问题

### Q1: 如何判断是否存在内存瓶颈？

检查内存带宽利用率，如果 > 80% 且性能仍不理想，则存在内存瓶颈

### Q2: 如何选择最优分块大小？

从推荐值开始，使用 autotune 搜索最优配置

### Q3: L1 Buffer 不够用怎么办？

减小分块大小，或使用多次遍历
