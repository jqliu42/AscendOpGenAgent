# 内存合并访问优化（Memory Coalescing）

内存合并访问是提升内存带宽利用率的关键优化技术，通过优化访问模式减少内存事务数量。

---

## 一、优化原理

### 1.1 什么是内存合并

```
非合并访问：
线程 0: load addr[0]
线程 1: load addr[100]
线程 2: load addr[200]
...
→ 每个线程独立访问，内存事务多

合并访问：
线程 0-31: load addr[0:32]
→ 一次内存事务完成多个线程的数据加载
```

### 1.2 合并访问的条件

| 条件 | 说明 |
|------|------|
| 连续访问 | 相邻线程访问相邻地址 |
| 对齐访问 | 起始地址对齐 |
| 相同大小 | 所有线程访问相同大小的数据 |

---

## 二、常见访问模式

### 2.1 连续访问（最优）

```python
@triton.jit
def continuous_access(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # 连续访问：offsets 连续
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask)  # ✓ 合并访问
    out = x * 2
    tl.store(out_ptr + offsets, out, mask=mask)  # ✓ 合并存储
```

### 2.2 固定步长访问

```python
@triton.jit
def strided_access(x_ptr, out_ptr, n, stride, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # 步长访问：stride != 1
    offsets = pid * BLOCK_SIZE * stride + tl.arange(0, BLOCK_SIZE) * stride
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask)  # ✗ 非合并访问
    ...
```

**优化方案**：重排数据或调整访问模式

### 2.3 随机访问

```python
@triton.jit
def random_access(x_ptr, idx_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # 随机索引
    idx = tl.load(idx_ptr + offsets, mask=mask)  # 索引可能是随机的
    x = tl.load(x_ptr + idx, mask=mask)  # ✗ 随机访问，无法合并
    ...
```

---

## 三、优化策略

### 3.1 数据布局优化

**问题**：矩阵按列访问但数据是行存储

```python
# 问题代码
@triton.jit
def matmul_column_access(a_ptr, b_ptr, c_ptr, M, N, K, ...):
    # A 矩阵按列访问（stride_am = 1）
    # 但数据是行存储（stride_am = K）
    for m in range(M):
        for k in range(K):
            a = tl.load(a_ptr + m * K + k)  # 行存储，按行访问 ✓
            b = tl.load(b_ptr + k * N + n)  # 行存储，按行访问 ✓
```

**优化**：使用块指针进行 2D 访问

```python
@triton.jit
def matmul_block_ptr(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 使用 make_block_ptr 进行 2D 块访问
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(K, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),  # 行优先
    )
    
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(N, 1),
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
    
    # 存储
    c_block_ptr = tl.make_block_ptr(...)
    tl.store(c_block_ptr, acc)
```

### 3.2 转置优化

**问题**：需要访问转置后的数据

```python
# 问题：直接访问转置数据
@triton.jit
def access_transpose(x_ptr, out_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    # 访问 x.T，即按列访问行存储数据
    for m in range(M):
        for n in range(N):
            x = tl.load(x_ptr + n * M + m)  # ✗ 非连续访问
            ...
```

**优化方案1**：预处理转置

```python
# 在 kernel 外部转置数据
x_t = x.T.contiguous()

# kernel 中访问转置后的数据
@triton.jit
def access_after_transpose(x_t_ptr, out_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    for m in range(M):
        for n in range(N):
            x = tl.load(x_t_ptr + m * N + n)  # ✓ 连续访问
            ...
```

**优化方案2**：使用块指针

```python
@triton.jit
def access_transpose_block_ptr(x_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # 使用块指针访问转置数据
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(N, M),  # 注意：shape 是转置后的
        strides=(M, 1),  # 原始数据的 stride
        offsets=(pid_n * BLOCK_N, pid_m * BLOCK_M),
        block_shape=(BLOCK_N, BLOCK_M),
        order=(1, 0),
    )
    x = tl.load(x_block_ptr)  # 自动处理转置
    ...
```

### 3.3 向量化加载

```python
@triton.jit
def vectorized_load(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # 向量化加载：一次加载多个元素
    x = tl.load(x_ptr + offsets, mask=mask)  # Triton 自动向量化
    
    # 处理
    out = x * 2
    
    # 向量化存储
    tl.store(out_ptr + offsets, out, mask=mask)
```

---

## 四、内存对齐

### 4.1 对齐的重要性

```
对齐访问：
地址 = 基地址 + 偏移
偏移 % 对齐粒度 = 0

对齐粒度：
- 32 字节（256 位）
- 64 字节（512 位）
- 128 字节（缓存行）
```

### 4.2 确保对齐

```python
@triton.jit
def aligned_access(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # 确保 BLOCK_SIZE * sizeof(dtype) 是对齐的
    # 例如：BLOCK_SIZE=128, dtype=float32 (4 bytes)
    # 128 * 4 = 512 bytes，是 64 字节的倍数 ✓
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask)
    ...
```

---

## 五、优化检查清单

### 5.1 访问模式检查

- [ ] 访问是否连续（stride = 1）
- [ ] 是否使用块指针（make_block_ptr）
- [ ] 是否正确处理边界（mask）

### 5.2 对齐检查

- [ ] BLOCK_SIZE * sizeof(dtype) 是否对齐
- [ ] 数据起始地址是否对齐

### 5.3 性能检查

- [ ] 内存带宽利用率是否 > 70%
- [ ] 是否存在不必要的重复加载

---

## 六、案例分析

### 案例：矩阵乘法的内存访问优化

```python
@triton.jit
def matmul_optimized(
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
    
    # 初始化累加器
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # K 维度分块
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        
        # A 矩阵：[BLOCK_M, BLOCK_K]，按行访问 ✓
        a = tl.load(a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak)
        
        # B 矩阵：[BLOCK_K, BLOCK_N]，按行访问 ✓
        b = tl.load(b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn)
        
        # 矩阵乘法
        acc += tl.dot(a, b)
    
    # 存储 C：[BLOCK_M, BLOCK_N]，按行存储 ✓
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc, mask=c_mask)
```

**优化要点**：
1. A 矩阵：按行访问，连续加载
2. B 矩阵：按行访问，连续加载
3. C 矩阵：按行存储，连续写入
4. 使用 tl.dot 触发 Cube Unit
