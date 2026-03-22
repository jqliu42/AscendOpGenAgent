# Ascend Cube Unit 优化指南

Cube Unit 是 Ascend NPU 的矩阵计算加速单元，是实现高性能矩阵运算的关键。

---

## 一、Cube Unit 概述

### 1.1 硬件架构

```
AI Core
├── Cube Unit（矩阵计算单元）
│   ├── 功能：矩阵乘法加速
│   ├── 触发：tl.dot
│   ├── 性能：TOPS 级别
│   └── 精度：FP16, BF16, INT8
│
├── Vector Unit（向量计算单元）
│   └── 功能：逐元素运算
│
└── Scalar Unit（标量计算单元）
    └── 功能：标量运算
```

### 1.2 Cube Unit 规格

| 架构 | Cube 吞吐量 (FP16) | Cube 吞吐量 (BF16) | 支持精度 |
|------|-------------------|-------------------|---------|
| 910B1 | 256 TOPS | 128 TOPS | FP16, BF16, INT8 |
| 910B2 | 280 TOPS | 140 TOPS | FP16, BF16, INT8 |
| 910B4 | 376+ TOPS | 188+ TOPS | FP16, BF16, INT8, FP32 |

---

## 二、触发 Cube Unit

### 2.1 使用 tl.dot

**tl.dot 是触发 Cube Unit 的唯一方式**

```python
# ✗ 错误：不触发 Cube Unit
@triton.jit
def matmul_no_cube(a_ptr, b_ptr, c_ptr, M, N, K):
    for m in range(M):
        for n in range(N):
            acc = 0
            for k in range(K):
                acc += a[m, k] * b[k, n]  # 逐元素乘法
            c[m, n] = acc

# ✓ 正确：触发 Cube Unit
@triton.jit
def matmul_with_cube(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # 加载块
    a_block = tl.load(...)  # [BLOCK_M, BLOCK_K]
    b_block = tl.load(...)  # [BLOCK_K, BLOCK_N]
    
    # 使用 tl.dot 触发 Cube Unit
    c_block = tl.dot(a_block, b_block)  # [BLOCK_M, BLOCK_N]
    
    tl.store(..., c_block)
```

### 2.2 tl.dot 使用要求

| 要求 | 说明 |
|------|------|
| 块大小 | BLOCK_SIZE 需要是 16 的倍数 |
| 输入类型 | FP16, BF16, INT8 |
| 输出类型 | FP32（累加器）或与输入相同 |
| 形状 | 2D 矩阵 |

```python
# tl.dot 语法
tl.dot(a, b, out_dtype=None, allow_tf32=True)

# 参数说明
# a: [M, K] 输入矩阵 A
# b: [K, N] 输入矩阵 B
# out_dtype: 输出数据类型，默认与输入相同
# allow_tf32: 是否允许使用 TF32（仅 FP32 输入时有效）

# 示例
@triton.jit
def dot_example(a_block, b_block):
    # BF16 输入，FP32 累加
    c = tl.dot(a_block, b_block, out_dtype=tl.float32)
    return c
```

---

## 三、块大小选择

### 3.1 Cube Unit 块大小要求

```
Cube Unit 要求：
- BLOCK_M, BLOCK_N, BLOCK_K 必须是 16 的倍数
- 推荐：32, 64, 128, 256

最优配置（经验值）：
- BLOCK_M = 128
- BLOCK_N = 128
- BLOCK_K = 64 或 128
```

### 3.2 架构特定建议

| 架构 | BLOCK_M | BLOCK_N | BLOCK_K | 说明 |
|------|---------|---------|---------|------|
| 910B1 | 64-128 | 64-128 | 32-64 | 较小块 |
| 910B2 | 128 | 128 | 64-128 | 中等块 |
| 910B4 | 128-256 | 128-256 | 128 | 可用更大的块 |

### 3.3 块大小与性能关系

```python
# 块太小：Cube Unit 利用率低
BLOCK_M = 16  # ✗ 太小
BLOCK_N = 16  # ✗ 太小
BLOCK_K = 16  # ✗ 太小

# 块适中：Cube Unit 利用率高
BLOCK_M = 128  # ✓ 适中
BLOCK_N = 128  # ✓ 适中
BLOCK_K = 64   # ✓ 适中

# 块太大：可能超出片上缓存
BLOCK_M = 512  # ? 可能太大
BLOCK_N = 512  # ? 可能太大
BLOCK_K = 256  # ? 可能太大
```

---

## 四、精度选择

### 4.1 精度性能对比

| 输入精度 | 累加精度 | 相对性能 | 精度 |
|---------|---------|---------|------|
| FP32 | FP32 | 1.0x | 最高 |
| BF16 | FP32 | 2-4x | 高 |
| FP16 | FP32 | 2-4x | 中 |
| INT8 | INT32 | 4-8x | 低 |

### 4.2 推荐精度配置

```python
@triton.jit
def matmul_optimal_precision(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 加载 BF16 输入
    a = tl.load(...).to(tl.bfloat16)  # [BLOCK_M, BLOCK_K]
    b = tl.load(...).to(tl.bfloat16)  # [BLOCK_K, BLOCK_N]
    
    # FP32 累加（避免精度损失）
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc += tl.dot(a, b, out_dtype=tl.float32)
    
    # 输出
    tl.store(..., acc)
```

### 4.3 精度选择建议

| 场景 | 输入精度 | 累加精度 | 输出精度 |
|------|---------|---------|---------|
| 训练前向 | BF16 | FP32 | BF16 |
| 训练反向 | FP32 | FP32 | FP32 |
| 推理 | BF16/FP16 | FP32 | BF16/FP16 |
| 量化推理 | INT8 | INT32 | INT8 |

---

## 五、内存访问优化

### 5.1 数据布局

```
Cube Unit 对数据布局的要求：
- 输入矩阵：行优先（Row Major）
- 连续访问：相邻元素在内存中连续
```

```python
# 确保输入是连续的
def ensure_contiguous(a: torch.Tensor, b: torch.Tensor):
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    return a, b
```

### 5.2 使用块指针

```python
@triton.jit
def matmul_block_ptr(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
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
        strides=(K, 1),  # 行优先
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),  # 行优先
    )
    
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(N, 1),  # 行优先
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
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, acc)
```

---

## 六、性能调优清单

### 6.1 必须检查项

- [ ] 是否使用 `tl.dot`
- [ ] BLOCK_SIZE 是否是 16 的倍数
- [ ] 输入数据是否连续
- [ ] 累加器是否使用 FP32

### 6.2 性能优化项

- [ ] BLOCK_SIZE 是否针对架构调优
- [ ] 精度选择是否合理
- [ ] 是否使用块指针
- [ ] Grid 是否充分利用 AI Core

---

## 七、常见问题

### Q1: 为什么性能不如预期？

检查项：
1. 是否使用 tl.dot
2. BLOCK_SIZE 是否合理
3. 输入是否连续
4. 精度是否合适

### Q2: 如何调试 Cube Unit 使用情况？

使用 NPU 性能分析工具检查 Cube Unit 利用率

### Q3: 块大小如何选择？

从 128x128x64 开始，使用 autotune 搜索最优配置
