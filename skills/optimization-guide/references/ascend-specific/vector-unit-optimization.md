# Ascend Vector Unit 优化指南

Vector Unit 是 Ascend NPU 的向量计算单元，用于加速逐元素运算。

---

## 一、Vector Unit 概述

### 1.1 硬件架构

```
AI Core
├── Cube Unit（矩阵计算单元）
│   └── 矩阵乘法
│
├── Vector Unit（向量计算单元）
│   ├── 功能：逐元素运算
│   ├── 触发：tl 逐元素 API
│   ├── 性能：高吞吐
│   └── 支持：加减乘除、激活函数、比较等
│
└── Scalar Unit（标量计算单元）
    └── 标量运算
```

### 1.2 Vector Unit 支持的操作

| 操作类型 | Triton API | 说明 |
|---------|-----------|------|
| 算术运算 | `+`, `-`, `*`, `/` | 加减乘除 |
| 数学函数 | `tl.exp`, `tl.log`, `tl.sqrt` | 指数、对数、开方 |
| 激活函数 | `tl.sigmoid`, `tl.relu` | Sigmoid, ReLU |
| 比较运算 | `tl.maximum`, `tl.minimum` | 最大最小值 |
| 类型转换 | `.to()` | 数据类型转换 |

---

## 二、向量化优化

### 2.1 向量化访问

```python
@triton.jit
def vectorized_elementwise(
    x_ptr, out_ptr, n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # 向量化加载：一次加载 BLOCK_SIZE 个元素
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask)  # 向量化加载
    
    # 向量化计算：对所有元素同时操作
    out = tl.exp(x) + tl.sqrt(x)  # Vector Unit 执行
    
    # 向量化存储
    tl.store(out_ptr + offsets, out, mask=mask)
```

### 2.2 避免标量操作

```python
# ✗ 错误：标量操作，效率低
@triton.jit
def scalar_operations(x_ptr, out_ptr, n):
    for i in range(n):
        x = tl.load(x_ptr + i)  # 逐个加载
        out = tl.exp(x)          # 逐个计算
        tl.store(out_ptr + i, out)

# ✓ 正确：向量化操作
@triton.jit
def vectorized_operations(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask)  # 向量化加载
    out = tl.exp(x)                           # 向量化计算
    tl.store(out_ptr + offsets, out, mask=mask)
```

---

## 三、常见 Vector Unit 操作

### 3.1 激活函数

```python
# ReLU
out = tl.maximum(x, 0.0)

# Sigmoid
out = tl.sigmoid(x)
# 或
out = 1.0 / (1.0 + tl.exp(-x))

# Tanh
out = tl.tanh(x)
# 或
out = (tl.exp(x) - tl.exp(-x)) / (tl.exp(x) + tl.exp(-x))

# GELU
out = 0.5 * x * (1.0 + tl.erf(x / 1.41421356237))

# SiLU (Swish)
out = x * tl.sigmoid(x)

# Softplus
out = tl.log(1.0 + tl.exp(x))
```

### 3.2 数学运算

```python
# 指数和对数
out = tl.exp(x)
out = tl.log(x)
out = tl.log2(x)
out = tl.log10(x)

# 幂运算
out = tl.pow(x, 2.0)
out = tl.sqrt(x)
out = tl.rsqrt(x)  # 1/sqrt(x)

# 三角函数
out = tl.sin(x)
out = tl.cos(x)
out = tl.tan(x)

# 取整
out = tl.floor(x)
out = tl.ceil(x)
out = tl.round(x)
```

### 3.3 比较和选择

```python
# 最大最小值
out = tl.maximum(a, b)
out = tl.minimum(a, b)

# 条件选择
out = tl.where(condition, a, b)

# 示例：Clamp
out = tl.minimum(tl.maximum(x, min_val), max_val)
```

---

## 四、精度优化

### 4.1 数据类型选择

```python
@triton.jit
def precision_optimized(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # 加载时转换精度
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)  # 计算精度
    
    # 高精度计算
    out = tl.exp(x) + tl.sqrt(x)
    
    # 存储时转换精度
    tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=mask)
```

### 4.2 精度转换建议

| 场景 | 计算精度 | 存储精度 |
|------|---------|---------|
| 训练前向 | FP32 | BF16 |
| 训练反向 | FP32 | FP32 |
| 推理 | FP32/BF16 | BF16/FP16 |

---

## 五、算子融合

### 5.1 融合优势

```
不融合：
x → Exp → temp → Add → out
    (kernel 1)  (kernel 2)
    内存读写 2 次

融合后：
x → Exp + Add → out
      (kernel 1)
    内存读写 1 次
```

### 5.2 融合示例

```python
@triton.jit
def fused_exp_add_kernel(
    x_ptr, y_ptr, out_ptr, n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # 融合：exp(x) + y
    out = tl.exp(x) + y
    
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def fused_gelu_mul_kernel(
    x_ptr, y_ptr, out_ptr, n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # 融合：GELU(x) * y
    sqrt2 = 1.41421356237
    gelu = 0.5 * x * (1.0 + tl.erf(x / sqrt2))
    out = gelu * y
    
    tl.store(out_ptr + offsets, out, mask=mask)
```

---

## 六、性能调优清单

### 6.1 必须检查项

- [ ] 是否使用向量化操作
- [ ] 是否避免标量循环
- [ ] 内存访问是否连续

### 6.2 性能优化项

- [ ] 是否可以融合多个算子
- [ ] 精度选择是否合理
- [ ] BLOCK_SIZE 是否合适

---

## 七、常见问题

### Q1: Vector Unit 性能不如预期？

检查项：
1. 是否使用向量化操作
2. 是否存在标量循环
3. 内存访问是否连续

### Q2: 如何选择 BLOCK_SIZE？

推荐 512-2048，根据数据规模调整

### Q3: 哪些操作应该融合？

相邻的逐元素操作应该融合，如：
- Exp + Add
- ReLU + Dropout
- LayerNorm 相关操作
