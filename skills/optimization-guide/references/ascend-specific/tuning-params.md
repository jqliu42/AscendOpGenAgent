# Ascend 调优参数建议

本文档提供针对不同架构和场景的调优参数建议。

---

## 一、BLOCK_SIZE 参数建议

### 1.1 按算子类型

| 算子类型 | 推荐 BLOCK_SIZE | 说明 |
|---------|----------------|------|
| MatMul | 128-256 | 需要是 16 的倍数 |
| Elementwise | 512-2048 | 较大块提高内存效率 |
| Reduce | 256-1024 | 根据数据规模调整 |
| Attention | 64-128 | Flash Attention 推荐 |

### 1.2 按数据规模

| 数据规模 | 推荐 BLOCK_SIZE |
|---------|----------------|
| < 10K | 64-128 |
| 10K - 100K | 128-256 |
| 100K - 1M | 256-512 |
| > 1M | 512-2048 |

### 1.3 按架构

| 架构 | 推荐 BLOCK_SIZE | 最大建议值 |
|------|----------------|-----------|
| 910B1 | 64-128 | 256 |
| 910B2 | 128-256 | 512 |
| 910B2C | 128-256 | 512 |
| 910B3 | 128-256 | 512 |
| 910B4 | 128-512 | 1024 |

---

## 二、MatMul 参数建议

### 2.1 标准 MatMul

| 矩阵规模 | BLOCK_M | BLOCK_N | BLOCK_K |
|---------|---------|---------|---------|
| 小 (< 512) | 64 | 64 | 32 |
| 中 (512-2048) | 128 | 128 | 64 |
| 大 (> 2048) | 128 | 128 | 128 |

### 2.2 Batch MatMul

| 场景 | BLOCK_M | BLOCK_N | BLOCK_K | Grid |
|------|---------|---------|---------|------|
| 小 Batch | 64 | 64 | 32 | (B, M/64, N/64) |
| 大 Batch | 32 | 32 | 32 | (B, M/32, N/32) |

### 2.3 架构特定

| 架构 | BLOCK_M | BLOCK_N | BLOCK_K |
|------|---------|---------|---------|
| 910B1 | 64 | 64 | 32 |
| 910B2 | 128 | 128 | 64 |
| 910B4 | 128 | 128 | 128 |

---

## 三、Elementwise 参数建议

### 3.1 单输入 Elementwise

| 数据规模 | BLOCK_SIZE | Grid |
|---------|-----------|------|
| < 100K | 512 | (n/512,) |
| 100K - 1M | 1024 | (n/1024,) |
| > 1M | 2048 | (n/2048,) |

### 3.2 双输入 Elementwise

| 数据规模 | BLOCK_SIZE | Grid |
|---------|-----------|------|
| < 100K | 512 | (n/512,) |
| 100K - 1M | 1024 | (n/1024,) |
| > 1M | 1024 | (n/1024,) |

---

## 四、Reduce 参数建议

### 4.1 行归约

| 列数 | BLOCK_SIZE | 说明 |
|------|-----------|------|
| < 1024 | 256 | 小数据 |
| 1024 - 4096 | 512 | 中等数据 |
| > 4096 | 1024 | 大数据 |

### 4.2 Softmax

| 列数 | BLOCK_SIZE | 说明 |
|------|-----------|------|
| < 1024 | 256 | 小数据 |
| 1024 - 4096 | 512 | 中等数据 |
| > 4096 | 1024 | 大数据 |

---

## 五、Attention 参数建议

### 5.1 Flash Attention

| 序列长度 | BLOCK_M | BLOCK_N | 说明 |
|---------|---------|---------|------|
| < 512 | 32 | 32 | 小序列 |
| 512 - 2048 | 64 | 64 | 中等序列 |
| > 2048 | 128 | 64 | 大序列 |

### 5.2 标准 Attention

| 序列长度 | BLOCK_SIZE | 说明 |
|---------|-----------|------|
| < 512 | 32 | 小序列 |
| 512 - 1024 | 64 | 中等序列 |
| > 1024 | 128 | 大序列 |

---

## 六、精度参数建议

### 6.1 按场景

| 场景 | 输入精度 | 累加精度 | 输出精度 |
|------|---------|---------|---------|
| 训练前向 | BF16 | FP32 | BF16 |
| 训练反向 | FP32 | FP32 | FP32 |
| 推理 | BF16/FP16 | FP32 | BF16/FP16 |
| 量化推理 | INT8 | INT32 | INT8 |

### 6.2 按算子类型

| 算子类型 | 推荐精度 |
|---------|---------|
| MatMul | BF16 输入，FP32 累加 |
| Elementwise | 与输入相同 |
| Reduce | FP32 累加 |
| Attention | BF16 输入，FP32 累加 |

---

## 七、Grid 配置建议

### 7.1 AI Core 数量

| 架构 | AI Core 数量 | 推荐 Grid 大小 |
|------|-------------|---------------|
| 910B1 | 32 | 64-128 |
| 910B2 | 30 | 60-120 |
| 910B3 | 32 | 64-128 |
| 910B4 | 40+ | 80-160 |

### 7.2 Grid 计算公式

```python
# 一维 Grid
grid = (triton.cdiv(n, BLOCK_SIZE),)

# 二维 Grid
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

# 三维 Grid（Batch）
grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

# 确保 Grid 大小 >= AI Core 数量 * 2
assert grid[0] * grid[1] >= AI_CORE_COUNT * 2
```

---

## 八、自动调优配置

### 8.1 MatMul Autotune

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_autotuned(...):
    ...
```

### 8.2 Elementwise Autotune

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
def elementwise_autotuned(...):
    ...
```

### 8.3 Attention Autotune

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}),
    ],
    key=['S', 'D'],
)
@triton.jit
def attention_autotuned(...):
    ...
```

---

## 九、性能调优清单

### 9.1 参数检查

- [ ] BLOCK_SIZE 是否是 2 的幂次
- [ ] BLOCK_SIZE 是否针对架构调优
- [ ] Grid 是否充分利用 AI Core
- [ ] 精度选择是否合理

### 9.2 性能检查

- [ ] 是否使用 autotune
- [ ] 内存带宽利用率是否 > 70%
- [ ] AI Core 利用率是否 > 70%

---

## 十、快速参考表

### 10.1 默认推荐配置

```python
# MatMul 默认配置
MATMUL_CONFIG = {
    'BLOCK_M': 128,
    'BLOCK_N': 128,
    'BLOCK_K': 64,
}

# Elementwise 默认配置
ELEMENTWISE_CONFIG = {
    'BLOCK_SIZE': 1024,
}

# Reduce 默认配置
REDUCE_CONFIG = {
    'BLOCK_SIZE': 512,
}

# Attention 默认配置
ATTENTION_CONFIG = {
    'BLOCK_M': 64,
    'BLOCK_N': 64,
}
```

### 10.2 架构特定配置

```python
# 910B1
CONFIG_910B1 = {
    'BLOCK_SIZE': 64,
    'GRID_MULTIPLIER': 2,
}

# 910B2
CONFIG_910B2 = {
    'BLOCK_SIZE': 128,
    'GRID_MULTIPLIER': 2,
}

# 910B4
CONFIG_910B4 = {
    'BLOCK_SIZE': 256,
    'GRID_MULTIPLIER': 2,
}
```
