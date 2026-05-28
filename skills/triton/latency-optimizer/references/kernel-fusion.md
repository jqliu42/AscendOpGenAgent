# Kernel Fusion 优化

## 概述

Kernel Fusion 是将多个串行执行的 Triton kernel 合并为一个 kernel 的优化技术。通过消除中间数据读写和 kernel 下发开销，可以显著提升性能。

| 优化手段 | 适用场景 | 核心思路 |
|---------|---------|---------|
| **Kernel Fusion** | 多阶段 kernel 串行执行 | 合并多个 kernel 到一个，减少中间 buffer 访问 |

---

## 问题描述

Agent 有时会将一个算子拆分成多个阶段实现，每个阶段对应一个独立的 Triton kernel。这些 kernel 串行执行，通过中间 buffer 传递数据。

```python
# 原始实现：两阶段 kernel

# Stage 1: 计算某种变换
kernel_transform[grid1](input_ptr, temp_buffer, ...)

# Stage 2: 计算另一种变换
kernel_scale[grid2](temp_buffer, output_ptr, ...)
```

**性能问题：**
1. **中间数据读写**：temp_buffer 需要先写回全局内存，再被下一个 kernel 读取
2. **Kernel 下发开销**：两次 kernel 启动有固定开销
3. **内存占用**：额外的 temp_buffer 消耗内存带宽

---

## 优化原理

将多个阶段的计算逻辑合并到一个 kernel 中，在单次 kernel 执行中完成所有变换。

```
原始流程：
input → [kernel1] → temp_buffer → [kernel2] → output
         ↓写入           ↓读写          ↓写入

优化后流程：
input → [fused_kernel] → output
           ↓一次性写入
```

### 收益分析

| 场景 | 原始延迟 | 优化后延迟 | 收益 |
|-----|---------|-----------|-----|
| 2-stage kernel | kernel1 + kernel2 + 中间访存 | fused_kernel | 减少 30-50% |
| 3-stage kernel | 3x kernel launch + 中间访存 | fused_kernel | 减少 50-70% |

---

## 优化方法

### 方法一：串行计算合并

将多个阶段的计算逻辑按顺序放在一个 kernel 的循环中。

```python
# 原始：两阶段
@triton.jit
def kernel_transform(input_ptr, temp_buffer, ...):
    # Stage 1: transform
    data = tl.load(input_ptr + offset)
    temp = tl.transform(data)
    tl.store(temp_buffer + offset, temp)

@triton.jit
def kernel_scale(temp_buffer, output_ptr, ...):
    # Stage 2: scale
    data = tl.load(temp_buffer + offset)
    result = data * scale
    tl.store(output_ptr, result)
```

```python
# 优化：合并为一个 kernel
@triton.jit
def kernel_fused(input_ptr, output_ptr, scale, ...):
    # 单次加载
    data = tl.load(input_ptr + offset)

    # Stage 1 + Stage 2 在一起
    temp = tl.transform(data)
    result = temp * scale

    # 单次存储
    tl.store(output_ptr, result)
```

### 方法二：流水线融合

当多个阶段处理不同数据块时，可以将它们融合成流水线结构。

```python
# 原始：按阶段处理
for block in blocks:
    temp[block] = kernel_stage1(data[block])

for block in blocks:
    result[block] = kernel_stage2(temp[block])
```

```python
# 优化：流水线融合
for block in blocks:
    temp_block = stage1(data[block])
    result[block] = stage2(temp_block)  # 立即处理，无中间存储
```

---

## 核间规约：融合的禁区

**⚠️ 重要警示**：如果合并 kernel 会引入核间规约（cross-core reduction），则不应融合。

### 什么是核间规约

核间规约是指规约操作（如 `tl.sum`、`tl.max`）的规约维度跨越多个 program_id，需要在不同 program 之间同步数据。

```python
# 核间规约示例
@triton.jit
def kernel_reduce(input_ptr, output_ptr, ...):
    pid = tl.program_id(0)  # 跨多核并行

    # 每个 program 计算局部结果
    local_sum = tl.sum(data, axis=某个维度)

    # 需要跨 program 同步才能得到最终结果
    # 在 Triton Ascend 上效率极低
```

### 如何判断是否会引入核间规约

| 判断维度 | 可融合 | 不可融合 |
|---------|-------|---------|
| **规约范围** | 单 program 内完成 | 需要跨 program 同步 |
| **Grid 配置** | 规约在 grid 内按 program 分区 | 规约需要所有 program 协作 |
| **数据依赖** | 各阶段无依赖 | 后续阶段依赖全局统计量 |

### 具体判断规则

1. **单 program 内规约** → 可融合
   ```python
   # 每个 program 独立处理一行，规约在单行内完成
   pid = tl.program_id(0)
   row_data = tl.load(input_ptr + pid * stride)
   row_sum = tl.sum(row_data)  # 单 program 内完成
   ```

2. **跨 program 规约** → 不可融合
   ```python
   # 所有 program 的数据需要汇总
   # 需要原子操作或显式跨 program 同步
   total_sum = tl.sum(all_data)  # 跨 program
   ```

3. **分阶段局部规约 + 全局汇总** → 谨慎融合
   ```python
   # 可以在单 kernel 内完成局部规约
   # 但全局汇总需要考虑是否值得
   ```

---

## 案例：LayerNorm Fusion

### 原始实现：2-pass

```python
# Pass 1: 计算 mean 和 variance
@triton.jit
def kernel_norm_stats(input_ptr, mean_ptr, var_ptr, N, ...):
    pid = tl.program_id(0)
    sum_val = 0.0
    sum_sq = 0.0

    for i in range(N):
        val = tl.load(input_ptr + pid * N + i)
        sum_val += val
        sum_sq += val * val

    mean = sum_val / N
    var = sum_sq / N - mean * mean

    tl.store(mean_ptr + pid, mean)
    tl.store(var_ptr + pid, var)

# Pass 2: 归一化
@triton.jit
def kernel_normalize(input_ptr, output_ptr, mean_ptr, var_ptr, N, ...):
    pid = tl.program_id(0)
    mean = tl.load(mean_ptr + pid)
    var = tl.load(var_ptr + pid)
    inv_std = 1.0 / tl.sqrt(var + eps)

    for i in range(N):
        val = tl.load(input_ptr + pid * N + i)
        output = (val - mean) * inv_std
        tl.store(output_ptr + pid * N + i, output)
```

### 优化实现：Fused Kernel

```python
@triton.jit
def kernel_layernorm_fused(input_ptr, output_ptr, N, eps: tl.float32, ...):
    pid = tl.program_id(0)

    # 单次遍历计算统计量
    sum_val = 0.0
    sum_sq = 0.0

    for i in range(N):
        val = tl.load(input_ptr + pid * N + i)
        sum_val += val
        sum_sq += val * val

    mean = sum_val / N
    var = sum_sq / N - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # 同一遍历进行归一化
    for i in range(N):
        val = tl.load(input_ptr + pid * N + i)
        output = (val - mean) * inv_std
        tl.store(output_ptr + pid * N + i, output)
```

**优化效果**：2-pass → 1-pass，减少了中间 buffer 访问和 kernel launch 开销

---

## 何时融合，何时分离

### 可融合的场景

| 场景 | 示例 | 原因 |
|-----|------|------|
| 线性流水线 | transform → scale → bias | 无依赖，可按序合并 |
| 元素级操作 | exp → sum → divide | 每元素独立，可向量化 |
| 独立块处理 | block1_kernel + block2_kernel | 无跨块依赖 |

### 不可融合的场景

| 场景 | 原因 | 替代方案 |
|-----|------|---------|
| 引入核间规约 | 核间同步效率低 | 保持分离 |
| 内存受限 | 融合后寄存器压力过大 | 保持分离 |
| 代码复杂度过高 | 难以维护 | 保持分离 |

### 决策流程

```
开始
  │
  ▼
是否存在多阶段 kernel？ ──否──→ 跳过，不适用
  │
  │是
  ▼
合并后是否会引入核间规约？ ──是──→ 跳过，不融合
  │
  │否
  ▼
融合后 kernel 是否还能保持高效？ ──否──→ 跳过
  │
  │是
  ▼
应用 Kernel Fusion 优化
  │
  ▼
结束
```

---

## 常见错误

### 错误 1：融合引入核间规约

```python
# ❌ 错误：融合后引入核间规约
@triton.jit
def kernel_fused_bad(input_ptr, output_ptr, ...):
    # 单 program 计算局部 sum
    local_sum = tl.sum(data, axis=0)

    # 需要跨 program 同步才能得到全局 sum
    # 这在 Triton Ascend 上效率极低
    global_sum = cross_program_reduce(local_sum)

# ✅ 正确：保持分离
kernel1[grid1](...)  # 计算局部统计量
kernel2[grid2](...)  # 全局汇总
```

### 错误 2：融合后寄存器溢出

```python
# ❌ 错误：融合过多阶段导致寄存器压力
@triton.jit
def kernel_overfused(input_ptr, output_ptr, ...):
    temp1 = stage1(data)
    temp2 = stage2(temp1)
    temp3 = stage3(temp2)
    temp4 = stage4(temp3)  # 太多中间变量
    ...

# ✅ 正确：分批融合或保持分离
```

### 错误 3：忽略数据依赖

```python
# ❌ 错误：忽略阶段间依赖
kernel1[grid](...)  # 计算全局 mean
kernel2[grid](...)  # 使用 mean 进行归一化

# 融合后：
# mean 需要跨 program 同步
# 导致核间规约，效率低下
```

---

## 总结

**Kernel Fusion 的核心原则：**

| 原则 | 说明 |
|-----|------|
| 减少中间访存 | 消除 temp_buffer 的读写 |
| 减少 launch 开销 | 单次 kernel 启动代替多次 |
| 避免核间规约 | 融合不能引入跨 program 同步 |
| 保持代码可维护 | 避免过度融合 |

**判断标准：**
- ✅ 多阶段 kernel 存在
- ✅ 阶段间无核间规约依赖
- ✅ 融合后 kernel 仍能高效执行

**不融合的情况：**
- ❌ 合并会引入核间规约
- ❌ 融合后寄存器压力导致性能下降
- ❌ 代码复杂度大幅增加