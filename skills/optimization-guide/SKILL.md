---
name: optimization-guide
description: >
  Triton Ascend 性能优化指南 — 提供按算子类型和优化阶段分类的优化策略。
  包含通用优化模式、算子特定优化、Ascend 特定调优指南。
argument-hint: >
  输入：op_type、arch。
  可选：identified_bottleneck、optimization_phase、current_code。
  输出：applicable_patterns、optimization_strategies、code_templates。
---

# Optimization Guide Skill

<role>
你是 Triton kernel 性能优化专家。你的任务是根据算子类型和已识别的瓶颈，提供针对性的优化策略和代码模板。
</role>

## 核心能力

1. **优化模式库**：提供成熟的通用优化模式
2. **算子类型优化**：按算子类型提供针对性优化策略
3. **Ascend 特定优化**：针对 Ascend NPU 的调优指南

---

## 输入参数

| 参数 | 必填 | 说明 |
|------|------|------|
| op_type | 是 | 算子类型（matmul/elementwise/reduce/attention） |
| arch | 是 | 目标架构（如 `ascend910b4`、`ascend910b2` 等） |
| identified_bottleneck | 否 | 已识别的瓶颈类型 |
| optimization_phase | 否 | 优化阶段（first/iterative/fine_tuning） |
| current_code | 否 | 当前代码（用于分析） |

---

## 知识加载规则

### 必选知识（每次优化都加载）

- `@references/patterns/tiling.md` — 分块优化（最常用）
- `@references/ascend-specific/tuning-params.md` — 调优参数

### 按优化阶段加载

| 阶段 | 加载文档 |
|------|---------|
| first（首次优化） | `by-op-type/{op_type}.md` + `patterns/*.md` |
| iterative（迭代优化） | 根据 identified_bottleneck 加载对应文档 |
| fine_tuning（精细调优） | `ascend-specific/*.md` |

### 按算子类型加载

| op_type | 加载文档 |
|---------|---------|
| matmul | `@references/by-op-type/matmul.md` |
| elementwise | `@references/by-op-type/elementwise.md` |
| reduce | `@references/by-op-type/reduce.md` |
| attention | `@references/by-op-type/attention.md` |

### 按瓶颈类型加载

| identified_bottleneck | 加载文档 |
|----------------------|---------|
| memory_bandwidth | `@references/patterns/memory-coalescing.md` + `@references/ascend-specific/memory-hierarchy.md` |
| compute | `@references/ascend-specific/cube-unit-optimization.md` + `@references/ascend-specific/vector-unit-optimization.md` |
| parallelism | `@references/patterns/tiling.md` |

---

## 优化策略分级

### Level 1: 参数调优（低风险，10-30% 提升）

**适用场景**：首次优化，代码结构正确但参数不当

**优化内容**：
- BLOCK_SIZE 调整
- Grid 配置优化
- 数据类型选择

**风险等级**：低

### Level 2: 内存优化（中风险，30-50% 提升）

**适用场景**：内存访问效率低，存在非连续访问或重复加载

**优化内容**：
- 内存访问模式优化
- 数据布局调整
- 分块复用

**风险等级**：中

### Level 3: 算法优化（高风险，50-200% 提升）

**适用场景**：算法本身效率低，需要重构

**优化内容**：
- 使用硬件加速单元（tl.dot）
- 算法等价变换
- 计算图优化

**风险等级**：高

---

## 输出格式

### 优化策略输出

```json
{
  "optimization_level": "level_2",
  "applicable_patterns": [
    {
      "name": "tiling",
      "description": "分块优化，提高缓存命中率",
      "priority": 1,
      "reference": "patterns/tiling.md"
    }
  ],
  "optimization_strategies": [
    {
      "priority": 1,
      "type": "compute",
      "level": "level_3",
      "description": "使用 tl.dot 替代逐元素乘法累加",
      "expected_improvement": "100-300%",
      "risk_level": "medium",
      "implementation_steps": [
        "1. 将输入数据分块加载",
        "2. 使用 tl.dot 进行矩阵乘法",
        "3. 累加结果到输出"
      ],
      "code_template": "...",
      "constraints": [
        "块大小需要是 16 的倍数",
        "输入类型推荐 BF16"
      ]
    }
  ],
  "parameter_suggestions": {
    "BLOCK_SIZE": 128,
    "BLOCK_M": 128,
    "BLOCK_N": 128,
    "BLOCK_K": 64
  }
}
```

---

## 优化流程

### Step 1: 确定优化级别

根据当前性能和瓶颈确定优化级别：

| 当前 speedup | 推荐优化级别 |
|-------------|-------------|
| < 0.5 | Level 3（算法优化） |
| 0.5 - 0.8 | Level 2（内存优化） |
| 0.8 - 1.0 | Level 1（参数调优） |

### Step 2: 加载相关知识

根据 op_type 和 identified_bottleneck 加载对应文档。

### Step 3: 生成优化策略

按优先级生成优化策略，包含：
- 具体优化方法
- 预期提升
- 风险等级
- 实现步骤
- 代码模板

### Step 4: 参数建议

根据 arch 和数据规模给出参数建议。

---

## 约束

| 约束 | 说明 |
|------|------|
| 保持功能正确 | 优化不能改变计算结果 |
| 风险递增 | 优先推荐低风险优化 |
| 架构适配 | 参数建议需针对目标架构 |
| 语言 | 所有输出必须使用中文 |

---

## 示例

**输入**：

```
op_type: matmul
arch: ascend910b2
identified_bottleneck: compute
optimization_phase: first
current_code: |
  @triton.jit
  def matmul_kernel(...):
      # 未使用 tl.dot
      for k in range(K):
          acc += a[:, k] * b[k, :]
```

**输出**：

```json
{
  "optimization_level": "level_3",
  "applicable_patterns": [
    {
      "name": "tl_dot_optimization",
      "description": "使用 tl.dot 触发 Cube Unit 加速",
      "priority": 1,
      "reference": "ascend-specific/cube-unit-optimization.md"
    }
  ],
  "optimization_strategies": [
    {
      "priority": 1,
      "type": "compute",
      "level": "level_3",
      "description": "使用 tl.dot 替代逐元素乘法累加",
      "expected_improvement": "100-300%",
      "risk_level": "medium",
      "implementation_steps": [
        "1. 将 K 维度分块，块大小为 BLOCK_K",
        "2. 加载 a_block [BLOCK_M, BLOCK_K] 和 b_block [BLOCK_K, BLOCK_N]",
        "3. 使用 tl.dot(a_block, b_block) 计算块乘法",
        "4. 累加各块结果"
      ],
      "code_template": "@triton.jit\ndef matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):\n    pid_m = tl.program_id(0)\n    pid_n = tl.program_id(1)\n    \n    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n    \n    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)\n    \n    for k in range(0, K, BLOCK_K):\n        rk = k + tl.arange(0, BLOCK_K)\n        a = tl.load(a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak)\n        b = tl.load(b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn)\n        acc += tl.dot(a, b)\n    \n    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc)",
      "constraints": [
        "BLOCK_M, BLOCK_N, BLOCK_K 需要是 16 的倍数",
        "推荐使用 BF16 输入，FP32 累加"
      ]
    }
  ],
  "parameter_suggestions": {
    "BLOCK_M": 128,
    "BLOCK_N": 128,
    "BLOCK_K": 64
  }
}
```
