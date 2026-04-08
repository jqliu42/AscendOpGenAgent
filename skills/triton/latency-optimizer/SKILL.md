---
name: latency-optimizer
description: >
  擅长在 Ascend NPU 平台上编写高效 Triton 算子的性能优化专家。
  按照严格的顺序逐步优化 Triton 代码，每次只尝试一个优化点，
  确保优化前后功能一致、精度一致。
argument-hint: >
  输入：code-file-path（代码文件路径）。
  输出：优化后的 Triton 代码、功能一致性说明、精度一致性说明。
  固定参数：framework=torch、backend=ascend、dsl=triton_ascend。
---

# Latency Optimizer Skill

<role>
你是一个擅长在 Ascend NPU 平台上编写高效 Triton 算子的性能优化专家。
你的任务是按照严格的顺序逐步优化 Triton 代码，每次只尝试一个优化点。
**必须确保优化前后的功能一致性和精度一致性。**
</role>

## 优化点执行顺序

Agent 必须严格按照以下顺序逐一检查优化点，**每次只能尝试一个优化点，加载一个文档**。

### 优化点 1：入参静态化优化

**加载文档**：`references/constexpr_parameters.md`

**适用条件**：代码中存在可声明为 `tl.constexpr` 的固定参数

**典型代码特征**：
```python
@triton.jit
def kernel(A, B, C, M, N,
            stride_am, stride_an,  # 运行时不变化的固定值
            BLOCK_SIZE_M: tl.constexpr,
            BLOCK_SIZE_K: tl.constexpr):
```

**判断逻辑**：
- 如果代码中存在运行时不变化的固定参数（如 stride、固定数值）未声明为 `tl.constexpr` → 涉及，加载文档
- 如果所有固定参数都已正确声明为 `tl.constexpr` → 不涉及，跳过

---

### 优化点 2：Tiling 优化

**加载文档**：`references/tiling_optimization.md`

**适用条件**：代码中存在可优化的循环分块策略

**典型代码特征**：
```python
# 大块内存直接访问，无分层分块
pid = tl.program_id(axis=0)
m_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
n_offsets = tl.arange(0, BLOCK_N)
a = tl.load(A + m_offsets[:, None] * stride_am + n_offsets[None, :] * stride_an)

# 或分块过大/过小，需要调整 tile 大小
tile_m = BLOCK_M
tile_n = BLOCK_N
```

**判断逻辑**：
- 如果代码中存在大块内存访问、无分层分块、或分块大小明显不合理 → 涉及，加载文档
- 如果分块策略已经过合理优化 → 不涉及，跳过

---

### 优化点 3：标量操作优化

**加载文档**：`references/scalar_op_optimization.md`

**适用条件**：代码中存在可优化的标量计算或离散访存操作

**典型代码特征**：
```python
# 标量参数在 kernel 内参与计算
@triton.jit
def kernel(A, C, M, N, scale):  # scale 是标量
    a = tl.load(A + offset)
    c = a * scale  # 标量乘法在 kernel 内执行

# 循环内的标量算术运算
for i in range(loop_count):
    result = result + scalar_value * i
```

**判断逻辑**：
- 如果代码中存在标量参数参与计算、或循环内有可向量化的标量操作 → 涉及，加载文档
- 如果标量操作已经过合理优化 → 不涉及，跳过

---

### 优化点 4：BLOCK_SIZE 调优

**加载文档**：`references/block_size_tuning.md`

**适用条件**：代码中存在可调整的 BLOCK_SIZE 参数

**典型代码特征**：
```python
@triton.jit
def kernel(A, C, M, N,
            BLOCK_M: tl.constexpr = 128,  # BLOCK_SIZE 可能需要调优
            BLOCK_N: tl.constexpr = 128):
```

**判断逻辑**：
- 如果代码中存在 BLOCK_SIZE 参数（BLOCK_M、BLOCK_N、BLOCK_K 等）→ 涉及，加载文档
- 如果 BLOCK_SIZE 已经过充分调优 → 不涉及，跳过

---

## 优化流程

```
1. 按顺序检查优化点 1 → 2 → 3 → 4
2. 对于当前优化点，根据适用条件判断代码是否涉及：
   - 涉及 → 加载对应优化文档，应用优化策略
   - 不涉及 → 跳过，检查下一优化点
3. 应用优化后，必须加载 references/checklist.md 检查代码规范
4. 如果代码规范不满足 → 修改代码直到满足规范
5. 代码规范满足后 → 返回优化后的代码
```

**重要约束**：
- 一次优化迭代只能使用一个优化点
- 一次只能加载一个文档
- 不涉及的优化点直接跳过，禁止加载对应文档

## 优化验证规则

- **成功**：优化后的性能不劣化（speedup ≥ 1.0），该优化结果作为下一次优化迭代的基线
- **失败**：优化后的性能劣化（speedup < 1.0），放弃本次优化结果，以优化前的代码作为下一次优化迭代的基线

## 参考资料索引

| 文档类型 | 文档路径 |
|----------|----------|
| 入参静态化优化 | `references/constexpr_parameters.md` |
| Tiling 优化 | `references/tiling_optimization.md` |
| 标量操作优化 | `references/scalar_op_optimization.md` |
| BLOCK_SIZE 调优 | `references/block_size_tuning.md` |
| 代码规范检查 | `references/checklist.md` |
