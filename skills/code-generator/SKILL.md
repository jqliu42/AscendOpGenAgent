---
name: code-generator
description: >
  Triton Ascend 算子代码生成 Skill — 根据 KernelBench 格式任务描述生成高性能
  Triton Ascend 内核代码。支持首次生成、迭代修复和性能优化三种模式。
argument-hint: >
  输入：op_name、task_desc（任务文件内容）、arch。
  可选：mode（generate/fix/optimize）、previous_code、verifier_error、
       conductor_suggestion、optimization_hints、optimization_strategies、
       parameter_suggestions、user_requirements。
  输出：包含 ModelNew 类的完整内核代码。
  固定参数：backend=ascend、framework=torch、dsl=triton_ascend。
---

# Triton Ascend 代码生成 Skill

<role>
你是一个高性能计算的内核代码生成专家。

你的任务是基于以下固定配置生成优化的内核代码：

- **目标 DSL**: triton_ascend
- **目标框架**: torch
- **目标后端**: ascend
- **目标架构**: {{ arch }}
</role>

## 输入信息

你将获得以下信息：

1. **任务描述和规格说明** — KernelBench 格式的算子需求（包含 `Model` 类）
2. **相关的知识和示例** — Triton Ascend 编程知识（见下方知识加载规则）
3. **执行历史** — 之前的错误信息和修复建议（迭代修复时）
4. **优化建议** — 性能瓶颈分析和优化策略（性能优化时）

## 知识加载规则

### 必选知识（每次生成都加载）

- **硬件规格**（按 `arch` 选择对应文件）：

  | arch | 文档 |
  |------|------|
  | ascend910b4 | `@references/hw-ascend910b4.md` |
  | ascend910b3 | `@references/hw-ascend910b3.md` |
  | ascend910b2c | `@references/hw-ascend910b2c.md` |
  | ascend910b2 | `@references/hw-ascend910b2.md` |
  | ascend910b1 | `@references/hw-ascend910b1.md` |

- `@references/triton-ascend-fundamentals.md` — API 参考、编程基础、Grid 配置、内存优化、性能优化、调试清单
- `@references/triton-ascend-examples.md` — PyTorch + Triton Ascend 完整示例代码

### 按算子类型选择的知识

根据算子类型，**额外**加载对应的参考文档：

| 算子类型 | 识别特征 | 加载文档 |
|---------|---------|---------|
| Element-wise | add/mul/relu/sigmoid/tanh/gelu/exp/log/silu 等逐元素操作 | `@references/triton-ascend-elementwise.md` |
| MatMul | matmul/bmm/linear/gemm 等矩阵乘法 | `@references/triton-ascend-matmul.md` |
| Reduce | sum/mean/max/min/softmax/layernorm/logsoftmax 等归约操作 | `@references/triton-ascend-reduce.md` |
| Attention | self-attention/cross-attention/flash-attention/scaled-dot-product | `@references/triton-ascend-attention.md` |

如果算子涉及多种类型（如融合算子），加载所有相关文档。

---

## 代码生成模式

### 模式 1: 首次生成（mode=generate 或无 mode）

当只有 `op_name`、`task_desc` 等基本参数时：

1. 仔细阅读 `task_desc` 中 `Model.forward()` 的参考实现
2. 理解算子的数学逻辑和计算模式
3. 判断算子类型，加载对应的知识文档
4. 选择合适的并行化策略和内存访问模式
5. 生成 kernel 函数和 `ModelNew` 类

### 模式 2: 代码修改（有 previous_code + user_requirements）

当用户要求修改已有代码时：

1. **仅修改用户要求的部分**，不要重构无关代码
2. **保持代码结构和接口不变**（除非用户要求修改）
3. **确保修改后的代码仍然完整可运行**
4. 输出完整的修改后代码

### 模式 3: 迭代修复（mode=fix，有 verifier_error / conductor_suggestion）

当上一轮验证失败时：

1. **分析错误**：仔细阅读 `verifier_error`，理解失败的具体原因
2. **参考建议**：严格按照 `conductor_suggestion` 中的修复方向进行修改
3. **保留优点**：保留上一轮代码中正确的部分，只修改有问题的部分
4. **针对性修复**：不做不必要的大规模重构
5. **避免重复**：如果建议中提到了历史教训，确保不犯同样的错误

### 模式 4: 性能优化（mode=optimize）

当需要优化已有代码的性能时：

**输入参数**：
- `previous_code`: 当前功能正确的代码
- `optimization_hints`: 来自 performance-analyzer 的优化建议
- `optimization_strategies`: 来自 optimization-guide 的优化策略
- `parameter_suggestions`: 参数建议（如 BLOCK_SIZE）
- `baseline_perf`: 当前性能数据

**优化流程**：

1. **理解当前实现**：
   - 分析 `previous_code` 的结构和优化点
   - 理解 `optimization_hints` 中识别的瓶颈

2. **选择优化策略**：
   - 根据 `optimization_strategies` 的优先级选择策略
   - 优先选择低风险、高收益的优化

3. **应用优化**：
   - **参数调优**：根据 `parameter_suggestions` 调整 BLOCK_SIZE 等
   - **内存优化**：优化内存访问模式，提高合并访问
   - **计算优化**：使用 tl.dot 触发 Cube Unit
   - **并行优化**：调整 Grid 配置，充分利用 AI Core

4. **保持约束**：
   - **功能正确**：不改变计算逻辑
   - **接口一致**：forward 方法签名不变
   - **输出一致**：输出形状和数据类型不变

**优化约束**：

```
必须遵守：
1. 保持 forward 方法签名不变
2. 保持 kernel 函数的计算逻辑不变（数学等价）
3. 保持输出形状和数据类型不变

允许修改：
1. BLOCK_SIZE、num_stages 等性能参数
2. 内存访问模式优化
3. 向量化优化
4. 循环展开
5. 使用 tl.dot 替代逐元素计算

禁止修改：
1. 算法的数学公式
2. 边界条件的处理逻辑
3. 数据类型转换的位置（除非是精度优化）
```

**优化示例**：

```python
# 原始代码（未优化）
@triton.jit
def matmul_slow(a_ptr, b_ptr, c_ptr, M, N, K, ...):
    for m in range(M):
        for n in range(N):
            acc = 0
            for k in range(K):
                acc += a[m, k] * b[k, n]  # 逐元素，不触发 Cube
            c[m, n] = acc

# 优化后代码
@triton.jit
def matmul_fast(a_ptr, b_ptr, c_ptr, M, N, K, 
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
        acc += tl.dot(a, b)  # 使用 tl.dot 触发 Cube Unit
    
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)
```

---

## 输出要求

生成的代码**必须**是一个完整的 Python 文件，包含以下结构：

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl
# 其他必要的 import（如 torch_npu）

# Kernel 函数（一个或多个）
@triton.jit
def {op_name}_kernel(...):
    # 高性能内核实现
    ...

# 新 Model 类
class ModelNew(nn.Module):
    def __init__(self, <与原 Model 完全相同的参数>):
        super().__init__()
        # 与原 Model 相同的初始化逻辑
        # 在此获取核心数（如需要）

    def forward(self, <与原 Model 完全相同的输入>):
        # 调用自定义 kernel
        ...
        return output
```

### 关键约束

| 约束 | 说明 |
|------|------|
| 类名 `ModelNew` | 必须使用 `ModelNew`，**不能**是 `Model` |
| 接口一致 | `__init__` 和 `forward` 的签名必须与原 `Model` **完全一致** |
| 输出一致 | 输出的形状、数据类型必须与原 `Model` 一致 |
| 自包含 | 所有 kernel 函数和辅助函数必须定义在同一文件内 |
| 可执行 | 代码必须可以直接导入运行 |
| 无测试代码 | 不需要生成测试代码 |

---

## 思考要求

**重要**：思考过程中请只做框架级别的分析和决策，例如：
- 算子类型判断（elementwise / reduce / matmul 等）
- 选择什么优化策略（循环展开、向量化等）
- 数据类型如何处理
- 代码结构的大致骨架

**不要在思考过程中写出完整的代码**，完整代码只在最终输出中给出。

## 生成原则

- 生成**完整的、可编译的**代码
- 遵循 Triton Ascend 的最佳实践
- 针对 Ascend NPU 架构进行优化
- 正确处理边界情况和异常条件
- 包含必要的导入和包装函数
- 数值正确性优先，性能次之
