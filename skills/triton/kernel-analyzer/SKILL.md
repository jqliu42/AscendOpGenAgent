---
name: kernel-analyzer
description: >
  Analyzes Triton kernel performance bottlenecks and identifies optimization opportunities.
  Invoke when user asks to analyze kernel performance, identify bottlenecks, or generate optimization suggestions.
argument-hint: >
  输入：code-file-path（代码文件路径）、todo-optim-path（输出路径）、arch（硬件架构）、optimization-result（可选，本轮优化结果）。
  输出：todo_optim.txt（包含识别出的瓶颈和可优化点列表）。
---

# Kernel Analyzer

<role>
你是一个专注于分析 Triton Kernel 性能瓶颈的专家。
你的任务是对给定的 Triton Kernel 代码进行深入分析，识别出当前 kernel 的性能瓶颈及可优化点。
**你是 todo-optim.txt 文件的唯一管理者，负责创建和更新该文件。**
</role>

## 输入参数

| 参数 | 必填 | 说明 |
|------|------|------|
| code_file_path | 是 | 待分析的 Triton Kernel 代码文件路径 |
| todo_optim_path | 是 | todo-optim.txt 输出路径 |
| arch | 是 | 硬件架构 |
| optimization_result | 否 | 本轮优化结果，用于更新 todo-optim.txt |

### optimization_result 格式

```json
{
  "optimization_point": "序号: 维度名称",
  "status": "success" | "failed",
  "speedup": 1.25,
  "reason": "失败原因（仅 status 为 failed 时需要）"
}
```

**字段说明**：

| 字段 | 必填 | 说明 |
|------|------|------|
| optimization_point | 是 | 优化点标识，格式为"序号: 维度名称" |
| status | 是 | 优化结果状态，"success" 或 "failed" |
| speedup | 否 | 加速比（仅 status 为 success 时），如 1.25 表示性能提升 25% |
| reason | 否 | 失败原因（仅 status 为 failed 时需要） |

## 分析流程

### 首次分析（无 optimization_result）

```
1. 加载待分析的 Triton Kernel 代码文件
2. 【IR 提取】运行 IR 提取脚本，编译 Triton Kernel 并提取编译器 IR
3. 【IR 分析】分析提取的 IR 文件，识别 IR 层面的优化机会
4. 【代码分析】按照分析维度逐一检查代码，识别所有可优化点
5. 按优先级从高到低排序优化点
6. 创建 todo-optim.txt 并写入所有优化点
7. 【清理】删除 IR 输出目录（ir_output/）
```

### 更新分析（有 optimization_result）

```
1. 读取现有的 todo-optim.txt
2. 根据 optimization_result 处理已执行的优化点：
   - status == "success" → 移除该优化点（已完成）
   - status == "failed" → 移除该优化点（尝试失败，不再重试）
3. 加载最新的代码文件
4. 【IR 提取】重新运行 IR 提取脚本
5. 【IR 分析】重新分析 IR 文件
6. 【代码分析】重新分析代码，识别新的优化机会
7. 按优先级从高到低排序剩余和新识别的优化点
8. 更新 todo-optim.txt
9. 【清理】删除 IR 输出目录（ir_output/）
```

## IR 提取与分析

### IR 提取步骤

使用 skill 目录下的 `scripts/extract_ir.py` 脚本提取 Triton 编译器 IR：

```bash
python <skill_dir>/scripts/extract_ir.py <code_file_path> --output-dir ./ir_output
```

该脚本会：
1. 设置 `TRITON_DEBUG=1` 和 `TRITON_ALWAYS_COMPILE=1` 触发编译
2. 运行 Triton 脚本，生成 dump 目录
3. 使用 `bishengir-compile --mlir-print-ir-after-all` 提取最后一个 pass 的 IR
4. 将 IR 文件保存到 `./ir_output/<kernel_name>_last_pass.mlir`

### IR 分析要点

分析提取的 IR 文件时，关注以下内容：

#### 1. 内存访问模式
- 检查 `hivm.intr.hivm.LOAD` / `hivm.intr.hivm.STORE` 指令
- 分析是否使用了向量化加载（`hivm.intr.hivm.LOAD.GM_TO_UB`）
- 检查是否存在离散访存模式

#### 2. 计算流水线
- 检查 Cube 指令（`hivm.intr.hivm.MAD` 等）用于矩阵运算
- 检查 Vector 指令（`hivm.intr.hivm.VADD` 等）用于元素级运算
- 分析 AIC（Cube）和 AIV（Vector）流水线的利用率

#### 3. 同步与屏障
- 检查 `hivm.intr.hivm.BARRIER` 指令数量
- 过多的同步可能影响并行度

#### 4. 循环展开与向量化
- 检查 IR 中的循环结构是否被正确展开
- 分析向量运算的宽度

#### 5. 内存层级使用
- 检查 L0A/L0B/L0C（Cube 专用内存）的使用
- 检查 UB（Unified Buffer，Vector 专用内存）的使用
- 分析 GM（Global Memory）与片上内存之间的数据传输

### IR 分析输出

IR 分析结果应作为优化点的一部分写入 todo-optim.txt，格式如下：

```markdown
### 可优化点 N：[IR 层面问题]

| 字段 | 内容 |
|------|------|
| **序号** | N |
| **问题描述** | [IR 中的具体问题，如：发现大量 BARRIER 指令] |
| **代码位置** | IR: <kernel_name>_last_pass.mlir |
| **预期收益** | [性能提升预估] |
| **优化建议** | [基于 IR 分析的具体优化方案] |
```

## 分析维度

### 维度 1：入参静态化检查

**检查内容**：代码中是否存在可声明为 `tl.constexpr` 但未声明的固定参数

**典型问题特征**：
```python
@triton.jit
def kernel(A, B, C, M, N,
            stride_am, stride_an,  # 运行时不变化的固定值，但未声明为 tl.constexpr
            BLOCK_SIZE_M: tl.constexpr,
            BLOCK_SIZE_K: tl.constexpr):
```

**判断逻辑**：
- 如果代码中存在运行时不变化的固定参数（如 stride、固定数值、BLOCK_SIZE等）未声明为 `tl.constexpr` → 标记为可优化点
- 如果所有固定参数都已正确声明为 `tl.constexpr` → 此维度无问题

---

### 维度 2：Tiling 策略检查

**检查内容**：检查是否存在 Tiling 不当导致的非连续内存访问

**典型问题特征**：
```python
@triton.jit
def kernel(input_ptr, output_ptr, dim1, dim2, ...):
    # 特征 1：向量化偏移 tl.arange 作用在非连续轴（如 dim1/M 轴）
    m_offsets = tl.arange(0, BLOCK_SIZE_M)
    # 特征 2：访存偏移计算中，向量化部分乘上了较大的 stride
    input_offset = m_offsets * stride_m + n_idx * stride_n
    # 特征 3：循环内部频繁进行还原操作（如 tl.sum）将向量压缩为标量
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    ...
    total_sum = tl.sum(acc, axis=0)
```

**判断逻辑**：
- 如果 `tl.load` 的偏移量计算中，`tl.arange` 产生的向量偏移量作用于 `stride > 1` 的轴，而存在 `stride = 1` 的轴仅被当作标量索引处理 → 标记为可优化点
- 如果 `tl.arange` 已经作用于内存最连续的轴（通常是最后一张量的最后一维），且实现了合并访存 → 此维度无问题

---

### 维度 3：BLOCK_SIZE 配置检查

**检查内容**：检查 BLOCK_SIZE 参数是否经过充分调优

**典型问题特征**：
```python
@triton.jit
def kernel(A, C, M, N,
            BLOCK_M: tl.constexpr = 128,  # BLOCK_SIZE 可能需要调优
            BLOCK_N: tl.constexpr = 128):
```

**判断逻辑**：
- 如果代码中存在 BLOCK_SIZE 参数（BLOCK_M、BLOCK_N、BLOCK_K 等）且未进行系统性调优 → 标记为可优化点
- 如果 BLOCK_SIZE 已经过充分调优（如通过 benchmark 确定了最优值）→ 此维度无问题

---

### 维度 4：向量化检查（Scalar 转 Vector）

**检查内容**：检查是否存在 scalar 操作可以转换为 vector 操作

**典型问题特征**：
```python
# 问题1：标量广播
scalar_val = 0.5
result = x * scalar_val  # scalar 广播，无法启用 vector 加速

# 问题2：标量规约
sum_val = 0.0
for n in range(N):
    val = tl.load(x_ptr + row_offset + n)
    sum_val += val  # 标量加法，循环依赖

# 问题3：int类型比较
is_invalid_tok = tok < 0  # i64/i32类型，退化为标量

# 问题4：int类型除法/取余
c = a // b  # i32标量除法
d = a % b   # i32标量取余
```

**判断逻辑**：
- 如果存在标量广播操作未使用 `tl.full` 转为 vector → 标记为可优化点
- 如果存在标量规约循环未使用 vector 分块规约 → 标记为可优化点
- 如果存在 int 类型比较未转为 float32 → 标记为可优化点
- 如果存在 int 类型除法/取余未优化 → 标记为可优化点

---

### 维度 5：循环不变式外提检查

**检查内容**：检查循环内部是否存在可以外提的计算

**典型问题特征**：
```python
# 问题：循环内重复加载相同的值
for outer_idx in range(outer_size):
    for inner_idx in range(inner_size):
        param_idx = outer_idx  # 只依赖外层变量
        val = tl.load(param_ptr + param_idx)  # 重复加载相同值
        ...

# 问题：索引通过整除映射
for block in range(num_blocks):
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    param_idx = offsets // SPATIAL_SIZE  # 映射到更粗粒度的索引
    val = tl.load(param_ptr + base + param_idx)  # 同一 param_idx 的多个元素重复加载
```

**判断逻辑**：
- 如果循环内存在索引不依赖内层变量的 `tl.load` → 标记为可优化点
- 如果存在内外层循环次数比例大且有重复加载 → 标记为可优化点

---

### 维度 6：维度合并检查

**检查内容**：检查是否存在可以合并的多层嵌套循环

**典型问题特征**：
```python
# 问题：3层循环
for n in range(N):
    for h in range(H):
        for w_start in range(0, W, BLOCK_SIZE):
            base_offset = n * stride_n + c * stride_c + h * stride_h
            data = tl.load(input_ptr + base_offset + ...)
```

**判断逻辑**：
- 如果存在多层嵌套循环处理连续维度（如 H×W）未合并 → 标记为可优化点
- 合并后可以减少外层循环次数、减少重复计算、提高内存连续性

---

### 维度 7：Pass 合并检查

**检查内容**：检查是否存在可以合并的多次遍历

**典型问题特征**：
```python
# 问题：多次遍历
for ...:  # Pass 1
    data = tl.load(...)
    mean += tl.sum(data)

for ...:  # Pass 2 - 再次遍历
    data = tl.load(...)
    var += tl.sum((data - mean) ** 2)

for ...:  # Pass 3 - 第三次遍历
    data = tl.load(...)
    tl.store(...)
```

**判断逻辑**：
- 如果存在多个统计量计算（mean + variance）需要多次遍历数据 → 标记为可优化点
- 可以利用数学公式同时计算多个统计量，减少遍历次数

---

### 维度 8：Load 指令重排序检查

**检查内容**：检查是否存在可以重排序以提高并行度的 load 指令

**典型问题特征**：
```python
# 问题：load B 在前，阻塞了 load A
for i in range(HEAD_NUM):
    p_B_index = B_index + i
    idx_B = tl.load(p_B_index)  # 在前，会阻塞
    p_B = B + idx_B
    b_B = tl.load(p_B)
    b_A = tl.load(p_A)  # 必须等 load B 完成
```

**判断逻辑**：
- 如果存在因依赖关系导致串行执行的 load 指令 → 标记为可优化点
- 调整 load 顺序可让无依赖的 load 与其他指令并行执行

---

### 维度 9：分核策略检查

**检查内容**：检查 Grid 大小是否与物理核数匹配

**典型问题特征**：
```python
# 问题1：Grid 远超物理核数
grid = (batch_size,)  # batch_size=128，远超 48 核

# 问题2：Grid 远小于物理核数
grid = (batch_size // 64,)  # 只有 2 核

# 问题3：Tile 过小
BLOCK_SIZE = 64  # UB 利用率低
```

**判断逻辑**：
- 如果 Grid 大小远大于或远小于物理核数（40-48 核）→ 标记为可优化点
- 如果 BLOCK_SIZE 过小（小于 1024）导致 UB 利用率低 → 标记为可优化点
- 如果 BLOCK_SIZE 过大导致 UB 溢出风险 → 标记为可优化点

---

### 维度 10：离散访存检查

**检查内容**：检查是否存在非连续或不可预测的索引访存

**典型问题特征**：
```python
# 问题：随机索引导致离散访存
offset = tl.load(offset_ptr)  # 随机标量
idx = tl.load(idx_ptr + rn * stride_idx)  # 随机向量
val = tl.load(x_ptr + offset + idx * stride_x)  # 直接从GM离散访问
```

**判断逻辑**：
- 如果存在完全无法预测的随机索引导致的离散访存 → 标记为可优化点
- 可考虑先整块读取到 UB，再使用 `tl.gather` 收集

---

### 维度 11：Libdevice 函数检查

**检查内容**：检查是否重复实现了已有的 libdevice 函数

**典型问题特征**：
```python
# 问题：重复造轮子
@triton.jit
def round_int8(x):
    return (x + 0.5).to(tl.int8)  # 逻辑不完整

# 问题：手写激活函数
out = tl.maximum(x, 0.0)  # 手写 relu
```

**判断逻辑**：
- 如果存在 round、trunc、pow 等数学函数的手写实现 → 标记为可优化点
- 如果存在激活函数（如 relu）的手写实现 → 标记为可优化点
- 应优先使用 `tl.extra.cann.libdevice` 中已有的优化函数

---

### 维度 12：自动调优（Autotune）检查

**检查内容**：检查是否使用了 `@triton.autotune` 装饰器

**典型问题特征**：
```python
# 问题：硬编码的 BLOCK_SIZE
@triton.jit
def kernel(A, C, M, N, BLOCK_SIZE: tl.constexpr = 128):
    ...

# 优化：使用 autotune 自动搜索最优配置
@triton.autotune(configs=[...], key=['M', 'N'])
@triton.jit
def kernel(A, C, M, N, BLOCK_SIZE: tl.constexpr):
    ...
```

**判断逻辑**：
- 如果存在可调整的参数（如 BLOCK_SIZE）但未使用 autotune → 标记为可优化点
- 如果已经使用 autotune 或参数已通过其他方式充分调优 → 此维度无问题

---

### 维度 13：IR 层面分析

**检查内容**：分析编译器生成的 IR，识别底层优化机会

**分析要点**：

1. **内存访问指令分析**
   - 检查 `hivm.intr.hivm.LOAD` / `STORE` 指令类型
   - 识别是否使用了高效的向量化加载（如 `LOAD.GM_TO_UB`）
   - 分析访存模式是否连续

2. **同步指令分析**
   - 统计 `hivm.intr.hivm.BARRIER` 指令数量
   - 过多的同步会降低并行度

3. **计算指令分析**
   - Cube 指令（`MAD` 等）：用于矩阵乘法
   - Vector 指令（`VADD`、`VMUL` 等）：用于元素级运算
   - 分析流水线利用率

4. **循环结构分析**
   - 检查循环是否被正确展开
   - 分析向量化宽度

**判断逻辑**：
- 如果 IR 中存在大量 BARRIER 指令 → 标记为可优化点，考虑减少同步
- 如果 IR 中存在标量化的访存指令 → 标记为可优化点，考虑向量化
- 如果 IR 中 Cube/Vector 流水线利用率低 → 标记为可优化点
- 如果 IR 中循环未正确展开 → 标记为可优化点

---

## 输出格式

**输出文件**：`todo_optim.txt`

**文件格式要求**：
```markdown
# Triton Kernel 性能分析报告
# 分析文件：<代码文件路径>
# 分析时间：<时间戳>

## 分析摘要

| 序号 | 维度 |
|------|------|
| 1 | 入参静态化 |
| 2 | Tiling策略 |
| ... | ... |

## 瓶颈及可优化点列表

### 可优化点 1：[优化维度名称]

| 字段 | 内容 |
|------|------|
| **序号** | 1 |
| **问题描述** | [具体问题说明] |
| **代码位置** | [文件名:行号] 或 IR: <kernel_name>_last_pass.mlir |
| **预期收益** | [性能提升预估] |
| **优化建议** | [具体的优化方案] |

---

### 可优化点 2：[优化维度名称]

| 字段 | 内容 |
|------|------|
| **序号** | 2 |
| **问题描述** | [具体问题说明] |
| **代码位置** | [文件名:行号] 或 IR: <kernel_name>_last_pass.mlir |
| **预期收益** | [性能提升预估] |
| **优化建议** | [具体的优化方案] |

---
...
```

**字段说明**：

| 字段 | 必填 | 说明 |
|------|------|------|
| 序号 | 是 | 优化点的唯一标识，从1开始递增，与摘要表格中的序号对应 |
| 问题描述 | 是 | 清晰描述当前代码存在的问题 |
| 代码位置 | 是 | 指明问题代码所在位置，格式：`文件名:行号` 或 `IR: <kernel_name>_last_pass.mlir` |
| 预期收益 | 是 | 预估优化后可获得的性能提升（如：减少20%访存、提升1.5x吞吐等） |
| 优化建议 | 是 | 具体可执行的优化方案 |

**摘要表格字段说明**：

| 字段 | 说明 |
|------|------|
| 序号 | 优化点的编号，从1开始，与详细描述中的序号对应 |
| 维度 | 分析维度名称 |

**注意**：todo-optim.txt 只保留未完成的优化点。已完成或失败的优化点会被移除。

## 重要约束

- ⚠️ **必须对所有 13 个维度逐一进行分析，不得遗漏**
- ⚠️ **每个发现的优化点都必须写入 todo-optim.txt**
- ⚠️ **优化建议必须具体、可执行**
- ⚠️ **只能使用本 skill 规定的优化方式进行识别，不要使用任何超出本 skill 之外的优化方式**
- ⚠️ **优化点必须按优先级从高到低排序，优先级高的优化点排在前面**
- ⚠️ **todo-optim.txt 只保留未完成的优化点，已完成或失败的优化点必须移除**
- ⚠️ **IR 分析完成后，必须删除 IR 输出目录（ir_output/）**
- 如果某个维度没有发现问题，仍需在报告中注明"该维度无明显问题"
