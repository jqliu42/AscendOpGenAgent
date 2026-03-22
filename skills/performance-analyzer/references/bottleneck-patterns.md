# 性能瓶颈模式库

本文档汇集 Triton kernel 常见的性能瓶颈模式，帮助快速识别和定位性能问题。

---

## 一、瓶颈分类总览

```
性能瓶颈
├── 内存瓶颈
│   ├── 内存带宽不足
│   ├── 非连续内存访问
│   ├── 内存不对齐
│   └── 重复数据加载
│
├── 计算瓶颈
│   ├── 未利用硬件加速单元
│   ├── 重复计算
│   ├── 低效算法
│   └── 精度转换开销
│
├── 并行度瓶颈
│   ├── Grid 配置不足
│   ├── 负载不均衡
│   └── 核间同步开销
│
└── 其他瓶颈
    ├── 启动开销
    ├── 数据传输
    └── 编译优化不足
```

---

## 二、内存瓶颈详解

### 2.1 内存带宽不足

**识别特征**：
- 性能随数据规模线性增长
- 计算强度低（FLOPs/Bytes < 10）
- 性能不随 BLOCK_SIZE 变化明显
- AI Core 利用率低，但内存带宽利用率高

**常见原因**：

| 原因 | 检查方法 | 优化方向 |
|------|---------|---------|
| 算子本身计算强度低 | 分析 FLOPs/Bytes | 考虑算子融合 |
| 过小的内存事务 | 检查 BLOCK_SIZE | 增大 BLOCK_SIZE |
| 内存访问碎片化 | 检查访问模式 | 合并访问 |

**优化策略**：

```python
# 问题：过小的内存事务
@triton.jit
def kernel_small_block(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # BLOCK_SIZE=32 太小，内存效率低
    ...

# 优化：增大 BLOCK_SIZE
@triton.jit
def kernel_large_block(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # BLOCK_SIZE=128 或更大
    ...
```

**Ascend 特定**：
- HBM 理论带宽：~1.2 TB/s
- 目标：带宽利用率 > 70%

---

### 2.2 非连续内存访问

**识别特征**：
- 访问 stride 不为 1
- 性能明显低于连续访问版本
- 内存带宽利用率低

**常见场景**：

| 场景 | 原因 | 解决方案 |
|------|------|---------|
| 矩阵转置 | 按列访问行存储数据 | 使用共享内存转置 |
| 高维张量切片 | 维度顺序不匹配 | 调整访问顺序或重排 |
| Stride 访问 | 步长不为 1 | 使用块访问 |

**优化策略**：

```python
# 问题：非连续访问
@triton.jit
def kernel_non_contiguous(x_ptr, out_ptr, M, N, stride_m, stride_n):
    pid = tl.program_id(0)
    # 按列访问，但数据是行存储
    for m in range(M):
        x = tl.load(x_ptr + m * stride_m + pid * stride_n)
        ...

# 优化方案1：使用块指针
@triton.jit
def kernel_block_ptr(x_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 使用 make_block_ptr 进行 2D 块访问
    x_block = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, N),
        strides=(stride_m, stride_n),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)  # 根据存储顺序调整
    )
    x = tl.load(x_block)
    ...

# 优化方案2：预处理转置
# 在 CPU/GPU 端先转置数据，使访问变为连续
```

**Ascend 特定**：
- 非连续访问可能导致 Cube Unit 无法使用
- 建议在 kernel 外部预处理数据布局

---

### 2.3 重复数据加载

**识别特征**：
- 同一数据在循环中多次加载
- 内存访问量远大于数据量
- 性能受内存带宽限制

**优化策略**：

```python
# 问题：重复加载
@triton.jit
def kernel_reload(a_ptr, b_ptr, c_ptr, M, N, K):
    for m in range(M):
        for n in range(N):
            acc = 0
            for k in range(K):
                # 每次循环都重新加载 a[m, k]
                a = tl.load(a_ptr + m * K + k)
                b = tl.load(b_ptr + k * N + n)
                acc += a * b
            tl.store(c_ptr + m * N + n, acc)

# 优化：分块加载，复用数据
@triton.jit
def kernel_tiled(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # 加载一块数据到 "共享内存"（通过寄存器）
    for k in range(0, K, BLOCK_K):
        a_block = tl.load(a_ptr + ...)  # 加载一次
        b_block = tl.load(b_ptr + ...)  # 加载一次
        
        # 在块内复用
        acc += tl.dot(a_block, b_block)
```

---

## 三、计算瓶颈详解

### 3.1 未利用硬件加速单元

**识别特征**：
- 矩阵运算未使用 `tl.dot`
- 性能远低于理论峰值
- AI Core 利用率低

**Ascend 硬件加速单元**：

| 单元 | 用途 | 触发条件 |
|------|------|---------|
| Cube Unit | 矩阵乘法 | 使用 `tl.dot` |
| Vector Unit | 逐元素运算 | 使用 `tl` 逐元素 API |
| Scalar Unit | 标量运算 | 普通标量操作 |

**优化策略**：

```python
# 问题：未使用 Cube Unit
@triton.jit
def matmul_slow(a_ptr, b_ptr, c_ptr, M, N, K):
    for m in range(M):
        for n in range(N):
            acc = 0
            for k in range(K):
                acc += a[m, k] * b[k, n]  # 逐元素乘法，不触发 Cube
            c[m, n] = acc

# 优化：使用 tl.dot 触发 Cube Unit
@triton.jit
def matmul_fast(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # 加载块
    a_block = tl.load(...)  # [BLOCK_M, BLOCK_K]
    b_block = tl.load(...)  # [BLOCK_K, BLOCK_N]
    
    # 使用 tl.dot 触发 Cube Unit
    c_block = tl.dot(a_block, b_block)  # [BLOCK_M, BLOCK_N]
    
    tl.store(..., c_block)
```

**关键要点**：
- `tl.dot` 是触发 Cube Unit 的唯一方式
- 块大小需要是 16 的倍数（Cube Unit 要求）
- 输入类型推荐 BF16，累加类型推荐 FP32

---

### 3.2 重复计算

**识别特征**：
- 相同计算在多处重复
- 循环内有不变的计算
- 计算量远大于理论需求

**优化策略**：

```python
# 问题：重复计算
@triton.jit
def kernel_recompute(x_ptr, out_ptr, n):
    for i in range(n):
        # 每次循环都计算相同的值
        scale = tl.sqrt(2.0)  # 常量，应该预计算
        x = tl.load(x_ptr + i)
        out = x / scale
        tl.store(out_ptr + i, out)

# 优化：预计算或提取常量
@triton.jit
def kernel_optimized(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # 常量在编译时计算
    INV_SQRT_2 = 1.0 / tl.sqrt(2.0)  # tl.constexpr
    
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask)
    out = x * INV_SQRT_2  # 乘法比除法快
    tl.store(out_ptr + offsets, out, mask=mask)
```

---

### 3.3 精度转换开销

**识别特征**：
- 频繁的数据类型转换
- `.to()` 操作过多
- 性能受转换开销影响

**优化策略**：

```python
# 问题：频繁精度转换
@triton.jit
def kernel_cast(x_ptr, out_ptr, n):
    x = tl.load(x_ptr + ...)  # FP32
    x = x.to(tl.bfloat16)     # 转换1
    y = compute1(x)
    y = y.to(tl.float32)      # 转换2
    z = compute2(y)
    z = z.to(tl.bfloat16)     # 转换3
    tl.store(out_ptr + ..., z)

# 优化：统一精度，减少转换
@triton.jit
def kernel_unified_precision(x_ptr, out_ptr, n):
    # 保持统一精度
    x = tl.load(x_ptr + ...).to(tl.bfloat16)  # 一次转换
    y = compute1(x)  # 保持 BF16
    z = compute2(y)  # 保持 BF16
    tl.store(out_ptr + ..., z)  # 直接存储
```

**Ascend 精度建议**：
- 计算精度：BF16（平衡精度和速度）
- 累加精度：FP32（避免精度损失）
- 存储精度：根据需求选择

---

## 四、并行度瓶颈详解

### 4.1 Grid 配置不足

**识别特征**：
- Grid 大小小于 AI Core 数量
- 部分 AI Core 空闲
- 性能随 Grid 增加而提升

**Ascend AI Core 数量**：

| 架构 | AI Core 数量 | 推荐 Grid 大小 |
|------|-------------|---------------|
| 910B1 | 32 | 64-128 |
| 910B2 | 30 | 60-120 |
| 910B4 | 40+ | 80-160 |

**优化策略**：

```python
# 问题：Grid 过小
def launch_kernel(x, y, n):
    BLOCK_SIZE = 4096  # 过大
    grid = (triton.cdiv(n, BLOCK_SIZE),)  # 可能只有几个核
    kernel[grid](x, y, n, BLOCK_SIZE=BLOCK_SIZE)

# 优化：减小 BLOCK_SIZE，增加 Grid
def launch_kernel_optimized(x, y, n):
    BLOCK_SIZE = 256  # 适中
    grid = (triton.cdiv(n, BLOCK_SIZE),)  # 更多核并行
    kernel[grid](x, y, n, BLOCK_SIZE=BLOCK_SIZE)
```

**Grid 配置原则**：
- Grid 大小 ≥ AI Core 数量 × 2
- 单个核的工作量适中（不要太大也不要太小）
- 考虑负载均衡

---

### 4.2 负载不均衡

**识别特征**：
- P99 延迟远大于 P50
- 部分核执行时间明显长于其他
- 整体性能受最慢核限制

**常见场景**：

| 场景 | 原因 | 解决方案 |
|------|------|---------|
| 变长数据处理 | 不同核处理不同长度 | 动态调度 |
| 边界处理 | 边界核处理更少数据 | 忽略（影响小） |
| 条件分支 | 不同核执行不同路径 | 重构算法 |

**优化策略**：

```python
# 问题：负载不均衡（变长数据）
@triton.jit
def kernel_variable_length(data_ptr, lengths_ptr, out_ptr, n):
    pid = tl.program_id(0)
    length = tl.load(lengths_ptr + pid)  # 每个核处理不同长度
    
    for i in range(length):  # 负载不均衡
        ...

# 优化：使用动态调度（work queue）
# 或重新设计数据布局使负载均衡
```

---

## 五、瓶颈诊断流程

### 快速诊断决策树

```
开始诊断
    │
    ├─ speedup < 0.3？
    │   └─ 是 → 严重问题：检查是否使用硬件加速单元
    │
    ├─ speedup 0.3-0.6？
    │   └─ 是 → 中等问题：检查并行度和内存访问
    │
    └─ speedup 0.6-1.0？
        └─ 是 → 轻微问题：检查参数调优

并行度检查：
    │
    ├─ Grid < AI Core 数量？
    │   └─ 是 → 增大 Grid（减小 BLOCK_SIZE）
    │
    └─ Grid 足够 → 检查内存访问

内存访问检查：
    │
    ├─ stride != 1？
    │   └─ 是 → 非连续访问，优化数据布局
    │
    ├─ 重复加载？
    │   └─ 是 → 分块复用
    │
    └─ 访问正常 → 检查计算

计算检查：
    │
    ├─ 矩阵运算未用 tl.dot？
    │   └─ 是 → 改用 tl.dot
    │
    ├─ 重复计算？
    │   └─ 是 → 预计算/缓存
    │
    └─ 计算正常 → 参数调优
```

### 诊断检查清单

**内存检查**：
- [ ] 访问是否连续（stride 检查）
- [ ] BLOCK_SIZE 是否合理（128-512）
- [ ] 是否有重复加载
- [ ] 内存对齐是否正确

**并行度检查**：
- [ ] Grid 是否充分利用 AI Core
- [ ] 是否存在负载不均衡
- [ ] 核间是否有同步等待

**计算检查**：
- [ ] 矩阵运算是否使用 tl.dot
- [ ] 是否有重复计算
- [ ] 精度转换是否过多

**Ascend 特定检查**：
- [ ] 是否针对目标架构调优
- [ ] Cube/Vector Unit 是否充分利用
- [ ] 参数是否符合硬件要求

---

## 六、瓶颈与优化策略映射

| 瓶颈类型 | 优先优化策略 | 预期提升 | 风险等级 |
|---------|-------------|---------|---------|
| 未使用 Cube Unit | 改用 tl.dot | 100-500% | 中 |
| Grid 过小 | 减小 BLOCK_SIZE | 30-100% | 低 |
| 非连续访问 | 优化数据布局 | 30-100% | 中 |
| BLOCK_SIZE 不当 | 参数调优 | 10-50% | 低 |
| 重复加载 | 分块复用 | 20-50% | 低 |
| 重复计算 | 预计算 | 10-30% | 低 |
| 精度转换过多 | 统一精度 | 5-20% | 低 |
