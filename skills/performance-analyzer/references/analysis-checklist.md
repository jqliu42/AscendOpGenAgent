# 性能分析检查清单

本文档提供系统化的性能分析检查清单，确保不遗漏任何可能的性能问题。

---

## 一、分析前准备

### 1.1 收集必要信息

- [ ] 获取完整的 kernel 代码
- [ ] 确认目标架构（arch）
- [ ] 获取性能测试结果（如有）
- [ ] 确认算子类型（matmul/elementwise/reduce/attention）

### 1.2 了解硬件规格

| 架构 | AI Core | HBM 带宽 | Cube 吞吐 |
|------|---------|---------|----------|
| 910B1 | 32 | ~1.0 TB/s | 基础 |
| 910B2 | 30 | ~1.2 TB/s | 中等 |
| 910B4 | 40+ | ~1.5 TB/s | 高 |

---

## 二、代码静态分析清单

### 2.1 内存访问分析

#### 访问模式检查
- [ ] **连续性检查**：`tl.load`/`tl.store` 的访问 stride 是否为 1
- [ ] **对齐检查**：访问地址是否 32/64 字节对齐
- [ ] **块访问检查**：是否使用 `tl.make_block_ptr` 进行 2D+ 访问
- [ ] **Mask 检查**：边界处理是否正确使用 mask

#### 内存效率检查
- [ ] **BLOCK_SIZE 检查**：是否在合理范围（通常 128-512）
- [ ] **重复加载检查**：同一数据是否多次加载
- [ ] **数据复用检查**：是否有机会复用已加载数据

**代码检查示例**：

```python
# 检查点1：stride 是否为 1
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # stride=1 ✓
offsets = pid * BLOCK_SIZE * 2 + tl.arange(0, BLOCK_SIZE)  # stride=2 ✗

# 检查点2：是否使用 mask
x = tl.load(x_ptr + offsets, mask=offsets < n)  # ✓
x = tl.load(x_ptr + offsets)  # ✗ 可能越界

# 检查点3：是否重复加载
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptr + ...)  # 每次循环都加载
    # 检查：a 是否可以在外层加载？
```

---

### 2.2 并行度分析

#### Grid 配置检查
- [ ] **Grid 大小**：是否 ≥ AI Core 数量 × 2
- [ ] **Grid 维度**：是否充分利用多维并行
- [ ] **BLOCK_SIZE 与 Grid 关系**：BLOCK_SIZE 是否导致 Grid 过小

#### 负载均衡检查
- [ ] **数据分配**：每个核处理的数据量是否相近
- [ ] **边界处理**：边界核是否处理明显更少数据
- [ ] **条件分支**：是否有导致负载不均衡的条件分支

**代码检查示例**：

```python
# 检查点1：Grid 大小
grid = (triton.cdiv(n, BLOCK_SIZE),)
# 计算：grid[0] >= AI_CORE_COUNT * 2 ?

# 检查点2：多维并行
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))  # 2D 并行 ✓
grid = (triton.cdiv(M * N, BLOCK_SIZE),)  # 1D 并行，可能不够 ✓

# 检查点3：BLOCK_SIZE 是否合理
BLOCK_SIZE = 4096  # 可能过大，导致 Grid 过小
BLOCK_SIZE = 128   # 较为合理
```

---

### 2.3 计算模式分析

#### 硬件加速检查
- [ ] **Cube Unit 检查**：矩阵运算是否使用 `tl.dot`
- [ ] **Vector Unit 检查**：逐元素运算是否使用 `tl` API
- [ ] **计算强度**：FLOPs/Bytes 比值是否合理

#### 计算效率检查
- [ ] **重复计算检查**：是否有重复计算可消除
- [ ] **循环效率检查**：循环是否有优化空间
- [ ] **精度转换检查**：是否有过多 `.to()` 操作

**代码检查示例**：

```python
# 检查点1：是否使用 tl.dot
# 矩阵乘法
c = tl.dot(a, b)  # ✓ 触发 Cube Unit
c = a @ b         # ✗ 可能不触发
for k in range(K):
    c += a[:, k:k+1] * b[k:k+1, :]  # ✗ 逐元素，不触发

# 检查点2：重复计算
for i in range(n):
    scale = tl.sqrt(2.0)  # ✗ 每次循环都计算常量
# 应该提取到循环外或使用 constexpr

# 检查点3：精度转换
x = x.to(tl.bfloat16).to(tl.float32).to(tl.bfloat16)  # ✗ 过多转换
```

---

### 2.4 参数配置分析

#### BLOCK_SIZE 检查
- [ ] **数值范围**：是否在 64-1024 范围内
- [ ] **2 的幂次**：是否为 2 的幂次（推荐）
- [ ] **架构适配**：是否针对目标架构调优

#### 其他参数检查
- [ ] **num_stages**：是否合理（Ascend 可能不支持）
- [ ] **数据类型**：是否选择合适的精度

**参数推荐值**：

| 架构 | 推荐 BLOCK_SIZE | 说明 |
|------|----------------|------|
| 910B1 | 64-128 | 较小块 |
| 910B2 | 128-256 | 中等块 |
| 910B4 | 128-512 | 可用更大块 |

---

## 三、性能数据分析清单

### 3.1 基础指标分析

#### 加速比分析
- [ ] **speedup < 0.3**：严重问题，需全面检查
- [ ] **speedup 0.3-0.6**：中等问题，针对性优化
- [ ] **speedup 0.6-1.0**：轻微问题，精细调优
- [ ] **speedup > 1.0**：达标，可考虑进一步优化

#### 延迟分析
- [ ] **P99 vs P50**：差距大则存在负载不均衡
- [ ] **平均延迟**：是否符合预期
- [ ] **延迟波动**：是否稳定

### 3.2 资源利用率分析

- [ ] **AI Core 利用率**：是否充分利用
- [ ] **内存带宽利用率**：是否达到理论值的 70%+
- [ ] **Cube Unit 利用率**：矩阵运算是否高效

---

## 四、瓶颈定位清单

### 4.1 按症状定位

| 症状 | 可能瓶颈 | 检查项 |
|------|---------|--------|
| speedup < 0.3 | 未用硬件加速 | 是否使用 tl.dot |
| speedup 随数据增长 | 内存瓶颈 | 内存访问模式 |
| P99 >> P50 | 负载不均衡 | Grid 配置 |
| 性能不随 BLOCK_SIZE 变化 | 并行度瓶颈 | Grid 大小 |

### 4.2 按算子类型定位

#### MatMul 类算子
- [ ] 是否使用 `tl.dot`
- [ ] BLOCK_M/N/K 是否合理
- [ ] 是否利用 Cube Unit

#### Elementwise 类算子
- [ ] 内存访问是否连续
- [ ] BLOCK_SIZE 是否合理
- [ ] 是否可向量化

#### Reduce 类算子
- [ ] Reduce 维度处理是否高效
- [ ] 是否有原子操作瓶颈
- [ ] 内存访问模式是否优化

#### Attention 类算子
- [ ] QK^T 计算是否高效
- [ ] Softmax 是否优化
- [ ] AV 计算是否高效

---

## 五、优化建议生成清单

### 5.1 建议优先级排序

按以下顺序生成建议：

1. **高优先级**：预期提升 > 50%，风险低
2. **中优先级**：预期提升 20-50%，风险低/中
3. **低优先级**：预期提升 < 20%，或风险高

### 5.2 建议内容要求

每条建议应包含：

- [ ] **问题描述**：清晰说明问题
- [ ] **优化方向**：具体的优化方向
- [ ] **预期提升**：量化的预期提升
- [ ] **风险等级**：低/中/高
- [ ] **实现提示**：代码层面的提示

### 5.3 风险评估

- [ ] **低风险**：参数调整、内存访问优化
- [ ] **中风险**：算法调整、使用 tl.dot
- [ ] **高风险**：重写 kernel、改变计算逻辑

---

## 六、分析报告模板

```json
{
  "analysis_summary": {
    "op_type": "matmul",
    "arch": "ascend910b2",
    "current_speedup": 0.35
  },
  
  "checklist_results": {
    "memory": {
      "contiguous_access": false,
      "block_size_reasonable": true,
      "duplicate_load": false
    },
    "parallelism": {
      "grid_sufficient": false,
      "load_balanced": true
    },
    "compute": {
      "uses_tl_dot": false,
      "redundant_compute": false
    }
  },
  
  "identified_bottlenecks": [
    {
      "type": "compute",
      "severity": "high",
      "description": "未使用 tl.dot，无法触发 Cube Unit"
    },
    {
      "type": "parallelism",
      "severity": "medium",
      "description": "Grid 大小不足，未充分利用 AI Core"
    }
  ],
  
  "optimization_suggestions": [
    {
      "priority": 1,
      "type": "compute",
      "description": "使用 tl.dot 替代逐元素乘法累加",
      "expected_improvement": "100-300%",
      "risk_level": "medium",
      "implementation_hint": "将 for 循环中的逐元素乘法改为 tl.dot(a_block, b_block)"
    }
  ],
  
  "risk_assessment": {
    "overall_risk": "medium",
    "notes": "改用 tl.dot 需要重构代码结构，但风险可控"
  }
}
```

---

## 七、快速检查脚本

```python
def quick_check(code: str, arch: str) -> dict:
    """快速检查代码中的常见问题"""
    
    issues = []
    
    # 检查1：是否使用 tl.dot
    if "matmul" in code.lower() or "dot" in code.lower():
        if "tl.dot" not in code:
            issues.append({
                "type": "compute",
                "severity": "high",
                "message": "矩阵运算未使用 tl.dot"
            })
    
    # 检查2：BLOCK_SIZE 范围
    import re
    block_sizes = re.findall(r'BLOCK_SIZE[:\s=]+(\d+)', code)
    for bs in block_sizes:
        bs_val = int(bs)
        if bs_val < 64 or bs_val > 1024:
            issues.append({
                "type": "config",
                "severity": "medium",
                "message": f"BLOCK_SIZE={bs_val} 可能不在最优范围"
            })
    
    # 检查3：是否使用 mask
    if "tl.load" in code and "mask=" not in code:
        issues.append({
            "type": "memory",
            "severity": "low",
            "message": "tl.load 未使用 mask，可能存在边界问题"
        })
    
    return {
        "arch": arch,
        "issues": issues,
        "issue_count": len(issues)
    }
```
