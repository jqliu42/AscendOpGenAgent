# Ascend NPU 性能指标参考

本文档提供 Ascend NPU 的性能指标和规格，帮助进行性能分析和优化。

---

## 一、硬件架构概览

### 1.1 AI Core 架构

```
AI Core
├── Cube Unit（矩阵计算单元）
│   ├── 功能：矩阵乘法加速
│   ├── 触发：tl.dot
│   └── 性能：TOPS 级别
│
├── Vector Unit（向量计算单元）
│   ├── 功能：逐元素运算
│   ├── 触发：tl 逐元素 API
│   └── 性能：高吞吐
│
└── Scalar Unit（标量计算单元）
    ├── 功能：标量运算
    └── 性能：基础
```

### 1.2 内存层次

```
内存层次
├── HBM（高带宽内存）
│   ├── 容量：32-64 GB
│   ├── 带宽：~1.0-1.5 TB/s
│   └── 延迟：高
│
├── L1 Buffer（片上缓存）
│   ├── 容量：~1 MB
│   ├── 带宽：极高
│   └── 延迟：低
│
└── L0 Buffer（寄存器级）
    ├── 容量：~KB 级别
    └── 延迟：最低
```

---

## 二、各架构规格对比

### 2.1 AI Core 数量

| 架构 | AI Core 数量 | Cube Unit | Vector Unit |
|------|-------------|-----------|-------------|
| 910B1 | 32 | 基础版 | 基础版 |
| 910B2 | 30 | 中等版 | 中等版 |
| 910B2C | 30 | 中等版 | 中等版 |
| 910B3 | 32 | 增强版 | 增强版 |
| 910B4 | 40+ | 高级版 | 高级版 |

### 2.2 内存规格

| 架构 | HBM 容量 | HBM 带宽 | L1 Buffer |
|------|---------|---------|-----------|
| 910B1 | 32 GB | ~1.0 TB/s | ~1 MB |
| 910B2 | 64 GB | ~1.2 TB/s | ~1 MB |
| 910B2C | 64 GB | ~1.2 TB/s | ~1 MB |
| 910B3 | 64 GB | ~1.4 TB/s | ~1.5 MB |
| 910B4 | 64 GB | ~1.5 TB/s | ~2 MB |

### 2.3 计算性能

| 架构 | FP16 TOPS | BF16 TOPS | INT8 TOPS |
|------|-----------|-----------|-----------|
| 910B1 | 256 | 128 | 512 |
| 910B2 | 280 | 140 | 560 |
| 910B2C | 280 | 140 | 560 |
| 910B3 | 320 | 160 | 640 |
| 910B4 | 376+ | 188+ | 752+ |

---

## 三、性能指标解读

### 3.1 理论峰值计算

#### Cube Unit 峰值

```python
# 矩阵乘法理论峰值
# FP16: TOPS = 2 × M × N × K / time (秒)
# BF16: TOPS = 2 × M × N × K / time (秒)

# 示例：计算 MatMul 的理论峰值
M, N, K = 4096, 4096, 4096
flops = 2 * M * N * K  # 137B FLOPs
time_ms = 10  # 实际测量时间
tops = flops / (time_ms / 1000) / 1e12  # TOPS
```

#### 内存带宽峰值

```python
# 内存带宽利用率
# 实际带宽 = 数据传输量 / 时间
# 利用率 = 实际带宽 / 理论带宽

# 示例：计算内存带宽利用率
data_size_bytes = 3 * M * N * 4  # 3 个 FP32 矩阵
actual_bandwidth = data_size_bytes / (time_ms / 1000)  # GB/s
utilization = actual_bandwidth / theoretical_bandwidth  # 910B2: ~1.2 TB/s
```

### 3.2 性能效率指标

| 指标 | 计算方式 | 目标值 |
|------|---------|--------|
| AI Core 利用率 | 实际 TOPS / 理论 TOPS | > 70% |
| 内存带宽利用率 | 实际带宽 / 理论带宽 | > 70% |
| 计算强度 | FLOPs / Bytes | > 10 (计算密集) |

---

## 四、性能基准参考

### 4.1 MatMul 性能基准

| 架构 | 矩阵规模 | PyTorch (ms) | 优化 Triton (ms) | 加速比 |
|------|---------|-------------|-----------------|--------|
| 910B2 | 1024×1024 | 1.65 | 2.95 | 0.56x |
| 910B2 | 2048×2048 | 3.64 | 9.70 | 0.38x |
| 910B2 | 4096×4096 | 15.2 | 18.5 | 0.82x |

### 4.2 Elementwise 性能基准

| 架构 | 操作 | 数据规模 | PyTorch (ms) | 优化 Triton (ms) | 加速比 |
|------|------|---------|-------------|-----------------|--------|
| 910B2 | ReLU | 10M | 0.85 | 0.72 | 1.18x |
| 910B2 | GELU | 10M | 2.35 | 2.35 | 1.00x |
| 910B2 | Add | 10M | 0.68 | 0.77 | 0.88x |

### 4.3 Reduce 性能基准

| 架构 | 操作 | 数据规模 | PyTorch (ms) | 优化 Triton (ms) | 加速比 |
|------|------|---------|-------------|-----------------|--------|
| 910B2 | Sum | 10M | 0.52 | 0.48 | 1.08x |
| 910B2 | Softmax | 10K×10K | 1.72 | 1.85 | 0.93x |

---

## 五、性能诊断指标

### 5.1 加速比解读

| 加速比范围 | 状态 | 建议 |
|-----------|------|------|
| > 1.5x | 优秀 | 性能优于 PyTorch |
| 1.0x - 1.5x | 良好 | 达到或略超 PyTorch |
| 0.8x - 1.0x | 可接受 | 轻微性能差距 |
| 0.5x - 0.8x | 需优化 | 存在明显问题 |
| < 0.5x | 严重问题 | 需全面检查 |

### 5.2 延迟指标解读

| 指标 | 健康值 | 异常值 | 可能原因 |
|------|--------|--------|---------|
| P50 vs P99 差异 | < 20% | > 50% | 负载不均衡 |
| 平均延迟波动 | < 10% | > 30% | 系统干扰 |
| 首次运行延迟 | 略高 | 高很多 | 编译开销 |

### 5.3 内存指标解读

| 指标 | 健康值 | 异常值 | 可能原因 |
|------|--------|--------|---------|
| 峰值内存 | 预期范围内 | 超出预期 | 内存泄漏 |
| 内存带宽利用率 | > 70% | < 50% | 访问效率低 |

---

## 六、优化参数参考

### 6.1 BLOCK_SIZE 推荐

| 数据规模 | 推荐 BLOCK_SIZE | 说明 |
|---------|----------------|------|
| < 1KB | 64-128 | 小数据 |
| 1KB - 1MB | 128-512 | 中等数据 |
| > 1MB | 256-1024 | 大数据 |

### 6.2 Grid 配置推荐

| 架构 | AI Core 数量 | 推荐 Grid 大小 |
|------|-------------|---------------|
| 910B1 | 32 | 64-128 |
| 910B2 | 30 | 60-120 |
| 910B3 | 32 | 64-128 |
| 910B4 | 40+ | 80-160 |

### 6.3 精度选择建议

| 场景 | 推荐精度 | 说明 |
|------|---------|------|
| 训练前向 | BF16 | 平衡精度和速度 |
| 训练反向 | FP32 | 保持梯度精度 |
| 推理 | BF16/FP16 | 速度优先 |
| 累加运算 | FP32 | 避免精度损失 |

---

## 七、性能分析工具

### 7.1 Triton 内置分析

```python
# 使用 Triton profiler
import triton

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
    ],
    key=['n_elements'],
)
@triton.jit
def kernel(...):
    ...
```

### 7.2 PyTorch Profiler

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.NPU,
    ],
    record_shapes=True,
    profile_memory=True,
) as prof:
    output = model(input)

print(prof.key_averages().table(sort_by="npu_time_total", row_limit=10))
```

### 7.3 NPU 特定工具

```python
# 使用 torch_npu 的性能分析
import torch_npu

# 获取 NPU 性能计数器
torch.npu.synchronize()
start = torch.npu.Event(enable_timing=True)
end = torch.npu.Event(enable_timing=True)

start.record()
output = model(input)
end.record()

torch.npu.synchronize()
elapsed_time = start.elapsed_time(end)  # 毫秒
```

---

## 八、常见性能问题速查

### 8.1 性能低于预期

| 症状 | 可能原因 | 检查方法 |
|------|---------|---------|
| speedup < 0.5 | 未用硬件加速 | 检查是否使用 tl.dot |
| speedup 随数据增长 | 内存瓶颈 | 检查内存访问模式 |
| speedup 波动大 | 系统干扰 | 多次测试取平均 |

### 8.2 内存问题

| 症状 | 可能原因 | 解决方案 |
|------|---------|---------|
| OOM | 内存不足 | 减小 BLOCK_SIZE |
| 内存带宽低 | 非连续访问 | 优化数据布局 |
| 内存泄漏 | 未释放资源 | 检查资源管理 |

### 8.3 编译问题

| 症状 | 可能原因 | 解决方案 |
|------|---------|---------|
| 编译时间长 | kernel 复杂 | 简化 kernel |
| 编译失败 | API 不支持 | 检查 API 兼容性 |
| 运行时错误 | 参数错误 | 检查参数配置 |
