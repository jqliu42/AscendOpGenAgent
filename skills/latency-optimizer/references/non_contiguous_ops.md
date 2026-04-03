# 高性能非连续维度还原算子生成准则

本准则旨在指导如何实现在 `(Batch, M, N)` 布局下，针对非连续维度 `M`（Stride > 1）进行高效还原（如 Mean, Sum, Argmax）的 Triton 算子。

## 1. 核心优化准则：访存合并 (Memory Coalescing)

**【原则】禁止在还原维度 `M` 上进行矢量化（`tl.arange`）。必须在物理内存最连续的维度 `N` 上进行矢量化。**

* **错误做法**：在一个 Program 中处理一个 `(b, n)` 坐标，然后在 `M` 维度上用 `arange` 加载数据。这会导致地址跳跃，产生离散访存。
* **正确做法**：在一个 Program 中处理 `BLOCK_N` 个连续的 `N` 索引。在 `M` 维度的循环中，每次 `tl.load` 读取一整行中连续的 `BLOCK_N` 个元素。这确保了显存带宽被合并利用。

## 2. 算子架构设计规范

### A. 并行策略 (Grid & Task Partition)
* **连续块划分 (Contiguous Block Partition)**：避免交织划分，每个任务处理连续的数据块，确保任务内的数据访存连续。
* **Grid 维度**：将 Grid 固定为 `(num_cores,)`，每个核心处理一个连续的数据分片。
* **任务映射**：将数据按 N 维度连续划分给各核心，每个核心独立处理其分片内的所有 M 维度数据。

### B. 寄存器累加 (In-Register Accumulation)
* 在循环外初始化一个长度为 `BLOCK_N` 的向量累加器。
* 在还原维度的 `for` 循环中，直接对向量进行累加，避免 Block 内部昂贵的规约指令。

---

## 3. 标准实现模板

以下是以 `Mean Reduction` 为例的标准高性能模板：

```python
import triton
import triton.language as tl

@triton.jit
def reduction_kernel(
    input_ptr, output_ptr,
    B, M, N,
    stride_b, stride_m, stride_n,
    num_cores: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 获取当前物理核心 ID
    pid = tl.program_id(0)

    # 计算 N 维度分块数和每个核心处理的起始块
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    blocks_per_core = tl.cdiv(num_n_blocks, num_cores)
    start_n_block = pid * blocks_per_core
    end_n_block = tl.minimum(start_n_block + blocks_per_core, num_n_blocks)

    # 连续处理 N 维度的分片（避免交织）
    for n_block_idx in range(start_n_block, end_n_block):
        # 1. 产生连续维度 N 的偏移量 (关键：实现访存合并)
        offsets_n = n_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offsets_n < N

        # 2. 按 Batch 维度顺序处理
        for b_idx in range(0, B):
            # 3. 初始化向量累加器 (长度为 BLOCK_N)
            acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

            # 4. 定位到当前 Batch 和 N 分片的起始指针
            # 注意：offsets_n * stride_n 保证了在连续维度上步进
            curr_input_ptr = input_ptr + b_idx * stride_b + offsets_n * stride_n

            # 5. 沿还原维度 M 进行标量循环
            for m_idx in range(0, M):
                # 这里的加载指令会一次性读取内存中连续的物理块
                vals = tl.load(curr_input_ptr + m_idx * stride_m, mask=mask_n, other=0.0)
                acc += vals

            # 6. 计算均值并写回结果
            mean_val = acc / M
            output_offset = b_idx * N + offsets_n
            tl.store(output_ptr + output_offset, mean_val, mask=mask_n)
```

## 4. 性能调优 Checklist

1. **确定 BLOCK_N 大小**：通常设置为 1024 或 2048。对于 dim2=4095 这种规模，建议 BLOCK_N 设为 4096 以便一次性处理整行。
2. **掩码保护 (Masking)**：由于 BLOCK_N 通常为 2 的幂，而实际维度（如 4095）可能不对齐，必须使用 mask_n 确保访存安全。
3. **避免物理转置**：本方案通过"逻辑转置读取"避免了物理转置带来的额外 $3\times$ 显存带宽消耗。
4. **硬件适配**：在 NPU 上，确保 num_cores 与硬件实际的 Vector Core 数量一致，以触发最优调度。

## 5. 使用说明

在生成任何还原类算子（Mean, Sum, Argmax, Min, Max）时，若还原维度不是内存中最连续的维度，应严格遵守此模板结构。