# Attention 类算子优化指南

Attention 是 Transformer 架构的核心算子，计算复杂度高，优化空间大。

---

## 一、优化策略概览

| 策略 | 适用场景 | 预期提升 | 风险等级 |
|------|---------|---------|---------|
| Flash Attention | 标准 Attention | 2-4x | 中 |
| 分块计算 | 大序列长度 | 50-200% | 中 |
| 内存优化 | 显存受限 | 30-50% | 低 |
| 精度优化 | FP32 计算 | 10-30% | 低 |

---

## 二、Attention 计算流程

```
标准 Attention:
Q [B, H, S, D] @ K.T [B, H, D, S] → Scores [B, H, S, S]
Scores / sqrt(D) → Scaled
Scaled → Softmax → Weights [B, H, S, S]
Weights @ V [B, H, S, D] → Output [B, H, S, D]

内存复杂度: O(S^2)
```

---

## 三、Flash Attention 优化

### 3.1 核心思想

```
Flash Attention:
1. 分块计算，避免存储完整的 S^2 矩阵
2. 在线 Softmax，增量更新
3. 内存复杂度从 O(S^2) 降到 O(S)

分块示意:
Q: [S, D] 分成 [S/BLOCK_M, BLOCK_M, D]
K: [S, D] 分成 [S/BLOCK_N, BLOCK_N, D]
V: [S, D] 分成 [S/BLOCK_N, BLOCK_N, D]

每次计算:
- 加载 Q_block [BLOCK_M, D]
- 加载 K_block [BLOCK_N, D]
- 计算 QK^T [BLOCK_M, BLOCK_N]
- 加载 V_block [BLOCK_N, D]
- 累加到 Output [BLOCK_M, D]
```

### 3.2 Flash Attention Kernel

```python
@triton.jit
def flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    B, H, S, D,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # 获取 batch 和 head 索引
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    # 计算 Q block 的起始位置
    q_start = pid_m * BLOCK_M
    q_offsets = q_start + tl.arange(0, BLOCK_M)
    
    # 加载 Q block [BLOCK_M, BLOCK_D]
    Q_block = tl.load(
        Q_ptr + pid_b * stride_qb + pid_h * stride_qh +
        q_offsets[:, None] * stride_qs + tl.arange(0, BLOCK_D)[None, :] * stride_qd
    )
    
    # 初始化累加器
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # log-sum-exp
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)  # max
    
    # 缩放因子
    scale = 1.0 / tl.sqrt(BLOCK_D * 1.0)
    
    # 遍历 K, V blocks
    for n_start in range(0, S, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        
        # 加载 K block [BLOCK_N, BLOCK_D]
        K_block = tl.load(
            K_ptr + pid_b * stride_kb + pid_h * stride_kh +
            n_offsets[:, None] * stride_ks + tl.arange(0, BLOCK_D)[None, :] * stride_kd
        )
        
        # 计算 QK^T [BLOCK_M, BLOCK_N]
        qk = tl.dot(Q_block, K_block.T) * scale
        
        # 在线 Softmax
        m_ij = tl.max(qk, axis=1)  # 当前 block 的 max
        m_new = tl.maximum(m_i, m_ij)  # 更新后的 max
        
        # 计算 exp(qk - m_new)
        p = tl.exp(qk - m_new[:, None])
        
        # 更新 l_i
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(p, axis=1)
        
        # 加载 V block [BLOCK_N, BLOCK_D]
        V_block = tl.load(
            V_ptr + pid_b * stride_vb + pid_h * stride_vh +
            n_offsets[:, None] * stride_vs + tl.arange(0, BLOCK_D)[None, :] * stride_vd
        )
        
        # 累加
        acc = acc * tl.exp(m_i - m_new)[:, None] + tl.dot(p.to(tl.float16), V_block)
        
        # 更新 m_i
        m_i = m_new
    
    # 归一化输出
    acc = acc / l_i[:, None]
    
    # 存储
    O_block = acc.to(tl.float16)
    tl.store(
        O_ptr + pid_b * stride_ob + pid_h * stride_oh +
        q_offsets[:, None] * stride_os + tl.arange(0, BLOCK_D)[None, :] * stride_od,
        O_block
    )


def flash_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    B, H, S, D = Q.shape
    
    O = torch.empty_like(Q)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D
    
    grid = (B, H, triton.cdiv(S, BLOCK_M))
    
    flash_attention_kernel[grid](
        Q, K, V, O,
        B, H, S, D,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        BLOCK_M, BLOCK_N, BLOCK_D,
    )
    
    return O
```

---

## 四、标准 Attention 优化

### 4.1 分块 MatMul Attention

```python
@triton.jit
def attention_qk_kernel(
    Q_ptr, K_ptr, Scores_ptr,
    B, H, S, D,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_sb, stride_sh, stride_ss, stride_sn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    pid_n = tl.program_id(3)
    
    # 加载 Q block [BLOCK_M, BLOCK_D]
    q_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    Q_block = tl.load(
        Q_ptr + pid_b * stride_qb + pid_h * stride_qh +
        q_offsets[:, None] * stride_qs + tl.arange(0, BLOCK_D)[None, :] * stride_qd
    )
    
    # 加载 K block [BLOCK_N, BLOCK_D]
    k_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    K_block = tl.load(
        K_ptr + pid_b * stride_kb + pid_h * stride_kh +
        k_offsets[:, None] * stride_ks + tl.arange(0, BLOCK_D)[None, :] * stride_kd
    )
    
    # QK^T
    scores = tl.dot(Q_block, K_block.T) / tl.sqrt(D * 1.0)
    
    # 存储
    tl.store(
        Scores_ptr + pid_b * stride_sb + pid_h * stride_sh +
        q_offsets[:, None] * stride_ss + k_offsets[None, :] * stride_sn,
        scores
    )


@triton.jit
def attention_av_kernel(
    Weights_ptr, V_ptr, O_ptr,
    B, H, S, D,
    stride_wb, stride_wh, stride_ws, stride_wn,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    # 初始化累加器
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # 遍历 N 维度
    for n_start in range(0, S, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        
        # 加载 Weights block [BLOCK_M, BLOCK_N]
        m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        W_block = tl.load(
            Weights_ptr + pid_b * stride_wb + pid_h * stride_wh +
            m_offsets[:, None] * stride_ws + n_offsets[None, :] * stride_wn
        )
        
        # 加载 V block [BLOCK_N, BLOCK_D]
        V_block = tl.load(
            V_ptr + pid_b * stride_vb + pid_h * stride_vh +
            n_offsets[:, None] * stride_vs + tl.arange(0, BLOCK_D)[None, :] * stride_vd
        )
        
        # 累加
        acc += tl.dot(W_block, V_block)
    
    # 存储
    tl.store(
        O_ptr + pid_b * stride_ob + pid_h * stride_oh +
        m_offsets[:, None] * stride_os + tl.arange(0, BLOCK_D)[None, :] * stride_od,
        acc
    )
```

---

## 五、参数调优

### 5.1 BLOCK_SIZE 选择

| 序列长度 | BLOCK_M | BLOCK_N | 说明 |
|---------|---------|---------|------|
| < 512 | 32 | 32 | 小块 |
| 512-2048 | 64 | 64 | 中等块 |
| > 2048 | 128 | 64 | 大块 |

### 5.2 架构特定建议

| 架构 | 推荐 BLOCK_SIZE | 说明 |
|------|----------------|------|
| 910B1 | 32-64 | 较小的块 |
| 910B2 | 64-128 | 中等块 |
| 910B4 | 64-128 | 可用更大的块 |

---

## 六、优化检查清单

- [ ] 是否使用 Flash Attention（推荐）
- [ ] 是否使用 tl.dot 触发 Cube Unit
- [ ] BLOCK_SIZE 是否合理
- [ ] 是否正确处理 Softmax 数值稳定性
- [ ] 内存访问是否连续

---

## 七、常见问题

### Q1: Attention 显存占用大怎么办？

使用 Flash Attention，避免存储完整的 S×S 矩阵

### Q2: 如何处理超长序列？

减小 BLOCK_M 和 BLOCK_N，增加分块数量

### Q3: 如何处理 Causal Attention？

在 QK 计算时添加 mask
