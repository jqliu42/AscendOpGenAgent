# Grid 并行数量限制 优化模式

## 概述

由于 NPU 比 GPU 核数少，应当限制 Triton 程序的 grid 并行数量不要超过物理核数，从而减少 kernel 下发的耗时。

## 优化原则

### 通用原则

- **Grid 并行数不应超过物理核数**：过多的 grid 并行会增加 kernel 下发的调度开销
- **选择合适的 grid 并行策略**：根据算子类型选择合适的并行度

### 按算子类型区分

| 算子类型 | Grid 并行数上限 |
|----------|-----------------|
| Vector 类算子 | Vector 单元总数 |
| Mix 类算子（Cube + Vector 混合） | Cube 单元总数 |

## 单元数量获取

可以通过以下程序获取 NPU 的 Cube 和 Vector 单元数量：

```python
from typing import Any, Dict, Tuple
import torch
import triton

device = torch.npu.current_device()
device_properties: Dict[str, Any] = (
    triton.runtime.driver.active.utils.get_device_properties(device)
)

num_aicore = device_properties.get("num_aicore", -1)
num_vectorcore = device_properties.get("num_vectorcore", -1)
```

其中：
- `num_aicore`：Cube 单元数量
- `num_vectorcore`：Vector 单元数量

## 性能收益

- 减少 kernel 下发的调度开销
- 避免过多的并行任务导致资源竞争
- 提升整体执行效率