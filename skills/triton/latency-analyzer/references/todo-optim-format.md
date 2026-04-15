# 性能优化待办事项清单

## 格式说明

此文件由主agent和维护，不允许优化执行subagent直接修改。

---

## 全局状态

```
[current_baseline_dir]: ./opt_iter_0
[current_best_dir]: ./opt_iter_0
[current_best_speedup]: 1.0
[loop_count]: 0
[optimization_active]: true
```

---

## 优化点列表

```
[optimization_points]

1. name: 入参静态化优化
   index: 1
   description: 将stride等固定参数声明为tl.constexpr
   status: pending  # pending | in_progress | completed | failed | skipped
   attempt_count: 0
   max_attempts: 2
   last_attempt_dir: ""
   result: ""  # success | failed | ""

2. name: Tiling优化
   index: 2
   description: 连续轴向量化，优化访存模式
   status: pending
   attempt_count: 0
   max_attempts: 2
   last_attempt_dir: ""
   result: ""

3. name: 分核优化
   index: 3
   description: 优化Grid配置，充分利用NPU硬件资源
   status: pending
   attempt_count: 0
   max_attempts: 2
   last_attempt_dir: ""
   result: ""

4. name: 离散访存优化
   index: 4
   description: 优化随机/不可预测索引访问
   status: pending
   attempt_count: 0
   max_attempts: 2
   last_attempt_dir: ""
   result: ""

5. name: Scalar转Vector优化
   index: 5
   description: 标量操作转换为向量操作
   status: pending
   attempt_count: 0
   max_attempts: 2
   last_attempt_dir: ""
   result: ""

6. name: Pass合并优化
   index: 6
   description: 合并多次遍历相同数据的操作
   status: pending
   attempt_count: 0
   max_attempts: 2
   last_attempt_dir: ""
   result: ""

7. name: 维度合并优化
   index: 7
   description: 合并多层嵌套循环
   status: pending
   attempt_count: 0
   max_attempts: 2
   last_attempt_dir: ""
   result: ""

8. name: Libdevice函数使用
   index: 8
   description: 使用libdevice优化版本替代手动实现
   status: pending
   attempt_count: 0
   max_attempts: 2
   last_attempt_dir: ""
   result: ""

9. name: 循环不变量外提
   index: 9
   description: 将内层循环中只依赖外层变量的load外提
   status: pending
   attempt_count: 0
   max_attempts: 2
   last_attempt_dir: ""
   result: ""

10. name: Load指令重排序
    index: 10
    description: 重排load指令以减少阻塞
    status: pending
    attempt_count: 0
    max_attempts: 2
    last_attempt_dir: ""
    result: ""

11. name: BLOCK_SIZE调优
    index: 11
    description: 系统性调优BLOCK_SIZE参数
    status: pending
    attempt_count: 0
    max_attempts: 2
    last_attempt_dir: ""
    result: ""

12. name: Autotune自动调优
    index: 12
    description: 使用autotune进行参数自动搜索
    status: pending
    attempt_count: 0
    max_attempts: 2
    last_attempt_dir: ""
    result: ""
```

---

## 状态更新规则

### 优化点状态流转

```
pending → in_progress: 主agent选择此优化点发送给优化执行subagent
in_progress → completed: 优化执行subagent返回成功（speedup ≥ 1.0）
in_progress → failed: 优化执行subagent返回失败（精度验证失败或speedup < 1.0）
failed → pending: 达到重试上限前可重试，或由主agent决定跳过
completed/skipped → (不变)
```

### 全局状态更新

```
当 optimization_points 中所有点的状态为 completed/failed/skipped 时:
    optimization_active = false

当任一优化点 result = success:
    更新 current_best_dir 为对应的 last_attempt_dir
    更新 current_best_speedup 为实际speedup
```

---

## 优化点选择规则

主agent每次选择优化点的逻辑：

1. 按 index 从小到大遍历 optimization_points
2. 选择第一个 status = pending 的优化点
3. 如果所有优化点都不是 pending，退出优化循环

---

## 目录命名规范

```
opt_iter_{N}/                           # 第N轮优化迭代
├── optimized_code.py                   # 优化后的代码
├── verify/
│   ├── {op_name}_torch.py             # PyTorch参考
│   ├── {op_name}_triton_baseline.py   # 基线代码
│   └── {op_name}_triton_optimized.py  # 优化后代码
├── baseline_perf_result.json           # 基线性能
├── optimized_perf_result.json          # 优化后性能
└── log.md                              # 本轮优化日志
```

---

## 优化执行subagent返回格式

```
{
    "success": true/false,
    "speedup": 1.05,          # 优化后的加速比，失败时为0
    "attempt_dir": "./opt_iter_0",
    "message": "优化说明"
}
```

---

## 终止条件

满足以下任一条件时，退出优化循环：

1. 所有优化点状态都不是 pending
2. 达到最大循环次数（可选，默认无限制）
