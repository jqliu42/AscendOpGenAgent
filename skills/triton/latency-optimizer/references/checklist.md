# 代码规范检查清单

在应用任何优化之前，必须检查代码是否满足以下规范：

## 必须遵循的规范

### 1. Kernel 定义规范
- [ ] 必须使用 `@triton.jit` 装饰器定义 kernel
- [ ] Kernel 函数签名中的 constexpr 参数必须正确声明

### 2. 内存访问规范
- [ ] 使用 `tl.load` 和 `tl.store` 进行内存访问
- [ ] 内存访问必须对齐（按照 tl.constexpr 对齐要求）
- [ ] 禁止在 kernel 内直接使用 PyTorch 操作（torch.*, F.*）

### 3. 变量声明规范
- [ ] 指针类型变量必须正确初始化
- [ ] 共享内存使用必须通过 `tl.shared_utils` 或 `jit` 函数参数

### 4. 计算规范
- [ ] 禁止在 kernel 内进行不必要的类型转换
- [ ] 数值计算必须使用 Triton 提供的高效 API

### 5. 并行度规范
- [ ] `tl.program_id` 和 `tl.num_programs` 的使用必须正确
- [ ] `grid` 配置必须与 kernel 设计匹配

## 检查流程

1. 加载本文件（checklist.md）
2. 逐一检查上述规范项
3. 如有不满足项 → 修改代码直到满足所有规范
4. 所有规范满足后 → 进行代码验证
