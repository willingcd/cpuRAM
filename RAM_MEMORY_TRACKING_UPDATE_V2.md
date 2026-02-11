# RAM 内存跟踪更新（第二次改动）

## 概述

本次更新在 C++ 和 Python 层面都添加了 RAM 内存跟踪功能，按照 vLLM 启动的时间顺序输出日志，所有日志都标记为"第二次改动"。

## 更新内容

### 1. Python 层面更新

#### 1.1 内存跟踪器更新 (`vllm/utils/ram_memory_tracker.py`)

**新增功能**:
- 所有日志输出都添加了"（第二次改动）"标记
- 新增 `log_ram_memory_from_cpp()` 函数，用于从 C++ 代码调用
- 改进了内存释放检测和日志输出

**日志格式**:
```
xxx 占用 <具体占用空间大小> M RAM内存 (累计: <累计值> M)（第二次改动）
yyy 占用 <具体占用空间大小> M RAM内存 (累计: <累计值> M)（第二次改动）
zzz 占用 <具体占用空间大小> M RAM内存 (累计: <累计值> M)（第二次改动）
```

#### 1.2 已添加内存跟踪的关键位置

**EngineCore 初始化** (`vllm/v1/engine/core.py`):
- 创建 ModelExecutor 之前/之后
- 初始化 KV Cache 之前/之后
- 创建 Scheduler 之前/之后
- 初始化 KV Cache 的各个步骤

**Worker 初始化** (`vllm/v1/worker/gpu_worker.py`):
- Worker 初始化开始
- WorkspaceManager 初始化
- GPUModelRunner 实例化
- 模型加载
- KV Cache 初始化

**GPUModelRunner** (`vllm/v1/worker/gpu_model_runner.py`):
- GPUModelRunner 初始化
- InputBatch 创建
- 模型加载过程
- KV Cache 初始化
- 权重转移到 GPU 后的内存释放检查

**InputBatch** (`vllm/v1/worker/gpu_input_batch.py`):
- token_ids_cpu_tensor 创建
- is_token_ids_tensor 创建
- num_computed_tokens_cpu_tensor 创建

### 2. C++ 层面更新

#### 2.1 内存跟踪辅助模块

**新增文件**:
- `csrc/cpu/ram_memory_tracker.h` - C++ 内存跟踪头文件
- `csrc/cpu/ram_memory_tracker.cpp` - C++ 内存跟踪实现

**功能**:
- 提供 `init_ram_memory_tracker()` 函数，用于从 Python 初始化回调
- 提供 `log_ram_memory_from_cpp()` 函数，用于在 C++ 代码中记录内存分配
- 通过 Python C API 调用 Python 的内存跟踪函数

#### 2.2 关键内存分配点（待添加跟踪）

**共享内存分配** (`csrc/cpu/shm.cpp`):
- `init_shm()` 函数中的 `mmap()` 调用
- `destroy_shm()` 函数中的 `munmap()` 调用

**CUDA 内存分配器** (`csrc/cumem_allocator.cpp`):
- `my_malloc()` 函数中的内存分配
- `my_free()` 函数中的内存释放

**注意**: C++ 层面的跟踪需要：
1. 在 Python 启动时初始化 C++ 内存跟踪回调
2. 在 C++ 代码的关键位置添加跟踪调用

## 使用方法

### 启用内存跟踪

内存跟踪默认启用，可以通过环境变量控制：

```bash
# 启用内存跟踪（默认）
export VLLM_ENABLE_RAM_TRACKING=1

# 禁用内存跟踪
export VLLM_ENABLE_RAM_TRACKING=0
```

### 运行 vLLM

```bash
vllm serve --model Qwen3-8B --gpu-memory-utilization 0.6 \
  --safetensors-load-strategy lazy --port 8000 \
  --max-model-len 5120 --tensor-parallel-size 1 --swap-space 0
```

### 查看日志输出

日志会按照启动时间顺序输出，格式如下：

```
INFO: EngineCore初始化开始 占用 0.00 M RAM内存 (累计: 0.00 M)（第二次改动）
INFO: 创建ModelExecutor之前 占用 50.23 M RAM内存 (累计: 50.23 M)（第二次改动）
INFO: 创建ModelExecutor之后 占用 120.45 M RAM内存 (累计: 120.45 M)（第二次改动）
INFO: InputBatch:创建token_ids_cpu_tensor之前 占用 5.12 M RAM内存 (累计: 125.57 M)（第二次改动）
INFO: InputBatch:创建token_ids_cpu_tensor之后 占用 5.24 M RAM内存 (累计: 130.81 M)（第二次改动）
INFO: GPUModelRunner:创建InputBatch之后 占用 8.50 M RAM内存 (累计: 139.31 M)（第二次改动）
...
```

## 内存跟踪的工作原理

### RSS (Resident Set Size) 跟踪

- 使用 `psutil.Process().memory_info().rss` 获取进程的物理内存占用
- RSS 反映的是实际占用的物理内存，不是虚拟内存
- 当内存被释放时，RSS 会减少
- 如果释放的内存被其他模块立即重用，RSS 可能不会明显变化

### 内存变化检测

1. **内存分配**: 记录每次内存增加
2. **内存释放**: 显式检测内存释放（如权重转移到 GPU）
3. **内存重用**: 识别内存被重用的情况

### 动态内存处理

- **临时分配**: 权重加载到 CPU 然后转移到 GPU（内存会释放）
- **持久分配**: InputBatch 的 CPU 缓冲区（内存不会释放）
- **重用分配**: 使用之前释放的内存空间（RSS 不增加）

## 代码位置总结

### Python 层面

| 文件 | 功能 |
|------|------|
| `vllm/utils/ram_memory_tracker.py` | 内存跟踪器核心实现 |
| `vllm/v1/engine/core.py` | EngineCore 初始化跟踪 |
| `vllm/v1/worker/gpu_worker.py` | Worker 初始化跟踪 |
| `vllm/v1/worker/gpu_model_runner.py` | GPUModelRunner 跟踪 |
| `vllm/v1/worker/gpu_input_batch.py` | InputBatch CPU 缓冲区跟踪 |

### C++ 层面

| 文件 | 功能 |
|------|------|
| `csrc/cpu/ram_memory_tracker.h` | C++ 内存跟踪头文件 |
| `csrc/cpu/ram_memory_tracker.cpp` | C++ 内存跟踪实现 |
| `csrc/cpu/shm.cpp` | 共享内存分配（待添加跟踪） |
| `csrc/cumem_allocator.cpp` | CUDA 内存分配器（待添加跟踪） |

## 下一步工作

### 待完成的任务

1. **C++ 内存跟踪集成**:
   - 在 Python 启动时初始化 C++ 内存跟踪回调
   - 在 `csrc/cpu/shm.cpp` 的 `init_shm()` 中添加跟踪
   - 在 `csrc/cumem_allocator.cpp` 的 `my_malloc()` 中添加跟踪

2. **编译配置**:
   - 确保 C++ 内存跟踪模块被正确编译
   - 处理条件编译（如果 C++ 代码在没有 Python 的环境中编译）

3. **测试验证**:
   - 验证所有内存跟踪点都能正常工作
   - 验证日志输出格式正确
   - 验证内存跟踪的准确性

## 注意事项

1. **性能影响**: 内存跟踪会略微影响性能，但影响很小
2. **内存测量**: RSS 反映的是物理内存，可能受到系统内存管理的影响
3. **内存重用**: 释放的内存可能被立即重用，导致 RSS 不减少
4. **垃圾回收**: Python 的垃圾回收可能延迟内存释放

## 相关文档

- `RAM_MEMORY_ALLOCATION_ANALYSIS.md` - 内存分配分析
- `RAM_MEMORY_DYNAMICS_EXPLANATION.md` - 动态内存变化说明
- `RAM_MEMORY_TRACKING_USAGE.md` - 使用说明
- `RAM_MEMORY_ALLOCATION_SOURCE_CODE.md` - 源码级别的内存分配分析
