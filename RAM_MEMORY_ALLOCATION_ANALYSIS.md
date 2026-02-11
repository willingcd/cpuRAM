# vLLM RAM 内存分配分析

## 概述
本文档分析了 vLLM 项目中会开辟 RAM 内存地址空间的代码部分。根据代码分析，即使设置了 `--swap-space 0`，vLLM 仍然会在多个地方分配 CPU RAM 内存。

## 主要内存分配位置

### 1. EngineCore 进程本身
**位置**: `vllm/v1/engine/core.py`

EngineCore 是 vLLM 的核心引擎进程，在后台运行。这个进程本身会占用一定的内存空间：
- 进程的代码段、数据段、堆栈等基础内存
- Python 解释器和运行时环境的内存
- 各种对象和数据结构的内存

**相关代码**:
- `EngineCore.__init__()` - 初始化时会创建多个对象
- `EngineCoreProc.run_engine_core()` - 后台进程的主循环

### 2. InputBatch CPU 缓冲区
**位置**: `vllm/v1/worker/gpu_input_batch.py`

在 `InputBatch` 类中，会分配多个 CPU 内存缓冲区：

```python
# 第 116-121 行
self.token_ids_cpu_tensor = torch.zeros(
    (max_num_reqs, max_model_len),
    device="cpu",
    dtype=torch.int32,
    pin_memory=False,
)
```

**内存大小**: `max_num_reqs * max_model_len * 4` 字节（int32）

```python
# 第 123-125 行
self.is_token_ids_tensor = torch.zeros(
    (max_num_reqs, max_model_len), device="cpu", dtype=bool, pin_memory=False
)
```

**内存大小**: `max_num_reqs * max_model_len * 1` 字节（bool）

这些缓冲区用于在 CPU 上准备输入数据，然后传输到 GPU。

### 3. 模型权重加载
**位置**: `vllm/model_executor/model_loader/`

在模型加载过程中，权重可能会临时存储在 CPU 内存中：
- 从磁盘读取模型权重时
- 权重在 CPU 和 GPU 之间传输时
- 使用 `safetensors-load-strategy lazy` 时，部分权重可能保留在 CPU 上

**相关代码**:
- `vllm/model_executor/model_loader/utils.py` - `device_loading_context()` 函数
- 第 148-157 行：使用 `torch.empty_strided()` 在 CPU 上分配内存

### 4. 共享内存 (Shared Memory)
**位置**: `csrc/cpu/shm.cpp`

vLLM 使用共享内存进行进程间通信，特别是在多进程场景下：

```cpp
// 第 329 行
munmap(_shared_mem_ptrs[i], compute_shm_size());
```

**相关代码**:
- `csrc/cpu/shm.cpp` - 共享内存的分配和管理
- 用于多进程数据并行（Data Parallel）场景

### 5. KV Cache 相关（即使 swap-space=0）
**位置**: `vllm/v1/worker/gpu_worker.py`

虽然设置了 `--swap-space 0`，但在某些情况下仍可能有 CPU 内存分配：

```python
# 第 175-177 行
def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
    self.cache_config.num_gpu_blocks = num_gpu_blocks
    self.cache_config.num_cpu_blocks = num_cpu_blocks
```

**注意**: 即使 `num_cpu_blocks=0`，相关的数据结构和元数据仍会占用内存。

### 6. 临时张量和缓冲区
**位置**: 多个文件

在模型执行过程中，会创建各种临时 CPU 张量：

- **Block Table CPU 张量**: `vllm/v1/worker/gpu/block_table.py`
  ```python
  # 第 93 行
  return ptrs_tensor_cpu.to(self.device, non_blocking=True)
  ```

- **Logprobs CPU 张量**: `vllm/v1/outputs.py`
  ```python
  # 第 67-69 行
  self.logprob_token_ids.to("cpu", non_blocking=True),
  self.logprobs.to("cpu", non_blocking=True),
  ```

- **Attention 元数据**: `vllm/v1/attention/backends/utils.py`
  ```python
  # 第 724-725 行
  query_start_loc=query_start_loc_cpu.to(device=device, non_blocking=True),
  seq_lens=seq_lens_cpu.to(device=device, non_blocking=True),
  ```

### 7. 序列化和通信缓冲区
**位置**: `vllm/v1/engine/core_client.py`

在 EngineCore 和客户端之间的通信中，会使用 ZMQ 和消息序列化：

```python
# 第 1078-1082 行
reuse_buffers: list[bytearray] = []
pending = deque[tuple[zmq.MessageTracker, Any, bytearray]]()
```

这些缓冲区用于消息的序列化和反序列化。

### 8. Python 对象和数据结构
**位置**: 整个代码库

Python 对象本身会占用内存：
- 字典、列表、集合等数据结构
- 请求对象、调度器状态等
- 缓存和哈希表

## 内存分配方式

### 1. PyTorch 张量分配
- `torch.zeros()` - 在 CPU 上分配零初始化张量
- `torch.empty()` - 分配未初始化张量
- `torch.empty_strided()` - 分配特定步长的张量

### 2. C++ 内存分配
- `malloc()` - 在 `csrc/cumem_allocator.cpp` 中使用
- `mmap()` / `munmap()` - 用于共享内存
- `shm_open()` / `shm_unlink()` - POSIX 共享内存

### 3. Python 对象分配
- Python 的垃圾收集器管理
- 各种数据结构的动态分配

## 估算内存占用

对于您的情况（Qwen3-8B，max-model-len=5120），主要内存占用可能来自：

1. **InputBatch CPU 缓冲区**:
   - `token_ids_cpu_tensor`: 假设 `max_num_reqs=256`，则 `256 * 5120 * 4 = 5.24 MB`
   - `is_token_ids_tensor`: `256 * 5120 * 1 = 1.31 MB`

2. **EngineCore 进程基础内存**: 通常 100-500 MB

3. **模型权重（如果 lazy loading）**: 取决于实际加载的权重数量

4. **临时缓冲区和对象**: 可能 100-500 MB

**总计估算**: 约 200 MB - 1 GB，这与您观察到的 2.3 GB 接近。

## 建议

1. **减少 max_num_reqs**: 如果可能，减少最大并发请求数
2. **减少 max_model_len**: 如果不需要 5120，可以降低
3. **检查 lazy loading**: 确认 `safetensors-load-strategy lazy` 是否真的延迟加载
4. **监控内存**: 使用 `memory_profiler` 或 `tracemalloc` 来精确定位内存分配

## 相关代码文件

- `vllm/v1/engine/core.py` - EngineCore 主逻辑
- `vllm/v1/worker/gpu_input_batch.py` - InputBatch CPU 缓冲区
- `vllm/v1/worker/gpu_worker.py` - Worker 初始化
- `vllm/v1/worker/gpu/block_table.py` - Block table CPU 张量
- `vllm/model_executor/model_loader/utils.py` - 模型加载
- `csrc/cpu/shm.cpp` - 共享内存
- `vllm/config/cache.py` - Cache 配置（swap_space）
