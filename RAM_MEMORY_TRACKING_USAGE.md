# RAM 内存跟踪功能使用说明

## 概述

我已经为 vLLM 添加了 RAM 内存跟踪功能，可以在启动过程中按时间顺序输出各个组件占用的 RAM 内存。

## 功能说明

### 1. 内存跟踪工具

创建了 `vllm/utils/ram_memory_tracker.py` 模块，提供：
- `RAMMemoryTracker` 类：跟踪进程的 RSS (Resident Set Size) 内存使用
- `log_ram_memory()` 函数：便捷的内存日志记录函数

### 2. 跟踪的关键位置

已在以下关键位置添加了内存跟踪日志：

#### EngineCore 初始化
- EngineCore初始化开始
- 创建ModelExecutor之前/之后
- 初始化KV Cache之前/之后
- 调用initialize_cache之前/之后
- 创建StructuredOutputManager之前/之后
- 创建Scheduler之前/之后

#### Worker 初始化
- Worker.initialize_cache之前/之后
- 初始化workspace_manager之前/之后
- 创建ModelRunner之前/之后
- Worker.load_model之前/之后

#### ModelRunner 初始化
- GPUModelRunner.__init__开始
- GPUModelRunner:创建InputBatch之前/之后
- ModelRunner.load_model:开始加载模型权重之前
- ModelRunner.load_model:调用model_loader.load_model之前/之后

#### InputBatch 创建
- InputBatch:创建token_ids_cpu_tensor之前/之后
- InputBatch:创建is_token_ids_tensor之前/之后

#### KV Cache 初始化
- _initialize_kv_caches:获取kv_cache_specs之前/之后
- _initialize_kv_caches:调用determine_available_memory之前/之后
- _initialize_kv_caches:调用get_kv_cache_configs之前/之后
- _initialize_kv_caches:调用initialize_from_config之前/之后

#### WorkerWrapper
- WorkerWrapper.initialize_from_config之前/之后
- WorkerWrapper.init_device之前/之后

## 使用方法

### 启用内存跟踪

内存跟踪默认是**启用**的。如果需要禁用，可以设置环境变量：

```bash
export VLLM_ENABLE_RAM_TRACKING=0
```

### 运行 vLLM

正常启动 vLLM，内存跟踪日志会自动输出：

```bash
vllm serve --model Qwen3-8B --gpu-memory-utilization 0.6 --safetensors-load-strategy lazy --port 8000 --max-model-len 5120 --tensor-parallel-size 1 --swap-space 0
```

### 日志输出示例

启动时会看到类似以下格式的日志：

```
INFO: EngineCore初始化开始 占用 0.00 M RAM内存 (累计: 0.00 M)
INFO: 创建ModelExecutor之前 占用 15.23 M RAM内存 (累计: 15.23 M)
INFO: 创建ModelExecutor之后 占用 8.45 M RAM内存 (累计: 23.68 M)
INFO: 初始化KV Cache之前 占用 2.11 M RAM内存 (累计: 25.79 M)
INFO: GPUModelRunner.__init__开始 占用 5.32 M RAM内存 (累计: 31.11 M)
INFO: GPUModelRunner:创建InputBatch之前 占用 1.89 M RAM内存 (累计: 33.00 M)
INFO: InputBatch:创建token_ids_cpu_tensor之前 占用 0.45 M RAM内存 (累计: 33.45 M)
INFO: InputBatch:创建token_ids_cpu_tensor之后 占用 12.34 M RAM内存 (累计: 45.79 M)
INFO: InputBatch:创建is_token_ids_tensor之前 占用 0.12 M RAM内存 (累计: 45.91 M)
INFO: InputBatch:创建is_token_ids_tensor之后 占用 3.08 M RAM内存 (累计: 48.99 M)
INFO: GPUModelRunner:创建InputBatch之后 占用 0.05 M RAM内存 (累计: 49.04 M)
INFO: Worker.load_model之前 占用 2.15 M RAM内存 (累计: 51.19 M)
INFO: ModelRunner.load_model:开始加载模型权重之前 占用 1.23 M RAM内存 (累计: 52.42 M)
INFO: ModelRunner.load_model:调用model_loader.load_model之前 占用 0.67 M RAM内存 (累计: 53.09 M)
INFO: ModelRunner.load_model:调用model_loader.load_model之后 占用 1234.56 M RAM内存 (累计: 1287.65 M)
...
```

## 日志格式说明

每条日志的格式为：
```
<组件名称> 占用 <增量内存> M RAM内存 (累计: <总内存> M)
```

- **组件名称**：正在初始化的组件或操作
- **增量内存**：相对于上一次检查点增加的内存（MB）
- **累计内存**：相对于启动时的基线增加的总内存（MB）

## 技术实现

### 内存测量方法

使用 `psutil` 库获取进程的 RSS (Resident Set Size)：
- RSS 是进程实际占用的物理内存
- 不包括交换空间（swap）
- 不包括共享库的内存（除非是进程独有的部分）

### 垃圾回收

在每次内存测量前会强制进行垃圾回收（`gc.collect()`），以确保测量准确性。

### 基线设置

内存跟踪器在首次调用时设置基线，后续所有测量都是相对于这个基线的增量。

## 注意事项

1. **内存测量精度**：由于 Python 的垃圾回收机制和系统内存管理，测量值可能有 ±1-2 MB 的误差。

2. **多进程场景**：在数据并行（DP）或多进程场景下，每个进程都会独立跟踪自己的内存使用。

3. **性能影响**：内存跟踪会略微影响启动性能（每次测量约 1-5ms），但影响很小。

4. **日志级别**：内存跟踪日志使用 `INFO` 级别，确保日志级别设置正确。

## 故障排查

### 看不到内存日志

1. 检查环境变量：确保 `VLLM_ENABLE_RAM_TRACKING` 未设置为 `0`
2. 检查日志级别：确保日志级别设置为 `INFO` 或更低
3. 检查日志输出：确认日志输出到正确的位置（控制台或文件）

### 内存值异常

1. **负值**：可能是垃圾回收释放了内存，这是正常的
2. **值过大**：可能是模型权重加载或其他大对象分配
3. **值过小**：可能是测量时机问题，某些分配在测量后发生

## 扩展功能

如果需要添加更多的内存跟踪点，可以在相应位置调用：

```python
from vllm.utils.ram_memory_tracker import log_ram_memory

log_ram_memory("您的组件名称")
```

## 相关文件

- `vllm/utils/ram_memory_tracker.py` - 内存跟踪工具实现
- `vllm/v1/engine/core.py` - EngineCore 中的跟踪点
- `vllm/v1/worker/gpu_worker.py` - Worker 中的跟踪点
- `vllm/v1/worker/gpu_model_runner.py` - ModelRunner 中的跟踪点
- `vllm/v1/worker/gpu_input_batch.py` - InputBatch 中的跟踪点
- `vllm/v1/worker/worker_base.py` - WorkerWrapper 中的跟踪点
