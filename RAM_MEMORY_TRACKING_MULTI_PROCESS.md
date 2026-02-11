# 多进程（TP>1）RAM 内存日志区分与分析

当你使用：

```bash
vllm serve ... --tensor-parallel-size 4 ...
```

vLLM 会启动 **多个 Python 进程**：
- **EngineCore 进程**：负责调度/协调
- **多个 Worker 进程**：每个 rank 负责一个 TP 分片（通常 rank=0..tp-1）

如果所有进程都把日志打到同一个 stdout，你会看到“混合日志”。为了解决这个问题，我们在 `vllm/utils/ram_memory_tracker.py` 中给每条 RAM 日志加了**进程上下文前缀**。

## 1. 新的日志前缀格式

每条 RAM 日志会以类似下面的前缀开头：

```
[pid=12345, role=enginecore, dp_rank=0] ...
[pid=12346, role=worker, rank=0, local_rank=0, is_driver_worker=True] ...
[pid=12347, role=worker, rank=1, local_rank=1, is_driver_worker=False] ...
```

含义：
- **pid**：进程号，可直接对照 `top` / `ps`
- **role**：
  - `enginecore`：EngineCore 子进程
  - `worker`：Worker 进程（TP/PP/DP 的执行进程）
- **rank/local_rank**：Worker 的全局 rank 和本机 GPU rank（通常 local_rank=GPU id）
- **dp_rank/local_dp_rank**：如果启用了 DP，会显示 DP 的 rank

## 2. 为什么这样就能“按进程分析”

因为 RSS 是**每个进程独立**的：EngineCore 的 RSS 与每个 worker 的 RSS 是不同的地址空间。

现在每条日志带 `pid/role/rank`，你可以：
- **按 role 分组**：只看 EngineCore 或只看 Worker
- **按 rank 分组**：只看 worker rank=0/1/2/3 的启动内存曲线
- **按 pid 对齐 top**：解释 “top 显示 2.3G” 对应哪一个进程

## 3. 常用过滤方式（推荐）

### 3.1 只看 EngineCore

用日志过滤（示例）：
- 关键字：`role=enginecore`

### 3.2 只看某个 worker rank

例如 rank=2：
- 关键字：`role=worker, rank=2`

### 3.3 对齐 top

1. 在 `top` 里记录目标进程 **pid**
2. 在日志里过滤 `pid=<那个pid>`，即可拿到该进程启动期间的 RAM 变化时间线

## 4. 代码修改点（你可以快速定位）

- **EngineCore 进程打标签**：`vllm/v1/engine/core.py` 的 `EngineCoreProc.run_engine_core`
- **Worker 进程打标签**：`vllm/v1/worker/worker_base.py` 的 `WorkerBase.__init__`
- **前缀拼接与输出格式**：`vllm/utils/ram_memory_tracker.py`

## 5. 你应该能看到的现象（TP=4）

- **EngineCore**：通常 RSS 比较稳定，主要是 Python runtime + 调度对象 + IPC buffer
- **Worker(rank=0..3)**：会发生更明显的 RSS 变化，尤其在：
  - 初始化 workspace_manager
  - 创建 ModelRunner / InputBatch
  - 加载权重（lazy 模式可能 RSS 增幅偏小）
  - 初始化 KV cache（如果有 CPU 侧结构/共享内存）

