# vLLM RAM 内存物理空间分配源码分析

## 概述

本文档详细分析 vLLM 源码中如何从操作系统开辟 RAM 内存物理空间。虽然我们无法直接查看操作系统层面的实现，但可以通过 vLLM 源码中使用的系统调用和 API 来理解底层的内存分配机制。

## 内存分配的调用链

### 调用层次结构

```
vLLM Python 代码
    ↓
PyTorch Python API (torch.zeros, torch.empty 等)
    ↓
PyTorch C++ 后端 (libtorch)
    ↓
C++ 标准库 / 系统调用
    ↓
操作系统内核 (brk, mmap, malloc 等)
    ↓
物理内存分配
```

## 1. PyTorch 张量分配（Python 层面）

### 1.1 InputBatch CPU 缓冲区

**位置**: `vllm/v1/worker/gpu_input_batch.py`

```python
# 第 119-124 行
self.token_ids_cpu_tensor = torch.zeros(
    (max_num_reqs, max_model_len),
    device="cpu",
    dtype=torch.int32,
    pin_memory=False,
)
```

**内存分配路径**:
1. `torch.zeros()` → PyTorch C++ 后端
2. PyTorch 调用 `THFloatTensor_newWithSize2d()` 或类似函数
3. 最终调用 `malloc()` 或 `posix_memalign()` (对于对齐内存)
4. `malloc()` 内部调用系统调用 `brk()` 或 `mmap()`

**操作系统调用**:
- Linux: `brk()` 或 `mmap(MAP_ANONYMOUS)` → 内核分配物理内存页
- macOS: `mmap()` → 内核分配物理内存页

### 1.2 模型权重加载

**位置**: `vllm/model_executor/model_loader/utils.py`

```python
# 第 148-155 行
cpu_data = torch.empty_strided(
    size=p.data.size(),
    stride=p.data.stride(),
    dtype=p.data.dtype,
    layout=p.data.layout,
    device="cpu",
    pin_memory=pin_memory,
)
```

**内存分配路径**:
1. `torch.empty_strided()` → PyTorch C++ 后端
2. 如果 `pin_memory=True`，使用 `cudaHostAlloc()` (CUDA pinned memory)
3. 如果 `pin_memory=False`，使用标准 `malloc()` 或 `posix_memalign()`
4. 最终调用系统调用分配物理内存

**操作系统调用**:
- `pin_memory=True`: `cudaHostAlloc()` → CUDA 驱动 → `mmap()` → 内核分配物理内存
- `pin_memory=False`: `malloc()` → `brk()`/`mmap()` → 内核分配物理内存

## 2. C++ 层面的直接内存分配

### 2.1 malloc() 分配

**位置**: `csrc/cumem_allocator.cpp`

```cpp
// 第 308 行
CUmemGenericAllocationHandle* p_memHandle =
    (CUmemGenericAllocationHandle*)malloc(
        sizeof(CUmemGenericAllocationHandle));
```

**内存分配路径**:
1. `malloc()` → C 标准库实现 (glibc/musl)
2. glibc 的 `malloc()` 实现：
   - 小内存块：使用 `brk()` 系统调用扩展堆
   - 大内存块：使用 `mmap(MAP_ANONYMOUS)` 直接映射
3. 内核分配物理内存页

**操作系统调用**:
- Linux: `brk()` 或 `mmap(MAP_ANONYMOUS | MAP_PRIVATE, PROT_READ | PROT_WRITE)`
- macOS: `mmap()` 类似

### 2.2 共享内存分配 (shm_open + mmap)

**位置**: `csrc/cpu/shm.cpp`

这是最直接可以看到操作系统调用的地方：

```cpp
// 第 283-284 行：创建共享内存对象
fd = shm_open(shm_name.c_str(), O_CREAT | O_EXCL | O_RDWR,
              S_IRUSR | S_IWUSR);

// 第 290 行：设置共享内存大小
if (ftruncate(fd, shm_size) == -1)
    TORCH_CHECK(false, "ftruncate in SHMManager failed. errno: " +
                       std::to_string(errno));

// 第 301-302 行：将共享内存映射到进程地址空间
void* shm_ptr = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_POPULATE, fd, 0);
```

**内存分配路径**:
1. `shm_open()` → 创建 POSIX 共享内存对象
   - Linux: 在 `/dev/shm/` 创建文件
   - macOS: 使用类似机制
2. `ftruncate()` → 设置共享内存大小
   - 系统调用：`ftruncate()` → 内核分配物理内存页
3. `mmap()` → 将共享内存映射到进程地址空间
   - 系统调用：`mmap()` → 内核建立虚拟地址到物理内存的映射
   - `MAP_POPULATE` 标志：立即分配物理内存页（预填充）

**操作系统调用链**:
```
shm_open() 
  → 系统调用: open() (在 /dev/shm/)
  → 内核: 创建共享内存对象

ftruncate(fd, shm_size)
  → 系统调用: ftruncate()
  → 内核: 分配物理内存页（如果支持）

mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0)
  → 系统调用: mmap()
  → 内核: 
     1. 建立虚拟地址到物理内存的映射
     2. 如果 MAP_POPULATE，立即分配物理内存页
     3. 返回虚拟地址
```

## 3. 内存分配的系统调用详解

### 3.1 brk() 系统调用

**用途**: 扩展进程的数据段（堆）

**调用场景**:
- `malloc()` 分配小内存块时
- Python 对象分配时

**工作原理**:
1. 进程有一个"程序中断点"（program break）
2. `brk()` 移动这个中断点，扩展堆空间
3. 内核分配新的物理内存页并映射到扩展的地址空间

**在 vLLM 中的使用**:
- 间接通过 `malloc()` 使用
- PyTorch 张量分配时可能使用

### 3.2 mmap() 系统调用

**用途**: 内存映射，可以映射文件或匿名内存

**调用场景**:
- 共享内存 (`csrc/cpu/shm.cpp`)
- 大内存块分配（`malloc()` 对于大块使用 `mmap()`）
- PyTorch 的 pinned memory（通过 CUDA API，底层也是 `mmap()`）

**mmap() 参数说明** (以 `csrc/cpu/shm.cpp` 为例):

```cpp
mmap(
    nullptr,                    // addr: 让系统选择地址
    shm_size,                  // length: 映射大小
    PROT_READ | PROT_WRITE,    // prot: 读写权限
    MAP_SHARED | MAP_POPULATE, // flags: 共享映射 + 预填充
    fd,                        // fd: 文件描述符（共享内存）
    0                          // offset: 偏移量
)
```

**MAP_POPULATE 标志**:
- 立即分配物理内存页
- 不等待页面错误（page fault）时才分配
- 确保内存真正被分配，而不是延迟分配

**操作系统行为**:
1. 内核在进程的虚拟地址空间中分配虚拟地址范围
2. 如果 `MAP_POPULATE`，立即分配物理内存页
3. 建立页表项（Page Table Entry），映射虚拟地址到物理地址
4. 返回虚拟地址指针

### 3.3 shm_open() 系统调用

**用途**: 创建或打开 POSIX 共享内存对象

**调用场景**:
- 多进程数据并行（Data Parallel）场景
- 进程间通信

**工作原理**:
1. 在 `/dev/shm/` (Linux) 或类似位置创建共享内存对象
2. 返回文件描述符
3. 可以多个进程通过同一个名称访问同一块内存

## 4. 物理内存分配的底层机制

### 4.1 虚拟内存到物理内存的映射

当调用 `mmap()` 或 `brk()` 时：

1. **虚拟地址分配**: 内核在进程的虚拟地址空间中分配地址范围
2. **物理页分配**: 内核从物理内存池中分配物理页（通常 4KB 或更大）
3. **页表建立**: 内核建立页表项，将虚拟地址映射到物理地址
4. **内存初始化**: 
   - 匿名映射：初始化为零（如果指定）
   - 文件映射：从文件读取数据

### 4.2 延迟分配（Lazy Allocation）

**默认行为**:
- `mmap()` 默认不立即分配物理内存
- 只有在首次访问时（页面错误）才分配物理内存

**立即分配**:
- 使用 `MAP_POPULATE` 标志（如 `csrc/cpu/shm.cpp`）
- 使用 `mlock()` 锁定内存
- 使用 `madvise(MADV_WILLNEED)` 提示内核

### 4.3 内存对齐

**代码示例**: `csrc/cpu/shm.cpp`

```cpp
// 第 314 行：检查内存对齐
TORCH_CHECK((size_t)shm_ptr % 64 == 0);
```

**原因**:
- CPU 缓存行对齐（通常 64 字节）
- SIMD 指令要求对齐
- 性能优化

## 5. 具体代码位置和调用链

### 5.1 InputBatch CPU 张量

**Python 代码**: `vllm/v1/worker/gpu_input_batch.py:119`
```python
torch.zeros((max_num_reqs, max_model_len), device="cpu", ...)
```

**调用链**:
```
torch.zeros()
  → PyTorch C++: THIntTensor_newWithSize2d()
  → PyTorch C++: THAllocator.allocate()
  → C 标准库: malloc() 或 posix_memalign()
  → 系统调用: brk() 或 mmap(MAP_ANONYMOUS)
  → 内核: 分配物理内存页
```

### 5.2 共享内存

**C++ 代码**: `csrc/cpu/shm.cpp:276-317`

**调用链**:
```
init_shm()
  → shm_open() → 系统调用: open() → 内核: 创建共享内存对象
  → ftruncate() → 系统调用: ftruncate() → 内核: 设置大小
  → mmap() → 系统调用: mmap() → 内核: 映射到虚拟地址空间 + 分配物理内存
```

### 5.3 模型权重（pinned memory）

**Python 代码**: `vllm/model_executor/model_loader/utils.py:148`
```python
torch.empty_strided(..., pin_memory=True)
```

**调用链**:
```
torch.empty_strided(pin_memory=True)
  → PyTorch C++: cudaHostAlloc()
  → CUDA 驱动: cudaHostAlloc()
  → 系统调用: mmap() (通过 CUDA 驱动)
  → 内核: 分配物理内存页（pinned，不可交换）
```

## 6. 内存分配的关键系统调用总结

| 系统调用 | 用途 | 在 vLLM 中的使用 | 物理内存分配时机 |
|---------|------|----------------|----------------|
| `brk()` | 扩展堆 | 通过 `malloc()` 间接使用 | 立即（小内存）或延迟（大内存） |
| `mmap()` | 内存映射 | 共享内存、大内存块 | 默认延迟，`MAP_POPULATE` 立即 |
| `shm_open()` | 创建共享内存 | `csrc/cpu/shm.cpp` | 不直接分配，配合 `mmap()` |
| `ftruncate()` | 设置文件大小 | `csrc/cpu/shm.cpp` | 可能触发物理内存分配 |
| `mlock()` | 锁定内存 | 可能用于 pinned memory | 确保物理内存不被交换 |

## 7. 如何验证内存分配

### 7.1 使用 strace 跟踪系统调用

```bash
# 跟踪 vLLM 进程的系统调用
strace -e trace=mmap,brk,shm_open,ftruncate -o vllm_syscalls.log \
  vllm serve --model Qwen3-8B ...
```

### 7.2 查看 /proc/PID/maps

```bash
# 查看进程的内存映射
cat /proc/$(pgrep -f "vllm serve")/maps
```

### 7.3 查看共享内存

```bash
# Linux: 查看共享内存对象
ls -lh /dev/shm/

# 查看共享内存使用情况
ipcs -m
```

## 8. 内存分配的关键代码位置

### 8.1 Python 层面

1. **InputBatch**: `vllm/v1/worker/gpu_input_batch.py:119, 128`
   - `torch.zeros()` → 最终调用系统分配

2. **模型权重**: `vllm/model_executor/model_loader/utils.py:148`
   - `torch.empty_strided()` → 可能使用 pinned memory

3. **权重卸载**: `vllm/model_executor/models/utils.py:551`
   - `torch.empty_strided(pin_memory=True)` → CUDA pinned memory

### 8.2 C++ 层面

1. **共享内存**: `csrc/cpu/shm.cpp:276-317`
   - `shm_open()` + `mmap()` → 直接系统调用

2. **CUDA 内存分配器**: `csrc/cumem_allocator.cpp:308, 321, 324`
   - `malloc()` → 间接系统调用

### 8.3 内存池

1. **TensorMemoryPool**: `vllm/distributed/kv_transfer/kv_connector/v1/p2p/tensor_memory_pool.py:93`
   - `torch.empty(..., pin_memory=True)` → CUDA pinned memory

## 9. 内存分配的实际流程示例

### 示例：InputBatch 创建 token_ids_cpu_tensor

```
1. Python: torch.zeros((256, 5120), device="cpu", dtype=torch.int32)
   ↓
2. PyTorch C++: THIntTensor_newWithSize2d(256, 5120)
   ↓
3. PyTorch C++: THAllocator.allocate(256 * 5120 * 4 = 5,242,880 字节)
   ↓
4. C 标准库: malloc(5,242,880)
   ↓
5. glibc malloc 实现:
   - 检查大小：5MB > 阈值（通常 128KB）
   - 使用 mmap() 而不是 brk()
   ↓
6. 系统调用: mmap(NULL, 5242880, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)
   ↓
7. 内核:
   - 在进程虚拟地址空间分配 5MB 地址范围
   - 分配物理内存页（5MB / 4KB = 1280 页）
   - 建立页表映射
   - 初始化内存为零（MAP_ANONYMOUS 的默认行为）
   ↓
8. 返回虚拟地址指针
   ↓
9. PyTorch 创建张量对象，包装这个指针
   ↓
10. Python 获得 torch.Tensor 对象
```

### 示例：共享内存分配

```
1. C++: init_shm(target_rank)
   ↓
2. shm_open("/vllm_shm_rank_0", O_CREAT|O_EXCL|O_RDWR, ...)
   → 系统调用: open("/dev/shm/vllm_shm_rank_0", ...)
   → 内核: 创建共享内存对象
   → 返回文件描述符 fd
   ↓
3. ftruncate(fd, shm_size)
   → 系统调用: ftruncate(fd, 8388608)  // 假设 8MB
   → 内核: 设置共享内存大小为 8MB
   → 可能触发物理内存分配（取决于实现）
   ↓
4. mmap(NULL, 8388608, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_POPULATE, fd, 0)
   → 系统调用: mmap(...)
   → 内核:
      - 在进程虚拟地址空间分配 8MB 地址范围
      - 由于 MAP_POPULATE，立即分配物理内存页（8MB / 4KB = 2048 页）
      - 建立页表映射（虚拟地址 → 物理地址）
      - 初始化内存为零
   → 返回虚拟地址指针
   ↓
5. C++ 代码使用这个指针访问内存
```

## 10. 关键系统调用的参数说明

### 10.1 mmap() 系统调用

```c
void *mmap(void *addr,           // 建议的地址（NULL 让系统选择）
           size_t length,         // 映射大小
           int prot,              // 保护标志：PROT_READ, PROT_WRITE
           int flags,             // 映射标志：MAP_SHARED, MAP_PRIVATE, MAP_ANONYMOUS, MAP_POPULATE
           int fd,                // 文件描述符（-1 表示匿名映射）
           off_t offset);         // 文件偏移量
```

**在 vLLM 中的使用**:
- `csrc/cpu/shm.cpp:301`: `mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0)`
  - `MAP_SHARED`: 多个进程可以共享
  - `MAP_POPULATE`: 立即分配物理内存

### 10.2 brk() 系统调用

```c
int brk(void *addr);  // 设置程序中断点
```

**工作原理**:
- 扩展或收缩进程的数据段（堆）
- 内核分配或释放物理内存页
- 通常由 `malloc()` 内部调用

## 11. 物理内存分配的实际时机

### 11.1 立即分配

**场景**:
- `mmap()` 使用 `MAP_POPULATE` 标志
- `mlock()` 锁定内存
- 首次访问内存时（页面错误）

**代码位置**:
- `csrc/cpu/shm.cpp:302`: `MAP_POPULATE` 确保立即分配

### 11.2 延迟分配（Copy-on-Write）

**场景**:
- `mmap()` 默认行为（无 `MAP_POPULATE`）
- `brk()` 扩展堆（可能延迟到首次访问）

**优势**:
- 节省物理内存（未使用的映射不分配物理页）
- 提高启动速度

## 12. 内存释放

### 12.1 munmap() 系统调用

**位置**: `csrc/cpu/shm.cpp:329`

```cpp
munmap(_shared_mem_ptrs[i], compute_shm_size());
```

**工作原理**:
1. 取消虚拟地址到物理内存的映射
2. 释放物理内存页（如果没有其他映射）
3. 释放虚拟地址空间

### 12.2 shm_unlink() 系统调用

**位置**: `csrc/cpu/shm.cpp:333`

```cpp
shm_unlink(_shm_names[i].c_str());
```

**工作原理**:
1. 删除共享内存对象名称
2. 当所有进程都取消映射时，物理内存被释放

## 13. 总结

### 13.1 内存分配的主要路径

1. **PyTorch 张量** → `torch.zeros/empty()` → PyTorch C++ → `malloc()` → `brk()`/`mmap()` → 内核
2. **共享内存** → `shm_open()` + `mmap()` → 直接系统调用 → 内核
3. **Pinned Memory** → `cudaHostAlloc()` → CUDA 驱动 → `mmap()` → 内核

### 13.2 关键系统调用

- **`mmap()`**: 最灵活，支持文件映射、匿名映射、共享内存
- **`brk()`**: 用于堆扩展，通常由 `malloc()` 内部使用
- **`shm_open()`**: 创建 POSIX 共享内存对象

### 13.3 物理内存分配时机

- **立即分配**: `MAP_POPULATE`, `mlock()`, 首次访问
- **延迟分配**: 默认的 `mmap()` 行为（按需分配）

### 13.4 在 vLLM 源码中的位置

| 分配方式 | 代码位置 | 系统调用 |
|---------|---------|---------|
| PyTorch 张量 | `vllm/v1/worker/gpu_input_batch.py:119` | 间接：`malloc()` → `brk()`/`mmap()` |
| 共享内存 | `csrc/cpu/shm.cpp:301` | 直接：`mmap(MAP_SHARED\|MAP_POPULATE)` |
| Pinned Memory | `vllm/model_executor/models/utils.py:551` | 间接：`cudaHostAlloc()` → `mmap()` |
| C++ 分配 | `csrc/cumem_allocator.cpp:308` | 间接：`malloc()` → `brk()`/`mmap()` |

## 14. 验证方法

### 14.1 使用 strace 跟踪

```bash
# 跟踪所有内存相关的系统调用
strace -e trace=mmap,munmap,brk,shm_open,shm_unlink,ftruncate \
  -o vllm_memory.log vllm serve --model Qwen3-8B ...
```

### 14.2 查看进程内存映射

```bash
# 查看进程的虚拟内存映射
cat /proc/$(pgrep -f "vllm serve")/maps

# 查看物理内存使用（RSS）
cat /proc/$(pgrep -f "vllm serve")/status | grep -E "VmSize|VmRSS|VmData"
```

### 14.3 查看共享内存

```bash
# 查看共享内存对象
ls -lh /dev/shm/ | grep vllm

# 查看共享内存统计
ipcs -m
```

通过这些方法，您可以在不直接查看操作系统源码的情况下，通过 vLLM 源码和系统调用跟踪来理解 RAM 内存物理空间的分配过程。
