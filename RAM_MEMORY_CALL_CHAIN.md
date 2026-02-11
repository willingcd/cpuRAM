# RAM 内存分配调用链（简化版）

## 快速参考：内存分配调用链

### 1. PyTorch 张量分配（最常见）

```
Python: torch.zeros((256, 5120), device="cpu")
    ↓
PyTorch C++: THIntTensor_newWithSize2d()
    ↓
PyTorch C++: THAllocator.allocate(5,242,880 字节)
    ↓
C 标准库: malloc(5,242,880)
    ↓
glibc malloc 实现:
  - 小内存 (<128KB): brk() 系统调用
  - 大内存 (>=128KB): mmap(MAP_ANONYMOUS) 系统调用
    ↓
Linux 内核:
  - brk(): 扩展堆，分配物理内存页
  - mmap(): 分配虚拟地址空间 + 物理内存页
    ↓
物理 RAM 内存被分配
```

### 2. 共享内存分配（最直接）

```
C++: init_shm() in csrc/cpu/shm.cpp
    ↓
shm_open("/vllm_shm_rank_0", O_CREAT|O_EXCL|O_RDWR, ...)
    ↓
系统调用: open("/dev/shm/vllm_shm_rank_0", ...)
    ↓
内核: 创建共享内存对象
    ↓
ftruncate(fd, 8388608)  // 8MB
    ↓
系统调用: ftruncate()
    ↓
内核: 设置共享内存大小
    ↓
mmap(NULL, 8388608, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_POPULATE, fd, 0)
    ↓
系统调用: mmap()
    ↓
内核:
  1. 在进程虚拟地址空间分配 8MB 地址范围
  2. 立即分配物理内存页（因为 MAP_POPULATE）
  3. 建立页表映射（虚拟地址 → 物理地址）
  4. 初始化内存为零
    ↓
返回虚拟地址指针
    ↓
C++ 代码使用这个指针
```

### 3. Pinned Memory（CUDA）

```
Python: torch.empty_strided(..., pin_memory=True)
    ↓
PyTorch C++: cudaHostAlloc()
    ↓
CUDA 驱动: cudaHostAlloc()
    ↓
系统调用: mmap() (通过 CUDA 驱动)
    ↓
内核: 分配物理内存页（pinned，不可交换到 swap）
    ↓
返回 pinned memory 指针
```

## 关键系统调用说明

### mmap() - 内存映射

**在 vLLM 中的使用**:
- `csrc/cpu/shm.cpp:301`: 共享内存映射
- PyTorch 大内存分配（通过 glibc）

**系统调用**:
```c
void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
```

**关键参数**:
- `MAP_ANONYMOUS`: 匿名映射（不关联文件）
- `MAP_SHARED`: 共享映射（多个进程可见）
- `MAP_POPULATE`: 立即分配物理内存（不延迟）

### brk() - 扩展堆

**在 vLLM 中的使用**:
- 通过 `malloc()` 间接使用
- PyTorch 小内存分配

**系统调用**:
```c
int brk(void *addr);
```

**工作原理**:
- 移动程序中断点（program break）
- 扩展进程的数据段（堆）
- 内核分配新的物理内存页

### shm_open() - POSIX 共享内存

**在 vLLM 中的使用**:
- `csrc/cpu/shm.cpp:283`: 创建共享内存对象

**系统调用**:
```c
int shm_open(const char *name, int oflag, mode_t mode);
```

**工作原理**:
- 在 `/dev/shm/` 创建共享内存对象
- 返回文件描述符
- 需要配合 `mmap()` 使用

## 代码位置速查

| 分配类型 | Python 代码 | C++ 代码 | 系统调用 |
|---------|------------|---------|---------|
| InputBatch CPU 张量 | `gpu_input_batch.py:119` | PyTorch C++ | `malloc()` → `brk()`/`mmap()` |
| 共享内存 | - | `csrc/cpu/shm.cpp:301` | `mmap(MAP_SHARED\|MAP_POPULATE)` |
| 模型权重 (pinned) | `model_loader/utils.py:148` | PyTorch C++ | `cudaHostAlloc()` → `mmap()` |
| C++ 小对象 | - | `csrc/cumem_allocator.cpp:308` | `malloc()` → `brk()` |

## 如何验证

### 使用 strace 跟踪系统调用

```bash
# 跟踪内存相关的系统调用
strace -e trace=mmap,munmap,brk,shm_open,shm_unlink \
  vllm serve --model Qwen3-8B ...
```

### 查看进程内存映射

```bash
# 查看虚拟内存映射
cat /proc/$(pgrep -f "vllm serve")/maps

# 查看物理内存使用
cat /proc/$(pgrep -f "vllm serve")/status | grep VmRSS
```
