# Zero-Copy Socket - 简化方案（基于参数判断）

## 核心思路

**不解析 endpoint 字符串，直接根据 `server_args` 参数判断是否使用零拷贝。**

## 决策逻辑

```python
def should_use_zerocopy(server_args) -> bool:
    """
    零拷贝仅在单机场景下有意义：
    1. 普通模式（无 DP）：总是单机
    2. DP 模式且 nnodes=1 且无 dist_init_addr：单机 DP
    3. DP 模式且 nnodes>1 或有 dist_init_addr：多机 DP
    """
    if not server_args.enable_dp_attention:
        return True  # 普通单机模式

    if server_args.nnodes == 1 and server_args.dist_init_addr is None:
        return True  # 单机 DP

    return False  # 多机 DP
```

**一行版本：**
```python
use_zerocopy = (not server_args.enable_dp_attention) or
               (server_args.nnodes == 1 and server_args.dist_init_addr is None)
```

## 对应关系

| 场景 | enable_dp_attention | nnodes | dist_init_addr | Endpoint | use_zerocopy |
|------|---------------------|--------|----------------|----------|--------------|
| 普通单机 | False | 1 | None | `ipc://...` | ✅ True |
| 单机 DP | True | 1 | None | `tcp://127.0.0.1:*` | ✅ True |
| 多机 DP | True | >1 | 远程地址 | `tcp://<remote>:*` | ❌ False |

## 集成方式

### 方式 1: 修改 PortArgs（推荐）

在 `server_args.py` 的 `PortArgs` 中添加标志：

```python
@dataclasses.dataclass
class PortArgs:
    tokenizer_ipc_name: str
    scheduler_input_ipc_name: str
    detokenizer_ipc_name: str
    nccl_port: int
    rpc_ipc_name: str
    metrics_ipc_name: str
    use_zerocopy: bool  # 新增字段
    tokenizer_worker_ipc_name: Optional[str] = None
```

然后在 `PortArgs.make()` 中设置：

```python
@staticmethod
def make(server_args: ServerArgs, dp_rank: Optional[int] = None, ...):
    # ... 现有逻辑 ...

    # 决定是否使用零拷贝
    use_zerocopy = should_use_zerocopy(server_args)

    return PortArgs(
        tokenizer_ipc_name=...,
        scheduler_input_ipc_name=...,
        detokenizer_ipc_name=...,
        nccl_port=nccl_port,
        rpc_ipc_name=...,
        metrics_ipc_name=...,
        use_zerocopy=use_zerocopy,  # 传递标志
        tokenizer_worker_ipc_name=...,
    )
```

使用时非常简单：

```python
# scheduler.py, tokenizer_manager.py, 等等
socket = get_zmq_socket(context, zmq.PULL, port_args.scheduler_input_ipc_name, True)
self.recv_from_tokenizer = ZeroCopySocket(socket, port_args.use_zerocopy)
```

### 方式 2: 修改 get_zmq_socket

在 `common.py` 的 `get_zmq_socket` 函数中添加参数：

```python
def get_zmq_socket(
    context: zmq.Context,
    socket_type: int,
    endpoint: str,
    is_bind: bool,
    server_args=None,  # 新增可选参数
) -> Union[zmq.Socket, ZeroCopySocket]:
    socket = context.socket(socket_type)
    config_socket(socket, socket_type)

    if is_bind:
        socket.bind(endpoint)
    else:
        socket.connect(endpoint)

    # 如果提供了 server_args，则包装为 ZeroCopySocket
    if server_args is not None:
        use_zerocopy = should_use_zerocopy(server_args)
        return ZeroCopySocket(socket, use_zerocopy)

    return socket
```

使用时：

```python
# 传递 server_args
self.recv_from_tokenizer = get_zmq_socket(
    context, zmq.PULL, port_args.scheduler_input_ipc_name, True, server_args
)
```

## 与端点解析方案对比

| 特性 | 参数判断方案 ✅ | 端点解析方案 |
|------|----------------|-------------|
| 代码复杂度 | **简单** (5行核心逻辑) | 复杂 (50+行解析) |
| 需要字符串解析 | ❌ 不需要 | ✅ 需要 (urlparse) |
| 边界情况 | ❌ 无 | ✅ 有 (tcp://*, IPv6, etc.) |
| 与设计契合度 | ✅ **直接映射** | 间接推断 |
| 可维护性 | ✅ **容易理解** | 需要理解解析逻辑 |
| 性能 | ✅ **更快** (无解析) | 需要解析字符串 |
| 依赖关系 | 需要 server_args | 独立工作 |

## 实现文件

完整实现见 `zero_copy_socket_simple.py`，包含：

1. **`TorchTensorPickler`**: 自定义 pickler 处理 torch tensor
2. **`torch_tensor_reducer`**: 将 tensor 移到 CPU
3. **`should_use_zerocopy()`**: 决策函数（核心）
4. **`ZeroCopySocket`**: Socket 包装类
5. **`create_zmq_socket()`**: 集成辅助函数

## 测试

运行测试：

```bash
python test_zerocopy_simple.py
```

输出：

```
Testing zero-copy decision logic based on server_args:
==========================================================================================
Status   Description                                   Result  Expected
==========================================================================================
✓ PASS   Normal single-node (no DP)                    True    True
✓ PASS   Single-node DP (nnodes=1, no dist_init_addr)  True    True
✓ PASS   Multi-node DP (nnodes=2)                      False   False
✓ PASS   Multi-node DP (dist_init_addr set)            False   False
==========================================================================================
✓ All tests passed!

Decision Logic:
  use_zerocopy = (not enable_dp_attention) or (nnodes == 1 and dist_init_addr is None)
```

## 迁移路径

### Step 1: 添加 should_use_zerocopy 函数

在 `common.py` 或新建 `zero_copy_socket.py` 中添加决策函数。

### Step 2: 修改 PortArgs（推荐）

在 `server_args.py` 中：
- 给 `PortArgs` 添加 `use_zerocopy: bool` 字段
- 在 `PortArgs.make()` 中调用 `should_use_zerocopy(server_args)` 并传递结果

### Step 3: 逐步替换 socket 创建

在各个 manager 中（scheduler.py, tokenizer_manager.py, detokenizer_manager.py）：

```python
# 原来
self.recv_from_tokenizer = get_zmq_socket(context, zmq.PULL, endpoint, True)

# 修改为
socket = get_zmq_socket(context, zmq.PULL, endpoint, True)
self.recv_from_tokenizer = ZeroCopySocket(socket, port_args.use_zerocopy)
```

### Step 4: 测试验证

分别测试三种场景：
1. 单机普通模式（最常见）
2. 单机 DP 模式
3. 多机 DP 模式

## 总结

**参数判断方案明显优于端点解析方案**，因为：

1. ✅ **更简单**：5 行核心逻辑 vs 50+ 行解析代码
2. ✅ **更直接**：直接映射设计意图，而不是反向推断
3. ✅ **更快**：无字符串解析开销
4. ✅ **更可靠**：无解析边界情况
5. ✅ **更易维护**：逻辑清晰，一目了然

推荐使用**方式 1（修改 PortArgs）**，因为：
- 集中式配置，清晰明了
- 使用时无需传递 server_args
- 与现有架构契合
