# SGLang ZMQ 通信场景分析

## 关键问题

**ZMQ 通信的组件是：Tokenizer Manager ↔ Scheduler ↔ Detokenizer Manager**

这些组件在什么情况下会跨节点通信？

## 部署场景分析

### 场景 1: 单机（无 DP）
```
配置: nnodes=1, tp_size=任意, enable_dp_attention=False
架构:
  [Node 0]
    HTTP Server → Tokenizer Manager → Scheduler → Detokenizer Manager
```
- ZMQ endpoint: `ipc://...`
- 零拷贝: ✅ YES

### 场景 2: 多机 TP（无 DP）
```
配置: nnodes>1, tp_size>1, enable_dp_attention=False
架构:
  [Node 0 - 用户请求入口]
    HTTP Server → Tokenizer Manager → Scheduler (TP rank 0) → Detokenizer Manager

  [Node 1+ - TP workers]
    Scheduler (TP rank 1+) - 通过 NCCL 与 Node 0 通信
```

**关键问题：其他节点有没有 Tokenizer/Detokenizer？**

让我检查代码...
