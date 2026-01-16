# 创建裁剪版 Qwen3VLMoe 模型用于快速验证

## 📋 问题背景

使用 HuggingFace 的 `AutoModelForCausalLM.from_config()` 创建 Qwen3VLMoeForConditionalGeneration 会遇到问题：
- ❌ 需要 flash_attn_2 依赖
- ❌ 不适用于多模态模型（需要用 AutoModelForVision2Seq）
- ❌ 可能加载不必要的代码依赖

## ✅ 解决方案

使用 SGLang 的配置和模型类来创建裁剪版模型，避免上述问题。

## 🚀 使用步骤

### 1. 准备原始模型目录

确保你有原始模型的 tokenizer 文件：
```bash
cd /path/to/original/qwen3vl-moe-model
ls tokenizer*  # 应该能看到 tokenizer.json 等文件
```

### 2. 运行创建脚本

```bash
# 在原始模型目录下运行
python /home/user/sglang/create_tiny_qwen3vl_moe.py

# 或者修改脚本中的路径
# original_model_path = "/path/to/original/model"
# tiny_model_path = "./tiny_qwen3vl_moe"
```

### 3. 测试模型加载

```bash
python /home/user/sglang/test_tiny_model.py
```

## 📊 裁剪对比

| 参数 | 原始模型 | 裁剪模型 | 说明 |
|------|---------|---------|------|
| Text layers | 24 | 2 | 减少层数 |
| Text hidden size | 2048 | 256 | 减少隐藏层维度 |
| MoE experts | 60 | 4 | 大幅减少专家数 |
| Vision layers | 27 | 4 | 减少视觉编码器层数 |
| Vision hidden size | 1152 | 256 | 减少视觉隐藏层维度 |
| **预计参数量** | ~30B | ~10M | 约 3000 倍缩减 |

## 💡 代码验证示例

```python
from sglang.srt.configs.qwen3_vl import Qwen3VLMoeConfig
from sglang.srt.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration

# 加载裁剪后的模型
config = Qwen3VLMoeConfig.from_pretrained("./tiny_qwen3vl_moe")
model = Qwen3VLMoeForConditionalGeneration(config, quant_config=None)

# 验证你的代码逻辑
# ... 你的测试代码 ...
```

## 🔧 自定义裁剪程度

如果需要更大/更小的模型，修改 `create_tiny_qwen3vl_moe.py` 中的参数：

```python
# 更激进的裁剪（超小模型，几百万参数）
text_config = Qwen3VLMoeTextConfig(
    hidden_size=128,           # 更小
    num_hidden_layers=1,       # 只用 1 层
    num_experts=2,             # 最少 2 个专家
    # ...
)

# 较温和的裁剪（中等模型，几十到几百 M 参数）
text_config = Qwen3VLMoeTextConfig(
    hidden_size=512,           # 稍大
    num_hidden_layers=4,       # 多几层
    num_experts=8,             # 多几个专家
    # ...
)
```

## ⚠️ 注意事项

1. **不用于实际推理**：裁剪后的模型使用随机权重，输出无意义
2. **仅用于代码验证**：测试数据流、形状、设备分配等逻辑
3. **保持 vocab_size**：必须与 tokenizer 匹配（151936）
4. **配置对齐**：text hidden_size 和 vision out_hidden_size 需要对齐

## 📚 相关文件

- `create_tiny_qwen3vl_moe.py` - 创建裁剪模型的脚本
- `test_tiny_model.py` - 测试模型加载的脚本
- `./tiny_qwen3vl_moe/` - 生成的裁剪模型目录

## 🐛 常见问题

### Q: 为什么不直接用 transformers 的 from_config?
A: 因为 Qwen3VLMoe 是多模态模型，而且有自定义代码依赖（如 flash_attn），直接用 transformers 会遇到依赖问题。SGLang 已经实现了兼容版本。

### Q: 可以用这个模型做实际推理吗？
A: 不可以，这个模型的权重是随机初始化的，输出没有意义。仅用于验证代码逻辑（形状、流程等）。

### Q: 如何进一步减小模型？
A: 修改配置中的 `hidden_size`、`num_hidden_layers`、`num_experts` 等参数，减小这些值。

### Q: 生成的模型文件有多大？
A: 配置文件很小（几 KB），空的 pytorch_model.bin 也很小。实际创建模型时才会占用内存（约几十 MB 到几百 MB）。
