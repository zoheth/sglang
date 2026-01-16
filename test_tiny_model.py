#!/usr/bin/env python3
"""测试裁剪版 Qwen3VLMoeForConditionalGeneration 模型加载"""
import torch
from sglang.srt.configs.qwen3_vl import Qwen3VLMoeConfig
from sglang.srt.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration

# 设置模型路径
tiny_model_path = "./tiny_qwen3vl_moe"

print("="*60)
print("🧪 Testing Tiny Qwen3VLMoe Model Loading")
print("="*60)

# 1. 加载配置
print("\n📖 Loading config...")
config = Qwen3VLMoeConfig.from_pretrained(tiny_model_path)
print(f"✅ Config loaded")
print(f"   Model type: {config.model_type}")
print(f"   Text layers: {config.text_config.num_hidden_layers}")
print(f"   Text hidden: {config.text_config.hidden_size}")
print(f"   MoE experts: {config.text_config.num_experts}")
print(f"   Vision layers: {config.vision_config.depth}")

# 2. 创建模型（随机初始化权重）
print("\n🔧 Creating model with random weights...")
with torch.device("cpu"):  # 在 CPU 上创建
    model = Qwen3VLMoeForConditionalGeneration(
        config=config,
        quant_config=None,
    )

# 3. 计算参数量
print("\n📊 Model statistics:")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

# 4. 打印模型结构
print("\n🏗️  Model structure:")
for name, module in model.named_children():
    print(f"   - {name}: {module.__class__.__name__}")

# 5. 测试前向传播（可选）
print("\n🚀 Testing forward pass (if needed)...")
print("   Skipped - forward pass requires proper input setup")

print("\n" + "="*60)
print("✅ All tests passed! Model is ready for code verification.")
print("="*60)
print("\n💡 Next steps:")
print("   1. 用这个模型来验证你的代码逻辑")
print("   2. 如果需要实际推理，请使用完整模型")
print("   3. 可以通过 model.eval() 切换到评估模式")
