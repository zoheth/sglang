#!/usr/bin/env python3
"""创建裁剪版 Qwen3VLMoeForConditionalGeneration 用于快速验证代码逻辑"""
import os
import shutil
import torch

# 导入 SGLang 配置和模型类
from sglang.srt.configs.qwen3_vl import (
    Qwen3VLMoeConfig,
    Qwen3VLMoeTextConfig,
    Qwen3VLMoeVisionConfig,
)

# 路径设置（可根据需要修改）
original_model_path = "./"  # 原始模型目录（用于复制 tokenizer 文件）
tiny_model_path = "./tiny_qwen3vl_moe"  # 裁剪后的存放目录

os.makedirs(tiny_model_path, exist_ok=True)

# 3. 创建裁剪后的文本配置 (Text Config)
text_config = Qwen3VLMoeTextConfig(
    vocab_size=151936,          # 保持原样（tokenizer需要）
    hidden_size=256,            # 大幅裁剪: 2048 -> 256
    intermediate_size=512,      # 大幅裁剪: 5632 -> 512
    num_hidden_layers=2,        # 大幅裁剪: 24 -> 2
    num_attention_heads=4,      # 大幅裁剪: 16 -> 4
    num_key_value_heads=4,      # 大幅裁剪: 16 -> 4
    num_experts=4,              # 大幅裁剪: 60 -> 4
    num_experts_per_tok=2,      # 裁剪: 4 -> 2
    moe_intermediate_size=256,  # 大幅裁剪: 1408 -> 256
    decoder_sparse_step=1,      # 保持原样
    mlp_only_layers=[],         # 保持原样
    max_position_embeddings=4096,  # 裁剪: 128000 -> 4096
    rms_norm_eps=1e-6,
    rope_theta=5000000.0,
    attention_bias=False,
    attention_dropout=0.0,
    norm_topk_prob=True,
    hidden_act="silu",
)

# 4. 创建裁剪后的视觉配置 (Vision Config)
vision_config = Qwen3VLMoeVisionConfig(
    depth=4,                    # 大幅裁剪: 27 -> 4
    hidden_size=256,            # 大幅裁剪: 1152 -> 256
    intermediate_size=512,      # 大幅裁剪: 4304 -> 512
    num_heads=4,                # 大幅裁剪: 16 -> 4
    out_hidden_size=256,        # 与text hidden_size对齐
    patch_size=16,
    spatial_merge_size=2,
    temporal_patch_size=2,
    in_channels=3,
    num_position_embeddings=1024,  # 裁剪: 2304 -> 1024
    deepstack_visual_indexes=[2],   # 裁剪: [8,16,24] -> [2]
    hidden_act="gelu_pytorch_tanh",
)

# 5. 创建完整的模型配置
config = Qwen3VLMoeConfig(
    text_config=text_config,
    vision_config=vision_config,
    image_token_id=151655,
    video_token_id=151656,
    vision_start_token_id=151652,
    vision_end_token_id=151653,
    tie_word_embeddings=False,
)

# 6. 保存配置文件
config.save_pretrained(tiny_model_path)
print(f"✅ Config saved to {tiny_model_path}/config.json")

# 7. 复制必要的 Tokenizer 文件
print("📋 Copying tokenizer files...")
files_to_copy = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.json",
    "generation_config.json",
]

for file in files_to_copy:
    src = os.path.join(original_model_path, file)
    if os.path.exists(src):
        shutil.copy(src, tiny_model_path)
        print(f"  ✓ {file}")
    else:
        print(f"  ⚠ {file} not found, skipping")

# 8. 创建空的权重文件（占位）
print("\n💾 Creating empty weight file...")
empty_state_dict = {}
torch.save(empty_state_dict, os.path.join(tiny_model_path, "pytorch_model.bin"))
print(f"  ✓ pytorch_model.bin (empty placeholder)")

# 9. 打印信息
print("\n" + "="*60)
print("🎉 Tiny model config created successfully!")
print("="*60)
print(f"📁 Location: {tiny_model_path}")
print(f"\n📝 Config summary:")
print(f"   Text layers: {text_config.num_hidden_layers}")
print(f"   Text hidden size: {text_config.hidden_size}")
print(f"   MoE experts: {text_config.num_experts} (use {text_config.num_experts_per_tok} per token)")
print(f"   Vision layers: {vision_config.depth}")
print(f"   Vision hidden size: {vision_config.hidden_size}")

print("\n💡 Quick test usage:")
print(f"   # 方式1: 直接用 SGLang 加载（会创建随机权重）")
print(f"   from sglang.srt.configs.qwen3_vl import Qwen3VLMoeConfig")
print(f"   from sglang.srt.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration")
print(f"   config = Qwen3VLMoeConfig.from_pretrained('{tiny_model_path}')")
print(f"   model = Qwen3VLMoeForConditionalGeneration(config, quant_config=None)")
print()
print(f"   # 方式2: 通过 SGLang 服务器启动")
print(f"   python -m sglang.launch_server --model-path {tiny_model_path} --port 30000")

