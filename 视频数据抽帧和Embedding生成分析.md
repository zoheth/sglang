# 视频数据抽帧和Embedding生成完整分析

## 📌 概览

视频数据在SGLang中的处理分为两个关键阶段：
1. **抽帧（Frame Sampling）**：从原始视频中选择若干关键帧
2. **Embedding生成**：将抽取的帧转换为模型可用的特征向量

---

## 🎬 一、视频抽帧（Frame Sampling）

### 1.1 核心抽帧函数

#### ① `sample_video_frames()` - 基础抽帧函数
**文件**: `python/sglang/srt/utils/common.py:963-978`

```python
def sample_video_frames(
    video: "VideoReader", *, desired_fps: int, max_frames: int
) -> list[int]:
    """使用均匀线性采样策略抽取视频帧"""
    total_frames = len(video)
    duration = total_frames / video.get_avg_fps()
    fps = min(desired_fps, video.get_avg_fps())

    # 计算要采样的帧数
    num_frames = math.floor(duration * fps)
    num_frames = min(max_frames, num_frames, total_frames)
    num_frames = max(1, num_frames)  # 至少1帧

    if num_frames == total_frames:
        return list(range(total_frames))
    else:
        # 关键：使用np.linspace进行均匀采样
        return np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
```

**采样策略**：均匀线性采样（np.linspace）
- 在视频总帧数范围内均匀分布采样点
- 示例：100帧视频，采样10帧 → 索引[0, 11, 22, 33, 44, 55, 66, 77, 88, 99]

---

#### ② `encode_video()` - 带帧数限制的抽帧
**文件**: `python/sglang/srt/utils/common.py:981-1005`

```python
def encode_video(video_path, frame_count_limit=None):
    """加载视频并抽帧，支持帧数限制"""
    from decord import VideoReader, cpu

    def uniform_sample(l, n):
        """内置均匀采样函数"""
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)
    frame_indices = [i for i in range(0, len(vr), sample_fps)]

    # 如果超过限制，使用uniform_sample二次采样
    if frame_count_limit is not None and len(frame_indices) > frame_count_limit:
        frame_indices = uniform_sample(frame_indices, frame_count_limit)

    frames = vr.get_batch(frame_indices).asnumpy()
    frames = [Image.fromarray(v.astype("uint8")) for v in frames]
    return frames
```

**特点**：
- 两阶段采样：先按FPS采样，再按帧数限制采样
- 返回PIL.Image格式的帧列表

---

#### ③ `preprocess_video()` - Qwen-VL模型专用
**文件**: `python/sglang/srt/multimodal/processors/qwen_vl.py:145-218`

```python
async def preprocess_video(
    vr,
    image_factor: int = IMAGE_FACTOR,
    video_config: dict = {},
) -> torch.Tensor:
    """完整的视频预处理流程，包括抽帧和分辨率调整"""

    # 1. 计算要采样的帧数
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    nframes = smart_nframes(
        video_config, total_frames=total_frames, video_fps=video_fps
    )

    # 2. 均匀采样帧索引
    idx = np.linspace(0, total_frames - 1, num=nframes, dtype=np.int64)
    idx = np.unique(idx)

    # 3. 批量获取帧数据
    video_np = vr.get_batch(idx).asnumpy()
    video = torch.from_numpy(video_np).pin_memory()
    video = video.permute(0, 3, 1, 2)  # THWC -> TCHW

    # 4. 智能调整分辨率
    nframes, _, height, width = video.shape
    min_pixels = video_config.get("min_pixels", VIDEO_MIN_PIXELS)
    max_pixels = ...  # 复杂计算逻辑

    resized_height, resized_width = smart_resize(
        height, width,
        factor=image_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    # 5. 使用torchvision进行双线性插值
    video = torchvision.transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BILINEAR,
    )

    # 6. 返回视频和元数据
    video_metadata = {
        "fps": video_fps,
        "duration": total_frames / video_fps,
        "total_num_frames": total_frames,
        "frames_indices": idx,  # 记录采样的帧索引
        "video_backend": "torchvision",
    }
    return video, video_metadata
```

**完整流程**：
```
VideoReader → 计算nframes → linspace采样 → 批量读取 → 格式转换 → 分辨率调整 → 返回Tensor
```

---

### 1.2 帧数计算逻辑

#### `smart_nframes()` - 智能帧数计算
**文件**: `python/sglang/srt/multimodal/processors/qwen_vl.py:96-141`

```python
def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """智能计算用于模型输入的视频帧数"""

    if "nframes" in ele:
        # 直接指定帧数
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        # 基于FPS计算
        fps = ele.get("fps", FPS)  # 默认2.0
        min_frames = ceil_by_factor(
            ele.get("min_frames", FPS_MIN_FRAMES),  # 默认4
            FRAME_FACTOR
        )
        max_frames = floor_by_factor(
            ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)),  # 默认768
            FRAME_FACTOR
        )

        # 核心公式
        nframes = total_frames / video_fps * fps

        # 应用约束
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)

    return nframes
```

**计算公式**：
```
nframes = (total_frames / video_fps) * desired_fps
约束条件: min_frames ≤ nframes ≤ min(max_frames, total_frames)
帧数因子: nframes 必须是 FRAME_FACTOR (2) 的倍数
```

**示例**：
- 原视频：300帧，30fps，时长10秒
- 目标FPS：2.0
- 计算：`nframes = 300 / 30 * 2 = 20帧`
- 采样结果：从300帧中均匀抽取20帧

---

### 1.3 配置参数

**文件**: `python/sglang/srt/multimodal/processors/qwen_vl.py:26-45`

```python
# 图像相关
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28 = 3,136
MAX_PIXELS = 4096 * 28 * 28 = 3,211,264

# 视频像素约束
VIDEO_TOTAL_PIXELS = 128000 * 28 * 28 * 0.9 = 89,446,400
VIDEO_MIN_PIXELS = 128 * 28 * 28 = 102,400
VIDEO_MAX_PIXELS = 768 * 28 * 28 = 614,400

# 帧数相关
FRAME_FACTOR = 2  # 帧数必须是2的倍数
FPS = 2.0         # 默认目标FPS
FPS_MIN_FRAMES = 4   # 最小帧数
FPS_MAX_FRAMES = 768 # 最大帧数
```

---

### 1.4 抽帧发生的时机

#### Pipeline中的位置

抽帧主要发生在以下两个地方：

**① 文本编码阶段（LLM推理路径）**
- 文件：`python/sglang/srt/multimodal/processors/qwen_vl.py`
- 时机：在tokenization之前
- 用途：为LLM准备视频输入

**② 输入验证阶段（视频生成路径）**
- 文件：`python/sglang/multimodal_gen/runtime/pipelines_core/stages/input_validation.py:209-221`
- 时机：在管道开始时
- 用途：加载condition视频/图像

```python
# 输入验证阶段
if batch.image_path is not None:
    if batch.image_path.endswith(".mp4"):
        image = load_video(batch.image_path)[0]  # 只取第一帧
    else:
        image = load_image(batch.image_path)
    batch.condition_image = image
```

**注意**：在视频生成路径中，如果是条件视频生成（I2V），通常只使用视频的第一帧作为条件图像。

---

## 🔮 二、Embedding生成

视频/图像的embedding生成分为两种类型：
1. **视觉特征Embedding**（image_embeds）- 使用视觉编码器
2. **隐空间Embedding**（image_latent）- 使用VAE编码器

---

### 2.1 视觉特征Embedding - ImageEncodingStage

#### 完整实现
**文件**: `python/sglang/multimodal_gen/runtime/pipelines_core/stages/image_encoding.py:40-177`

```python
class ImageEncodingStage(PipelineStage):
    """使用视觉编码器生成image embeddings"""

    def __init__(
        self,
        image_processor,     # 图像预处理器
        image_encoder=None,  # 视觉编码器（如CLIP）
        text_encoder=None,   # 文本编码器（某些模型使用）
    ):
        self.image_processor = image_processor
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """将condition_image编码为image_embeds"""

        # 1. 检查是否有条件图像
        if batch.condition_image is None:
            return batch

        # 2. 移动到GPU
        cuda_device = get_local_torch_device()
        self.move_to_device(cuda_device)

        image = batch.condition_image

        # 3. 图像预处理（PIL/Tensor → 标准化Tensor）
        image_processor_kwargs = (
            server_args.pipeline_config.prepare_image_processor_kwargs(batch)
        )
        image_inputs = self.image_processor(
            images=image,
            return_tensors="pt",
            **image_processor_kwargs
        ).to(cuda_device)

        # 4. 视觉编码器推理
        if self.image_encoder:
            with set_forward_context(current_timestep=0, attn_metadata=None):
                outputs = self.image_encoder(
                    **image_inputs,
                    **server_args.pipeline_config.image_encoder_extra_args,
                )
                # 5. 后处理提取embeddings
                image_embeds = server_args.pipeline_config.postprocess_image(outputs)

            # 6. 存储到batch
            batch.image_embeds.append(image_embeds)

        # 7. 移回CPU释放GPU内存
        self.move_to_device("cpu")
        return batch
```

#### 处理流程
```
batch.condition_image (PIL.Image)
    ↓
image_processor() → 预处理（resize, normalize, totensor）
    ↓
image_inputs (Dict[str, Tensor])
    例如: {"pixel_values": Tensor[1, 3, 224, 224]}
    ↓
image_encoder.forward() → CLIP/其他视觉编码器推理
    ↓
outputs (BaseEncoderOutput)
    包含: last_hidden_state, pooler_output等
    ↓
postprocess_image() → 提取所需的embedding部分
    通常: outputs.last_hidden_state
    ↓
batch.image_embeds (List[Tensor])
    形状示例: [1, 257, 1024]  # [batch, tokens, hidden_dim]
```

---

### 2.2 隐空间Embedding - ImageVAEEncodingStage

#### 完整实现
**文件**: `python/sglang/multimodal_gen/runtime/pipelines_core/stages/image_encoding.py:180-301`

```python
class ImageVAEEncodingStage(PipelineStage):
    """使用VAE编码器将图像编码到隐空间"""

    def __init__(self, vae: ParallelTiledVAE):
        self.vae = vae

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """将condition_image编码为image_latent"""

        # 1. 检查条件图像
        if batch.condition_image is None:
            return batch

        num_frames = batch.num_frames
        self.vae = self.vae.to(get_local_torch_device())

        # 2. 预处理：PIL → Tensor，归一化到[-1, 1]
        image = batch.condition_image
        image = self.preprocess(image).to(
            get_local_torch_device(),
            dtype=torch.float32
        )
        # image shape: [B, C, H, W]

        # 3. 添加时间维度: [B, C, H, W] → [B, C, 1, H, W]
        image = image.unsqueeze(2)

        # 4. 处理多帧情况
        if num_frames == 1:
            video_condition = image
        else:
            # 首帧 + 零填充帧
            video_condition = torch.cat(
                [
                    image,  # 第一帧
                    image.new_zeros(  # 补充的零帧
                        image.shape[0],
                        image.shape[1],
                        num_frames - 1,  # 补充 num_frames-1 帧
                        image.shape[3],
                        image.shape[4],
                    ),
                ],
                dim=2,
            )
        # video_condition shape: [B, C, T, H, W]

        # 5. VAE编码（核心步骤）
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        with torch.autocast(device_type="cuda", dtype=vae_dtype):
            if server_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()  # 启用tiling减少内存

            # VAE编码到隐空间
            encoder_output: DiagonalGaussianDistribution = self.vae.encode(
                video_condition
            )

        # 6. 从高斯分布采样
        generator = batch.generator
        sample_mode = server_args.pipeline_config.vae_config.encode_sample_mode()
        latent_condition = self.retrieve_latents(
            encoder_output,
            generator,
            sample_mode=sample_mode
        )

        # 7. 缩放和平移（VAE标准化）
        scaling_factor, shift_factor = (
            server_args.pipeline_config.get_decode_scale_and_shift(
                device=latent_condition.device,
                dtype=latent_condition.dtype,
                vae=self.vae,
            )
        )
        latent_condition -= shift_factor
        latent_condition = latent_condition * scaling_factor

        # 8. 后处理（添加mask等）
        batch.image_latent = server_args.pipeline_config.postprocess_image_latent(
            latent_condition, batch
        )

        self.vae.to("cpu")
        return batch
```

#### VAE编码详细流程
```
condition_image (PIL.Image, 例如 1080x720)
    ↓
preprocess() → PIL → NumPy → Tensor, 归一化[-1, 1]
    ↓
unsqueeze(2) → 添加时间维度
    ↓
Tensor shape: [1, 3, 1, 1080, 720]
    ↓
如果num_frames > 1:
    Tensor shape: [1, 3, T, 1080, 720]  (补零帧)
    ↓
vae.encode() → VAE编码器推理
    ├─ spatial_compression_ratio: 8
    ├─ temporal_compression_ratio: 4
    └─ latent_channels: 16
    ↓
DiagonalGaussianDistribution (mean, logvar)
    ↓
sample() → 从高斯分布采样
    ↓
latent_condition shape: [1, 16, T/4, 1080/8, 720/8]
                      = [1, 16, T/4, 135, 90]
    ↓
scaling & shifting → VAE标准化
    latent = (latent - shift) * scale
    ↓
postprocess_image_latent() → 添加mask等
    ↓
batch.image_latent shape: [1, 17, T/4, 135, 90]
                         (16通道latent + 1通道mask)
```

---

### 2.3 preprocess() 方法详解

**文件**: `python/sglang/multimodal_gen/runtime/pipelines_core/stages/image_encoding.py:316-331`

```python
def preprocess(
    self,
    image: torch.Tensor | PIL.Image.Image,
) -> torch.Tensor:
    """将PIL Image或Tensor转换为VAE输入格式"""

    if isinstance(image, PIL.Image.Image):
        # PIL → NumPy
        image = pil_to_numpy(image)
        # 转换为 [0, 1] 范围的 float32

        # NumPy → PyTorch
        image = numpy_to_pt(image)
        # 转置 [H, W, C] → [C, H, W]

    # 归一化到 [-1, 1]
    do_normalize = True
    if image.min() < 0:
        do_normalize = False

    if do_normalize:
        image = normalize(image)  # image * 2 - 1

    return image
```

**辅助函数**：
```python
# vision_utils.py
def pil_to_numpy(image: PIL.Image) -> np.ndarray:
    """PIL.Image → NumPy [0, 1]"""
    return np.array(image).astype(np.float32) / 255.0

def numpy_to_pt(images: np.ndarray) -> torch.Tensor:
    """NumPy → PyTorch, [H,W,C] → [C,H,W]"""
    return torch.from_numpy(images.transpose(0, 3, 1, 2))

def normalize(images) -> torch.Tensor:
    """[0, 1] → [-1, 1]"""
    return 2.0 * images - 1.0
```

---

### 2.4 视觉编码器实现 - CLIPVisionModel

**文件**: `python/sglang/multimodal_gen/runtime/models/encoders/clip.py:658-686`

```python
class CLIPVisionModel(ImageEncoder):
    """CLIP视觉编码器"""

    def forward(
        self,
        pixel_values: torch.Tensor,
        feature_sample_layers: list[int] | None = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        # 调用内部vision_model
        base_encoder_output = self.vision_model(
            pixel_values,
            output_hidden_states=output_hidden_states,
            feature_sample_layers=feature_sample_layers,
        )
        return base_encoder_output
```

**CLIPVisionEmbeddings**（文件同上，第43-84行）:
```python
def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
    """将图像转换为patch embeddings"""
    batch_size = pixel_values.shape[0]
    target_dtype = self.patch_embedding.weight.dtype

    # 1. Patch Embedding（卷积实现）
    patch_embeds = self.patch_embedding(
        pixel_values.to(dtype=target_dtype)
    )  # shape: [B, hidden_size, grid, grid]
    patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
    # shape: [B, num_patches, hidden_size]

    # 2. 添加CLS token
    class_embeds = self.class_embedding.expand(batch_size, 1, -1)
    embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
    # shape: [B, 1 + num_patches, hidden_size]

    # 3. 添加位置编码
    embeddings = embeddings + self.position_embedding(self.position_ids)

    return embeddings
```

---

### 2.5 VAE编码器实现 - ParallelTiledVAE

**文件**: `python/sglang/multimodal_gen/runtime/models/vaes/common.py:76-92`

```python
class ParallelTiledVAE:
    """支持tiling的VAE编码器，处理大分辨率视频"""

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        """
        编码输入视频/图像到隐空间
        Args:
            x: shape [B, C, T, H, W]
        Returns:
            DiagonalGaussianDistribution (latent分布)
        """
        batch_size, num_channels, num_frames, height, width = x.shape
        latent_num_frames = (
            (num_frames - 1) // self.temporal_compression_ratio + 1
        )

        # 选择编码策略
        if (
            self.use_tiling
            and self.use_temporal_tiling
            and num_frames > self.tile_sample_min_num_frames
        ):
            # 时空tiling编码（处理长视频）
            latents = self.tiled_encode(x)[:, :, :latent_num_frames]
        elif self.use_tiling and (
            width > self.tile_sample_min_width
            or height > self.tile_sample_min_height
        ):
            # 空间tiling编码（处理大分辨率）
            latents = self.spatial_tiled_encode(x)[:, :, :latent_num_frames]
        else:
            # 直接编码
            latents = self._encode(x)[:, :, :latent_num_frames]

        return DiagonalGaussianDistribution(latents)
```

---

## 📊 三、数据流总结

### 3.1 完整Pipeline流程

```
HTTP Request (视频文件上传)
    ↓
保存到: outputs/uploads/{request_id}_video.mp4
    ↓
batch.image_path = "outputs/uploads/{request_id}_video.mp4"
    ↓
═══════════════════════════════════════════════════
Pipeline 开始执行
═══════════════════════════════════════════════════
    ↓
【1. InputValidationStage】
    ├─ load_video(image_path)
    │   └─ 使用Decord加载视频
    ├─ 取第一帧作为条件图像
    └─ batch.condition_image = PIL.Image
    ↓
【2. ImageEncodingStage】 ← 视觉特征Embedding
    ├─ image_processor(condition_image)
    │   └─ 预处理（resize, normalize）
    ├─ image_encoder.forward()
    │   └─ CLIP视觉编码器推理
    └─ batch.image_embeds = Tensor[1, 257, 1024]
    ↓
【3. ImageVAEEncodingStage】 ← 隐空间Embedding
    ├─ preprocess(condition_image)
    │   └─ PIL → Tensor, 归一化到[-1, 1]
    ├─ 添加时间维度和零帧
    │   └─ [B,C,H,W] → [B,C,T,H,W]
    ├─ vae.encode()
    │   ├─ 空间压缩: 1080x720 → 135x90 (÷8)
    │   ├─ 时间压缩: T → T÷4
    │   └─ 通道扩展: 3 → 16
    ├─ sample from Gaussian
    └─ batch.image_latent = Tensor[1, 17, T/4, 135, 90]
    ↓
【4-7. 其他阶段】
    └─ TextEncoding, LatentPreparation, Denoising, Decoding...
```

### 3.2 两种Embedding的用途

| 类型 | 生成阶段 | 形状示例 | 用途 |
|------|---------|---------|------|
| **image_embeds** | ImageEncodingStage | [1, 257, 1024] | 作为cross-attention的condition输入DiT |
| **image_latent** | ImageVAEEncodingStage | [1, 17, T/4, 135, 90] | 与噪声latent混合，控制生成内容 |

---

## 📁 四、关键文件位置汇总

### 抽帧相关
| 功能 | 文件路径 | 行号 |
|------|---------|------|
| 基础抽帧函数 | `python/sglang/srt/utils/common.py` | 963-978 |
| 带限制抽帧 | `python/sglang/srt/utils/common.py` | 981-1005 |
| Qwen预处理 | `python/sglang/srt/multimodal/processors/qwen_vl.py` | 145-218 |
| 帧数计算 | `python/sglang/srt/multimodal/processors/qwen_vl.py` | 96-141 |
| 配置常量 | `python/sglang/srt/multimodal/processors/qwen_vl.py` | 26-45 |

### Embedding生成相关
| 功能 | 文件路径 | 行号 |
|------|---------|------|
| 视觉特征Stage | `python/sglang/multimodal_gen/runtime/pipelines_core/stages/image_encoding.py` | 40-177 |
| VAE隐空间Stage | `python/sglang/multimodal_gen/runtime/pipelines_core/stages/image_encoding.py` | 180-301 |
| CLIP编码器 | `python/sglang/multimodal_gen/runtime/models/encoders/clip.py` | 658-686 |
| VAE编码器 | `python/sglang/multimodal_gen/runtime/models/vaes/common.py` | 76-92 |
| 视觉工具函数 | `python/sglang/multimodal_gen/runtime/models/vision_utils.py` | 全文件 |
| 输入验证Stage | `python/sglang/multimodal_gen/runtime/pipelines_core/stages/input_validation.py` | 162-233 |

---

## 🔍 五、关键技术细节

### 5.1 抽帧策略对比

| 方法 | 策略 | 优点 | 缺点 |
|------|------|------|------|
| `sample_video_frames` | 均匀线性采样 | 简单高效，覆盖全视频 | 可能错过关键帧 |
| `encode_video` | 两阶段采样 | 支持帧数限制 | 稍微复杂 |
| `preprocess_video` | 智能参数计算 | 考虑像素约束 | 计算开销较大 |

### 5.2 Embedding维度对比

**示例配置**（1080x720视频，120帧）:

```python
# 原始视频
输入: 1080 x 720 x 120 frames x 3 channels

# 抽帧后（FPS=2.0, 原视频30fps）
抽帧: 8 frames (120/30*2)

# 视觉特征Embedding (CLIP)
image_embeds: [1, 257, 1024]
  - 1: batch size
  - 257: 1 CLS token + 256 patch tokens (16x16 patches)
  - 1024: hidden dimension

# VAE隐空间Embedding
image_latent: [1, 17, 30, 135, 90]
  - 1: batch size
  - 17: 16 latent channels + 1 mask channel
  - 30: 120 frames / 4 (temporal compression)
  - 135: 1080 / 8 (spatial compression)
  - 90: 720 / 8 (spatial compression)
```

### 5.3 内存优化技术

**VAE Tiling**:
- 用途：处理超大分辨率视频
- 策略：将视频分割为小块，分别编码后拼接
- 配置：`server_args.pipeline_config.vae_tiling = True`

**Mixed Precision**:
- 用途：减少显存占用，加速推理
- 配置：`vae_precision = "bf16"` 或 `"fp16"`
- 实现：使用 `torch.autocast`

---

## 🎯 六、常见问题

### Q1: 为什么视频生成只使用第一帧？
**A**: 在条件视频生成（I2V）中，通常使用参考视频的第一帧作为起始帧，生成后续动作。完整的时序信息通过prompt或其他机制提供。

### Q2: 如何控制抽帧的密度？
**A**: 通过以下参数：
- `fps`: 目标采样FPS（默认2.0）
- `min_frames`: 最小帧数（默认4）
- `max_frames`: 最大帧数（默认768）

### Q3: image_embeds 和 image_latent 的区别？
**A**:
- `image_embeds`: 高层语义特征，用于cross-attention引导生成
- `image_latent`: 像素级隐表示，直接参与diffusion过程

### Q4: 如何支持更长的视频？
**A**:
1. 启用VAE tiling: `vae_tiling=True`
2. 调整最大帧数: `max_frames=1024`
3. 减少采样FPS: `fps=1.0`

---

## 📚 七、参考配置

### 默认配置（Qwen-VL）
```python
VIDEO_CONFIG = {
    "fps": 2.0,
    "min_frames": 4,
    "max_frames": 768,
    "min_pixels": 128 * 28 * 28,
    "max_pixels": 768 * 28 * 28,
    "total_pixels": 128000 * 28 * 28 * 0.9,
}

VAE_CONFIG = {
    "spatial_compression_ratio": 8,
    "temporal_compression_ratio": 4,
    "latent_channels": 16,
    "vae_tiling": True,
    "vae_precision": "bf16",
}
```

---

## 总结

1. **抽帧发生在加载阶段**，使用 `np.linspace` 均匀采样策略
2. **Embedding生成分为两个阶段**：
   - ImageEncodingStage：生成高层视觉特征（image_embeds）
   - ImageVAEEncodingStage：生成像素级隐表示（image_latent）
3. **两种embedding协同工作**：
   - image_embeds 通过 cross-attention 引导生成方向
   - image_latent 通过像素级约束控制生成内容
4. **优化技术**：
   - VAE tiling 处理大分辨率
   - Mixed precision 减少显存
   - 智能参数计算平衡质量和性能
