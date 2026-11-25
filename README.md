# Irodori TTS Server (Genie-Compatible CPU Inference Service)

与 `genie-server` HTTP 接口完全兼容的独立纯 CPU TTS 推理服务，底层使用 **Irodori-TTS MeanFlow**（`clarity_fix` 蒸馏模型）+ 29 角色 `49 KB` speaker embeddings。

部署与运维说明见 [DEPLOYMENT.md](DEPLOYMENT.md)。

## 核心特性

- **接口 100% 兼容 `genie-server`**：`POST /tasks`、`GET /tasks/{task_id}`、`GET /health`、每 IP 每日限流、503 队列保护、WAV + Opus(48k) OGG 双格式上传 R2、Cloudflare Worker 代理零改动切换。
- **音色策略（`ref_wav` 优先，`ref_embed` 兜底）**：
  - 请求传入非空 `reference_audio_id` 时，从 `REFERENCE_RESOURCE_SERVER` 下载 `.m4a` 并转为 48 kHz 单声道 WAV（本地缓存）进行 `ref_wav` 克隆；
  - 若 `reference_audio_id` 为空，或下载/推理失败（`FALLBACK_TO_EMBED_ON_REF_ERROR=true`），自动回退到该角色的 49 KB `*.speaker.safetensors`；
  - `character_name` 自动兼容英文 ID（`hazuki`）、日文短名（`はづき`）、日文全名（`七草はづき`）及中文名（`七草叶月`）。
- **推理默认配置**：
  - 默认启动即加载单份模型权重与 DACVAE 编解码器并执行一次 1 步预热推理（`PRELOAD_ON_STARTUP=true`），就绪后立即以稳定延迟响应首请求；
  - 默认 `6` 步（`DEFAULT_NUM_STEPS=6`），不做长文本自动切割；
  - 默认关闭 SilentCipher 水印（`ENABLE_WATERMARK=false`），保留原生 48 kHz 波形并省去 CPU 水印耗时；
  - `MODEL_PRECISION=auto`：在支持 `amx_bf16` / `avx512_bf16` 的 Xeon/EPYC CPU 上自动启用安全 BF16 推理（保留 RoPE 复数缓存），无硬件 BF16 时自动使用 FP32。

## 快速启动

1. 安装依赖（纯 CPU 生产环境）：

   ```bash
   uv sync --extra cpu
   ```

2. 配置环境变量：

   ```bash
   cp .env.example .env
   ```

   填写 `MODEL_CHECKPOINT`、`SPEAKERS_DIR`、R2 凭据和 `BASE_STATIC_URL`。

3. 启动服务：

   ```bash
   uv run uvicorn server:app --host 127.0.0.1 --port 12451 --workers 1
   ```

## API 说明

- `POST /tasks`

  完全兼容 `genie-server` 请求体：

  ```json
  {
    "character_name": "はづき",
    "reference_audio_id": "special_communications/490300191/4903001910010",
    "reference_audio_text": "お誕生日おめでとうございます、プロデューサーさん♪",
    "text": "要合成的日文台词"
  }
  ```

  同时支持向后兼容的可选扩展字段（不传均走默认值）：
  - `reference_audio_id` / `reference_audio_text` 可省略或为空字符串（直接走 `ref_embed` 兜底）；
  - `caption`：可选日文情绪/风格描述；
  - `num_steps`：采样步数（默认 `6`，范围 `1..MAX_NUM_STEPS`）；
  - `seed`：随机种子；
  - `duration_scale`：语速/时长缩放系数（默认 `1.0`）。

- `GET /tasks/{task_id}`：查询任务状态（`pending` / `running` / `completed` / `failed`，不存在返回 `404`）。
- `GET /health`：就绪探针，返回 `{"status": "ok"}`。
