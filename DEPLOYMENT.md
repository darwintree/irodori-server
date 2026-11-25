# Irodori TTS Server 部署指南（纯 CPU 节点）

本服务与 `genie-server` 接口完全兼容，可直接复用现有的 Cloudflare Worker + Workers VPC + Cloudflare Tunnel + R2 架构（监听 `127.0.0.1:12451`）。

```text
客户端
  -> Cloudflare Worker
       -> /audio/wav/*、/audio/ogg/* -> R2
       -> 其他请求 -> Workers VPC Service
                      -> Cloudflare Tunnel
                      -> 127.0.0.1:12451 Irodori TTS API (CPU)
```

## 1. 系统依赖（Debian 13 / Ubuntu）

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl ffmpeg git jq libgomp1 libsndfile1
ffmpeg -hide_banner -encoders 2>/dev/null | grep libopus
```

安装 `uv` 并安装纯 CPU 依赖：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd /path/to/irodori-server
~/.local/bin/uv python install 3.10
~/.local/bin/uv sync --extra cpu --python 3.10
```

## 2. 准备模型、Speaker Embeddings 与离线缓存

把以下资产放置到计算节点（示例目录 `/opt/irodori-assets`）：

1. **MeanFlow 蒸馏模型包**（解压 `model_meanflow_s8000_clarity_fix_export.tar`，含 `model_meanflow_s8000_clarity_fix.safetensors` 和同级 `tokenizer/`）；
2. **29 角色 Speaker Embeddings**（`*.speaker.safetensors`，共 29 个 × 49 KB ≈ 1.4 MB）；
3. **DACVAE 编解码器权重缓存**（首次带网预热一次即可离线运行）：

```bash
uv run python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('Aratako/Semantic-DACVAE-Japanese-32dim', 'weights.pth'))"
```

完成后在 `.env` 中保持 `HF_HUB_OFFLINE=1`。

## 3. 配置 `.env`

```bash
cp .env.example .env
chmod 600 .env
```

关键生产项建议（以 8 核 Xeon Sapphire/Emerald Rapids、16 GB RAM 为例）：

```dotenv
MODEL_CHECKPOINT=/opt/irodori-assets/clarity_fix/model_meanflow_s8000_clarity_fix.safetensors
SPEAKERS_DIR=/opt/irodori-assets/speakers_s8000
CODEC_REPO=Aratako/Semantic-DACVAE-Japanese-32dim

MODEL_DEVICE=cpu
MODEL_PRECISION=auto
CODEC_DEVICE=cpu
CODEC_PRECISION=fp32
CPU_THREADS=8
DEFAULT_NUM_STEPS=6
MAX_NUM_STEPS=16
ENABLE_WATERMARK=false
PRELOAD_ON_STARTUP=true
HF_HUB_OFFLINE=1

REFERENCE_RESOURCE_SERVER=https://service.sc-viewer.top/convert/direct/sounds/voice/events
FALLBACK_TO_EMBED_ON_REF_ERROR=true
TMP_REFERENCE_DIR=./tmp_references
OUTPUT_DIR=./output
MAX_PENDING_TASKS=20
TASK_RATE_LIMIT=100/day

R2_ACCOUNT_ID=<CLOUDFLARE_ACCOUNT_ID>
R2_BUCKET=<R2_BUCKET_NAME>
R2_ACCESS_KEY_ID=<R2_ACCESS_KEY_ID>
R2_SECRET_ACCESS_KEY=<R2_SECRET_ACCESS_KEY>
BASE_STATIC_URL=<WORKER_URL>/audio
```

> `MODEL_PRECISION=auto` 会检查 `/proc/cpuinfo` 的 `amx_bf16` / `avx512_bf16` 标志：在 Xeon 节点自动启用 BF16（RSS ~3.5 GB，6 步约 1.4–1.9s）；在无硬件 BF16 的机器自动回退 FP32。

## 4. 配置 systemd 服务

创建 `/etc/systemd/system/irodori-tts.service`：

```ini
[Unit]
Description=Irodori TTS CPU API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=<LINUX_USER>
WorkingDirectory=<REPO_PATH>
EnvironmentFile=<REPO_PATH>/.env
Environment=PYTHONUNBUFFERED=1
ExecStart=<REPO_PATH>/.venv/bin/uvicorn server:app --host 127.0.0.1 --port 12451 --workers 1
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

必须保持 `--workers 1`（任务队列、限流状态、单份模型内存均在进程内维护，避免多线程争抢 OpenMP 核数）。

启动并验证：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now irodori-tts
curl -fsS http://127.0.0.1:12451/health
```

## 5. Worker 复用说明

若你已按 `genie-server` 部署好 Cloudflare Worker、R2 与 `cloudflared`（VPC 指向 `127.0.0.1:12451`），只需停掉原 `genie` 服务并启动 `irodori-tts` 即可无缝切换，Worker 端无需重新部署。若需部署全新的独立 Worker，可参考 `worker/wrangler.example.jsonc` 配置后执行 `cd worker && npm ci && npm test && npm run deploy`。
