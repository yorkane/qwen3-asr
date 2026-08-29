# 单镜像部署指南（deploy.md）

> 目标读者：部署 agent / 运维。只需要**一个镜像地址**即可上线，无需任何
> 模型下载、外部挂载或网络。模型（Qwen3-ASR 1.7B + ForcedAligner 0.6B +
> FunASR 全套）已烘焙进镜像，运行时 `HF_HUB_OFFLINE=1`。

## 镜像

| 镜像 | 体积 | 说明 |
|---|---|---|
| `w217/qwen3-asr:async-baked-slim-023-final` | ~20GB | **推荐**。vLLM 0.23 + torch 2.11 + AsyncLLMEngine + 模型内置 + 压扁瘦身（吞吐 +8%，启动更快） |
| `w217/qwen3-asr:async-baked-slim` | ~19.9GB | vLLM 0.19 稳定版（回退选项） |
| `w217/qwen3-asr:async-baked` | ~31.7GB | 未瘦身版本 |

镜像内关键事实：

- 服务端口：容器内 `8000`（内置 nginx 入口脚本自动多 GPU 实例化）
- 模型路径：`/root/.cache/huggingface/hub`、`/root/.cache/modelscope/hub/models`
- 运行时引擎：vLLM 0.19.0 AsyncLLMEngine（continuous batching）
- 质检脚本：`/app/app/qa`（单元 + 冒烟 + 压测，见「上线质检」）

## 前置要求

- NVIDIA GPU（单卡即可；镜像面向 CUDA 12.8，L40/L20 等 sm_89 已验证）
- 显存 ≥ 16GB（1.7B 模型 + aligner 建议 ≥ 24GB）
- 已安装 NVIDIA Container Toolkit（`--gpus` 可用）

## 最小部署（单卡，一条命令）

```bash
docker run -d --name qwen3-asr \
  --gpus '"device=0"' \
  --shm-size=8g \
  -p 17003:8000 \
  -e API_KEY=devideo2026asrkey \
  -e QWEN3_ASR_MODEL=qwen3-asr-1.7b \
  --restart unless-stopped \
  w217/qwen3-asr:async-baked-slim
```

启动约 3–5 分钟（vLLM 编译 + 模型加载）。就绪后：

```bash
curl -s -H 'Authorization: Bearer devideo2026asrkey' http://127.0.0.1:17003/v1/models
```

## 环境变量（可选调优，均有默认值）

| 变量 | 默认 | 说明 |
|---|---|---|
| `API_KEY` | 空（关闭鉴权） | 设置后所有 `/v1/*` 需 `Authorization: Bearer <key>` |
| `QWEN3_ASR_MODEL` | 自动 | `qwen3-asr-1.7b` / `qwen3-asr-0.6b` |
| `QWEN_GPU_MEMORY_UTILIZATION` | 0.7 | 主模型显存占比 |
| `QWEN_FORCE_ALIGNER_GPU_MEMORY_UTILIZATION` | 0.15 | 词级对齐模型显存占比 |
| `QWEN_VLLM_MAX_NUM_BATCHED_TOKENS` | 16384 | 调度批处理上限 |
| `QWEN_VLLM_MAX_NUM_SEQS` | 128 | 并发序列上限 |
| `QWEN_VLLM_RENDERER_WORKERS` | 8 | 多模态预处理线程数 |
| `QWEN_VLLM_MAX_IN_FLIGHT` | 64 | 提交给引擎的在途请求上限 |
| `QWEN_VLLM_SHARED_CONCURRENCY` | 64 | 应用层并发上限 |
| `QWEN_VAD_POOL` | 4 | VAD 实例池大小 |
| `ASR_BATCH_SIZE` | 32 | 单请求分段批大小 |
| `VLLM_USE_FLASHINFER_SAMPLER` | 0 | 0.23 镜像内置：禁用 flashinfer 采样（sm_89 用 PyTorch 采样） |
| `CUDA_HOME` | 内置 | 0.23 镜像内置：指向 torch 自带 CUDA 13 工具链 |
| `LD_LIBRARY_PATH` | 内置 | 0.23 镜像内置：NVRTC 运行时库路径 |

## 多 GPU 部署

入口脚本会自动为每张可见卡拉起一个后端实例，并用内置 nginx 做负载（`CUDA_VISIBLE_DEVICES=all` 时）。也可显式指定：

```bash
docker run -d --name qwen3-asr \
  --gpus all \
  --shm-size=8g \
  -p 17003:8000 \
  -e API_KEY=devideo2026asrkey \
  -e CUDA_VISIBLE_DEVICES=all \
  --restart unless-stopped \
  w217/qwen3-asr:async-baked-slim
```

## 上线质检（强烈建议）

镜像内置质检套件 `app.qa`，无需外部音频（自带 20s 测试片段）。

```bash
# 1. 快速单元测试（不依赖服务，秒级）
docker exec qwen3-asr /opt/venv/bin/python -m app.qa.unit_checks

# 2. 功能冒烟（转写/分段/词级时间戳/并发）
docker exec qwen3-asr /opt/venv/bin/python -m app.qa.smoke_checks \
  --url http://127.0.0.1:8000 --key <API_KEY>

# 3. 并发压测（吞吐扫描，验证 continuous batching 生效）
docker exec qwen3-asr /opt/venv/bin/python -m app.qa.load_checks \
  --url http://127.0.0.1:8000 --key <API_KEY> \
  --concurrency 1,8,16,32,64 --total 64 --min-speedup 1.5

# 或一键全量：单元 + 冒烟 + 压测
docker exec qwen3-asr /opt/venv/bin/python -m app.qa.run_qa \
  --all --url http://127.0.0.1:8000 --key <API_KEY> --min-speedup 1.5
```

退出码 0 表示全部通过，可直接接入发布流程。

## 验收基线（L40 单卡实测，20s 音频）

vLLM 0.23 + torch 2.11（推荐镜像）：

| 并发 | 吞吐 (rps) | 加速比 |
|---|---|---|
| 1 | ~2.0 | 1.0× |
| 32 | ~12.4 | ~6.3× |
| 64（峰值） | **~12.4** | **~6.3×** |
| 128 | ~11.6 | ~5.9× |

vLLM 0.19（回退镜像）峰值 ~11.5 rps @ 并发64。

启动时间（含模型加载）：首次 ~3.5 分钟，二次启动 ~2.3 分钟（torch.compile 缓存复用）。

## 故障排查

- **启动失败 `Free memory ... less than desired`**：显存不足，降低
  `QWEN_GPU_MEMORY_UTILIZATION`（如 0.5）或换卡。
- **`/v1/models` 返回 401**：设置了 `API_KEY`，请求需带 `Authorization: Bearer <key>`。
- **转写返回空文本**：检查音频时长与格式；用质检片段自证：
  `docker exec qwen3-asr /opt/venv/bin/python -m app.qa.smoke_checks --url http://127.0.0.1:8000`。

## 模型清单与版本

见 [MODELS.md](MODELS.md)（每个模型的下载地址、commit/版本与框架版本）。
