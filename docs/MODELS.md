# 模型清单（下载地址与版本来源）

本镜像（`w217/qwen3-asr:async-baked`）已将全部运行时模型烘焙进镜像，运行时
`HF_HUB_OFFLINE=1`，不访问任何网络。以下为模型来源与版本，供审计与离线重下。

## Hugging Face（vLLM 主链路）

| 模型 | 版本/commit | 用途 | 下载命令 |
|---|---|---|---|
| [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | `7278e1e70fe206f11671096ffdd38061171dd6e5` | ASR 主模型（vLLM AsyncLLMEngine） | `huggingface-cli download Qwen/Qwen3-ASR-1.7B` |
| [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | `c7cbfc2048c462b0d63a45797104fc9db3ad62b7` | 词级时间戳强制对齐（vLLM pooling） | `huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B` |

存放路径：`/root/.cache/huggingface/hub/`（HF 缓存布局，`HF_HUB_CACHE` 指向此处）。

## ModelScope（FunASR 链路）

| 模型 | 版本 | 用途 | 下载命令 |
|---|---|---|---|
| [speech_fsmn_vad_zh-cn-16k-common-pytorch](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch) | v2.0.2 | VAD 语音分段 | `modelscope download --model damo/speech_fsmn_vad_zh-cn-16k-common-pytorch` |
| [speech_campplus_speaker-diarization_common](https://modelscope.cn/models/iic/speech_campplus_speaker-diarization_common) | latest | CAM++ 说话人分离 | `modelscope download --model iic/speech_campplus_speaker-diarization_common` |
| [speech_campplus_sv_zh-cn_16k-common](https://modelscope.cn/models/damo/speech_campplus_sv_zh-cn_16k-common) | latest | CAM++ 说话人验证 | `modelscope download --model damo/speech_campplus_sv_zh-cn_16k-common` |
| [speech_campplus-transformer_scl_zh-cn_16k-common](https://modelscope.cn/models/damo/speech_campplus-transformer_scl_zh-cn_16k-common) | latest | CAM++ Transformer | `modelscope download --model damo/speech_campplus-transformer_scl_zh-cn_16k-common` |
| [speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online) | latest | Paraformer 实时流式 | `modelscope download --model iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online` |
| [punc_ct-transformer_zh-cn-common-vocab272727-pytorch](https://modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch) | latest | 标点恢复（离线） | `modelscope download --model iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch` |
| [punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727](https://modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727) | latest | 标点恢复（实时） | `modelscope download --model iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727` |

存放路径：`/root/.cache/modelscope/hub/models/`（ModelScope 缓存布局）。

> ModelScope 中未固定 revision 的模型按 `latest` 烘焙；版本固化见
> `app/services/asr/model_capabilities.py` 的 `revision` 字段。

## 运行时框架版本

| 组件 | 版本 | 来源 |
|---|---|---|
| vLLM | 0.19.0（`vllm[audio]`） | PyPI |
| PyTorch | 2.10.0+cu128 | `https://download.pytorch.org/whl/cu128` |
| torchaudio | 2.10.0 | 同上 |
| transformers | 4.57.6 | PyPI |
| FunASR | 1.3.1 | PyPI |
| librosa | 0.11.0 | PyPI |
| CUDA | 12.8（cudnn9） | `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime` 基础镜像 |
| Python | 3.12 | 基础镜像 |
