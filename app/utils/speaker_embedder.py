# -*- coding: utf-8 -*-
"""ERes2NetV2 声纹 embedding 提取（新增能力，与 CAM++ 并列的第二路声纹模型）。

为什么用它：多模型融合实测能把声纹可分性显著抬高（LOO-NN 行级同人率
CAM++ 单路 88.2% -> 双路融合 90.8%，逐行最近邻错误 19->13；
详见 doc/声纹双模型融合实验报告_20261004.md）。ERes2NetV2 主打短时长 utterance，
正好补 CAM++ 在短句上的短板。

实现口径（与 devideo/pipeline/speaker_embed2.py 逐位对齐，勿擅改）：
  - 网络构造 ERes2NetV2(feat_dim=80, embedding_size=192, m_channels=64,
    baseWidth=26, scale=2, expansion=2)，load_state_dict(strict=True)。
  - 前端 torchaudio.compliance.kaldi.fbank(num_mel_bins=80,
    sample_frequency=16000, dither=0.0, frame_length=25, frame_shift=10)。
  - 不做 mean-norm。3D-Speaker 官方 infer_sv 用 mean_nor=True，与 devideo 已冻结的
    npy 口径不同（同一 4s 音频两种口径 cos 仅 0.5621），混用会让分数完全不可比。
  - 行窗 PAD=0.06 / MINLEN=0.5 / MAXLEN=6.0；6s 是实测红线，加长窗污染率陡增。
  - 输出逐条 L2 归一化的 192 维向量。

网络定义来源：优先 funasr>=1.4 自带实现，缺失时回退 app/vendor/eres2net
（3D-Speaker speakerlab 最小闭包：ERes2NetV2 + pooling_layers.TSTP + fusion.AFF）。
funasr 1.3.1 没有该模块，故两条路都要留着。
"""

from __future__ import annotations

import threading
import os
from typing import Any, List

import numpy as np
from loguru import logger

from ..core.config import settings
from ..core.device import detect_device
from ..core.exceptions import DefaultServerErrorException

SAMPLE_RATE = 16000
EMBEDDING_DIM = 192
PAD_SEC = 0.06
MIN_LEN_SEC = 0.5
MAX_LEN_SEC = 6.0   # 实测红线：加长窗污染率陡增，勿放大

_global_eres2net_model: Any | None = None
_eres2net_model_lock = threading.Lock()
# ERes2NetV2 前向很短（GPU batch 实测 0.0044 s/行），限一路只为避免显存尖峰；
# 与 CAM++ 的 _diarization_inference_semaphore 同形，二者互不阻塞。
_eres2net_inference_semaphore = threading.BoundedSemaphore(1)


def _resolve_device() -> str:
    return detect_device(getattr(settings, "SPEAKER_EMB_DEVICE", None) or settings.DEVICE)


def _resolve_ckpt_path() -> str:
    """定位 ERes2NetV2 权重 .ckpt。

    依次尝试：SPEAKER_ERES2NETV2_CKPT 显式文件 -> 直接路径 -> ModelScope 缓存目录
    内的常见文件名。容器内需把权重烘焙到 MODELSCOPE_PATH 下。
    """
    import os
    from pathlib import Path

    env_ckpt = (os.getenv("SPEAKER_ERES2NETV2_CKPT") or "").strip()
    if env_ckpt:
        p = Path(env_ckpt)
        if p.is_file():
            return str(p)
        raise DefaultServerErrorException(f"SPEAKER_ERES2NETV2_CKPT 指向的文件不存在: {env_ckpt}")

    model_id = (getattr(settings, "SPEAKER_ERES2NETV2_MODEL", "") or "").strip()
    if not model_id:
        raise DefaultServerErrorException(
            "未配置 ERes2NetV2 权重：请设 SPEAKER_ERES2NETV2_MODEL 或 SPEAKER_ERES2NETV2_CKPT"
        )

    candidate_dirs: List[Path] = []
    direct = Path(model_id)
    if direct.is_file():
        return str(direct)
    if direct.is_dir():
        candidate_dirs.append(direct)
    else:
        candidate_dirs.append(Path(settings.MODELSCOPE_PATH) / model_id)

    for d in candidate_dirs:
        if not d.is_dir():
            continue
        for name in ("pretrained_eres2netv2.ckpt", "eres2netv2.ckpt"):
            f = d / name
            if f.is_file():
                return str(f)
        hits = sorted(d.glob("*.ckpt")) + sorted(d.glob("*.pt"))
        if hits:
            return str(hits[0])

    raise DefaultServerErrorException(
        "未找到 ERes2NetV2 权重（查找 %s）。权重来源 ModelScope "
        "iic/speech_eres2netv2_sv_zh-cn_16k-common，约 71MB。" % [str(x) for x in candidate_dirs]
    )


def _import_eres2net_cls():
    """返回 (ERes2NetV2 类, 实现来源标记)。先 funasr>=1.4，再回退 vendored。"""
    force_vendor = (os.getenv("SPEAKER_FORCE_VENDOR") or "").strip().lower() in ("1", "true", "yes")
    try:
        if force_vendor:
            raise ImportError("SPEAKER_FORCE_VENDOR=1：跳过 funasr，直接验证 vendored 实现")
        from funasr.models.eres2net.eres2netv2 import ERes2NetV2  # noqa: PLC0415

        return ERes2NetV2, "funasr"
    except Exception as exc:  # funasr<1.4 或结构变动
        logger.debug("funasr 自带 ERes2NetV2 不可用（{}），回退 vendored 实现", exc)

    import importlib
    import importlib.machinery
    import sys
    from pathlib import Path

    vendor_root = Path(__file__).resolve().parent.parent / "vendor"
    # 注册 speakerlab.models.eres2net 命名空间包并指向 vendored 目录，
    # 使 ERes2NetV2.py 内部对 pooling_layers / fusion 的导入无需改源码即可解析。
    pkg_paths = {
        "speakerlab": vendor_root,
        "speakerlab.models": vendor_root / "eres2net",
        "speakerlab.models.eres2net": vendor_root / "eres2net",
    }
    for name, path in pkg_paths.items():
        if name not in sys.modules:
            spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
            mod = importlib.util.module_from_spec(spec)
            mod.__path__ = [str(path)]  # type: ignore[attr-defined]
            sys.modules[name] = mod
    try:
        from speakerlab.models.eres2net.ERes2NetV2 import ERes2NetV2  # noqa: PLC0415

        return ERes2NetV2, "vendor"
    except Exception as exc:
        raise DefaultServerErrorException(
            "ERes2NetV2 网络定义不可用（funasr 与 vendored 均失败）：%s" % exc
        ) from exc


def get_global_eres2net_model() -> Any:
    """懒加载全局 ERes2NetV2 单例（fp32, eval）。"""
    global _global_eres2net_model

    if _global_eres2net_model is not None:
        return _global_eres2net_model

    with _eres2net_model_lock:
        if _global_eres2net_model is not None:
            return _global_eres2net_model

        import torch

        ckpt = _resolve_ckpt_path()
        cls, source = _import_eres2net_cls()
        device = _resolve_device()
        logger.info("正在加载 ERes2NetV2 声纹模型: {}, device={}, 实现={}", ckpt, device, source)
        try:
            model = cls(
                feat_dim=80,
                embedding_size=EMBEDDING_DIM,
                m_channels=64,
                baseWidth=26,
                scale=2,
                expansion=2,
            )
            state = torch.load(ckpt, map_location="cpu")
            first = list(state.values())[:3] if hasattr(state, "values") else []
            if any(isinstance(v, torch.Tensor) for v in first):
                model.load_state_dict(state, strict=True)
            else:  # 兼容外层包了一层 state_dict 的 ckpt
                model.load_state_dict(state["state_dict"], strict=True)
            model.eval().to(device)
            torch.set_num_threads(8)
        except Exception as exc:
            logger.error(f"ERes2NetV2 加载失败: {exc}")
            raise DefaultServerErrorException(f"ERes2NetV2 模型加载失败: {exc}")

        n_params = sum(p.numel() for p in model.parameters())
        logger.info(
            "ERes2NetV2 加载成功（{} 参数, device={}, 实现={}）",
            "%.2fM" % (n_params / 1e6), device, source,
        )
        _global_eres2net_model = model
        return _global_eres2net_model


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def extract_embeddings(
    segments: List[np.ndarray],
    batch_size: int = 32,
) -> np.ndarray:
    """对一组 16k float32 单声道波形批量抽 ERes2NetV2 embedding。

    Args:
        segments: 每条为 16kHz float32 mono（内部按 6s 上限截断，空段补 1s 静音）。
        batch_size: 前向批大小。

    Returns:
        (n, 192) float32，逐条 L2 归一。空输入返回 (0, 192)。
    """
    import torch
    import torchaudio

    if not segments:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    model = get_global_eres2net_model()
    device = next(model.parameters()).device
    max_samples = int(MAX_LEN_SEC * SAMPLE_RATE)

    feats = []
    for x in segments:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            arr = np.zeros(SAMPLE_RATE, dtype=np.float32)
        arr = arr[:max_samples]
        w = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)
        feats.append(
            torchaudio.compliance.kaldi.fbank(
                w,
                num_mel_bins=80,
                sample_frequency=SAMPLE_RATE,
                dither=0.0,
                frame_length=25,
                frame_shift=10,
            )
        )

    out = np.zeros((len(segments), EMBEDDING_DIM), dtype=np.float32)
    bs = max(1, int(batch_size))
    with _eres2net_inference_semaphore, torch.no_grad():
        for start in range(0, len(feats), bs):
            chunk = feats[start:start + bs]
            fmax = max(int(f.shape[0]) for f in chunk)
            batch = torch.zeros((len(chunk), fmax, 80), dtype=torch.float32)
            for i, f in enumerate(chunk):
                batch[i, : f.shape[0], :] = f
            emb = model(batch.to(device)).detach().float().cpu().numpy()
            out[start:start + len(chunk)] = np.asarray(emb, dtype=np.float32).reshape(len(chunk), -1)
    return _l2(out).astype(np.float32)


def extract_embedding_from_waveform(waveform: np.ndarray) -> np.ndarray:
    """单条便捷入口，返回 (192,) 已 L2 归一。"""
    e = extract_embeddings([waveform])
    if len(e) == 0:
        raise DefaultServerErrorException("ERes2NetV2 embedding 提取失败：空输入")
    return e[0]


def load_waveform_16k(audio_path: str) -> np.ndarray:
    """读音频为 16k float32 mono。"""
    import librosa

    data, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    return np.asarray(data, dtype=np.float32)


def reset_global_model_for_tests() -> None:
    """测试用：释放单例（换设备/换权重后重载）。"""
    global _global_eres2net_model
    with _eres2net_model_lock:
        _global_eres2net_model = None


# ---------------------------------------------------------------------------
# 第二路：CAM++ embedding（供双模型融合用；与 devideo speaker_embed2 同口径）
# ---------------------------------------------------------------------------

_global_campplus_sv_model: Any | None = None
_campplus_sv_lock = threading.Lock()


def _resolve_campplus_sv_bin() -> str:
    """定位 CAM++ SV 权重 campplus_cn_common.bin。"""
    import os
    from pathlib import Path

    env_bin = (os.getenv("SPEAKER_CAMPPLUS_SV_BIN") or "").strip()
    if env_bin:
        p = Path(env_bin)
        if p.is_file():
            return str(p)
        raise DefaultServerErrorException(
            f"SPEAKER_CAMPPLUS_SV_BIN 指向的文件不存在: {env_bin}"
        )

    model_id = (getattr(settings, "SPEAKER_CAMPPLUS_SV_MODEL", "") or "").strip()
    if not model_id:
        model_id = "damo/speech_campplus_sv_zh-cn_16k-common"

    dirs = []
    direct = Path(model_id)
    if direct.is_file():
        return str(direct)
    if direct.is_dir():
        dirs.append(direct)
    else:
        dirs.append(Path(settings.MODELSCOPE_PATH) / model_id)

    for d in dirs:
        if not d.is_dir():
            continue
        for name in ("campplus_cn_common.bin", "campplus.bin"):
            f = d / name
            if f.is_file():
                return str(f)
        hits = sorted(d.glob("*.bin"))
        if hits:
            return str(hits[0])

    raise DefaultServerErrorException(
        "未找到 CAM++ SV 权重（查找 %s）。来源 ModelScope "
        "damo/speech_campplus_sv_zh-cn_16k-common。" % [str(x) for x in dirs]
    )


def get_global_campplus_sv_model() -> Any:
    """懒加载全局 CAM++ SV 单例（funasr 前端 + CAMPPlus，fp32 eval）。"""
    global _global_campplus_sv_model

    if _global_campplus_sv_model is not None:
        return _global_campplus_sv_model

    with _campplus_sv_lock:
        if _global_campplus_sv_model is not None:
            return _global_campplus_sv_model

        import torch
        from funasr.frontends.wav_frontend import WavFrontend  # noqa: PLC0415
        from funasr.models.campplus.model import CAMPPlus  # noqa: PLC0415

        ckpt = _resolve_campplus_sv_bin()
        device = _resolve_device()
        logger.info("正在加载 CAM++ SV 声纹模型: {}, device={}", ckpt, device)
        try:
            model = CAMPPlus(
                feat_dim=80,
                embedding_size=EMBEDDING_DIM,
                growth_rate=32,
                bn_size=4,
                init_channels=128,
                config_str="batchnorm-relu",
                memory_efficient=True,
                output_level="segment",
            )
            model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
            model.eval().to(device)
            frontend = WavFrontend(fs=SAMPLE_RATE)
        except Exception as exc:
            logger.error(f"CAM++ SV 加载失败: {exc}")
            raise DefaultServerErrorException(f"CAM++ SV 模型加载失败: {exc}")

        _global_campplus_sv_model = (model, frontend, device)
        logger.info("CAM++ SV 加载成功（device={}）", device)
        return _global_campplus_sv_model


def extract_campplus_embeddings(
    segments: List[np.ndarray],
    batch_size: int = 32,
) -> np.ndarray:
    """对一组 16k float32 mono 波形批量抽 CAM++ embedding，返回 (n, 192) L2 归一。"""
    import torch

    if not segments:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    model, frontend, device = get_global_campplus_sv_model()
    max_samples = int(MAX_LEN_SEC * SAMPLE_RATE)
    out = np.zeros((len(segments), EMBEDDING_DIM), dtype=np.float32)
    bs = max(1, int(batch_size))

    with _eres2net_inference_semaphore, torch.no_grad():
        for start in range(0, len(segments), bs):
            chunk = segments[start:start + bs]
            waves = []
            for x in chunk:
                arr = np.asarray(x, dtype=np.float32).reshape(-1)
                if arr.size == 0:
                    arr = np.zeros(SAMPLE_RATE, dtype=np.float32)
                arr = arr[:max_samples]
                waves.append(torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0))
            lengths = torch.tensor([int(w.shape[1]) for w in waves], dtype=torch.long)
            fmax = int(max(int(w.shape[1]) for w in waves))
            batch = torch.zeros((len(waves), fmax), dtype=torch.float32)
            for i, w in enumerate(waves):
                batch[i, : w.shape[1]] = w[0]
            feats, _ = frontend(batch, lengths)
            emb = model(feats.to(device)).detach().float().cpu().numpy()
            out[start:start + len(chunk)] = np.asarray(emb, dtype=np.float32).reshape(len(chunk), -1)
    return _l2(out).astype(np.float32)


def reset_campplus_sv_for_tests() -> None:
    global _global_campplus_sv_model
    with _campplus_sv_lock:
        _global_campplus_sv_model = None
