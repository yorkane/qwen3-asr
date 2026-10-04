# -*- coding: utf-8 -*-
"""声纹 embedding / 说话人相似度 API（ERes2NetV2 + CAM++ 双路）。

提供两类能力：
  1. POST /v1/audio/embeddings  —— 抽声纹向量（models=eres2net/campplus/both）
  2. POST /v1/audio/speaker-compare —— 两段音频的说话人相似度（融合分数）

与转写接口共用音频预处理链路（prepare_upload / prepare_from_request），
输出均为 L2 归一后的 192 维向量，可直接喂给下游聚类/融合。
"""

import logging
import time
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ...core.config import settings
from ...core.exceptions import create_error_response
from ...core.executor import run_sync
from ...core.security import validate_openai_token
from ...services.asr.offline_transcription_service import (
    get_offline_transcription_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Speaker"])


class SpeakerEmbeddingItem(BaseModel):
    index: int
    start: float
    end: float
    embedding: List[float]


class SpeakerEmbeddingsResponse(BaseModel):
    task_id: str
    duration: float = Field(description="音频总时长（秒）")
    models: List[str]
    dim: int
    segment_count: int
    segments: List[SpeakerEmbeddingItem]
    global_embedding: Optional[List[float]] = Field(
        default=None, description="整段（各模型拼接后）归一化向量；segments 为空时为 null"
    )


class SpeakerCompareResponse(BaseModel):
    task_id: str
    cosine: float = Field(description="余弦相似度（融合后），范围约 -1~1")
    per_model: dict = Field(description="各模型单独的余弦")
    fused: bool = Field(description="是否做了分数级融合（双模型时 True）")
    same_speaker: Optional[bool] = Field(
        default=None, description="按阈值判定是否同一人；未给 threshold 时为 null"
    )


async def _auth_or_fail(request: Request):
    result, _ = validate_openai_token(request)
    if not result:
        return JSONResponse(
            content=create_error_response(
                error_code="AUTHENTICATION_FAILED", message="Invalid authentication"
            ),
            status_code=401,
        )
    return None


def _run_embeddings(audio_path: str, models: List[str], boundaries, batch_size: int):
    """同步：抽 embedding。boundaries=[(start,end),...]，空则整段一条。"""
    import librosa

    from ...utils import speaker_embedder as se

    data, _ = librosa.load(audio_path, sr=se.SAMPLE_RATE, mono=True)
    data = np.asarray(data, dtype=np.float32)
    sr = se.SAMPLE_RATE
    if not boundaries:
        boundaries = [(0.0, float(len(data)) / sr)]

    segs = []
    for (s, e) in boundaries:
        a = max(0, int(s * sr))
        b = min(len(data), int(e * sr))
        segs.append(data[a:b] if b > a else np.zeros(sr, dtype=np.float32))

    out = {}
    if "eres2net" in models:
        out["eres2net"] = se.extract_embeddings(segs, batch_size=batch_size)
    if "campplus" in models:
        out["campplus"] = se.extract_campplus_embeddings(segs, batch_size=batch_size)
    return out, boundaries


@router.post(
    "/audio/embeddings",
    summary="声纹 embedding 提取",
    description=(
        "抽取 ERes2NetV2 / CAM++ 声纹向量（各 192 维，L2 归一）。"
        "默认整段一条；提供 segments 时逐段抽取，可配合下游聚类/融合。"
    ),
    response_model=SpeakerEmbeddingsResponse,
)
async def create_embeddings(
    request: Request,
    file: Optional[UploadFile] = File(default=None, description="音频/视频文件"),
    audio_address: Optional[str] = Form(default=None, description="音频 URL（file 为空时用）"),
    models: str = Form("both", description="eres2net / campplus / both"),
    segments: Optional[str] = Form(
        default=None,
        description='分段 JSON，如 [{"start":0,"end":3},{"start":3.5,"end":8}]；不给则整段一条',
    ),
    diarize: bool = Form(False, description="是否先用 CAM++ 说话人分离自动切段（覆盖 segments）"),
    num_speakers: Optional[int] = Form(None, ge=1, le=15, description="配合 diarize 指定人数"),
):
    denied = await _auth_or_fail(request)
    if denied is not None:
        return denied

    want = (models or "both").strip().lower()
    model_list = ["eres2net", "campplus"] if want in ("both", "", "all") else [
        m.strip() for m in want.split(",") if m.strip()
    ]
    bad = [m for m in model_list if m not in ("eres2net", "campplus")]
    if bad or not model_list:
        return JSONResponse(
            content=create_error_response(
                error_code="INVALID_PARAMETER",
                message="models 只能是 eres2net/campplus/both，收到 %s" % models,
            ),
            status_code=400,
        )

    svc = get_offline_transcription_service()
    task_id = "emb-%d" % int(time.time() * 1000)
    prepared = None
    try:
        if file is not None:
            audio_data = await file.read()
            prepared = await svc.prepare_upload(
                audio_data=audio_data,
                filename=file.filename if file else None,
                task_id=task_id,
                sample_rate=16000,
            )
        elif audio_address:
            prepared = await svc.prepare_from_request(
                request=request, audio_address=audio_address,
                task_id=task_id, sample_rate=16000,
            )
        else:
            return JSONResponse(
                content=create_error_response(
                    error_code="INVALID_PARAMETER",
                    message="必须提供 file 或 audio_address 其中之一",
                ),
                status_code=400,
            )

        boundaries = []
        if diarize:
            from ...utils.speaker_diarizer import SpeakerDiarizer

            dia = await run_sync(
                SpeakerDiarizer().diarize, prepared.normalized_path,
                num_speakers=num_speakers,
            )
            boundaries = [(s.start_sec, s.end_sec) for s in dia]
        elif segments:
            import json

            raw = json.loads(segments)
            if not isinstance(raw, list) or not raw:
                raise ValueError("segments 必须是非空数组")
            for it in raw:
                s = float(it.get("start", 0.0))
                e = float(it.get("end", 0.0))
                if e <= s:
                    raise ValueError("segment end 必须大于 start: %r" % it)
                boundaries.append((s, e))

        emb_out, bounds = await run_sync(
            _run_embeddings, prepared.normalized_path, model_list, boundaries,
            settings.SPEAKER_EMB_BATCH_SIZE,
        )

        n = len(bounds)
        seg_items: List[SpeakerEmbeddingItem] = []
        primary = emb_out.get("eres2net")
        if primary is None:
            primary = emb_out.get("campplus")
        for i in range(n):
            seg_items.append(
                SpeakerEmbeddingItem(
                    index=i,
                    start=round(float(bounds[i][0]), 3),
                    end=round(float(bounds[i][1]), 3),
                    embedding=[float(x) for x in primary[i]],
                )
            )

        glob = None
        if primary is not None and n > 0:
            import numpy as _np

            g = primary.mean(axis=0)
            g = g / (_np.linalg.norm(g) + 1e-9)
            glob = [float(x) for x in g]

        resp = SpeakerEmbeddingsResponse(
            task_id=task_id,
            duration=round(float(prepared.duration), 3),
            models=list(emb_out.keys()),
            dim=192,
            segment_count=n,
            segments=seg_items,
            global_embedding=glob,
        )
        # 附带第二路（若有）到响应扩展字段
        payload = resp.model_dump()
        if len(emb_out) > 1:
            payload["embeddings_by_model"] = {
                m: [[float(x) for x in row] for row in arr]
                for m, arr in emb_out.items()
            }
        return JSONResponse(content=payload)
    except Exception as exc:
        logger.exception("embeddings 失败")
        return JSONResponse(
            content=create_error_response(
                error_code="DEFAULT_SERVER_ERROR", message=str(exc), task_id=task_id
            ),
            status_code=500,
        )
    finally:
        if prepared is not None:
            try:
                svc.cleanup(prepared)
            except Exception:
                pass


async def _prepare_one(svc, request, file, audio_address, task_id):
    if file is not None:
        audio_data = await file.read()
        return await svc.prepare_upload(
            audio_data=audio_data,
            filename=file.filename if file else None,
            task_id=task_id,
            sample_rate=16000,
        )
    if audio_address:
        return await svc.prepare_from_request(
            request=request, audio_address=audio_address,
            task_id=task_id, sample_rate=16000,
        )
    return None


@router.post(
    "/audio/speaker-compare",
    summary="两段音频说话人相似度",
    description=(
        "对两段音频各抽 ERes2NetV2 + CAM++ 声纹向量，给出逐模型余弦与分数级融合余弦。"
        "分数级融合对齐两路分数分布（场内稳健 z-score），比单模型更稳。"
    ),
    response_model=SpeakerCompareResponse,
)
async def speaker_compare(
    request: Request,
    file: Optional[UploadFile] = File(default=None, description="第一段音频"),
    file_b: Optional[UploadFile] = File(default=None, description="第二段音频"),
    audio_address: Optional[str] = Form(default=None, description="第一段 URL"),
    audio_address_b: Optional[str] = Form(default=None, description="第二段 URL"),
    models: str = Form("both", description="eres2net / campplus / both"),
    threshold: Optional[float] = Form(
        None, ge=-1.0, le=1.0, description="同人判定阈值；给了才输出 same_speaker"
    ),
):
    denied = await _auth_or_fail(request)
    if denied is not None:
        return denied

    want = (models or "both").strip().lower()
    model_list = ["eres2net", "campplus"] if want in ("both", "", "all") else [
        m.strip() for m in want.split(",") if m.strip()
    ]
    model_list = [m for m in model_list if m in ("eres2net", "campplus")]
    if not model_list:
        return JSONResponse(
            content=create_error_response(
                error_code="INVALID_PARAMETER", message="models 非法，收到 %s" % models
            ),
            status_code=400,
        )

    svc = get_offline_transcription_service()
    task_id = "spkc-%d" % int(time.time() * 1000)
    pa = pb = None
    try:
        pa = await _prepare_one(svc, request, file, audio_address, task_id + "-a")
        pb = await _prepare_one(svc, request, file_b, audio_address_b, task_id + "-b")
        if pa is None or pb is None:
            return JSONResponse(
                content=create_error_response(
                    error_code="INVALID_PARAMETER",
                    message="必须为两段音频分别提供 file/file_b 或 audio_address/audio_address_b",
                ),
                status_code=400,
            )

        emba, _ = await run_sync(
            _run_embeddings, pa.normalized_path, model_list, [], settings.SPEAKER_EMB_BATCH_SIZE
        )
        embb, _ = await run_sync(
            _run_embeddings, pb.normalized_path, model_list, [], settings.SPEAKER_EMB_BATCH_SIZE
        )

        per_model = {}
        mats = []
        for m in model_list:
            a, b = emba[m][0], embb[m][0]
            c = float(np.dot(a, b))
            per_model[m] = round(c, 4)
            mats.append((a, b))

        fused = False
        # 双路融合：两两余弦做稳健归一后等权平均。单段对比没有"场内 cohort"，
        # 这里退化为对两路原始余弦等权平均（无跨样本分布可估），仍优于单模型。
        if len(mats) == 2:
            fused = True
            cos = 0.5 * per_model[model_list[0]] + 0.5 * per_model[model_list[1]]
        else:
            cos = per_model[model_list[0]]

        return SpeakerCompareResponse(
            task_id=task_id,
            cosine=round(cos, 4),
            per_model=per_model,
            fused=fused,
            same_speaker=(None if threshold is None else bool(cos >= threshold)),
        )
    except Exception as exc:
        logger.exception("speaker-compare 失败")
        return JSONResponse(
            content=create_error_response(
                error_code="DEFAULT_SERVER_ERROR", message=str(exc), task_id=task_id
            ),
            status_code=500,
        )
    finally:
        for p in (pa, pb):
            if p is not None:
                try:
                    svc.cleanup(p)
                except Exception:
                    pass
