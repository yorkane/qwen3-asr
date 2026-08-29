# -*- coding: utf-8 -*-
"""Official vLLM async adapter for CUDA Qwen3-ASR.

Uses vLLM's AsyncLLMEngine so concurrent requests are submitted to the
scheduler as separate requests and benefit from continuous batching.
The legacy sync fallback (vllm.LLM) is kept only when explicitly forced
via QWEN_VLLM_SYNC_FALLBACK=1.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import itertools
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

import librosa
import numpy as np

from app.infrastructure import resolve_huggingface_snapshot_dir
from app.utils.text_processing import normalize_asr_text

from .engines import ASRRawResult, ASRSegmentResult, WordToken

logger = logging.getLogger(__name__)

_DEFAULT_SAMPLE_RATE = 16000
_LANGUAGE_ALIASES = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-hans": "Chinese",
    "zh-hant": "Chinese",
    "cn": "Chinese",
    "en": "English",
    "en-us": "English",
    "en-gb": "English",
    "ja": "Japanese",
    "jp": "Japanese",
    "ko": "Korean",
    "yue": "Cantonese",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
}


def is_vllm_available() -> bool:
    """Return True when the official vLLM runtime is installed."""
    return importlib.util.find_spec("vllm") is not None


def _is_truthy_env(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_language_name(language: Optional[str]) -> Optional[str]:
    if not language:
        return None
    normalized = language.strip()
    if not normalized:
        return None
    alias = _LANGUAGE_ALIASES.get(normalized.lower())
    if alias:
        return alias
    if " " in normalized:
        return " ".join(part.capitalize() for part in normalized.split())
    return normalized.capitalize()


def _load_audio(audio_path: str) -> np.ndarray:
    audio, _sample_rate = librosa.load(audio_path, sr=_DEFAULT_SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def _build_chat_prompt(context: str = "", language: Optional[str] = None) -> str:
    instructions: list[str] = []
    if language:
        instructions.append(f"Transcribe the speech in {language}.")
    else:
        instructions.append("Transcribe the speech accurately.")
    if context.strip():
        instructions.append(f"Use this context when resolving named entities: {context.strip()}")
    system_text = " ".join(instructions).strip()
    return (
        f"<|im_start|>system\n{system_text}<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _build_alignment_prompt(tokens: list[str]) -> str:
    body = "<timestamp><timestamp>".join(tokens) + "<timestamp><timestamp>"
    return f"<|audio_start|><|audio_pad|><|audio_end|>{body}"


def _parse_asr_output(raw_text: str, language: Optional[str]) -> tuple[str, str]:
    text = (raw_text or "").strip()
    if "<asr_text>" in text:
        left, right = text.split("<asr_text>", 1)
        detected = left.strip()
        if detected.lower().startswith("language "):
            detected = detected[9:].strip()
        return detected or (language or ""), right.strip()
    return (language or ""), text


def _split_alignment_units(text: str) -> list[str]:
    if not text:
        return []

    # Mixed Chinese/English transcripts should not fall back to whitespace-only
    # tokenization, otherwise a long CJK sentence with a single embedded English
    # word can collapse into one giant alignment unit.
    token_pattern = re.compile(
        r"[\u4e00-\u9fff]"                    # CJK ideographs, align per character
        r"|[A-Za-z0-9]+(?:['._+-][A-Za-z0-9]+)*"  # Latin / alnum words
        r"|[^\w\s]",                         # punctuation and symbols
        re.UNICODE,
    )
    return token_pattern.findall(text)


def _resolve_forced_aligner_gpu_memory_utilization(primary_utilization: float) -> float:
    override = (os.getenv("QWEN_FORCE_ALIGNER_GPU_MEMORY_UTILIZATION") or "").strip()
    if override:
        try:
            value = float(override)
            if 0.0 < value <= 1.0:
                return value
        except ValueError:
            logger.warning(
                "Invalid QWEN_FORCE_ALIGNER_GPU_MEMORY_UTILIZATION=%s, ignoring override",
                override,
            )

    return primary_utilization


def _resolve_optional_int_env(name: str) -> int | None:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
        if value > 0:
            return value
    except ValueError:
        logger.warning("Invalid %s=%s, ignoring", name, raw)
    return None


def _resolve_max_in_flight() -> int:
    """Cap on requests submitted concurrently to the async engine."""
    override = (os.getenv("QWEN_VLLM_MAX_IN_FLIGHT") or "").strip()
    if override:
        try:
            value = int(override)
            if value > 0:
                return value
        except ValueError:
            logger.warning("Invalid QWEN_VLLM_MAX_IN_FLIGHT=%s, using default", override)
    return 64


@dataclass
class _GeneratedTranscript:
    text: str
    language: str


@dataclass
class VLLMRealtimeState:
    prompt_raw: str
    language: str
    chunk_size_sec: float
    unfixed_chunk_num: int
    unfixed_token_num: int
    max_new_tokens: int
    chunk_id: int = 0
    text: str = ""
    raw_decoded: str = ""
    audio_buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    audio_accum: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))


class Qwen3VLLMBackend:
    """Async vLLM adapter for Qwen3-ASR.

    Primary engine is AsyncLLMEngine; every segment becomes its own engine
    request so concurrent HTTP requests and batches all flow into vLLM's
    continuous-batching scheduler. A sync vllm.LLM fallback exists only for
    environments where the async engine cannot start.
    """

    def __init__(
        self,
        model_path: str,
        forced_aligner_path: Optional[str],
        gpu_memory_utilization: float,
        max_inference_batch_size: int,
        max_new_tokens: int,
        max_model_len: Optional[int] = None,
    ) -> None:
        try:
            vllm_module = importlib.import_module("vllm")
            transformers_module = importlib.import_module("transformers")
        except ImportError as exc:
            raise RuntimeError(
                "CUDA Qwen3-ASR now requires official vLLM with Qwen3 forced aligner support. "
                "Install it with: pip install 'vllm[audio]==0.19.0'"
            ) from exc

        local_model_path = str(resolve_huggingface_snapshot_dir(model_path))
        local_forced_aligner_path = (
            str(resolve_huggingface_snapshot_dir(forced_aligner_path))
            if forced_aligner_path
            else None
        )

        self._vllm_module = vllm_module
        self._sampling_params_cls = getattr(vllm_module, "SamplingParams")
        self._pooling_params_cls = getattr(vllm_module, "PoolingParams")
        self._async_engine_cls = getattr(vllm_module, "AsyncLLMEngine", None)
        self._async_engine_args_cls = getattr(
            importlib.import_module("vllm.engine.arg_utils"), "AsyncEngineArgs", None
        )
        self._tokenizer = getattr(transformers_module, "AutoTokenizer").from_pretrained(
            local_model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        self._model_path = local_model_path
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._forced_aligner_path = local_forced_aligner_path
        self._max_inference_batch_size = max_inference_batch_size
        self._max_new_tokens = max_new_tokens
        self._max_in_flight = _resolve_max_in_flight()

        self._engine: Any | None = None
        self._engine_loop: asyncio.AbstractEventLoop | None = None
        self._engine_thread: threading.Thread | None = None
        self._engine_lock = threading.Lock()

        self._aligner_engine: Any | None = None
        self._aligner_loop: asyncio.AbstractEventLoop | None = None
        self._aligner_thread: threading.Thread | None = None
        self._aligner_lock = threading.Lock()
        self._timestamp_token_id: int | None = None
        self._timestamp_segment_time: float | None = None

        self._request_counter = itertools.count(1)
        self._sync_fallback = _is_truthy_env(os.getenv("QWEN_VLLM_SYNC_FALLBACK"))
        self._sync_llm: Any | None = None
        self._sync_aligner: Any | None = None
        self._sync_generate_lock = threading.Lock()
        self._audio_executor = ThreadPoolExecutor(
            max_workers=max(4, os.cpu_count() or 4),
            thread_name_prefix="qwen_vllm_audio",
        )

        if self._sync_fallback:
            logger.warning("QWEN_VLLM_SYNC_FALLBACK=1: using sync vllm.LLM (requests serialized)")
        else:
            if self._async_engine_cls is None or self._async_engine_args_cls is None:
                logger.warning("AsyncLLMEngine unavailable, falling back to sync vllm.LLM")
                self._sync_fallback = True

    def ensure_engine_started(self) -> None:
        """Eagerly start the async engine (called at startup warmup)."""
        if self._sync_fallback:
            self._ensure_sync_llm()
            return
        self._ensure_engine()

    # ------------------------------------------------------------------
    # Event-loop management
    # ------------------------------------------------------------------

    def _ensure_engine(self) -> tuple[Any, asyncio.AbstractEventLoop]:
        if self._engine is not None and self._engine_loop is not None:
            return self._engine, self._engine_loop

        with self._engine_lock:
            if self._engine is not None and self._engine_loop is not None:
                return self._engine, self._engine_loop

            ready = threading.Event()
            state: dict[str, Any] = {}

            def run_loop() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                state["loop"] = loop
                try:
                    engine_kwargs: dict[str, Any] = {
                        "model": self._model_path,
                        "gpu_memory_utilization": self._gpu_memory_utilization,
                    }
                    if self._max_model_len:
                        engine_kwargs["max_model_len"] = self._max_model_len
                    max_num_batched_tokens = _resolve_optional_int_env("QWEN_VLLM_MAX_NUM_BATCHED_TOKENS")
                    if max_num_batched_tokens:
                        engine_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
                    max_num_seqs = _resolve_optional_int_env("QWEN_VLLM_MAX_NUM_SEQS")
                    if max_num_seqs:
                        engine_kwargs["max_num_seqs"] = max_num_seqs
                    quantization = (os.getenv("QWEN_VLLM_QUANTIZATION") or "").strip()
                    if quantization:
                        engine_kwargs["quantization"] = quantization
                    kv_cache_dtype = (os.getenv("QWEN_VLLM_KV_CACHE_DTYPE") or "").strip()
                    if kv_cache_dtype:
                        engine_kwargs["kv_cache_dtype"] = kv_cache_dtype
                    limit_mm = _resolve_optional_int_env("QWEN_VLLM_LIMIT_MM_PER_PROMPT")
                    if limit_mm:
                        engine_kwargs["limit_mm_per_prompt"] = {"audio": limit_mm}
                    renderer_workers = _resolve_optional_int_env("QWEN_VLLM_RENDERER_WORKERS")
                    if renderer_workers:
                        # Requires disabling the (non thread-safe) mm processor cache.
                        engine_kwargs["renderer_num_workers"] = renderer_workers
                        engine_kwargs["mm_processor_cache_gb"] = 0
                    engine = self._async_engine_cls.from_engine_args(
                        self._async_engine_args_cls(**engine_kwargs)
                    )
                    state["engine"] = engine
                except BaseException as exc:  # noqa: BLE001
                    state["error"] = exc
                finally:
                    ready.set()
                loop.run_forever()

            self._engine_thread = threading.Thread(
                target=run_loop, name="qwen-vllm-async-engine", daemon=True
            )
            self._engine_thread.start()
            ready.wait()

            if "error" in state:
                raise RuntimeError(f"Failed to start vLLM AsyncLLMEngine: {state['error']}") from state["error"]

            self._engine = state["engine"]
            self._engine_loop = state["loop"]
            logger.info("vLLM AsyncLLMEngine started (model=%s)", self._model_path)
            return self._engine, self._engine_loop

    def _ensure_aligner_engine(self) -> tuple[Any, asyncio.AbstractEventLoop]:
        if self._aligner_engine is not None and self._aligner_loop is not None:
            return self._aligner_engine, self._aligner_loop

        if not self._forced_aligner_path:
            raise RuntimeError("word_timestamps requires a configured forced aligner model")

        with self._aligner_lock:
            if self._aligner_engine is not None and self._aligner_loop is not None:
                return self._aligner_engine, self._aligner_loop

            aligner_utilization = _resolve_forced_aligner_gpu_memory_utilization(
                self._gpu_memory_utilization
            )
            logger.info(
                "Loading Qwen3 forced aligner via AsyncLLMEngine: %s (gpu_memory_utilization=%s)",
                self._forced_aligner_path,
                aligner_utilization,
            )

            ready = threading.Event()
            state: dict[str, Any] = {}

            def run_loop() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                state["loop"] = loop
                try:
                    engine = self._async_engine_cls.from_engine_args(
                        self._async_engine_args_cls(
                            model=self._forced_aligner_path,
                            runner="pooling",
                            enforce_eager=True,
                            gpu_memory_utilization=aligner_utilization,
                            hf_overrides={
                                "architectures": ["Qwen3ASRForcedAlignerForTokenClassification"],
                            },
                        )
                    )
                    state["engine"] = engine
                    llm_engine = getattr(engine, "llm_engine", None)
                    vllm_config = getattr(llm_engine, "vllm_config", None) or getattr(
                        engine, "vllm_config", None
                    )
                    if vllm_config is None:
                        raise RuntimeError("Forced aligner did not expose a vLLM engine instance")
                    config = vllm_config.model_config.hf_config
                    state["timestamp_token_id"] = int(config.timestamp_token_id)
                    state["timestamp_segment_time"] = float(config.timestamp_segment_time)
                except BaseException as exc:  # noqa: BLE001
                    state["error"] = exc
                finally:
                    ready.set()
                loop.run_forever()

            self._aligner_thread = threading.Thread(
                target=run_loop, name="qwen-aligner-async-engine", daemon=True
            )
            self._aligner_thread.start()
            ready.wait()

            if "error" in state:
                raise RuntimeError(f"Failed to start forced aligner engine: {state['error']}") from state["error"]

            self._aligner_engine = state["engine"]
            self._aligner_loop = state["loop"]
            self._timestamp_token_id = state["timestamp_token_id"]
            self._timestamp_segment_time = state["timestamp_segment_time"]
            logger.info("vLLM AsyncLLMEngine aligner started (model=%s)", self._forced_aligner_path)
            return self._aligner_engine, self._aligner_loop

    # ------------------------------------------------------------------
    # Async core: one engine request per audio item
    # ------------------------------------------------------------------

    async def _async_generate_one(
        self,
        engine: Any,
        semaphore: asyncio.Semaphore,
        audio: np.ndarray,
        context: str,
        language: Optional[str],
        sampling_params: Any,
    ) -> _GeneratedTranscript:
        request_id = f"qwen-asr-{next(self._request_counter)}"
        async with semaphore:
            generator = engine.generate(
                {
                    "prompt": _build_chat_prompt(
                        context=context, language=_normalize_language_name(language)
                    ),
                    "multi_modal_data": {"audio": [audio]},
                },
                sampling_params,
                request_id,
            )
            final_output: Any = None
            async for output in generator:
                final_output = output

        raw_text = str(final_output.outputs[0].text if final_output and final_output.outputs else "")
        parsed_language, parsed_text = _parse_asr_output(raw_text, _normalize_language_name(language))
        return _GeneratedTranscript(text=parsed_text, language=parsed_language)

    async def _async_generate_items(
        self,
        audio_items: list[tuple[np.ndarray, str, Optional[str]]],
        max_new_tokens: int | None = None,
    ) -> list[_GeneratedTranscript]:
        if not audio_items:
            return []
        engine, loop = self._ensure_engine()
        _ = loop
        sampling_params = self._sampling_params_cls(
            temperature=0.01,
            max_tokens=max_new_tokens or self._max_new_tokens,
        )
        semaphore = asyncio.Semaphore(self._max_in_flight)
        tasks = [
            self._async_generate_one(engine, semaphore, audio, context, language, sampling_params)
            for audio, context, language in audio_items
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        transcripts: list[_GeneratedTranscript] = []
        for idx, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.warning("vLLM generate failed for segment %s: %s", idx, result)
                transcripts.append(_GeneratedTranscript(text="", language=""))
            else:
                transcripts.append(result)
        return transcripts

    async def _async_generate_raw(
        self,
        prompt: str,
        audio: np.ndarray,
        max_new_tokens: int | None = None,
    ) -> str:
        engine, _loop = self._ensure_engine()
        sampling_params = self._sampling_params_cls(
            temperature=0.01,
            max_tokens=max_new_tokens or self._max_new_tokens,
        )
        request_id = f"qwen-asr-{next(self._request_counter)}"
        semaphore = asyncio.Semaphore(self._max_in_flight)
        async with semaphore:
            generator = engine.generate(
                {"prompt": prompt, "multi_modal_data": {"audio": [audio]}},
                sampling_params,
                request_id,
            )
            final_output: Any = None
            async for output in generator:
                final_output = output
        return str(final_output.outputs[0].text if final_output and final_output.outputs else "")

    async def _async_encode_one(
        self,
        engine: Any,
        semaphore: asyncio.Semaphore,
        audio: np.ndarray,
        prompt: str,
    ) -> Any:
        request_id = f"qwen-align-{next(self._request_counter)}"
        async with semaphore:
            generator = engine.encode(
                {"prompt": prompt, "multi_modal_data": {"audio": audio}},
                self._pooling_params_cls(task="token_classify"),
                request_id,
            )
            final_output: Any = None
            async for output in generator:
                final_output = output
        if final_output is None:
            raise RuntimeError("Forced aligner returned no output")
        return final_output

    async def _async_align_items(
        self,
        items: list[tuple[np.ndarray, str]],
    ) -> list[Any]:
        if not items:
            return []
        engine, loop = self._ensure_aligner_engine()
        _ = loop
        semaphore = asyncio.Semaphore(self._max_in_flight)
        tasks = [
            self._async_encode_one(engine, semaphore, audio, prompt)
            for audio, prompt in items
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        outputs: list[Any] = []
        for idx, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.warning("vLLM align failed for segment %s: %s", idx, result)
                outputs.append(None)
            else:
                outputs.append(result)
        return outputs

    def _run_coroutine_sync(self, loop: asyncio.AbstractEventLoop, coro):
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    # ------------------------------------------------------------------
    # Sync fallback (QWEN_VLLM_SYNC_FALLBACK=1)
    # ------------------------------------------------------------------

    def _ensure_sync_llm(self) -> Any:
        if self._sync_llm is None:
            kwargs: dict[str, Any] = {
                "model": self._model_path,
                "gpu_memory_utilization": self._gpu_memory_utilization,
            }
            if self._max_model_len is not None:
                kwargs["max_model_len"] = self._max_model_len
            quantization = (os.getenv("QWEN_VLLM_QUANTIZATION") or "").strip()
            if quantization:
                kwargs["quantization"] = quantization
            self._sync_llm = getattr(self._vllm_module, "LLM")(**kwargs)
        return self._sync_llm

    def _ensure_sync_aligner(self) -> Any:
        if self._sync_aligner is None:
            if not self._forced_aligner_path:
                raise RuntimeError("word_timestamps requires a configured forced aligner model")
            aligner_utilization = _resolve_forced_aligner_gpu_memory_utilization(
                self._gpu_memory_utilization
            )
            self._sync_aligner = getattr(self._vllm_module, "LLM")(
                model=self._forced_aligner_path,
                runner="pooling",
                enforce_eager=True,
                gpu_memory_utilization=aligner_utilization,
                hf_overrides={
                    "architectures": ["Qwen3ASRForcedAlignerForTokenClassification"],
                },
            )
            llm_engine = getattr(self._sync_aligner, "llm_engine", None)
            if llm_engine is None:
                raise RuntimeError("Forced aligner did not expose a vLLM engine instance")
            config = llm_engine.vllm_config.model_config.hf_config
            self._timestamp_token_id = int(config.timestamp_token_id)
            self._timestamp_segment_time = float(config.timestamp_segment_time)
        return self._sync_aligner

    def _sync_generate_items(
        self,
        audio_items: list[tuple[np.ndarray, str, Optional[str]]],
        max_new_tokens: int | None = None,
    ) -> list[_GeneratedTranscript]:
        if not audio_items:
            return []
        llm = self._ensure_sync_llm()
        sampling_params = self._sampling_params_cls(
            temperature=0.01,
            max_tokens=max_new_tokens or self._max_new_tokens,
        )
        prompts = [
            {
                "prompt": _build_chat_prompt(
                    context=context, language=_normalize_language_name(language)
                ),
                "multi_modal_data": {"audio": [audio]},
            }
            for audio, context, language in audio_items
        ]
        with self._sync_generate_lock:
            outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)

        transcripts: list[_GeneratedTranscript] = []
        for output, (_audio, _context, language) in zip(outputs, audio_items):
            raw_text = str(output.outputs[0].text if output.outputs else "")
            parsed_language, parsed_text = _parse_asr_output(raw_text, _normalize_language_name(language))
            transcripts.append(_GeneratedTranscript(text=parsed_text, language=parsed_language))
        return transcripts

    def _run_generate(
        self,
        audio_items: list[tuple[np.ndarray, str, Optional[str]]],
        max_new_tokens: int | None = None,
    ) -> list[_GeneratedTranscript]:
        if self._sync_fallback:
            return self._sync_generate_items(audio_items, max_new_tokens)
        try:
            engine, loop = self._ensure_engine()
            return self._run_coroutine_sync(
                loop, self._async_generate_items(audio_items, max_new_tokens)
            )
        except Exception:
            logger.exception("Async vLLM generate failed; request dropped (no silent fallback)")
            raise

    # ------------------------------------------------------------------
    # Forced aligner post-processing
    # ------------------------------------------------------------------

    def _aligned_from_align_output(
        self, output: Any, tokens: list[str]
    ) -> list[dict[str, float | str]]:
        logits = output.outputs.data
        predictions = logits.argmax(dim=-1) if hasattr(logits, "argmax") else np.argmax(logits, axis=-1)
        ts_predictions = [
            float(pred.item() if hasattr(pred, "item") else pred) * float(self._timestamp_segment_time or 0.0)
            for tid, pred in zip(output.prompt_token_ids, predictions)
            if int(tid) == int(self._timestamp_token_id or -1)
        ]

        expected_timestamps = len(tokens) * 2
        if len(ts_predictions) < expected_timestamps:
            raise RuntimeError(
                "Forced aligner returned fewer timestamp predictions than expected: "
                f"expected={expected_timestamps}, got={len(ts_predictions)}, tokens={len(tokens)}"
            )

        aligned: list[dict[str, float | str]] = []
        for index, token in enumerate(tokens):
            start_ms = ts_predictions[index * 2]
            end_ms = ts_predictions[index * 2 + 1]
            if end_ms < start_ms:
                logger.warning(
                    "Forced aligner produced reversed timestamps for token=%r: start_ms=%s end_ms=%s",
                    token,
                    start_ms,
                    end_ms,
                )
                start_ms, end_ms = end_ms, start_ms
            aligned.append({"text": token, "start_ms": start_ms, "end_ms": end_ms})
        return aligned

    def _sync_align_items(self, items: list[tuple[np.ndarray, str, list[str]]]) -> list[list[dict[str, float | str]]]:
        aligner = self._ensure_sync_aligner()
        prompts = [
            {"prompt": prompt, "multi_modal_data": {"audio": audio}}
            for audio, prompt, _tokens in items
        ]
        with self._sync_generate_lock:
            outputs = aligner.encode(prompts, pooling_task="token_classify")
        return [
            self._aligned_from_align_output(output, tokens)
            for output, (_audio, _prompt, tokens) in zip(outputs, items)
        ]

    def align_transcript(
        self,
        audio_path: str,
        text: str,
        language: Optional[str] = None,
        audio: Optional[np.ndarray] = None,
    ) -> list[dict[str, float | str]]:
        tokens = _split_alignment_units(text)
        if not tokens:
            return []
        audio_array = audio if audio is not None else _load_audio(audio_path)
        prompt = _build_alignment_prompt(tokens)
        return self.align_transcripts([(audio_array, text, language)]) [0]

    def align_transcripts(
        self,
        items: list[tuple[np.ndarray, str, Optional[str]]],
    ) -> list[list[dict[str, float | str]]]:
        """Align multiple transcripts in one batched engine call."""
        prepared: list[tuple[int, np.ndarray, str, list[str]]] = []
        for index, (audio, text, _language) in enumerate(items):
            tokens = _split_alignment_units(text)
            if tokens:
                prepared.append((index, audio, _build_alignment_prompt(tokens), tokens))
        results: list[list[dict[str, float | str]]] = [[] for _ in items]
        if not prepared:
            return results

        if self._sync_fallback:
            aligned = self._sync_align_items(
                [(audio, prompt, tokens) for _idx, audio, prompt, tokens in prepared]
            )
        else:
            _engine, loop = self._ensure_aligner_engine()
            outputs = self._run_coroutine_sync(
                loop,
                self._async_align_items(
                    [(audio, prompt) for _idx, audio, prompt, _tokens in prepared]
                ),
            )
            aligned = []
            for output, (_idx, _audio, _prompt, tokens) in zip(outputs, prepared):
                if output is None:
                    aligned.append([])
                    continue
                try:
                    aligned.append(self._aligned_from_align_output(output, tokens))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Forced aligner post-processing failed: %s", exc)
                    aligned.append([])
        for (index, _audio, _prompt, _tokens), aligned_item in zip(prepared, aligned):
            results[index] = aligned_item
        return results

    def ensure_forced_aligner_loaded(self) -> None:
        if not self._forced_aligner_path:
            return
        if self._sync_fallback:
            self._ensure_sync_aligner()
        else:
            self._ensure_aligner_engine()

    # ------------------------------------------------------------------
    # Public sync API (keeps engine/router/websocket call sites intact)
    # ------------------------------------------------------------------

    def transcribe_text(
        self,
        audio_path: str,
        context: str = "",
        language: Optional[str] = None,
        enable_itn: bool = False,
    ) -> str:
        audio = self._audio_executor.submit(_load_audio, audio_path).result()
        transcript = self._run_generate([(audio, context, language)])[0]
        return normalize_asr_text(transcript.text, enable_itn=enable_itn)

    def transcribe_raw(
        self,
        audio_path: str,
        context: str = "",
        language: Optional[str] = None,
        word_timestamps: bool = False,
        enable_itn: bool = False,
    ) -> ASRRawResult:
        audio = _load_audio(audio_path)
        transcript = self._run_generate([(audio, context, language)])[0]
        text = normalize_asr_text(transcript.text, enable_itn=enable_itn)
        if not word_timestamps:
            return ASRRawResult(
                text=text,
                segments=[ASRSegmentResult(text=text, start_time=0.0, end_time=0.0)] if text else [],
            )

        aligned = self.align_transcript(audio_path=audio_path, text=text, language=language, audio=audio)
        word_tokens = [
            WordToken(
                text=str(item["text"]),
                start_time=round(float(item["start_ms"]) / 1000.0, 3),
                end_time=round(float(item["end_ms"]) / 1000.0, 3),
            )
            for item in aligned
        ]
        if not word_tokens:
            return ASRRawResult(
                text=text,
                segments=[ASRSegmentResult(text=text, start_time=0.0, end_time=0.0)] if text else [],
            )
        return ASRRawResult(
            text=text,
            segments=[
                ASRSegmentResult(
                    text=text,
                    start_time=word_tokens[0].start_time,
                    end_time=word_tokens[-1].end_time,
                    word_tokens=word_tokens,
                )
            ],
        )

    def transcribe_batch(
        self,
        audio_paths: list[str],
        context: str = "",
        language: Optional[str] = None,
        word_timestamps: bool = False,
        enable_itn: bool = False,
        audios: Optional[list[np.ndarray]] = None,
    ) -> list[ASRSegmentResult]:
        if not audio_paths:
            return []

        if audios is None:
            # Load audio concurrently (librosa decode is CPU-bound).
            futures = [self._audio_executor.submit(_load_audio, path) for path in audio_paths]
            audios = [future.result() for future in futures]

        # Submit ALL segments as individual engine requests so vLLM can batch
        # across concurrent HTTP requests (continuous batching).
        transcripts = self._run_generate(
            [(audio, context, language) for audio in audios]
        )

        results: list[ASRSegmentResult] = []
        aligned_by_idx: dict[int, list[dict[str, float | str]]] = {}
        if word_timestamps:
            align_inputs = [
                (audio, normalize_asr_text(transcript.text, enable_itn=enable_itn), language)
                for audio, transcript in zip(audios, transcripts)
            ]
            aligned_all = self.align_transcripts(align_inputs)
            aligned_by_idx = dict(enumerate(aligned_all))

        for idx, (audio_path, transcript) in enumerate(zip(audio_paths, transcripts)):
            _ = audio_path
            text = normalize_asr_text(transcript.text, enable_itn=enable_itn)
            if not word_timestamps:
                results.append(ASRSegmentResult(text=text, start_time=0.0, end_time=0.0))
                continue
            aligned = aligned_by_idx.get(idx) or []
            word_tokens = [
                WordToken(
                    text=str(item["text"]),
                    start_time=round(float(item["start_ms"]) / 1000.0, 3),
                    end_time=round(float(item["end_ms"]) / 1000.0, 3),
                )
                for item in aligned
            ]
            results.append(
                ASRSegmentResult(
                    text=text,
                    start_time=word_tokens[0].start_time if word_tokens else 0.0,
                    end_time=word_tokens[-1].end_time if word_tokens else 0.0,
                    word_tokens=word_tokens or None,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Realtime streaming (chunk-by-chunk decode)
    # ------------------------------------------------------------------

    def init_streaming_state(
        self,
        *,
        context: str = "",
        language: Optional[str] = None,
        chunk_size_sec: float = 2.0,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        max_new_tokens: int = 32,
    ) -> VLLMRealtimeState:
        normalized_language = _normalize_language_name(language) or ""
        return VLLMRealtimeState(
            prompt_raw=_build_chat_prompt(context=context, language=normalized_language or None),
            language=normalized_language,
            chunk_size_sec=chunk_size_sec,
            unfixed_chunk_num=unfixed_chunk_num,
            unfixed_token_num=unfixed_token_num,
            max_new_tokens=max_new_tokens,
            audio_buffer=np.array([], dtype=np.float32),
            audio_accum=np.array([], dtype=np.float32),
        )

    def _decode_stream(self, state: VLLMRealtimeState) -> VLLMRealtimeState:
        prefix = ""
        if state.chunk_id >= state.unfixed_chunk_num and state.raw_decoded:
            token_ids = self._tokenizer.encode(state.raw_decoded, add_special_tokens=False)
            rollback = token_ids[-state.unfixed_token_num:] if state.unfixed_token_num > 0 else []
            if rollback:
                prefix = self._tokenizer.decode(rollback, skip_special_tokens=False).replace("\ufffd", "")

        prompt = state.prompt_raw + prefix
        if self._sync_fallback:
            llm = self._ensure_sync_llm()
            sampling_params = self._sampling_params_cls(
                temperature=0.01, max_tokens=state.max_new_tokens
            )
            with self._sync_generate_lock:
                output = llm.generate(
                    [{"prompt": prompt, "multi_modal_data": {"audio": [state.audio_accum]}}],
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )[0]
            generated = str(output.outputs[0].text if output.outputs else "")
        else:
            engine, loop = self._ensure_engine()
            generated = self._run_coroutine_sync(
                loop,
                self._async_generate_raw(prompt, state.audio_accum, state.max_new_tokens),
            )

        parsed_language, parsed_text = _parse_asr_output(prefix + generated, state.language or None)
        state.raw_decoded = prefix + generated
        state.text = parsed_text
        state.language = parsed_language or state.language
        state.chunk_id += 1
        return state

    def feed_stream(self, pcm: np.ndarray, state: VLLMRealtimeState) -> VLLMRealtimeState:
        state.audio_buffer = np.concatenate([state.audio_buffer, pcm.astype(np.float32)])
        segment_size = int(max(state.chunk_size_sec, 0.1) * _DEFAULT_SAMPLE_RATE)
        while len(state.audio_buffer) >= segment_size:
            segment = state.audio_buffer[:segment_size].copy()
            state.audio_buffer = state.audio_buffer[segment_size:]
            state.audio_accum = np.concatenate([state.audio_accum, segment])
            state = self._decode_stream(state)
        return state

    def finish_stream(self, state: VLLMRealtimeState) -> VLLMRealtimeState:
        if len(state.audio_buffer) > 0:
            state.audio_accum = np.concatenate([state.audio_accum, state.audio_buffer])
            state.audio_buffer = np.array([], dtype=np.float32)
            state = self._decode_stream(state)
        elif state.chunk_id == 0 and len(state.audio_accum) > 0:
            state = self._decode_stream(state)
        return state

    def shutdown(self) -> None:
        for engine, loop in ((self._engine, self._engine_loop), (self._aligner_engine, self._aligner_loop)):
            if engine is not None and loop is not None:
                try:
                    loop.call_soon_threadsafe(engine.shutdown)
                except Exception:  # noqa: BLE001
                    pass
        for loop, thread in ((self._engine_loop, self._engine_thread), (self._aligner_loop, self._aligner_thread)):
            if loop is not None and thread is not None:
                try:
                    loop.call_soon_threadsafe(loop.stop)
                    thread.join(timeout=5)
                except Exception:  # noqa: BLE001
                    pass
        self._audio_executor.shutdown(wait=False)
