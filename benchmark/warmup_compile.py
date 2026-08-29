#!/usr/bin/env python3
"""Build-time warmup: load the ASR engine like production does and run one
transcription, so torch.compile artifacts are baked into /root/.cache/vllm.

Runs inside docker build (no GPU required for cache write? -- NO, this needs
a GPU; build on the L40 host with --gpus during 'docker build' via buildkit
gpu passthrough, or run this step as a post-build container and commit).
"""
import asyncio
import glob
import os
import sys
import time

t0 = time.time()


def resolve_snapshot(repo_dirname: str) -> str:
    matches = glob.glob(f"/root/.cache/huggingface/hub/{repo_dirname}/snapshots/*/")
    if not matches:
        raise FileNotFoundError(repo_dirname)
    return sorted(matches)[-1].rstrip("/")


MODEL = resolve_snapshot("models--Qwen--Qwen3-ASR-1.7B")

from vllm import SamplingParams  # noqa: E402
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
from vllm.engine.async_llm_engine import AsyncLLMEngine  # noqa: E402

# Production-equivalent engine settings (max_model_len from models.json)
args = AsyncEngineArgs(
    model=MODEL,
    gpu_memory_utilization=float(os.getenv("QWEN_GPU_MEMORY_UTILIZATION", "0.7")),
    max_model_len=8192,
    max_num_batched_tokens=16384,
    max_num_seqs=128,
    limit_mm_per_prompt={"audio": 1},
    renderer_num_workers=8,
    mm_processor_cache_gb=0,
)
engine = AsyncLLMEngine.from_engine_args(args)
print(f"[warmup t={time.time()-t0:.1f}s] engine created", flush=True)


async def main() -> None:
    import numpy as np

    audio = np.zeros(16000 * 2, dtype=np.float32)
    prompt = (
        "<|im_start|>system\nTranscribe the speech accurately.<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    sp = SamplingParams(temperature=0.01, max_tokens=16)
    req = {"prompt": prompt, "multi_modal_data": {"audio": [audio]}}
    out = None
    async for o in engine.generate(req, sp, "warmup-compile"):
        out = o
    print(f"[warmup t={time.time()-t0:.1f}s] output: {out.outputs[0].text!r}", flush=True)


asyncio.run(main())
print(f"[warmup t={time.time()-t0:.1f}s] DONE", flush=True)

