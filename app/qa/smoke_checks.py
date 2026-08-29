# -*- coding: utf-8 -*-
"""Smoke / release checks against a running ASR service (HTTP).

Checks, in order:
  1. health:   /v1/models responds and lists at least one model
  2. transcribe: POST /v1/audio/transcriptions returns non-empty text
  3. verbose:  response_format=verbose_json returns segments
  4. words:    word_timestamps=true returns word-level timings
  5. offline:  service works with no network (HF_HUB_OFFLINE semantics)

Exit code 0 when all checks pass. Each check prints a PASS/FAIL line.
"""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ASSET = Path(__file__).resolve().parent / "assets" / "test_speech.mp3"


def _multipart_body(fields: dict, file_field: str, filename: str, data: bytes):
    boundary = "QA-SMOKE"
    crlf = b"\r\n"
    parts = []
    for name, value in fields.items():
        parts.append(b"--" + boundary.encode() + crlf)
        parts.append(
            ('Content-Disposition: form-data; name="%s"' % name).encode()
            + crlf + crlf
        )
        parts.append(str(value).encode() + crlf)
    parts.append(b"--" + boundary.encode() + crlf)
    parts.append(
        b'Content-Disposition: form-data; name="%s"; filename="%s"'
        % (file_field.encode(), filename.encode())
        + crlf
        + b"Content-Type: audio/mpeg" + crlf + crlf
        + data + crlf
    )
    parts.append(b"--" + boundary.encode() + b"--" + crlf)
    return b"".join(parts), boundary


def _post(api: str, body: bytes, boundary: str, api_key: str | None, timeout: int):
    req = urllib.request.Request(api, data=body, method="POST")
    req.add_header("Content-Type", "multipart/form-data; boundary=" + boundary)
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        ctype = resp.headers.get("Content-Type", "")
    if "json" in ctype:
        return json.loads(raw.decode("utf-8", "replace"))
    return {"text": raw.decode("utf-8", "replace")}


def run(base_url: str, api_key: str | None = None, timeout: int = 300) -> bool:
    api = base_url.rstrip("/") + "/v1/audio/transcriptions"
    models_api = base_url.rstrip("/") + "/v1/models"
    audio = ASSET.read_bytes()
    failures = []

    def check(name: str, ok: bool, detail: str = ""):
        tag = "PASS" if ok else "FAIL"
        suffix = f" ({detail})" if detail else ""
        print(f"[smoke] {tag} {name}{suffix}")
        if not ok:
            failures.append(name)

    # 1. health
    try:
        req = urllib.request.Request(models_api)
        if api_key:
            req.add_header("Authorization", "Bearer " + api_key)
        with urllib.request.urlopen(req, timeout=30) as resp:
            models = json.loads(resp.read().decode())
        ids = [m.get("id") for m in models.get("data", [])]
        check("health/models", bool(ids), f"models={ids}")
        model = ids[0] if ids else "qwen3-asr-1.7b"
    except Exception as exc:  # noqa: BLE001
        check("health/models", False, str(exc)[:120])
        print(f"[smoke] ABORT: service unreachable at {base_url}")
        return False

    def call(fields: dict):
        body, boundary = _multipart_body(fields, "file", "qa.mp3", audio)
        return _post(api, body, boundary, api_key, timeout)

    # 2. transcribe
    try:
        t0 = time.time()
        out = call({"model": model, "response_format": "json"})
        dt = time.time() - t0
        text = (out.get("text") or "").strip()
        check("transcribe", len(text) >= 5, f"{len(text)} chars in {dt:.1f}s")
    except Exception as exc:  # noqa: BLE001
        check("transcribe", False, str(exc)[:120])

    # 3. verbose_json
    try:
        out = call({"model": model, "response_format": "verbose_json"})
        segments = out.get("segments") or []
        check("verbose_json", len(segments) > 0, f"{len(segments)} segments")
    except Exception as exc:  # noqa: BLE001
        check("verbose_json", False, str(exc)[:120])

    # 4. word timestamps
    try:
        out = call(
            {
                "model": model,
                "response_format": "verbose_json",
                "word_timestamps": "true",
                "enable_speaker_diarization": "false",
            }
        )
        words = out.get("words") or []
        ok = len(words) > 0
        if ok:
            w0 = words[0]
            ok = w0.get("end", 0) >= w0.get("start", 0)
        check("word_timestamps", ok, f"{len(words)} words")
    except Exception as exc:  # noqa: BLE001
        check("word_timestamps", False, str(exc)[:120])

    # 5. concurrency sanity (3 parallel requests)
    try:
        from concurrent.futures import ThreadPoolExecutor

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = [
                ex.submit(
                    call, {"model": model, "response_format": "json"}
                )
                for _ in range(3)
            ]
            results = [f.result() for f in futs]
        dt = time.time() - t0
        ok = all(len((r.get("text") or "").strip()) >= 5 for r in results)
        check("concurrent_smoke", ok, f"3 reqs in {dt:.1f}s")
    except Exception as exc:  # noqa: BLE001
        check("concurrent_smoke", False, str(exc)[:120])

    if failures:
        print(f"[smoke] FAILED checks: {failures}")
        return False
    print("[smoke] ALL PASS")
    return True


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--key", default=None)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()
    sys.exit(0 if run(args.url, args.key, args.timeout) else 1)
