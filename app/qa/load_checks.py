# -*- coding: utf-8 -*-
"""Concurrency load test against a running ASR service.

Sweeps concurrency levels and reports throughput (rps), latency
percentiles and success rate per level. Uses the bundled audio asset so
it runs fully offline inside the image.

Usage:
    python -m app.qa.load_checks --url http://127.0.0.1:8000 \
        [--key KEY] [--concurrency 1,4,8,16,32,64] [--total 64]
"""

import argparse
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ASSET = Path(__file__).resolve().parent / "assets" / "test_speech.mp3"


def _build_body(audio: bytes, model: str):
    boundary = "QA-LOAD"
    crlf = b"\r\n"
    parts = []

    def field(name, value):
        parts.append(b"--" + boundary.encode() + crlf)
        parts.append(
            ('Content-Disposition: form-data; name="%s"' % name).encode()
            + crlf + crlf
        )
        parts.append(str(value).encode() + crlf)

    field("model", model)
    field("response_format", "json")
    field("enable_speaker_diarization", "false")
    parts.append(b"--" + boundary.encode() + crlf)
    parts.append(
        b'Content-Disposition: form-data; name="file"; filename="qa.mp3"'
        + crlf
        + b"Content-Type: audio/mpeg" + crlf + crlf
        + audio + crlf
    )
    parts.append(b"--" + boundary.encode() + b"--" + crlf)
    return b"".join(parts), boundary


def _one_req(api: str, body: bytes, boundary: str, key: str | None, timeout: int):
    req = urllib.request.Request(api, data=body, method="POST")
    req.add_header("Content-Type", "multipart/form-data; boundary=" + boundary)
    if key:
        req.add_header("Authorization", "Bearer " + key)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
        return time.time() - t0, "ok"
    except urllib.error.HTTPError as exc:
        return time.time() - t0, f"http_{exc.code}"
    except Exception as exc:  # noqa: BLE001
        return time.time() - t0, f"err:{type(exc).__name__}"


def _percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * pct / 100))]


def run(
    base_url: str,
    api_key: str | None = None,
    concurrency: str = "1,4,8,16,32,64",
    total: int = 64,
    timeout: int = 900,
    model: str | None = None,
    min_speedup: float | None = None,
) -> bool:
    api = base_url.rstrip("/") + "/v1/audio/transcriptions"
    audio = ASSET.read_bytes()

    if model is None:
        import json

        req = urllib.request.Request(base_url.rstrip("/") + "/v1/models")
        if api_key:
            req.add_header("Authorization", "Bearer " + api_key)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
            model = ids[0] if ids else "qwen3-asr-1.7b"
        except Exception:  # noqa: BLE001
            model = "qwen3-asr-1.7b"

    body, boundary = _build_body(audio, model)
    print(f"asset={ASSET.name} ({len(audio) / 1e6:.1f}MB) model={model}")
    print(
        "%-6s %7s %7s %7s %7s %7s %8s %8s"
        % ("conc", "ok/tot", "avg_s", "p50_s", "p95_s", "max_s", "rps", "speedup")
    )

    base_rps = None
    peak = {"conc": 0, "rps": 0.0}
    all_ok = True
    for level in [int(x) for x in concurrency.split(",")]:
        lats = []
        ok = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=level) as ex:
            futs = [
                ex.submit(_one_req, api, body, boundary, api_key, timeout)
                for _ in range(total)
            ]
            for fut in futs:
                dt, status = fut.result()
                lats.append(dt)
                if status == "ok":
                    ok += 1
        wall = time.time() - t0
        rps = ok / wall if wall > 0 else 0.0
        if base_rps is None and ok > 0:
            base_rps = rps
        if ok == total and rps > peak["rps"]:
            peak = {"conc": level, "rps": rps}
        if ok < total:
            all_ok = False
        speedup = rps / base_rps if base_rps else 0.0
        print(
            "%-6d %4d/%-3d %7.2f %7.2f %7.2f %7.2f %8.3f %8.2f"
            % (
                level, ok, total,
                sum(lats) / len(lats) if lats else 0.0,
                _percentile(lats, 50), _percentile(lats, 95),
                max(lats) if lats else 0.0,
                rps, speedup,
            )
        )

    print(
        f"[load] peak throughput {peak['rps']:.2f} rps at concurrency "
        f"{peak['conc']} (baseline {base_rps or 0:.2f} rps @ conc 1)"
    )

    if not all_ok:
        print("[load] FAIL: some requests failed")
        return False
    if min_speedup is not None and base_rps:
        if peak["rps"] / base_rps < min_speedup:
            print(
                f"[load] FAIL: speedup {peak['rps'] / base_rps:.2f}x below "
                f"threshold {min_speedup}x (continuous batching not effective)"
            )
            return False
    print("[load] PASS")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--key", default=None)
    ap.add_argument("--concurrency", default="1,4,8,16,32,64")
    ap.add_argument("--total", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--model", default=None)
    ap.add_argument(
        "--min-speedup",
        type=float,
        default=None,
        help="fail if peak rps / baseline rps is below this value",
    )
    args = ap.parse_args()
    sys.exit(
        0
        if run(
            args.url, args.key, args.concurrency, args.total, args.timeout,
            args.model, args.min_speedup,
        )
        else 1
    )
