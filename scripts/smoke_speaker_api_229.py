#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""229 ASR 声纹端点冒烟：segments / diarize / threshold 三条分支。"""
import json
import subprocess
import sys

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:17003"  # 在 229 宿主机执行: python3 scripts/smoke_speaker_api_229.py
HDR = "Authorization: Bearer devideo2026asrkey"
WAV = "/data/tmp/moss-test.wav"


def post(path, forms):
    cmd = ["curl", "-s", "-m", "300", "-X", "POST", BASE + path, "-H", HDR]
    for f in forms:
        cmd += ["-F", f]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    return json.loads(out)


fails = 0

# 1) 显式 segments
d = post("/v1/audio/embeddings", [
    "file=@" + WAV,
    'segments=[{"start":0,"end":5},{"start":30,"end":60}]',
    "models=eres2net",
])
bounds = [(s["start"], s["end"]) for s in d.get("segments", [])]
ok = d.get("segment_count") == 2 and bounds == [(0.0, 5.0), (30.0, 60.0)] and d.get("models") == ["eres2net"]
print("[1] segments:", "OK" if ok else "FAIL", d.get("segment_count"), bounds, d.get("models"))
fails += 0 if ok else 1

# 2) diarize 自动切段
d = post("/v1/audio/embeddings", ["file=@" + WAV, "diarize=true", "models=both"])
n = d.get("segment_count")
by_model = d.get("embeddings_by_model", {})
ok = (n or 0) > 1 and set(d.get("models", [])) == {"eres2net", "campplus"} \
    and set(by_model.keys()) == {"eres2net", "campplus"} \
    and len(by_model.get("eres2net", [])) == n
print("[2] diarize:", "OK" if ok else "FAIL", "n=", n, "models=", d.get("models"), "err=", d.get("error_code"))
fails += 0 if ok else 1

# 3) threshold 判定
d = post("/v1/audio/speaker-compare", ["file=@" + WAV, "file_b=@" + WAV, "threshold=0.5"])
ok = d.get("same_speaker") is True and abs(d.get("cosine", 0) - 1.0) < 0.05
print("[3] compare-threshold:", "OK" if ok else "FAIL", d.get("same_speaker"), d.get("cosine"), d.get("per_model"))
fails += 0 if ok else 1

print("SMOKE_" + ("PASS" if fails == 0 else "FAIL"))
sys.exit(fails)
