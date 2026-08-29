#!/usr/bin/env python3
"""将 qwen3-asr 服务的 verbose_json 转写结果转换为 talk session JSON 格式。

用法:
  # 直接转写音频文件(调用本地服务)
  python3 scripts/verbose_to_session.py --audio demo2/talk1.mp3 --output demo2/talk1_converted.json

  # 或从已有 verbose_json 转换
  python3 scripts/verbose_to_session.py --verbose-json demo2/talk1_raw_verbose.json \
      --audio demo2/talk1.mp3 --output demo2/talk1_converted.json

session 格式与 demo2/talk1.json 一致: 顶层含 sessionId/fullText/segments/speakers,
每个 segment 含 startTime/endTime/speakerName/wordTokens(绝对时间戳) 等字段。
"""
import argparse
import json
import mimetypes
import os
import sys
import time
import urllib.request
import uuid
from datetime import datetime

DEFAULT_API = "http://127.0.0.1:17003/v1/audio/transcriptions"
DEFAULT_MODEL = "qwen3-asr-1.7b"

# 说话人配色板(与 talk1.json 一致, 按出现顺序分配)
SPEAKER_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
                  "#DDA0DD", "#F4A460", "#87CEEB"]


def call_api(audio_path: str, api: str, api_key: str, model: str) -> dict:
    """调用 OpenAI 兼容转写接口, 返回 verbose_json。"""
    boundary = "----SessionConvert"
    CRLF = b"\r\n"
    with open(audio_path, "rb") as f:
        audio = f.read()

    def field(name: str, value: str) -> bytes:
        return (b"--" + boundary.encode() + CRLF +
                f'Content-Disposition: form-data; name="{name}"'.encode() + CRLF + CRLF +
                value.encode() + CRLF)

    body = (field("model", model) +
            field("response_format", "verbose_json") +
            field("word_timestamps", "true") +
            field("enable_speaker_diarization", "true") +
            b"--" + boundary.encode() + CRLF +
            b'Content-Disposition: form-data; name="file"; filename="audio"' + CRLF +
            b"Content-Type: " + (mimetypes.guess_type(audio_path)[0] or "application/octet-stream").encode() + CRLF + CRLF +
            audio + CRLF +
            b"--" + boundary.encode() + b"--" + CRLF)

    req = urllib.request.Request(api, data=body, method="POST")
    req.add_header("Content-Type", "multipart/form-data; boundary=" + boundary)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        data = json.load(r)
    print(f"转写完成: {time.time() - t0:.1f}s, {len(data.get('segments', []))} 段",
          file=sys.stderr)
    return data


def assign_words_to_segments(segments: list, words: list) -> list:
    """把全局词级时间戳切分到各段, 返回每段的 wordTokens 列表。

    优先按累计字符数顺序对齐(词与字符 1:1 时精确);
    数量不一致时退化为按词起点时间归属。
    """
    seg_chars = [len(s["text"].replace(" ", "")) for s in segments]
    result = [[] for _ in segments]
    if not words:
        return result

    if sum(seg_chars) == len(words):
        # 顺序对齐: 词流按段文本字符数依次切分
        idx = 0
        for i, cnt in enumerate(seg_chars):
            for w in words[idx:idx + cnt]:
                result[i].append({"text": w["word"],
                                  "startTime": w["start"],
                                  "endTime": w["end"]})
            idx += cnt
        return result

    # 退化: 按词起点时间归属
    j = 0
    n = len(segments)
    for w in words:
        ws = w["start"]
        while j < n - 1 and ws >= segments[j + 1]["start"] - 1e-6:
            j += 1
        result[j].append({"text": w["word"],
                          "startTime": w["start"],
                          "endTime": w["end"]})
    return result


def speaker_sort_key(name: str):
    """按'说话人N'的数字排序。"""
    digits = "".join(c for c in name if c.isdigit())
    return int(digits) if digits else 0


def build_session(verbose: dict, audio_path: str | None, title: str | None) -> dict:
    segs_in = verbose.get("segments", [])
    words = verbose.get("words") or []
    word_tokens = assign_words_to_segments(segs_in, words)

    # 校验: 每段 token 数应与文本字符数一致(含标点)
    for i, (s, wt) in enumerate(zip(segs_in, word_tokens)):
        if wt and len(wt) != len(s["text"].replace(" ", "")):
            print(f"警告: 段{i} token数({len(wt)}) != 文本长度({len(s['text'])})",
                  file=sys.stderr)

    segments = []
    for i, s in enumerate(segs_in):
        speaker_name = s.get("speaker") or "说话人1"
        segments.append({
            "startTime": s["start"],
            "endTime": s["end"],
            "speakerId": None,
            "speakerName": speaker_name,
            "speakerLabel": speaker_name,
            "rawText": s["text"],
            "correctedText": None,
            "polishedText": None,
            "modifiedText": None,
            "displayText": s["text"],
            "wordTokens": word_tokens[i],
            "confidence": 1.0,
            "isEdited": False,
            "editSource": None,
            "editedAt": None,
            "reviewStatus": "valid",
        })

    # 说话人列表: 按说话人编号排序分配 id 和颜色
    names = []
    for s in segs_in:
        name = s.get("speaker") or "说话人1"
        if name not in names:
            names.append(name)
    names.sort(key=speaker_sort_key)
    speakers = [{"id": f"speaker_{i}",
                 "name": name,
                 "color": SPEAKER_COLORS[i % len(SPEAKER_COLORS)]}
                for i, name in enumerate(names)]

    created = datetime.now().isoformat()
    file_size = os.path.getsize(audio_path) if audio_path else None

    return {
        "sessionId": str(uuid.uuid4()),
        "mode": "file",
        "state": "completed",
        "title": title or (os.path.basename(audio_path) if audio_path else "audio"),
        "language": verbose.get("language", "zh"),
        "duration": verbose.get("duration", 0.0),
        "fullText": "\n".join(s["text"] for s in segs_in),
        "segments": segments,
        "speakers": speakers,
        "fileId": str(uuid.uuid4()),
        "fileSize": file_size,
        "enableSpeakerDiarization": True,
        "meetingDate": None,
        "meetingType": "business",
        "visibility": "private",
        "createdByCode": "demo@asrclient.dev",
        "createdByName": "Demo User",
        "createdAt": created,
        "completedAt": datetime.now().isoformat(),
        "error": None,
    }


def main():
    ap = argparse.ArgumentParser(description="verbose_json -> talk session JSON")
    ap.add_argument("--audio", help="音频文件(获取文件大小/标题; 无 --verbose-json 时调用API转写)")
    ap.add_argument("--verbose-json", help="已有的 verbose_json 结果文件")
    ap.add_argument("--output", required=True, help="输出 session JSON 路径")
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--api-key", default=os.environ.get("ASR_API_KEY", "devideo2026asrkey"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    if args.verbose_json:
        with open(args.verbose_json) as f:
            verbose = json.load(f)
    elif args.audio:
        verbose = call_api(args.audio, args.api, args.api_key, args.model)
    else:
        ap.error("需要 --audio 或 --verbose-json")

    session = build_session(verbose, args.audio, args.title)
    with open(args.output, "w") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)
    print(f"已写入 {args.output}: {len(session['segments'])} 段, "
          f"{len(session['speakers'])} 个说话人", file=sys.stderr)


if __name__ == "__main__":
    main()
