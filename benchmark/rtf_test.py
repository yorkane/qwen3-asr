#!/usr/bin/env python3
"""Single-request RTF test: measures end-to-end latency for different audio durations."""
import argparse, json, os, subprocess, sys, time, urllib.request

def get_duration(path):
    out = subprocess.run(["ffprobe","-v","quiet","-show_entries","format=duration","-of","csv=p=0",path],
                         capture_output=True, text=True).stdout.strip()
    return float(out)

def transcribe_once(url, key, audio_path, model, extra_fields=None):
    boundary = "RTFBOUNDARY"
    crlf = b"\r\n"
    parts = []
    def field(name, value):
        parts.append(f"--{boundary}{crlf.decode()}Content-Disposition: form-data; name=\"{name}\"{crlf.decode()}{crlf.decode()}".encode())
        parts.append(str(value).encode() + crlf)
    field("model", model)
    field("response_format", "json")
    for k, v in (extra_fields or {}).items():
        field(k, v)
    with open(audio_path, "rb") as f:
        audio = f.read()
    parts.append(f"--{boundary}{crlf.decode()}Content-Disposition: form-data; name=\"file\"; filename=\"test.mp3\"{crlf.decode()}Content-Type: audio/mpeg{crlf.decode()}{crlf.decode()}".encode())
    parts.append(audio)
    parts.append(crlf)
    parts.append(f"--{boundary}--{crlf.decode()}".encode())
    body = b"".join(parts)
    req = urllib.request.Request(url + "/v1/audio/transcriptions", data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    elapsed = time.perf_counter() - t0
    return elapsed, data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--key", default=None)
    ap.add_argument("--model", default="qwen3-asr-1.7b")
    ap.add_argument("--audio", nargs="+", required=True)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--verbose-json", action="store_true")
    ap.add_argument("--word-timestamps", action="store_true")
    ap.add_argument("--speaker", action="store_true")
    args = ap.parse_args()
    extra = {}
    if args.verbose_json:
        extra["response_format"] = "verbose_json"
    if args.word_timestamps:
        extra["timestamp_granularities[]"] = "word"
    if args.speaker:
        extra["enable_speaker_diarization"] = "true"
    print(f"{'file':<50} {'dur_s':>7} {'mode':<12} {'avg_s':>8} {'p95_s':>8} {'rtf':>6} {'xrt':>6}")
    results = []
    for audio in args.audio:
        dur = get_duration(audio)
        mode = "plain"
        if args.word_timestamps: mode = "wordts"
        elif args.verbose_json: mode = "verbose"
        if args.speaker: mode += "+spk"
        times = []
        for _ in range(args.repeat):
            elapsed, _ = transcribe_once(args.url, args.key, audio, args.model, extra)
            times.append(elapsed)
        times.sort()
        avg = sum(times)/len(times)
        p95 = times[int(0.95*(len(times)-1))]
        rtf = avg / dur
        xrt = dur / avg
        print(f"{os.path.basename(audio):<50} {dur:>7.1f} {mode:<12} {avg:>8.3f} {p95:>8.3f} {rtf:>6.3f} {xrt:>6.2f}")
        results.append({"file": audio, "duration": dur, "mode": mode, "avg": avg, "p95": p95, "rtf": rtf})
    print(json.dumps({"results": results}))

if __name__ == "__main__":
    main()
