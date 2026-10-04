# -*- coding: utf-8 -*-
"""Route A step 2: zero-shot label music segments with CLAP.

Reads the speech/music/noise timeline from step 1, then slides a window over
each 'music' segment and scores candidate mood/style prompts with CLAP.
"""
import argparse, json, os
import numpy as np
import soundfile as sf

LABELS = ["激昂昂扬", "悲伤哀愁", "欢乐喜悦", "史诗宏大", "紧张悬疑", "舒缓宁静", "浪漫温柔"]
LABEL_PROMPTS = {
    "激昂昂扬": "uplifting triumphant energetic music",
    "悲伤哀愁": "sad melancholy sorrowful music",
    "欢乐喜悦": "happy joyful cheerful music",
    "史诗宏大": "epic grand cinematic orchestral music",
    "紧张悬疑": "tense suspense thriller music",
    "舒缓宁静": "calm relaxing peaceful ambient music",
    "浪漫温柔": "romantic gentle tender music",
}

def load_audio(path, sr=48000):
    data, orig_sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, orig_sr

def resample(data, orig_sr, target_sr):
    if orig_sr == target_sr:
        return data
    import resampy
    return resampy.resample(data, orig_sr, target_sr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="full audio wav used in step 1")
    ap.add_argument("--segments", default="segments.json")
    ap.add_argument("-o", "--output", default="music_labels.json")
    ap.add_argument("--window", type=float, default=5.0, help="window seconds")
    ap.add_argument("--hop", type=float, default=2.5, help="hop seconds")
    ap.add_argument("--model", default="laion/clap-htsat-unfused")
    ap.add_argument("--min-duration", type=float, default=3.0)
    args = ap.parse_args()

    with open(args.segments) as f:
        segs = json.load(f)

    data, orig_sr = load_audio(args.audio)
    sr = 48000
    data = resample(data, orig_sr, sr)

    from transformers import pipeline
    pipe = pipeline("zero-shot-audio-classification", model=args.model)

    prompts = [LABEL_PROMPTS[l] for l in LABELS]
    timeline = []
    for s in segs:
        if s["label"] != "music":
            continue
        dur = s["end"] - s["start"]
        if dur < args.min_duration:
            continue
        t = s["start"]
        while t + args.window <= s["end"] + 1e-6:
            t0 = max(t, s["start"])
            t1 = min(t + args.window, s["end"])
            i0, i1 = int(t0 * sr), int(t1 * sr)
            chunk = data[i0:i1]
            if len(chunk) < sr * 1:  # need >= 1s
                break
            # CLAP expects fixed 10s input; pad/truncate handled by pipeline via sampling_rate
            res = pipe({"array": chunk, "sampling_rate": sr}, candidate_labels=prompts, top_k=3)
            top = []
            for r in res[:3]:
                # map prompt back to chinese label
                lab = [k for k, v in LABEL_PROMPTS.items() if v == r["label"]][0]
                top.append({"label": lab, "score": round(float(r["score"]), 4)})
            timeline.append({"start": round(t0, 2), "end": round(t1, 2), "top": top})
            t += args.hop

    with open(args.output, "w") as f:
        json.dump(timeline, f, ensure_ascii=False, indent=2)
    for item in timeline:
        best = item["top"][0] if item["top"] else {"label": "?", "score": 0}
        print(f"{item['start']:8.2f} - {item['end']:8.2f}  {best['label']} ({best['score']:.3f})")
    print(f"[done] {len(timeline)} windows -> {args.output}")

if __name__ == "__main__":
    main()
