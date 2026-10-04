# -*- coding: utf-8 -*-
"""Route A step 1: inaSpeechSegmenter -> speech/music/noise timeline."""
import argparse, json
from inaSpeechSegmenter import Segmenter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("-o", "--output", default="segments.json")
    args = ap.parse_args()
    seg = Segmenter(detect_gender=False)
    result = seg(args.audio)
    out = []
    for label, start, end in result:
        out.append({"label": label, "start": round(start, 2), "end": round(end, 2)})
    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    for item in out:
        print(f"{item['start']:8.2f} - {item['end']:8.2f}  {item['label']}")
    print(f"[done] {len(out)} segments -> {args.output}")

if __name__ == "__main__":
    main()
