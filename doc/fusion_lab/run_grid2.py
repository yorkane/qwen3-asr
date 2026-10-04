#!/usr/bin/env python3
import json, sys
import numpy as np
sys.path.insert(0, "/data/tmp/fusion_lab")
from fusion_lib import load_dataset, build_sim, evaluate

ROOT = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else "x"
ds = load_dataset(ROOT)
print("rows=%d scenes=%d models=%s win_dur_median=%.2f short(<1.5s)=%d" % (
    len(ds["lids"]), len(set(ds["scenes"])), list(ds["embs"]),
    float(np.median(ds["win_dur"])), int((ds["win_dur"] < 1.5).sum())))
M = ["eres2net", "campplus"]
specs = [
    ("eres only",        dict(kind="single", models=["eres2net"])),
    ("camp only",        dict(kind="single", models=["campplus"])),
    ("concat",           dict(kind="concat", models=M)),
    ("concat w.6/.4",    dict(kind="concat", models=M, w=0.6)),
    ("score z w=.5",     dict(kind="score", models=M, w=0.5, znorm=True)),
    ("score z w=.6",     dict(kind="score", models=M, w=0.6, znorm=True)),
    ("score z w=.7",     dict(kind="score", models=M, w=0.7, znorm=True)),
    ("score raw w=.5",   dict(kind="score", models=M, w=0.5, znorm=False)),
    ("score z len .5/.7", dict(kind="score", models=M, w=0.5, short_w=0.7, znorm=True, lenaware=True)),
    ("score z len .5/.8", dict(kind="score", models=M, w=0.5, short_w=0.8, znorm=True, lenaware=True)),
]
out = {}
for method in ("spectral", "average", "kmeans"):
    print("### method=" + method)
    print("%-18s %6s %8s %7s  %-20s %-20s" % ("variant", "acc", "ARI", "LOO", "margin_short", "margin_long"))
    for name, spec in specs:
        try:
            S = build_sim(ds, spec)
            r = evaluate(ds, S, method=method)
            out["%s|%s" % (method, name)] = r
            print("%-18s %5.1f%% %8.3f %6.1f%%  %-20s %-20s" % (
                name, 100 * r["acc"], r["macro_ari"], 100 * r["loo_nn"],
                "/".join(str(v) for v in r["margin_short"]),
                "/".join(str(v) for v in r["margin_long"])))
        except Exception as e:
            print("%-18s ERROR %r" % (name, e))
    print()
json.dump(out, open("/data/tmp/fusion_lab/results_%s.json" % TAG, "w"), ensure_ascii=False, indent=1)
print("wrote results_%s.json" % TAG)

