#!/usr/bin/env python3
"""修正 k（按真值人数）后重测：融合收益是否显现。"""
import sys, collections
import numpy as np
sys.path.insert(0, "/data/tmp/fusion_lab")
from fusion_lib import load_dataset, build_sim, to_nonneg_sim, cluster, evaluate

ds = load_dataset("/data/tmp/zhouxc_0925_run")
truth = np.array(ds["truth"]); scenes = np.array(ds["scenes"])
M = ["eres2net", "campplus"]
specs = {"eres": dict(kind="single", models=["eres2net"]),
         "camp": dict(kind="single", models=["campplus"]),
         "concat": dict(kind="concat", models=M),
         "scorez.5": dict(kind="score", models=M, w=0.5, znorm=True),
         "scorez.6": dict(kind="score", models=M, w=0.6, znorm=True)}
S = {n: to_nonneg_sim(build_sim(ds, sp)) for n, sp in specs.items()}

def run(kmode, method):
    tot = collections.defaultdict(lambda: [0, 0]); per = {}
    for g in sorted(set(scenes.tolist())):
        ii = [i for i in range(len(truth)) if scenes[i] == g]
        if len(ii) < 2: continue
        nk = len({truth[i] for i in ii})
        k = nk if kmode == "oracle" else min(ds["k_by_scene"].get(str(g), 2), len(ii))
        k = min(max(k, 1), len(ii))
        for n in specs:
            lab = cluster(S[n][np.ix_(ii, ii)], k, method)
            cnt = collections.defaultdict(collections.Counter)
            for l, i in zip(lab, ii): cnt[l][truth[i]] += 1
            ok = sum(c.most_common(1)[0][1] for c in cnt.values())
            tot[n][0] += ok; tot[n][1] += len(ii)
        per[g] = (len(ii), ds["k_by_scene"].get(str(g)), nk)
    return {n: round(v[0]/v[1], 4) for n, v in tot.items()}, per

for method in ("spectral", "average"):
    a, per = run("prod", method)
    b, _ = run("oracle", method)
    print("### method=%s" % method)
    print("%-10s %-8s %-8s %s" % ("variant", "k=生产", "k=真值", "delta"))
    for n in specs:
        print("%-10s %7.1f%% %7.1f%%   %+.1fpt" % (n, 100*a[n], 100*b[n], 100*(b[n]-a[n])))
    if method == "spectral":
        print("  每场 (n, k生产, k真值):", per)
    print()

