#!/usr/bin/env python3
"""逐行翻盘分析：融合 vs 最佳单模型，看谁救回谁、以及按场/时长的分布。"""
import sys, json, collections
import numpy as np
sys.path.insert(0, "/data/tmp/fusion_lab")
from fusion_lib import load_dataset, build_sim, to_nonneg_sim, loo_nn_detail

ds = load_dataset(sys.argv[1] if len(sys.argv) > 1 else "/data/tmp/zhouxc_0925_run")
M = ["eres2net", "campplus"]
variants = {
    "eres": dict(kind="single", models=["eres2net"]),
    "camp": dict(kind="single", models=["campplus"]),
    "concat": dict(kind="concat", models=M),
    "scorez": dict(kind="score", models=M, w=0.5, znorm=True),
}
res = {}
for name, spec in variants.items():
    S = to_nonneg_sim(build_sim(ds, spec))
    _, sa, la, wrong = loo_nn_detail(S, ds["scenes"], ds["truth"], ds["win_dur"], lids=ds["lids"])
    res[name] = {(w["line_id"], w["scene"]) for w in wrong}
    print("%-7s errors=%3d  short_err=%3d" % (name, len(res[name]),
          sum(1 for k in res[name] if ds["win_dur"][ds["lids"].index(k[0])] < 1.5)))
base = res["camp"] | res["eres"]
print()
for name in ("concat", "scorez"):
    fixed = base - res[name]
    broke = res[name] - (res["camp"] & res["eres"])
    print("== %s: 两模型都错->融合救回 %d 行；融合新错(单模型原本至少一个对) %d 行" % (name, len(fixed), len(broke)))
    sc = collections.Counter(k[1] for k in fixed); scb = collections.Counter(k[1] for k in broke)
    print("   救回分布", dict(sc))
    print("   新错分布", dict(scb))
    lids = {l: i for i, l in enumerate(ds["lids"])}
    for k in sorted(fixed):
        i = lids[k[0]]
        print("   FIX sc%s %s dur=%.2f truth=%s" % (k[1], k[0], ds["win_dur"][i], ds["truth"][i]))
    for k in sorted(broke):
        i = lids[k[0]]
        print("   BROKE sc%s %s dur=%.2f truth=%s" % (k[1], k[0], ds["win_dur"][i], ds["truth"][i]))
print()
print("=== 每场：单模型(camp) LOO vs 融合 concat，以及 k vs 真值人数 ===")
S_c = to_nonneg_sim(build_sim(ds, variants["camp"]))
S_f = to_nonneg_sim(build_sim(ds, variants["concat"]))
for g in sorted(set(ds["scenes"])):
    ii = [i for i in range(len(ds["lids"])) if ds["scenes"][i] == g]
    if len(ii) < 2:
        print("sc%s n=%d (skip)" % (g, len(ii))); continue
    nchar = len({ds["truth"][i] for i in ii})
    def acc_of(S):
        _, _, _, wrong = loo_nn_detail(S, ds["scenes"], ds["truth"], ds["win_dur"], lids=ds["lids"])
        bad = {w["line_id"] for w in wrong}
        good = sum(1 for i in ii if ds["lids"][i] not in bad)
        return good, len(ii)
    gc, tc = acc_of(S_c); gf, tf = acc_of(S_f)
    print("sc%s n=%-3d k=%s nchar=%d  camp %d/%d=%.2f  fusion %d/%d=%.2f" % (
        g, len(ii), ds["k_by_scene"].get(g), nchar, gc, tc, gc/tc, gf, tf, gf/tf))

