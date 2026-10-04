#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S10 双模型声纹融合实验台（离线，只读 npy，不占 GPU）。

对照：单模型 / 拼接融合 / 分数级加权融合(robust z) / 时长感知权重。
指标：cluster_acc(每簇多数标签, 行加权)、macro ARI、LOO-NN 同人率、attribution margin 分位。
"""
import collections
import json
import os

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score


def l2(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + 1e-9)


def load_dataset(root, models=("eres2net", "campplus")):
    rows = json.load(open(os.path.join(root, "rows_attr.json"), encoding="utf-8"))
    ref_path = os.path.join(root, "s10_ref0918.json")
    use_ref = os.path.exists(ref_path)
    truth = {}
    if use_ref:
        ref = json.load(open(ref_path, encoding="utf-8"))
        for lid, v in ref.items():
            truth[lid] = v["name"] if isinstance(v, dict) else str(v)
    else:
        for r in rows:
            truth[r["line_id"]] = r["character_name"]
    kb_path = os.path.join(root, "k_by_scene.json")
    k_by_scene = {}
    if os.path.exists(kb_path):
        kb = json.load(open(kb_path, encoding="utf-8"))
        k_by_scene = {str(key): int(val) for key, val in kb.items()}
    else:
        by_scene = collections.defaultdict(set)
        for r in rows:
            by_scene[str(r["scene_no"])].add(truth[r["line_id"]])
        k_by_scene = {s: max(1, len(v)) for s, v in by_scene.items()}
    lids = [r["line_id"] for r in rows]
    scenes = [str(r["scene_no"]) for r in rows]
    starts = [float(r.get("start", 0.0)) for r in rows]
    ends = [float(r.get("end", 0.0)) for r in rows]
    line_dur = [max(0.0, e - s) for s, e in zip(starts, ends)]
    win_dur = list(line_dur)
    wpath = os.path.join(root, "s10_emb", "row_windows.json")
    if os.path.exists(wpath):
        wmap = {w["line_id"]: w["window"] for w in json.load(open(wpath, encoding="utf-8"))}
        win_dur = [
            max(0.0, wmap[l][1] - wmap[l][0]) if l in wmap else d
            for l, d in zip(lids, line_dur)
        ]
    embs = {}
    for m in models:
        p = os.path.join(root, "s10_emb", "emb_%s.npy" % m)
        if not os.path.exists(p):
            p2 = os.path.join(root, "s10_emb3", "emb_%s.npy" % m)
            p = p2 if os.path.exists(p2) else p
        if not os.path.exists(p):
            continue
        X = np.load(p).astype(np.float64)
        if len(X) != len(lids):
            raise SystemExit("row count mismatch: %s %d vs %d" % (p, len(X), len(lids)))
        embs[m] = l2(X)
    return {
        "lids": lids, "scenes": scenes, "truth": [truth[l] for l in lids],
        "k_by_scene": k_by_scene, "line_dur": np.array(line_dur),
        "win_dur": np.array(win_dur), "embs": embs, "rows": rows,
    }


def robust_z(S, triu_mask):
    off = S[triu_mask]
    med = np.median(off)
    iqr = np.subtract(*np.percentile(off, [75, 25])) / 1.349
    iqr = max(iqr, 1e-3)
    Z = (S - med) / iqr
    return Z


def build_sim(ds, spec):
    """spec: dict(kind=single|concat|score, models=[...], w=?, znorm=bool, lenaware=bool, short_thr=1.5)"""
    N = len(ds["lids"])
    E = ds["embs"]
    if spec["kind"] == "concat":
        blocks = [E[m] for m in spec["models"]]
        return l2(np.hstack(blocks))
    if spec["kind"] == "single":
        return E[spec["models"][0]]
    ms = np.triu(np.ones((N, N), dtype=bool), 1)
    mats = []
    per_scene = bool(spec.get("scene_norm"))
    for m in spec["models"]:
        S = E[m] @ E[m].T
        if not spec.get("znorm", True):
            mats.append(S)
        elif per_scene:
            Sn = S.copy()
            sc = np.array(ds["scenes"])
            for g in sorted(set(sc.tolist())):
                ii = [i for i in range(N) if sc[i] == g]
                if len(ii) < 3:
                    continue
                sub = S[np.ix_(ii, ii)]
                mask = np.triu(np.ones((len(ii), len(ii)), dtype=bool), 1)
                Sn[np.ix_(ii, ii)] = robust_z(sub, mask)
            mats.append(Sn)
        else:
            mats.append(robust_z(S, ms))
    if spec.get("lenaware"):
        thr = spec.get("short_thr", 1.5)
        d = ds["win_dur"]
        short = d[:, None] < thr
        base_w = spec.get("w", 0.5)
        short_w = spec.get("short_w", 0.7)
        long_w = 1.0 - base_w
        wij = np.where(short, short_w, long_w)
        wij = 0.5 * (wij + wij.T)
        s0, s1 = mats
        S = wij * s0 + (1 - wij) * s1
    else:
        w = spec.get("w", 0.5)
        S = (w * mats[0] + (1 - w) * mats[1]) if len(mats) == 2 else sum(mats) / len(mats)
    return S


def to_sim_matrix(S):
    """统一成 N×N 相似度矩阵：输入若是 embedding (N,D) 且 D!=N 则取余弦。"""
    n = S.shape[0]
    if S.ndim == 2 and S.shape[1] == n:
        return S
    return l2(S) @ l2(S).T


def to_nonneg_sim(S):
    """拉回 [0,1] 非负相似度尺度（保序），供 spectral / margin 使用。"""
    n = S.shape[0]
    if S.ndim == 2 and S.shape[1] == n:
        lo, hi = float(np.nanmin(S)), float(np.nanmax(S))
        if lo >= 0.0 and hi <= 1.0 + 1e-9:
            return S
        return (S - lo) / max(hi - lo, 1e-9)
    X = l2(S)
    return (X @ X.T + 1.0) / 2.0


def cluster(S, k, method):
    n = S.shape[0]
    k = int(max(1, min(k, n)))
    if k <= 1 or n < 2:
        return np.zeros(n, dtype=int)
    if method == "spectral":
        return SpectralClustering(
            n_clusters=k, affinity="precomputed", assign_labels="kmeans",
            random_state=0, n_init=20,
        ).fit_predict(S)
    if method == "kmeans":
        return KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(l2(S))
    D = np.clip(1.0 - S, 0, None)
    np.fill_diagonal(D, 0.0)
    return AgglomerativeClustering(
        n_clusters=k, metric="precomputed", linkage="average"
    ).fit_predict(D)


def evaluate(ds, S, method="spectral"):
    S = to_sim_matrix(S)
    # margin 需在 [0,1] 可比尺度上报告：z-score 矩阵先做 min-max 拉回
    Sn = S
    if float(np.nanmin(S)) < -1e-6 or float(np.nanmax(S)) > 1.0 + 1e-6:
        lo, hi = float(np.nanmin(S)), float(np.nanmax(S))
        Sn = (S - lo) / max(hi - lo, 1e-9)
    scenes = np.array(ds["scenes"])
    truth = np.array(ds["truth"])
    kb = ds["k_by_scene"]
    pred = np.full(len(truth), -1, dtype=int)
    hit = tot = 0
    aris = []
    margins = {}
    for g in sorted(set(scenes.tolist())):
        ii = [i for i in range(len(truth)) if scenes[i] == g]
        if len(ii) < 2:
            continue
        k = kb.get(str(g), 2)
        k = min(k, len(ii))
        if method == "spectral":
            # spectral 的 affinity 必须非负：z 矩阵含负值会 NaN，用保序的 min-max 尺度
            lab = cluster(Sn[np.ix_(ii, ii)], k, method)
        else:
            lab = cluster(S[np.ix_(ii, ii)], k, method)
        for j, i in enumerate(ii):
            pred[i] = lab[j]
        y = truth[ii]
        cnt = collections.defaultdict(collections.Counter)
        for l, yy in zip(lab, y):
            cnt[l][yy] += 1
        ok = sum(c.most_common(1)[0][1] for c in cnt.values())
        hit += ok
        tot += len(ii)
        aris.append(adjusted_rand_score(y, lab))
        cen = np.stack([l2(Sn[ii][lab == c].mean(0)) for c in sorted(set(lab))])
        cen_norm = cen / (np.linalg.norm(cen, axis=1, keepdims=True) + 1e-9)
        for pos, i in enumerate(ii):
            sc = cen_norm @ Sn[ii][pos]
            top = np.sort(sc)[::-1]
            margins[i] = float(top[0] - top[1]) if len(top) > 1 else 1.0
    acc = hit / max(1, tot)
    macro_ari = float(np.mean(aris)) if aris else 0.0
    loo = loo_nn(S, scenes, truth)
    mvals = np.array(list(margins.values())) if margins else np.array([0.0])
    dur = ds["win_dur"]
    short = [margins[i] for i in margins if dur[i] < 1.5]
    long_ = [margins[i] for i in margins if dur[i] >= 1.5]
    q = lambda a: [round(float(np.percentile(a, p)), 3) for p in (10, 50, 90)] if len(a) else [None] * 3
    return {
        "acc": round(acc, 4), "macro_ari": round(macro_ari, 4), "loo_nn": round(loo, 4),
        "margin_all": q(mvals), "margin_short": q(np.array(short)), "margin_long": q(np.array(long_)),
        "n_rows": tot, "n_scenes": len(aris), "n_short": len(short),
    }


def loo_nn_detail(S, scenes, truth, dur=None, short_thr=1.5, lids=None):
    """逐行 LOO-NN：返回 (overall, short_acc, long_acc, wrong_lines)。"""
    hit = tot = 0
    sh_hit = sh_tot = lo_hit = lo_tot = 0
    wrong = []
    scenes = np.array(scenes)
    truth = np.array(truth)
    for g in sorted(set(scenes.tolist())):
        ii = [i for i in range(len(truth)) if scenes[i] == g]
        if len(ii) < 2:
            continue
        sub = S[np.ix_(ii, ii)]
        for a in range(len(ii)):
            row = sub[a].copy()
            row[a] = -np.inf
            b = int(np.argmax(row))
            ok = int(truth[ii[b]] == truth[ii[a]])
            hit += ok
            tot += 1
            is_short = dur is not None and dur[ii[a]] < short_thr
            if dur is None:
                pass
            elif is_short:
                sh_hit += ok
                sh_tot += 1
            else:
                lo_hit += ok
                lo_tot += 1
            if not ok:
                wrong.append({
                    "line_id": lids[ii[a]] if lids is not None else ii[a],
                    "row": ii[a], "scene": str(scenes[ii[a]]),
                    "truth": str(truth[ii[a]]), "nn_pred": str(truth[ii[b]]),
                    "sim": round(float(sub[a][b]), 3),
                    "dur": round(float(dur[ii[a]]), 2) if dur is not None else None,
                })
    return (hit / max(1, tot),
            sh_hit / max(1, sh_tot) if sh_tot else None,
            lo_hit / max(1, lo_tot) if lo_tot else None,
            wrong)


def loo_nn(S, scenes, truth):
    return loo_nn_detail(S, scenes, truth)[0]
