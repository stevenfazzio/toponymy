"""
How much is choosing among candidate labels worth, independent of label length?

Zero API calls. Uses saved Sonnet lineups, which keep three listener samples per label, so a label
can be picked on one sample and scored on the other two (cross-fitting). That removes winner's curse
from listener noise. It does NOT remove document-sampling noise, since all samples share one frozen
document draw; the last section checks that separately against fresh documents and another listener.
"""
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon

REPO = Path("/Users/stevenfazzio/repos/toponymy")
B = "experiment/naming-features:experiments/label_quality/data/"


def saved(name):
    r = subprocess.run(["git", "show", B + name], cwd=REPO, capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 else None


def fracs(u):
    out = []
    for s in u.get("samples") or []:
        total = sum(s["scores"]) or 1
        out.append(s["scores"][s["order"].index(u["true"])] / total)
    return out


def usable(u):
    return u.get("mode") == "nn" and u.get("k") == 5 and u.get("run") == 1 and len(u.get("samples") or []) == 3


def words(s):
    return len(s.split())


def room(pools):
    """Within-cluster SD of true score, after removing listener noise; and length's share of it."""
    total, noise, x, y = [], [], [], []
    for cands in pools.values():
        if len(cands) < 2:
            continue
        m = np.array([np.mean(c) for c in cands.values()])
        total.append(m.var(ddof=1))
        noise.append(np.mean([np.var(c, ddof=1) for c in cands.values()]) / 3)
        w = np.log([words(k) for k in cands])
        x.extend(w - w.mean())
        y.extend(m - m.mean())
    x, y = np.array(x), np.array(y)
    r2 = (x @ y) ** 2 / ((x @ x) * (y @ y)) if (x @ x) > 0 else float("nan")
    true_var = max(np.mean(total) - np.mean(noise), 0)
    return float(np.sqrt(np.mean(total))), float(np.sqrt(true_var)), float(r2)


def crossfit(pools, reference=None):
    """Pick the top label on one sample, score it on the other two, across the three folds."""
    picked, rand, ref, w_pick, w_rand = [], [], [], [], []
    for key, cands in pools.items():
        if len(cands) < 2:
            continue
        labs = list(cands)
        a = np.array([cands[k] for k in labs])
        ours = []
        for s in range(3):
            rest = [t for t in range(3) if t != s]
            j = int(np.argmax(a[:, s]))
            ours.append(a[j, rest].mean())
            w_pick.append(words(labs[j]))
        picked.append(np.mean(ours))
        rand.append(a.mean())
        w_rand.append(np.mean([words(k) for k in labs]))
        if reference and key in reference and reference[key] in cands:
            ref.append((np.mean(ours), np.mean(cands[reference[key]])))
    out = dict(clusters=len(picked), picked=float(np.mean(picked)), random=float(np.mean(rand)),
               words_picked=float(np.mean(w_pick)), words_random=float(np.mean(w_rand)))
    d = np.array(picked) - np.array(rand)
    out["gain_vs_random"] = float(d.mean())
    out["p_vs_random"] = float(wilcoxon(d[d != 0]).pvalue)
    if ref:
        r = np.array(ref)
        out["gain_vs_stock"] = float((r[:, 0] - r[:, 1]).mean())
        out["stock"] = float(r[:, 1].mean())
        out["n_stock"] = len(r)
    return out


def show(name, pools, reference=None):
    usable_pools = {k: v for k, v in pools.items() if len(v) >= 2}
    n_c = np.mean([len(v) for v in usable_pools.values()])
    raw, true, r2 = room(usable_pools)
    c = crossfit(usable_pools, reference)
    extra = f" | vs stock {c['gain_vs_stock']:+.3f} (stock {c['stock']:.3f}, n={c['n_stock']})" if "gain_vs_stock" in c else ""
    print(f"{name:34s} {c['clusters']:3d} clusters, {n_c:.1f} cands | true SD {true:.3f} (raw {raw:.3f}), "
          f"length explains {r2:.1%} | picked {c['picked']:.3f} at {c['words_picked']:.1f}w, "
          f"random {c['random']:.3f} at {c['words_random']:.1f}w, gain {c['gain_vs_random']:+.3f} (p={c['p_vs_random']:.2g}){extra}")


def analyze(cell):
    print(f"\n========== {cell} ==========")
    controller = saved(f"length_controller_{cell}.json")
    stock = {(r["L"], r["i"]): r["stock"] for r in controller["rows"]}
    ladder, draws, evidence = defaultdict(dict), defaultdict(dict), defaultdict(dict)

    for u in saved(f"wayfinding_{cell}_ladder.json")["units"].values():
        if usable(u):
            ladder[(u["L"], u["i"])][u["label"]] = fracs(u)
    gold = {}
    for u in saved(f"wayfinding_{cell}_battery.json")["units"].values():
        if usable(u) and u["kind"] == "gold":
            gold[(u["L"], u["i"])] = (u["label"], fracs(u))
    for key, (lab, f) in gold.items():
        evidence[key][lab] = f
        if key in stock and stock[key] in ladder.get(key, {}):
            draws[key][stock[key]] = ladder[key][stock[key]]
            draws[key].setdefault(lab, f)
    for fname in (f"wayfinding_{cell}_features.json", f"wayfinding_{cell}_subtopics.json"):
        data = saved(fname)
        for u in (data or {"units": {}})["units"].values():
            if usable(u):
                # these two arXiv runs used the other numbering for clusters 18 and 19 at layer 1
                i = {18: 19, 19: 18}.get(u["i"], u["i"]) if cell == "arxiv_home" and u["L"] == 1 else u["i"]
                evidence[(u["L"], i)][u["label"]] = fracs(u)

    show("length ladder, all rungs", ladder, stock)
    for lo, hi in ((2, 3), (4, 6), (7, 12)):
        band = {k: {l: f for l, f in v.items() if lo <= words(l) <= hi} for k, v in ladder.items()}
        show(f"ladder, only {lo}-{hi} word candidates", band)
    show("same prompt, two draws", draws)
    show("different naming evidence", evidence)

    combined = defaultdict(dict)
    for src in (ladder, evidence):
        for k, v in src.items():
            combined[k].update(v)
    if cell == "20ng":
        pairs = saved("wayfinding_20ng_pairs.json")
        alt = defaultdict(dict)
        for u in pairs["units"].values():
            if usable(u):
                alt[(u["L"], u["i"])][u["label"]] = fracs(u)
        show("different namers (haiku, gpt-4o-mini)", alt)
        for k, v in alt.items():
            combined[k].update(v)
    show("everything pooled", combined, stock)

    # ---- do per-cluster differences survive fresh documents and a different listener?
    ho = saved(f"wayfinding_{cell}_heldout.json")["units"].values()
    by = defaultdict(dict)
    for u in ho:
        by[(u["L"], u["i"])][u["kind"]] = u["pm"]
    rows = []
    for r in controller["rows"]:
        key = (r["L"], r["i"])
        if cell == "arxiv_home" and r["L"] == 1 and r["i"] in (18, 19):
            continue
        h = by.get(key, {})
        if r["differs"] and "ho-chosen" in h and "ho-stock" in h and r["stock_pm"] is not None:
            rows.append((r["chosen_pm"] - r["stock_pm"], h["ho-chosen"] - h["ho-stock"]))
    a = np.array(rows)
    big = np.abs(a[:, 0]) >= controller["band"]
    agree = np.mean(np.sign(a[big, 0]) == np.sign(a[big, 1]))
    slope = (a[:, 0] @ a[:, 1]) / (a[:, 0] @ a[:, 0])
    print(f"fresh docs + gpt-4o-mini, chosen minus stock, n={len(a)}: pearson {pearsonr(a[:, 0], a[:, 1])[0]:+.3f}, "
          f"spearman {spearmanr(a[:, 0], a[:, 1])[0]:+.3f}, sign agreement {agree:.0%} on {big.sum()} band-clearing, slope {slope:+.2f}")
    # the same check on the raw per-label score, to separate label effects from cluster effects
    level = []
    for r in controller["rows"]:
        key = (r["L"], r["i"])
        if cell == "arxiv_home" and r["L"] == 1 and r["i"] in (18, 19):
            continue
        h = by.get(key, {})
        if "ho-chosen" in h:
            level.append((r["chosen_pm"], h["ho-chosen"]))
    lv = np.array(level)
    print(f"label score level, in-run vs fresh: spearman {spearmanr(lv[:, 0], lv[:, 1])[0]:+.3f} (n={len(lv)})")


if __name__ == "__main__":
    for c in ("20ng", "arxiv_home"):
        analyze(c)
