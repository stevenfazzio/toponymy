"""
Compare Haiku's picks with Sonnet's picks and stock on the same clusters and fresh documents.

Primary result: Haiku pick minus stock on the fresh draw over the clusters in the subset, where a
pick equal to stock counts as zero. Sonnet's picks are scored on exactly the same clusters.
Reads from argv[1] only.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from haiku_test import frac, haiku_scores, in_subset, picks_from, words

SP = Path(sys.argv[1])


def boot(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    b = [x[rng.integers(len(x), size=len(x))].mean() for _ in range(n)]
    return [float(v) for v in np.percentile(b, [2.5, 97.5])]


def fresh_lookup():
    pairs = json.loads((SP / "fresh_pairs.json").read_text())
    samples = defaultdict(list)
    labels = {}
    for c, v in pairs.items():
        for u in v["units"]:
            labels[u["uid"]] = (c, u["L"], u["i"], u["label"])
    for r in map(json.loads, (SP / "fresh_results.jsonl").read_text().splitlines()):
        c, L, i, label = labels[r["uid"]]
        samples[(c, L, i, label)].append(frac(r, i))
    for r in map(json.loads, (SP / "haiku_rate_results.jsonl").read_text().splitlines()):
        samples[(r["cell"], r["L"], r["i"], r["label"])].append(frac(r, r["i"]))
    out = {}
    for k, v in samples.items():
        assert len(v) == 3, k
        out[k] = float(np.mean(v))
    return pairs, out


def within(rows):
    out = []
    for r in rows:
        a = np.array(r, float)
        if len(a) >= 4 and a[:, 0].std() > 0 and a[:, 1].std() > 0:
            out.append(spearmanr(a[:, 0], a[:, 1])[0])
    return float(np.mean(out)), len(out), float(wilcoxon(out).pvalue)


def main():
    pairs, fresh = fresh_lookup()
    keep = in_subset(SP)
    picks = picks_from(SP)
    scores = haiku_scores(SP)
    units = json.loads((SP / "haiku_units.json").read_text())
    summary = {}
    for cell, v in pairs.items():
        print(f"\n========== {cell} ==========")
        sonnet_means = defaultdict(dict)
        for u in units[cell]:
            sonnet_means[(u["L"], u["i"])][u["label"]] = u["sonnet"]
        H, S, rows, same_s, same_st = [], [], [], 0, 0
        layer = defaultdict(list)
        frozen_h, frozen_s, frozen_r, w_h, w_st, w_s = [], [], [], [], [], []
        for p in v["pairs"]:
            key = (p["L"], p["i"])
            if (cell, *key) not in keep:
                continue
            pick, stock, spick = picks[cell][key], p["stock"], p["pick"]
            f = lambda lab: fresh[(cell, *key, lab)]
            h = 0.0 if pick == stock else f(pick) - f(stock)
            s = 0.0 if spick == stock else f(spick) - f(stock)
            H.append(h)
            S.append(s)
            layer[p["L"]].append((h, s))
            same_s += pick == spick
            same_st += pick == stock
            pool = sonnet_means[key]
            frozen_h.append(np.mean(pool[pick]))
            frozen_s.append(np.mean(pool[spick]))
            frozen_r.append(np.mean([np.mean(x) for x in pool.values()]))
            w_h.append(words(pick))
            w_s.append(words(spick))
            w_st.append(words(stock))
        H, S = np.array(H), np.array(S)
        keys = [k for k in sonnet_means if (cell, *k) in keep]
        agree = within([[(scores[cell][k][lab], np.mean(x)) for lab, x in sonnet_means[k].items()] for k in keys])
        ref = within([[(x[0], np.mean(x[1:])) for x in sonnet_means[k].values()] for k in keys])
        ci, cid = boot(H), boot(H - S)
        nz = H[H != 0]
        d = (H - S)
        p_d = wilcoxon(d[d != 0]).pvalue
        print(f"clusters {len(H)}; Haiku pick equals Sonnet's in {same_s}, equals stock in {same_st}")
        print(f"within-cluster rho, Haiku single sample vs Sonnet mean {agree[0]:+.3f} (n={agree[1]}, p={agree[2]:.2g}); "
              f"Sonnet single sample vs its other two {ref[0]:+.3f}")
        print(f"HAIKU fresh gain over stock {H.mean():+.3f} (95% CI {ci[0]:+.3f} to {ci[1]:+.3f}), wilcoxon p={wilcoxon(nz).pvalue:.3g}")
        print(f"SONNET fresh gain over stock, same clusters {S.mean():+.3f} (95% CI {boot(S)[0]:+.3f} to {boot(S)[1]:+.3f})")
        print(f"Haiku minus Sonnet {H.mean() - S.mean():+.3f} (95% CI {cid[0]:+.3f} to {cid[1]:+.3f}), wilcoxon p={p_d:.3g}; "
              f"Haiku keeps {H.mean() / S.mean():.0%} of Sonnet's gain")
        print(f"pick beats stock on fresh documents in {np.mean(nz > 0):.0%} of differing clusters")
        print(f"frozen documents, Sonnet-rated: Haiku pick {np.mean(frozen_h):.3f}, random candidate {np.mean(frozen_r):.3f}, Sonnet pick {np.mean(frozen_s):.3f} (in-sample)")
        print(f"words: Haiku pick {np.mean(w_h):.1f}, Sonnet pick {np.mean(w_s):.1f}, stock {np.mean(w_st):.1f}")
        for L in sorted(layer):
            a = np.array(layer[L])
            print(f"  L{L}: {len(a)} clusters, Haiku {a[:, 0].mean():+.3f}, Sonnet {a[:, 1].mean():+.3f}")
        summary[cell] = dict(clusters=len(H), same_as_sonnet=same_s, same_as_stock=same_st, within=agree[0], reference=ref[0],
                             haiku=float(H.mean()), haiku_ci=ci, sonnet=float(S.mean()), sonnet_ci=boot(S),
                             difference=float(H.mean() - S.mean()), difference_ci=cid, difference_p=float(p_d),
                             share=float(H.mean() / S.mean()), win_rate=float(np.mean(nz > 0)), haiku_p=float(wilcoxon(nz).pvalue),
                             frozen_haiku=float(np.mean(frozen_h)), frozen_random=float(np.mean(frozen_r)), frozen_sonnet=float(np.mean(frozen_s)),
                             words_haiku=float(np.mean(w_h)), words_sonnet=float(np.mean(w_s)), words_stock=float(np.mean(w_st)))
    t = {}
    for name in ("haiku_results.jsonl", "haiku_rate_results.jsonl"):
        rows = [json.loads(l) for l in (SP / name).read_text().splitlines()]
        t[name] = (len(rows), sum(r["input_tokens"] for r in rows), sum(r["output_tokens"] for r in rows))
    print(f"\ncalls and tokens (calls, input, output): {t}")
    (SP / "haiku_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
