"""
Do picks made on the frozen documents beat stock on documents they have never seen?

Primary result: mean pick minus stock on the fresh draw, over all clusters, where a cluster whose
pick is the stock label counts as zero. Reads fresh_pairs.json and fresh_results.jsonl from argv[1].
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon

SP = Path(sys.argv[1])
CROSSFIT = {"20ng": 0.048, "arxiv_home": 0.045}  # selection_value_report.txt, everything pooled vs stock


def frac(r, true):
    total = sum(r["scores"]) or 1
    return r["scores"][r["order"].index(true)] / total


def main():
    data = json.loads((SP / "fresh_pairs.json").read_text())
    rows = [json.loads(l) for l in (SP / "fresh_results.jsonl").read_text().splitlines()]
    scores = defaultdict(list)
    for r in rows:
        if r["scores"] is not None:
            scores[r["uid"]].append(r)
    summary = {}
    for cell, v in data.items():
        print(f"\n========== {cell} ==========")
        fresh = {}
        for u in v["units"]:
            s = scores[u["uid"]]
            assert len(s) == 3, u["uid"]
            fresh[(u["L"], u["i"], u["kind"])] = float(np.mean([frac(r, u["true"]) for r in s]))
        table = []
        for p in v["pairs"]:
            key = (p["L"], p["i"])
            frozen = np.mean(p["pick_frozen"]) - np.mean(p["stock_frozen"])
            if p["pick"] == p["stock"]:
                table.append(dict(L=p["L"], differs=False, frozen=0.0, fresh=0.0))
                continue
            a, b = fresh[(*key, "fresh-pick")], fresh[(*key, "fresh-stock")]
            table.append(dict(L=p["L"], differs=True, frozen=float(frozen), fresh=a - b, pick=a, stock=b,
                              pick_frozen=float(np.mean(p["pick_frozen"])), stock_frozen=float(np.mean(p["stock_frozen"])),
                              pick_words=p["pick_words"], stock_words=p["stock_words"]))
        d = [t for t in table if t["differs"]]
        f = np.array([t["fresh"] for t in d])
        z = np.array([t["frozen"] for t in d])
        allf = np.array([t["fresh"] for t in table])
        allz = np.array([t["frozen"] for t in table])
        rng = np.random.default_rng(0)
        boots = [allf[rng.integers(len(allf), size=len(allf))].mean() for _ in range(10000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p = wilcoxon(f[f != 0]).pvalue
        print(f"clusters {len(table)}, pick differs from stock in {len(d)}")
        print(f"fresh documents: pick {np.mean([t['pick'] for t in d]):.3f}, stock {np.mean([t['stock'] for t in d]):.3f} (differing clusters)")
        print(f"frozen documents: pick {np.mean([t['pick_frozen'] for t in d]):.3f}, stock {np.mean([t['stock_frozen'] for t in d]):.3f} (in-sample, inflated)")
        print(f"PRIMARY fresh gain over all clusters {allf.mean():+.3f} (95% CI {lo:+.3f} to {hi:+.3f}), wilcoxon p={p:.3g}")
        print(f"on the frozen draw: in-sample {allz.mean():+.3f}, cross-fit {CROSSFIT[cell]:+.3f}; "
              f"fresh gain is {allf.mean() / CROSSFIT[cell]:.0%} of cross-fit")
        print(f"pick beats stock on fresh documents in {np.mean(f > 0):.0%} of differing clusters")
        print(f"cluster-level: frozen vs fresh difference pearson {pearsonr(z, f)[0]:+.3f}, spearman {spearmanr(z, f)[0]:+.3f}, "
              f"slope {(z @ f) / (z @ z):+.2f}")
        print(f"words: pick {np.mean([t['pick_words'] for t in d]):.1f}, stock {np.mean([t['stock_words'] for t in d]):.1f}")
        for L in sorted({t["L"] for t in table}):
            sub = [t for t in table if t["L"] == L]
            print(f"  L{L}: {len(sub)} clusters, fresh gain {np.mean([t['fresh'] for t in sub]):+.3f}")
        summary[cell] = dict(clusters=len(table), differ=len(d), gain=float(allf.mean()), ci=[float(lo), float(hi)], p=float(p),
                             in_sample=float(allz.mean()), crossfit=CROSSFIT[cell], pick=float(np.mean([t["pick"] for t in d])),
                             stock=float(np.mean([t["stock"] for t in d])), win_rate=float(np.mean(f > 0)),
                             spearman=float(spearmanr(z, f)[0]), pearson=float(pearsonr(z, f)[0]), slope=float((z @ f) / (z @ z)),
                             pick_words=float(np.mean([t["pick_words"] for t in d])), stock_words=float(np.mean([t["stock_words"] for t in d])))
    pooled = []
    for c, s in summary.items():
        pooled.append(s["gain"] * s["clusters"])
    print(f"\ninput tokens {sum(r['input_tokens'] for r in rows):,}, output tokens {sum(r['output_tokens'] for r in rows):,}, calls {len(rows)}")
    (SP / "fresh_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
