"""
Phase 1 gate (b) -- does a cheap metric track the grounded judge?

Joins per-candidate metric scores (metric_scores_*.json from metrics.py) with grounded-judge
ratings (judge_ratings_*.json from judge_quality.py) and reports, per reference-point metric:

  * SPEARMAN(metric cosine, judge field) across all judged candidates -- does the metric rank
    quality the way the judge does? (the graded test)
  * PAIRWISE AGREEMENT with the judge on gold-vs-variant: does sign(metric_gold - metric_var)
    match sign(judge_gold - judge_var)?  Overall and per variant -- especially `verbose`, where
    raw centroid is expected to DISAGREE with the judge (it rewards padding; the judge penalises it).

A metric passes gate (b) to the extent it correlates with / agrees with the judge -- which is the
independent quality signal the intrusion gate lacked.

  uv run python experiments/label_quality/validate_gate_b.py --judge sonnet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path("/Users/stevenfazzio/repos/toponymy/experiments/label_quality")
METRICS = ["centroid", "whitened", "medoid", "exemplar"]

try:
    from scipy.stats import spearmanr
except ImportError:  # numpy fallback (rank then Pearson)
    def spearmanr(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1]), float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="sonnet")
    ap.add_argument("--dataset", default="20ng")
    ap.add_argument("--field", default="overall", help="judge field to correlate against")
    args = ap.parse_args()

    sfile = HERE / "data" / f"metric_scores_{args.dataset}.json"
    jfile = HERE / "data" / f"judge_ratings_{args.dataset}_{args.judge}.json"
    if not jfile.exists():
        raise SystemExit(f"no judge ratings yet at {jfile} -- run judge_quality.py first")
    S = {(r["layer"], r["idx"], r["type"]): r for r in json.loads(sfile.read_text())}
    J = {(r["layer"], r["idx"], r["type"]): r for r in json.loads(jfile.read_text())}

    keys = [k for k in J if k in S and J[k].get(args.field) is not None]
    print(f"[{args.dataset}] judge={args.judge} field={args.field} | {len(keys)} joined candidates")

    print("\njudge mean by candidate type (sanity):")
    for t in ["gold", "verbose", "sibling", "ancestor", "generic"]:
        v = [J[k][args.field] for k in keys if k[2] == t]
        if v:
            print(f"  {t:<9} {np.mean(v):.2f}  (n={len(v)})")

    jv = np.array([J[k][args.field] for k in keys])
    print(f"\nSPEARMAN( metric cosine , judge {args.field} )  across {len(keys)} candidates")
    for m in METRICS:
        rho, p = spearmanr(np.array([S[k][m] for k in keys]), jv)
        print(f"  {m:<10} rho={rho:+.3f}  (p={p:.1e})")

    variants = sorted({k[2] for k in keys if k[2] != "gold"})
    print("\nPAIRWISE AGREEMENT with judge on gold-vs-variant  (sign of metric diff matches judge)")
    print(f"{'metric':<10} " + "  ".join(f"{v:>8}" for v in variants) + "    overall")
    for m in METRICS:
        per = {v: [] for v in variants}
        for (L, i, t) in keys:
            if t == "gold":
                continue
            gk = (L, i, "gold")
            if gk not in S or J.get(gk, {}).get(args.field) is None:
                continue
            jd = J[gk][args.field] - J[(L, i, t)][args.field]   # judge: gold - variant
            if jd == 0:
                continue                                        # judge tie -> no ground-truth direction
            md = S[gk][m] - S[(L, i, t)][m]                     # metric: gold - variant
            per[t].append(1.0 if (md > 0) == (jd > 0) else 0.0)
        cells = "  ".join(f"{np.mean(per[v])*100:7.1f}%" if per[v] else f"{'-':>8}" for v in variants)
        allv = [x for v in variants for x in per[v]]
        print(f"{m:<10} {cells}    {np.mean(allv)*100:5.1f}%")

    out = HERE / "data" / f"gate_b_{args.dataset}_{args.judge}.json"
    res = {
        "spearman": {m: float(spearmanr(np.array([S[k][m] for k in keys]), jv)[0]) for m in METRICS},
        "n": len(keys),
    }
    out.write_text(json.dumps(res, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
