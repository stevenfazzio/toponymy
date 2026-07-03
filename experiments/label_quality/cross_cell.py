"""
Cross-cell synthesis of the embedder x dataset robustness study (the 2x2:
{minilm, cohere} x {20ng, arxiv}). Reads each data/home_<ds>_<emb>.json and reports, per cell:

  - GATE-B        : Spearman(metric, judge) over the battery (gold + perturbation variants),
                    for raw centroid and whitened centroid. "Does the metric track the judge?"
  - INTRUSION     : gold-beats-variant rate per variant type + strict top-1, both reference points.
                    The `verbose` column is the discriminator (on-topic padding -- the documented
                    failure mode whitening is supposed to fix).
  - ABLATION      : drop a naming feature, re-name, judge. Per feature: judge-Δ (feature's
                    contribution), metric-Δ, Spearman(metric-Δ, judge-Δ), sign-agree%. "Is the
                    metric a useful regression guard?" (across 3 cells so far it was blind for MiniLM)
  - FEATURE       : mean judge-Δ per ablated feature, the embedder-INDEPENDENT contribution finding
                    (exemplars >> keyphrases on 20ng/minilm).

Then a synthesis block answering the three robustness questions the study exists to settle:
  (1) Is whitening's verbose-intrusion win embedder-dependent (shrinks/reverses for the stronger,
      less-anisotropic embedder)?
  (2) Is the ablation-blindness MiniLM-specific or universal?
  (3) Is exemplars >> keyphrases robust across all four cells?

  uv run python experiments/label_quality/cross_cell.py

Writes data/cross_cell_summary.json. Skips (and lists) any cell whose JSON is missing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

CELLS = [("20ng", "minilm"), ("20ng", "cohere"), ("arxiv", "minilm"), ("arxiv", "cohere")]
VARIANTS = ["ancestor", "sibling", "distant", "generic", "verbose"]
METRICS = ["centroid", "whitened"]
FEATURES = ["exemplars", "keyphrases"]


def _rho(xs, ys):
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return float("nan"), len(xs)
    return float(spearmanr(xs, ys)[0]), len(xs)


def analyze(cell: dict) -> dict:
    rows = cell["rows"]
    # index by cluster -> kind -> row
    by = {}
    for r in rows:
        by.setdefault((r["L"], r["i"]), {})[r["kind"]] = r
    judged = [r for r in rows if r["overall"] is not None]
    battery = [r for r in judged if not r["kind"].startswith("abl:")]

    out = {"dataset": cell["dataset"], "embedder": cell["embedder"], "counts": cell["counts"],
           "n_clusters": int(sum(cell["counts"])), "n_judged": len(judged), "n_battery": len(battery)}

    # --- GATE-B: Spearman(metric, judge) over the battery ---
    out["gate_b"] = {}
    for m in METRICS:
        rho, n = _rho([r[m] for r in battery], [r["overall"] for r in battery])
        out["gate_b"][m] = {"spearman": rho, "n": n}

    # --- INTRUSION: gold beats each variant (per reference point) ---
    out["intrusion"] = {}
    for m in METRICS:
        per_variant, strict_hits, strict_tot = {}, 0, 0
        for (L, i), d in by.items():
            if "gold" not in d:
                continue
            g = d["gold"][m]
            present = [v for v in VARIANTS if v in d]
            for v in present:
                per_variant.setdefault(v, []).append(g > d[v][m])
            if present:
                strict_tot += 1
                strict_hits += all(g > d[v][m] for v in present)
        out["intrusion"][m] = {
            v: (float(np.mean(hits)) if hits else float("nan")) for v, hits in per_variant.items()
        }
        out["intrusion"][m]["strict_top1"] = strict_hits / strict_tot if strict_tot else float("nan")
        out["intrusion"][m]["_n"] = strict_tot

    # --- ABLATION detection + FEATURE contribution ---
    out["ablation"] = {}
    for feat in FEATURES:
        key = f"abl:{feat}"
        jd, mdc, mdw = [], {m: [] for m in METRICS}, None
        mdelta = {m: [] for m in METRICS}
        for (L, i), d in by.items():
            if "gold" not in d or key not in d:
                continue
            if d["gold"]["overall"] is None or d[key]["overall"] is None:
                continue
            if d["gold"]["label"] == d[key]["label"]:  # feature didn't change the name
                continue
            jd.append(d["gold"]["overall"] - d[key]["overall"])
            for m in METRICS:
                mdelta[m].append(d["gold"][m] - d[key][m])
        rec = {"judge_delta_mean": float(np.mean(jd)) if jd else float("nan"), "n": len(jd)}
        for m in METRICS:
            md = mdelta[m]
            rho, _ = _rho(md, jd)
            sign = [(a > 0) == (b > 0) for a, b in zip(md, jd) if b != 0]
            rec[m] = {
                "metric_delta_mean": float(np.mean(md)) if md else float("nan"),
                "spearman_vs_judge": rho,
                "sign_agree": float(np.mean(sign)) if sign else float("nan"),
            }
        out["ablation"][feat] = rec
    return out


def fmt_pct(x):
    return "  n/a" if x != x else f"{x*100:4.0f}%"


def fmt_rho(x):
    return " n/a " if x != x else f"{x:+.2f}"


def main():
    results, missing = [], []
    for ds, emb in CELLS:
        p = DATA / f"home_{ds}_{emb}.json"
        if not p.exists():
            missing.append(f"{ds}/{emb}")
            continue
        results.append(analyze(json.loads(p.read_text())))

    label = lambda r: f"{r['dataset']}/{r['embedder']}"
    W = 16

    print("=" * 84)
    print("EMBEDDER x DATASET ROBUSTNESS  (2x2)")
    if missing:
        print(f"  MISSING CELLS (not yet run): {', '.join(missing)}")
    print("=" * 84)

    print("\nGATE-B  Spearman(metric, grounded-judge) over battery [gold + 5 perturbations]")
    print(f"  {'cell':<{W}} {'centroid':>10} {'whitened':>10}   {'n':>5}  clusters")
    for r in results:
        g = r["gate_b"]
        print(f"  {label(r):<{W}} {fmt_rho(g['centroid']['spearman']):>10} "
              f"{fmt_rho(g['whitened']['spearman']):>10}   {g['centroid']['n']:>5}  {r['counts']}")

    for m in METRICS:
        print(f"\nINTRUSION  gold beats variant  [{m}]  (strict = gold beats ALL present variants)")
        print(f"  {'cell':<{W}} " + " ".join(f"{v[:7]:>8}" for v in VARIANTS) + f"{'strict':>9}")
        for r in results:
            iz = r["intrusion"][m]
            print(f"  {label(r):<{W}} " + " ".join(fmt_pct(iz.get(v, float('nan'))) for v in VARIANTS)
                  + f"  {fmt_pct(iz['strict_top1'])}")

    print("\nABLATION  judge-Δ = feature contribution (+ = full better) | metric-Δ + sign-agree = is metric a guard?")
    for feat in FEATURES:
        print(f"  [{feat}]")
        print(f"    {'cell':<{W}} {'judgeΔ':>7} | {'cen-Δ':>7} {'cen ρ':>6} {'cen sgn':>8} | "
              f"{'wht-Δ':>7} {'wht ρ':>6} {'wht sgn':>8}  {'n':>4}")
        for r in results:
            a = r["ablation"].get(feat)
            if not a:
                continue
            c, w = a["centroid"], a["whitened"]
            print(f"    {label(r):<{W}} {a['judge_delta_mean']:+7.2f} | "
                  f"{c['metric_delta_mean']:+7.3f} {fmt_rho(c['spearman_vs_judge']):>6} {fmt_pct(c['sign_agree']):>8} | "
                  f"{w['metric_delta_mean']:+7.3f} {fmt_rho(w['spearman_vs_judge']):>6} {fmt_pct(w['sign_agree']):>8}  {a['n']:>4}")

    # ---------------- SYNTHESIS ----------------
    print("\n" + "=" * 84)
    print("SYNTHESIS")
    print("=" * 84)

    def cell(ds, emb):
        return next((r for r in results if r["dataset"] == ds and r["embedder"] == emb), None)

    # Q1: whitening's verbose-intrusion win, minilm vs cohere
    print("\n(1) Whitening's verbose-intrusion edge (whitened − centroid, pp) by embedder:")
    for ds in ["20ng", "arxiv"]:
        for emb in ["minilm", "cohere"]:
            r = cell(ds, emb)
            if not r:
                continue
            cv = r["intrusion"]["centroid"].get("verbose", float("nan"))
            wv = r["intrusion"]["whitened"].get("verbose", float("nan"))
            edge = (wv - cv) * 100
            print(f"    {ds}/{emb:<7} verbose: centroid {fmt_pct(cv)}  whitened {fmt_pct(wv)}  "
                  f"edge {edge:+5.1f}pp")
    print("    -> if the edge shrinks/reverses minilm->cohere, whitening is a weak-embedder patch.")

    # Q2: ablation blindness — sign-agree vs 50% chance, per cell/feature/metric
    print("\n(2) Ablation guard: is metric-Δ sign-agree with judge-Δ above chance (50%)?")
    for r in results:
        bits = []
        for feat in FEATURES:
            a = r["ablation"].get(feat)
            if a and a["n"] >= 4:
                bits.append(f"{feat[:3]}: cen {fmt_pct(a['centroid']['sign_agree'])} "
                            f"wht {fmt_pct(a['whitened']['sign_agree'])} (jΔ{a['judge_delta_mean']:+.2f},n{a['n']})")
        print(f"    {label(r):<{W}} " + " | ".join(bits))
    print("    -> near 50% everywhere = blind for all embedders (not just MiniLM).")

    # Q3: exemplars vs keyphrases contribution across cells
    print("\n(3) Feature contribution (judge-Δ, + = dropping it HURT) across cells:")
    print(f"    {'cell':<{W}} {'exemplars':>10} {'keyphrases':>11}")
    for r in results:
        ex = r["ablation"].get("exemplars", {}).get("judge_delta_mean", float("nan"))
        kp = r["ablation"].get("keyphrases", {}).get("judge_delta_mean", float("nan"))
        print(f"    {label(r):<{W}} {ex:+10.2f} {kp:+11.2f}")
    print("    -> exemplars consistently + (help); keyphrases ~0 or - = robust 'don't earn place'.")

    outp = DATA / "cross_cell_summary.json"
    outp.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {outp}")


if __name__ == "__main__":
    main()
