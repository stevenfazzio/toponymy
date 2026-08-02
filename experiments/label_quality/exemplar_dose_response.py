"""
Phase 6b -- exemplar dose-response (all layers), plus the keyphrase x exemplar interaction.

The lineup said identification is flat: dropping exemplars entirely (k=0, keyphrases retained) costs
+0.011 prob-mass, indistinguishable from stock k=8. So the identification version of "how many
exemplars do you need" is answered at both endpoints with nothing to interpolate. Fit is the axis
where dose demonstrably matters -- the same k=0 ablation costs -0.47 judge-points -- so this sweeps
the dose and asks where fit degrades.

Two things this measures that the first (layer-0-only) pass did not:

  ALL LAYERS. `n_exemplars` is a single GLOBAL default, so a recommendation to halve it has to hold
  at coarse layers too. Coarse clusters are larger and more heterogeneous and may need more evidence
  per name; if the knee moves, the advice is "layer-dependent", not "set it to 4".

  THE INTERACTION (rung `4_nokp`). Upstream is weighing turning keyphrase extraction off; we
  separately find exemplars can be halved. Both were measured with the OTHER feature at its default.
  If the exemplar channel is what compensates for missing keyphrases, doing both could regress in a
  way neither measurement predicts. With the stock cell (gold), the k=4 cell, and the keyphrases-off
  cell (ablation.json), this completes a 2x2.

Naming uses stock machinery (same clusterer, same keyphrase channel unless ablated, same
disambiguation pass); scoring is the grounded judge (sonnet, 0-4 rubric after Preiss et al. 2024,
majority of 3) against the SAME deterministic document sample the gold labels were judged on, so
every comparison is paired per cluster. k=8 is stock (ratings reused from
judge_ratings_20ng_sonnet.json); k=0 is the exemplars ablation in ablation.json.

Haiku naming is mostly-but-not-fully deterministic, so `--stage name` reports layer-0 drift against
the archived layer-0-only run and the judge cache is keyed by label text -- a cached rating is
reused only when the label string still matches.

Exemplar selection is strictly nested (verified 74/74 across k), so lower rungs are subsets of the
stock 8. The sweep goes DOWN from the default deliberately: k=16 would put 87 namer exemplars into
the lineups' held-out pools across 47 of 74 clusters.

  uv run python experiments/label_quality/exemplar_dose_response.py --stage name
  uv run python experiments/label_quality/exemplar_dose_response.py --stage judge
  uv run python experiments/label_quality/exemplar_dose_response.py --stage report
Writes data/dose_names_20ng.json and data/dose_judge_20ng.json (both resumable).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
NIBLING = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(NIBLING))

DOSE_RUNGS = [1, 2, 4]
INTERACTION = "4_nokp"          # k=4 exemplars AND keyphrases blanked
STOCK_K = 8
K_SAMPLES = 3
NAMES = HERE / "data" / "dose_names_20ng.json"
JUDGE = HERE / "data" / "dose_judge_20ng.json"
DRAW1_NAMES = HERE / "data" / "dose_names_20ng.draw1.json"
DRAW1_JUDGE = HERE / "data" / "dose_judge_20ng.draw1.json"


def rung_keys():
    return [str(k) for k in DOSE_RUNGS] + [INTERACTION]


def rung_k(key: str) -> int:
    return int(key.split("_")[0])


def gold_labels() -> dict:
    b = json.loads((HERE / "data" / "battery_20ng.json").read_text())
    return {(it["layer"], it["idx"]): it["gold"] for it in b}


def gold_ratings() -> dict:
    r = json.loads((HERE / "data" / "judge_ratings_20ng_sonnet.json").read_text())
    return {(x["layer"], x["idx"]): x["overall"] for x in r
            if x["type"] == "gold" and x.get("overall") is not None}


def ablation_ratings(feature: str) -> dict:
    """k=0 (feature='exemplars') / keyphrases-off-at-stock-k (feature='keyphrases')."""
    rows = json.loads((HERE / "data" / "ablation.json").read_text())[feature]
    return {(r["L"], r["i"]): r["j_abl"] for r in rows}


# ------------------------------------------------------------------ stage: name

def stage_name():
    from ab_harness import load_dataset, make_embedder, make_namer
    from ablation import ablate
    from contextlib import nullcontext

    from toponymy.cluster_layer import ClusterLayerText
    from toponymy.clustering import ToponymyClusterer
    from toponymy.toponymy import Toponymy

    objects, emb, coords, meta = load_dataset("20ng", None)
    out = json.loads(NAMES.read_text()) if NAMES.exists() else {}
    # the archived run stored layer 0 only, as a flat list; drop it so we re-name saving all layers
    out = {k: v for k, v in out.items() if v and isinstance(v[0], list)}

    for key in rung_keys():
        if key in out:
            print(f"{key}: cached ({[len(l) for l in out[key]]})")
            continue
        k, nokp = rung_k(key), key.endswith("_nokp")
        print(f"\n=== naming at k={k} exemplars{', keyphrases OFF' if nokp else ''} ===", flush=True)
        clusterer = ToponymyClusterer(min_clusters=4, base_min_cluster_size=25, verbose=False)
        clusterer.fit_predict(coords, emb, ClusterLayerText)
        counts = [int(l.centroid_vectors.shape[0]) for l in clusterer.cluster_layers_]
        assert counts == [74, 24, 9], f"replay mismatch {counts} -- do not proceed"
        for layer in clusterer.cluster_layers_:
            layer.n_exemplars = k
        model = Toponymy(make_namer("haiku"), make_embedder(meta["emb_model"]),
                         clusterer=clusterer, object_description=meta["obj"],
                         corpus_description=meta["corpus"], verbose=False)
        with (ablate("keyphrases") if nokp else nullcontext()):
            model.fit(objects, emb, coords)
        out[key] = [list(l) for l in model.topic_names_]
        NAMES.write_text(json.dumps(out, indent=2))
        print(f"  layers {[len(l) for l in out[key]]} written")

    # determinism check against the archived layer-0-only run
    if DRAW1_NAMES.exists():
        old = json.loads(DRAW1_NAMES.read_text())
        print("\nlayer-0 drift vs the archived run (haiku is not fully deterministic):")
        for key in [str(k) for k in DOSE_RUNGS]:
            if key not in old:
                continue
            same = sum(1 for a, b in zip(old[key], out[key][0]) if a.strip() == b.strip())
            print(f"  k={key}: {same}/{len(old[key])} layer-0 names identical across runs")

    gold = gold_labels()
    print("\nchanged vs stock (k=8), by layer:")
    for key in rung_keys():
        parts = []
        for L, names in enumerate(out[key]):
            ch = sum(1 for i, n in enumerate(names) if n.strip() != gold[(L, i)].strip())
            parts.append(f"L{L} {ch}/{len(names)}")
        wl = np.mean([len(n.split()) for n in out[key][0]])
        print(f"  {key:>7}: {'  '.join(parts)} | L0 mean {wl:.1f} words")


# ------------------------------------------------------------------ stage: judge

def migrate_old_judge() -> dict:
    """Archived cache was {rung: {idx: score}} for layer 0; re-key by label so it stays valid."""
    if not (DRAW1_JUDGE.exists() and DRAW1_NAMES.exists()):
        return {}
    oj, on = json.loads(DRAW1_JUDGE.read_text()), json.loads(DRAW1_NAMES.read_text())
    out = {}
    for key, scores in oj.items():
        out[key] = {f"0.{i}": {"label": on[key][int(i)], "overall": v}
                    for i, v in scores.items() if v is not None and int(i) < len(on[key])}
    return out


def stage_judge(concurrency: int):
    from async_judge import rate_many
    from judge_fair import sample_docs

    from ab_harness import load_dataset

    _, _, _, meta = load_dataset("20ng", None)
    names = json.loads(NAMES.read_text())
    gold, gr = gold_labels(), gold_ratings()

    done = json.loads(JUDGE.read_text()) if JUDGE.exists() else {}
    # discard the archived flat schema ({rung: {idx: float}}): entries must be
    # {"label", "overall"} so a cached rating can be validated against the current label text
    dropped = 0
    for key in list(done):
        clean = {k: v for k, v in done[key].items() if isinstance(v, dict)}
        dropped += len(done[key]) - len(clean)
        done[key] = clean
    if dropped:
        print(f"dropped {dropped} cache entries in the archived flat schema (no label to verify)")
    if not any(done.values()):
        done = migrate_old_judge()
        if done:
            n = sum(len(v) for v in done.values())
            print(f"migrated {n} cached layer-0 ratings from the archived run (validated by label)")

    for key in rung_keys():
        done.setdefault(key, {})
        todo = []
        for L, layer_names in enumerate(names[key]):
            for i, lab in enumerate(layer_names):
                ck = f"{L}.{i}"
                if not lab or lab == "Unlabelled":
                    continue
                cached = done[key].get(ck)
                if cached and cached.get("label", "").strip() == lab.strip():
                    continue                                    # same label, rating still valid
                if lab.strip() == gold[(L, i)].strip():
                    done[key][ck] = {"label": lab, "overall": gr.get((L, i))}
                    continue                                    # identical to stock, reuse gold
                todo.append((L, i, lab))
        print(f"{key}: {len(todo)} to judge ({len(done[key])} cached/unchanged)", flush=True)
        if not todo:
            continue
        tasks = [(lab, sample_docs("20ng", L, i, n_near=10, n_rand=5)) for L, i, lab in todo]
        res = rate_many(tasks, meta["obj"], "anthropic/claude-sonnet-4-6",
                        k=K_SAMPLES, concurrency=concurrency)
        for (L, i, lab), r in zip(todo, res):
            done[key][f"{L}.{i}"] = {"label": lab, "overall": r["overall"]}
        JUDGE.write_text(json.dumps(done, indent=2))
        print(f"  done ({sum(1 for v in done[key].values() if v['overall'] is not None)} rated)")


# ------------------------------------------------------------------ stage: report

def _scores(judged, key, keys):
    return [judged[key][f"{L}.{i}"]["overall"] for (L, i) in keys]


def report():
    names = json.loads(NAMES.read_text())
    judged = json.loads(JUDGE.read_text())
    gold, gr = gold_labels(), gold_ratings()
    k0, kp_off = ablation_ratings("exemplars"), ablation_ratings("keyphrases")

    ok = set(gr)
    for key in rung_keys():
        ok &= {tuple(map(int, s.split("."))) for s, v in judged[key].items()
               if v["overall"] is not None}
    all_keys = sorted(ok & set(k0) & set(kp_off))
    by_layer = defaultdict(list)
    for L, i in all_keys:
        by_layer[L].append((L, i))

    print("\n======== EXEMPLAR DOSE-RESPONSE BY LAYER, GROUNDED JUDGE (20NG, sonnet) ========")
    print("paired per cluster; same document sample at every rung; k=8 = stock, k=0 = ablation")

    for scope, keys in [("ALL", all_keys)] + [(f"L{L}", v) for L, v in sorted(by_layer.items())]:
        if len(keys) < 5:
            print(f"\n-- {scope}: n={len(keys)}, too few to test --")
            continue
        base = [gr[k] for k in keys]
        print(f"\n-- {scope} (n={len(keys)}) --")
        print(f"   {'k':>4} {'judge':>8} {'vs stock':>10} {'95% CI':>18} {'Wilcoxon':>10}")
        for k, sc in ([(0, [k0[x] for x in keys])]
                      + [(rung_k(s), _scores(judged, s, keys)) for s in
                         [str(r) for r in DOSE_RUNGS]]
                      + [(STOCK_K, base)]):
            d = np.array([a - b for a, b in zip(sc, base)])
            m, se = d.mean(), (d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0)
            p = float("nan")
            if np.any(d != 0):
                try:
                    p = wilcoxon(d).pvalue
                except ValueError:
                    pass
            ci = f"[{m-1.96*se:+.3f},{m+1.96*se:+.3f}]" if se else " " * 17
            print(f"   {k:>4} {np.mean(sc):>8.3f} {m:>+10.3f} {ci:>18} {p:>10.3g}")

    # ---- the 2x2 interaction ----
    keys = all_keys
    base = [gr[k] for k in keys]
    cells = {
        ("k=8", "kp on"): base,
        ("k=4", "kp on"): _scores(judged, "4", keys),
        ("k=8", "kp off"): [kp_off[k] for k in keys],
        ("k=4", "kp off"): _scores(judged, INTERACTION, keys),
    }
    print(f"\n======== KEYPHRASE x EXEMPLAR INTERACTION (n={len(keys)}) ========")
    print("   does halving exemplars AND dropping keyphrases compound?\n")
    print(f"   {'':>8} {'kp on':>18} {'kp off':>18}")
    for kk in ["k=8", "k=4"]:
        row = "   " + f"{kk:>8}"
        for kp in ["kp on", "kp off"]:
            v = np.mean(cells[(kk, kp)])
            row += f" {v:>10.3f} ({np.mean([a-b for a, b in zip(cells[(kk,kp)], base)]):+.3f})"
        print(row)
    stock = np.array(base)
    k4 = np.array(cells[("k=4", "kp on")])
    kpo = np.array(cells[("k=8", "kp off")])
    both = np.array(cells[("k=4", "kp off")])

    def line(name, a, c):
        d = a - c
        se = d.std(ddof=1) / np.sqrt(len(d))
        p = wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
        print(f"   {name:<32} {d.mean():+.3f}  SE {se:.3f}  "
              f"[{d.mean()-1.96*se:+.3f},{d.mean()+1.96*se:+.3f}]  p={p:.3g}")

    print("\n   vs stock (k=8, keyphrases on):")
    line("k=4 alone", k4, stock)
    line("keyphrases-off alone", kpo, stock)
    line("BOTH", both, stock)
    print("\n   the decision-relevant comparison:")
    line("BOTH vs keyphrases-off alone", both, kpo)

    # paired 2x2 interaction: per-cluster contrast, so the SE is the SE of that contrast
    D = both - k4 - kpo + stock
    se = D.std(ddof=1) / np.sqrt(len(D))
    p = wilcoxon(D).pvalue
    print(f"\n   interaction contrast  D = both - k4 - kp_off + stock")
    line("D", both - k4 - kpo + stock, np.zeros(len(D)))
    if p < 0.05 and D.mean() < 0:
        print("   -> SUPER-ADDITIVE HARM. Each change alone is free-or-better, but the exemplar")
        print("      channel is what compensates for the missing keyphrases. Shipping both is not")
        print("      a regression against stock -- it forfeits the gain keyphrases-off alone buys.")
        print("      Practical rule: drop keyphrases OR halve exemplars, not both.")
    else:
        print("   -> no evidence the two changes compound")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["name", "judge", "report"])
    ap.add_argument("--concurrency", type=int, default=24)
    args = ap.parse_args()
    if args.stage == "name":
        stage_name()
    elif args.stage == "judge":
        stage_judge(args.concurrency)
    else:
        report()


if __name__ == "__main__":
    main()
