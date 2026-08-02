"""
Phase 6c -- the arXiv replication of the naming-feature work.

Everything in phase 6 so far is 20NG-only, which is below this project's own standard: the #173
study was run as a 2x2 precisely because two earlier conclusions fell apart when the setup varied,
and only findings that survived every cell were stated as findings. jc-healy also flagged that these
experiments "can be quite data dependent". So before any of it goes upstream, the three claims get
a second corpus:

  (a) keyphrases are identification-inert          -- lineup, vs the arXiv frozen battery
  (b) exemplars are identification-inert but       -- lineup + judge
      fit-critical, with the knee at k=4
  (c) keyphrases-off x exemplars-halved is         -- judge, paired 2x2
      super-additively harmful

Substrate: the at-home arXiv/MiniLM cell (embeddings re-computed from document text, so the #176
row-misalignment in the bundled example vectors cannot bite). Its gold labels, gold judge ratings,
and frozen k=5 lineups already exist from phase 4-5:
  data/battery_arxiv_home.json, data/judge_ratings_arxiv_home_sonnet.json,
  data/wayfinding_arxiv_home_battery.json

⚠ Judge documents MUST be sampled in the home cell's OWN geometry (nearest-centroid by that
embedder's vectors, n_near=10 / n_rand=5 / maxlen=280, seeded 1000+i) exactly as home_pipeline.py
did when the gold ratings were produced. judge_fair.sample_docs is NOT usable here -- it replays the
canonical *20NG* fit, so using it would silently compare labels against the wrong documents.

  uv run python experiments/label_quality/arxiv_naming_features.py --stage name
  uv run python experiments/label_quality/arxiv_naming_features.py --stage judge
  uv run python experiments/label_quality/arxiv_naming_features.py --stage lineup
  uv run python experiments/label_quality/arxiv_naming_features.py --stage report
Writes data/arxiv_feature_names.json, data/arxiv_feature_judge.json,
data/wayfinding_arxiv_home_features.json (all resumable).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from numpy.linalg import norm
from scipy.stats import wilcoxon

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
NIBLING = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(NIBLING))

TAG = "arxiv_minilm"
SUBSAMPLE = 7000
STOCK_K = 8
K_SAMPLES = 3
N_REPEAT = 30
# naming conditions: the two single-feature ablations, the dose rungs, and the interaction cell
# `4_draw2` is an independent naming draw at k=4: naming runs at temperature 0.4 (~76% label churn
# between identical runs), and k=4 is the rung that overturns the 20NG recommendation, so it gets
# the same two-draw treatment 20NG got rather than resting on a single draw.
CONDITIONS = ["abl_exemplars", "abl_keyphrases", "1", "2", "4", "4_nokp", "4_draw2"]
LINEUP_CONDITIONS = ["abl_exemplars", "abl_keyphrases"]

NAMES = HERE / "data" / "arxiv_feature_names.json"
JUDGE = HERE / "data" / "arxiv_feature_judge.json"
LINEUP = HERE / "data" / "wayfinding_arxiv_home_features.json"


def cond_k(c: str) -> int | None:
    return None if c.startswith("abl_") else int(c.split("_")[0])


# ------------------------------------------------------------------ substrate

def substrate():
    """The at-home arXiv/MiniLM cell, replayed exactly as home_pipeline.py built it."""
    from ab_harness import load_dataset
    from sentence_transformers import SentenceTransformer

    emb = np.load(HERE / "data" / f"home_{TAG}_emb.npy")
    coords = np.load(HERE / "data" / f"home_{TAG}_coords.npy")
    objects, _, _, meta = load_dataset("arxiv", emb.shape[0])
    meta = dict(meta, emb_model="all-MiniLM-L6-v2")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return objects, emb, coords, meta, embedder


def fresh_clusterer(coords, emb, n_exemplars=None):
    from toponymy.cluster_layer import ClusterLayerText
    from toponymy.clustering import ToponymyClusterer

    cl = ToponymyClusterer(min_clusters=4, base_min_cluster_size=25, verbose=False)
    cl.fit_predict(coords, emb, ClusterLayerText)
    if n_exemplars is not None:
        for layer in cl.cluster_layers_:
            layer.n_exemplars = n_exemplars
    return cl


def home_docs(clusterer, objects, emb, n_near=10, n_rand=5, maxlen=280):
    """Grounding docs per cluster in the HOME geometry -- the sampling the gold ratings used."""
    emb64 = emb.astype(np.float64)
    counts = [int(l.centroid_vectors.shape[0]) for l in clusterer.cluster_layers_]
    out = {}
    for L in range(len(counts)):
        cls = clusterer.cluster_layers_[L].cluster_labels
        C = clusterer.cluster_layers_[L].centroid_vectors
        for i in range(counts[L]):
            mem = np.where(cls == i)[0]
            if mem.size == 0:
                out[(L, i)] = []
                continue
            E = emb64[mem] / (norm(emb64[mem], axis=1, keepdims=True) + 1e-12)
            c = C[i].astype(np.float64) / (norm(C[i]) + 1e-12)
            near = mem[np.argsort(-(E @ c))[:n_near]]
            rest = np.array([m for m in mem if m not in set(near.tolist())])
            rng = np.random.default_rng(1000 + i)          # matches home_pipeline.py exactly
            rand = (rng.choice(rest, size=min(n_rand, rest.size), replace=False)
                    if rest.size else np.array([], int))
            out[(L, i)] = [" ".join(str(objects[j]).split())[:maxlen]
                           for j in list(near) + list(rand)]
    return out


def gold_labels() -> dict:
    b = json.loads((HERE / "data" / "battery_arxiv_home.json").read_text())
    return {(it["layer"], it["idx"]): it["gold"] for it in b}


def gold_ratings() -> dict:
    r = json.loads((HERE / "data" / "judge_ratings_arxiv_home_sonnet.json").read_text())
    return {(x["layer"], x["idx"]): x["overall"] for x in r
            if x["type"] == "gold" and x.get("overall") is not None}


# ------------------------------------------------------------------ stage: name

def stage_name():
    from ab_harness import make_namer
    from ablation import ablate

    from toponymy.toponymy import Toponymy

    objects, emb, coords, meta, embedder = substrate()
    out = json.loads(NAMES.read_text()) if NAMES.exists() else {}

    for cond in CONDITIONS:
        if cond in out:
            print(f"{cond}: cached ({[len(l) for l in out[cond]]})")
            continue
        k, nokp = cond_k(cond), cond.endswith("_nokp")
        feat = cond[4:] if cond.startswith("abl_") else None
        desc = f"ablate {feat}" if feat else f"k={k}{', keyphrases OFF' if nokp else ''}"
        print(f"\n=== naming arXiv-home: {desc} ===", flush=True)

        clusterer = fresh_clusterer(coords, emb, n_exemplars=k)
        counts = [int(l.centroid_vectors.shape[0]) for l in clusterer.cluster_layers_]
        print(f"  clusters {counts}", flush=True)
        model = Toponymy(make_namer("haiku"), embedder, clusterer=clusterer,
                         object_description=meta["obj"], corpus_description=meta["corpus"],
                         verbose=False)
        ctx = ablate(feat) if feat else (ablate("keyphrases") if nokp else nullcontext())
        with ctx:
            model.fit(objects, emb, coords)
        out[cond] = [list(x) for x in model.topic_names_]
        NAMES.write_text(json.dumps(out, indent=2))
        print(f"  layers {[len(l) for l in out[cond]]} written")

    gold = gold_labels()
    print("\nchanged vs stock, by layer:")
    for cond in CONDITIONS:
        parts = []
        for L, names in enumerate(out[cond]):
            ch = sum(1 for i, n in enumerate(names)
                     if (L, i) in gold and n.strip() != gold[(L, i)].strip())
            parts.append(f"L{L} {ch}/{len(names)}")
        print(f"  {cond:>14}: {'  '.join(parts)}")


# ------------------------------------------------------------------ stage: judge

def stage_judge(concurrency: int):
    from async_judge import rate_many

    objects, emb, coords, meta, _ = substrate()
    clusterer = fresh_clusterer(coords, emb)
    docs = home_docs(clusterer, objects, emb)
    names = json.loads(NAMES.read_text())
    gold, gr = gold_labels(), gold_ratings()
    done = json.loads(JUDGE.read_text()) if JUDGE.exists() else {}

    for cond in CONDITIONS:
        done.setdefault(cond, {})
        todo = []
        for L, layer_names in enumerate(names[cond]):
            for i, lab in enumerate(layer_names):
                ck = f"{L}.{i}"
                if not lab or lab == "Unlabelled" or (L, i) not in gr or not docs.get((L, i)):
                    continue
                cached = done[cond].get(ck)
                if cached and cached.get("label", "").strip() == lab.strip():
                    continue
                if lab.strip() == gold[(L, i)].strip():
                    done[cond][ck] = {"label": lab, "overall": gr[(L, i)]}
                    continue
                todo.append((L, i, lab))
        print(f"{cond}: {len(todo)} to judge ({len(done[cond])} cached/unchanged)", flush=True)
        if not todo:
            continue
        tasks = [(lab, docs[(L, i)]) for L, i, lab in todo]
        res = rate_many(tasks, meta["obj"], "anthropic/claude-sonnet-4-6",
                        k=K_SAMPLES, concurrency=concurrency)
        for (L, i, lab), r in zip(todo, res):
            done[cond][f"{L}.{i}"] = {"label": lab, "overall": r["overall"]}
        JUDGE.write_text(json.dumps(done, indent=2))
        print(f"  done ({sum(1 for v in done[cond].values() if v['overall'] is not None)} rated)")


# ------------------------------------------------------------------ stage: lineup

def stage_lineup(concurrency: int):
    from wayfinding import Cell, K_DEFAULT, MODELS, make_unit, run_all

    cell = Cell("arxiv_home")
    print(f"replayed arXiv-home fit: {cell.counts}")
    names = json.loads(NAMES.read_text())
    gold = gold_labels()

    units = []
    for cond in LINEUP_CONDITIONS:
        for L, layer_names in enumerate(names[cond]):
            for i, lab in enumerate(layer_names):
                if not lab or lab == "Unlabelled" or (L, i) not in gold:
                    continue
                if lab.strip() == gold[(L, i)].strip():
                    continue                       # identical label => identical pm, no call
                u = make_unit(cell, L, i, cond, lab, K_DEFAULT, "nn", run=1)
                u.update(feature=cond[4:], full=gold[(L, i)])
                units.append(u)
    rng = np.random.default_rng(6161)
    pick = rng.choice(len(units), min(N_REPEAT, len(units)), replace=False)
    units += [dict(units[j], run=2, uid=units[j]["uid"][:-1] + "2") for j in pick]

    print(f"{len(units)} units x {K_SAMPLES} samples on sonnet")
    asyncio.run(run_all(units, cell, MODELS["sonnet"], concurrency, LINEUP))


# ------------------------------------------------------------------ stage: report

def gold_pm() -> dict:
    b = json.loads((HERE / "data" / "wayfinding_arxiv_home_battery.json").read_text())["units"]
    return {(u["L"], u["i"]): u["pm"] for u in b.values()
            if u["kind"] == "gold" and u["mode"] == "nn" and u.get("pm") is not None}


def report():
    names = json.loads(NAMES.read_text())
    judged = json.loads(JUDGE.read_text())
    gold, gr = gold_labels(), gold_ratings()

    print("\n############ arXiv-HOME REPLICATION (MiniLM, sonnet judge + listener) ############")

    # ---------- (a)+(b) identification: the two ablations through the frozen lineups ----------
    if LINEUP.exists():
        gp = gold_pm()
        units = json.loads(LINEUP.read_text())["units"]
        us = [u for u in units.values() if u.get("pm") is not None]
        r1 = {u["uid"][:-1]: u for u in us if u["run"] == 1}
        r2 = {u["uid"][:-1]: u for u in us if u["run"] == 2}
        rep = [abs(r1[x]["pm"] - r2[x]["pm"]) for x in r2 if x in r1]
        band = float(np.percentile(rep, 90)) if rep else float("nan")
        print(f"\n=== IDENTIFICATION (lineup) ===   repeat band p90 |d pm| = {band:.3f} "
              f"(n={len(rep)})")
        for cond in LINEUP_CONDITIONS:
            recs = [dict(u, dpm=gp[(u["L"], u["i"])] - u["pm"]) for u in us
                    if u["run"] == 1 and u["kind"] == cond and (u["L"], u["i"]) in gp]
            n_unchanged = sum(1 for (L, i), g in gold.items()
                              if L < len(names[cond]) and i < len(names[cond][L])
                              and names[cond][L][i].strip() == g.strip())
            d = [r["dpm"] for r in recs] + [0.0] * n_unchanged
            m = float(np.mean(d))
            se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
            p = wilcoxon([r["dpm"] for r in recs]).pvalue if recs else float("nan")
            print(f"  {cond:<15} n={len(d):>3}  d pm {m:+.3f}  SE {se:.3f}  "
                  f"[{m-1.96*se:+.3f},{m+1.96*se:+.3f}]  p={p:.3g}")
        print("  (positive d pm = the ablation HURT identification)")
    else:
        print("\n=== IDENTIFICATION: not run yet (--stage lineup) ===")

    # ---------- fit: dose-response ----------
    ok = set(gr)
    for cond in CONDITIONS:
        ok &= {tuple(map(int, s.split("."))) for s, v in judged.get(cond, {}).items()
               if v["overall"] is not None}
    keys = sorted(ok)
    by_layer = defaultdict(list)
    for L, i in keys:
        by_layer[L].append((L, i))

    def sc(cond, ks):
        return np.array([judged[cond][f"{L}.{i}"]["overall"] for (L, i) in ks])

    print(f"\n=== FIT: exemplar dose-response (grounded judge) ===")
    for scope, ks in [("ALL", keys)] + [(f"L{L}", v) for L, v in sorted(by_layer.items())]:
        if len(ks) < 5:
            print(f"\n-- {scope}: n={len(ks)}, too few --")
            continue
        base = np.array([gr[k] for k in ks])
        print(f"\n-- {scope} (n={len(ks)}) --")
        print(f"   {'k':>4} {'judge':>8} {'vs stock':>10} {'95% CI':>18} {'Wilcoxon':>10}")
        for label, arr in ([("0", sc("abl_exemplars", ks))]
                           + [(c, sc(c, ks)) for c in ["1", "2", "4"]]
                           + [("8", base)]):
            d = arr - base
            m, se = d.mean(), (d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0)
            p = wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
            ci = f"[{m-1.96*se:+.3f},{m+1.96*se:+.3f}]" if se else " " * 17
            print(f"   {label:>4} {arr.mean():>8.3f} {m:>+10.3f} {ci:>18} {p:>10.3g}")

    # ---------- draw-to-draw stability at k=4 ----------
    if "4_draw2" in judged and len(keys) >= 5:
        base = np.array([gr[k] for k in keys])
        a, b = sc("4", keys), sc("4_draw2", keys)
        n_same = sum(1 for (L, i) in keys
                     if names["4"][L][i].strip() == names["4_draw2"][L][i].strip())
        print(f"\n=== k=4 DRAW STABILITY (n={len(keys)}) ===")
        print(f"   identical label text across the two draws: {n_same}/{len(keys)} "
              f"({n_same/len(keys):.0%})")
        for nm, arr in [("draw 1", a), ("draw 2", b)]:
            d = arr - base
            se = d.std(ddof=1) / np.sqrt(len(d))
            print(f"   {nm} vs stock: {d.mean():+.3f}  SE {se:.3f}  "
                  f"[{d.mean()-1.96*se:+.3f},{d.mean()+1.96*se:+.3f}]  "
                  f"p={wilcoxon(d).pvalue:.3g}")
        dd = b - a
        se = dd.std(ddof=1) / np.sqrt(len(dd))
        print(f"   draw2 - draw1: {dd.mean():+.3f}  SE {se:.3f}  "
              f"p={wilcoxon(dd).pvalue if np.any(dd != 0) else float('nan'):.3g}")
        both_neg = (a - base).mean() < 0 and (b - base).mean() < 0
        print(f"   -> {'k=4 deficit reproduces across draws' if both_neg else 'draws DISAGREE on the sign -- single-draw noise'}")

    # ---------- (c) the interaction ----------
    if len(keys) >= 5:
        base = np.array([gr[k] for k in keys])
        k4, kpo, both = sc("4", keys), sc("abl_keyphrases", keys), sc("4_nokp", keys)
        print(f"\n=== KEYPHRASE x EXEMPLAR INTERACTION (n={len(keys)}) ===\n")
        print(f"   {'':>8} {'kp on':>18} {'kp off':>18}")
        print(f"   {'k=8':>8} {base.mean():>10.3f} (+0.000) {kpo.mean():>10.3f} "
              f"({(kpo-base).mean():+.3f})")
        print(f"   {'k=4':>8} {k4.mean():>10.3f} ({(k4-base).mean():+.3f}) {both.mean():>10.3f} "
              f"({(both-base).mean():+.3f})")

        def line(name, d):
            se = d.std(ddof=1) / np.sqrt(len(d))
            p = wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
            print(f"   {name:<32} {d.mean():+.3f}  SE {se:.3f}  "
                  f"[{d.mean()-1.96*se:+.3f},{d.mean()+1.96*se:+.3f}]  p={p:.3g}")

        print()
        line("k=4 alone", k4 - base)
        line("keyphrases-off alone", kpo - base)
        line("BOTH", both - base)
        line("BOTH vs keyphrases-off alone", both - kpo)
        D = both - k4 - kpo + base
        line("interaction D", D)
        se = D.std(ddof=1) / np.sqrt(len(D))
        print(f"   -> {'super-additive harm replicates' if (D.mean() < 0 and wilcoxon(D).pvalue < 0.05) else 'interaction does NOT replicate on arXiv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["name", "judge", "lineup", "report"])
    ap.add_argument("--concurrency", type=int, default=24)
    args = ap.parse_args()
    {"name": lambda: stage_name(),
     "judge": lambda: stage_judge(args.concurrency),
     "lineup": lambda: stage_lineup(args.concurrency),
     "report": lambda: report()}[args.stage]()


if __name__ == "__main__":
    main()
