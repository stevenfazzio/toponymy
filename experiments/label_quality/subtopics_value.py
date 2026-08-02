"""
Phase 6f -- what are subcluster names (subtopics) actually worth?

Subtopics are the third naming-prompt feature and the least examined. The #173 ablation reported
judge-delta -0.033 ("no measurable effect"), but that number averages over ALL layers, and layer 0
has no children -- so 73 of its 106 rows are structurally a no-op. Worse, they are not quiet: 54 of
73 layer-0 labels still "changed" under the ablation, which is just temperature-0.4 redraw noise.
Restricted to the layers where subtopics exist, 20NG gives n=33, judge-delta -0.131, SE 0.121,
CI [-0.369, +0.106] -- not a null, an absence of measurement.

There is also a mechanism confound. `topic_name_prompt` has a `[!SKIP!]` path: a cluster with
exactly one named child inherits that child's name with no LLM call at all. Blanking subtopics
disables it, so single-child clusters take a different code path rather than the same prompt minus
a feature. On 20NG that is 12 of 33 coarse clusters (9 of 24 at L1, 3 of 9 at L2), so the two
mechanisms are reported separately here.

This measures subtopics properly: coarse layers only, split single-child vs multi-child, on both
axes (grounded judge for fit, wayfinding lineup for identification), on both corpora.

POWER CEILING, stated up front: pooling both corpora gives ~67 coarse clusters, and the
single/multi split cuts that to roughly 45 and 22. That resolves an effect around 0.3 or larger and
cannot resolve 0.1. Toponymy hierarchies simply do not have many coarse clusters -- that is the
point of a hierarchy -- so this is a structural limit, not one more spending fixes.

  uv run python experiments/label_quality/subtopics_value.py --stage name     # arXiv arm
  uv run python experiments/label_quality/subtopics_value.py --stage judge    # arXiv arm
  uv run python experiments/label_quality/subtopics_value.py --stage lineup   # both corpora
  uv run python experiments/label_quality/subtopics_value.py --stage report
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
NIBLING = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(NIBLING))

ARXIV_NAMES = HERE / "data" / "subtopics_names_arxiv.json"
ARXIV_JUDGE = HERE / "data" / "subtopics_judge_arxiv.json"
LINEUP = {"20ng": HERE / "data" / "wayfinding_20ng_subtopics.json",
          "arxiv_home": HERE / "data" / "wayfinding_arxiv_home_subtopics.json"}
K_SAMPLES, N_REPEAT = 3, 20


def child_counts(tree) -> dict:
    """(L,i) -> number of children in the cluster tree (0 for leaves at layer 0)."""
    return {k: len(v) for k, v in tree.items()}


def fits():
    """Both substrates, with their cluster trees, for the single-child split."""
    from perturbations import load_fit
    from arxiv_naming_features import fresh_clusterer, substrate

    cl20, *_ = load_fit("20ng", None, 25, 4)
    objects, emb, coords, meta, embedder = substrate()
    clax = fresh_clusterer(coords, emb)
    return {"20ng": cl20.cluster_tree_, "arxiv_home": clax.cluster_tree_}


# ------------------------------------------------------------------ arXiv arm

def stage_name():
    from ab_harness import make_namer
    from ablation import ablate
    from arxiv_naming_features import fresh_clusterer, substrate

    from toponymy.toponymy import Toponymy

    if ARXIV_NAMES.exists():
        print("arXiv subtopics naming: cached")
        return
    objects, emb, coords, meta, embedder = substrate()
    clusterer = fresh_clusterer(coords, emb)
    print(f"clusters {[int(l.centroid_vectors.shape[0]) for l in clusterer.cluster_layers_]}")
    model = Toponymy(make_namer("haiku"), embedder, clusterer=clusterer,
                     object_description=meta["obj"], corpus_description=meta["corpus"],
                     verbose=False)
    with ablate("subtopics"):
        model.fit(objects, emb, coords)
    ARXIV_NAMES.write_text(json.dumps([list(x) for x in model.topic_names_], indent=2))
    print("written")


def stage_judge(concurrency: int):
    from async_judge import rate_many
    from arxiv_naming_features import (fresh_clusterer, gold_labels, gold_ratings, home_docs,
                                       substrate)

    objects, emb, coords, meta, _ = substrate()
    clusterer = fresh_clusterer(coords, emb)
    docs = home_docs(clusterer, objects, emb)
    names = json.loads(ARXIV_NAMES.read_text())
    gold, gr = gold_labels(), gold_ratings()
    done = json.loads(ARXIV_JUDGE.read_text()) if ARXIV_JUDGE.exists() else {}

    todo = []
    for L in range(1, len(names)):                       # coarse layers only
        for i, lab in enumerate(names[L]):
            ck = f"{L}.{i}"
            if not lab or lab == "Unlabelled" or (L, i) not in gr or not docs.get((L, i)):
                continue
            if done.get(ck, {}).get("label", "").strip() == lab.strip():
                continue
            if lab.strip() == gold[(L, i)].strip():
                done[ck] = {"label": lab, "overall": gr[(L, i)]}
                continue
            todo.append((L, i, lab))
    print(f"arXiv subtopics: {len(todo)} coarse labels to judge ({len(done)} cached)")
    if todo:
        res = rate_many([(lab, docs[(L, i)]) for L, i, lab in todo], meta["obj"],
                        "anthropic/claude-sonnet-4-6", k=K_SAMPLES, concurrency=concurrency)
        for (L, i, lab), r in zip(todo, res):
            done[f"{L}.{i}"] = {"label": lab, "overall": r["overall"]}
        ARXIV_JUDGE.write_text(json.dumps(done, indent=2))
    print(f"  rated {sum(1 for v in done.values() if v['overall'] is not None)}")


# ------------------------------------------------------------------ lineups (both corpora)

def stage_lineup(concurrency: int):
    from wayfinding import Cell, K_DEFAULT, MODELS, make_unit, run_all

    for ds in ["20ng", "arxiv_home"]:
        if ds == "20ng":
            rows = json.loads((HERE / "data" / "ablation.json").read_text())["subtopics"]
            labels = {(r["L"], r["i"]): r["abl"] for r in rows if r["L"] >= 1}
            gold = {(r["L"], r["i"]): r["full"] for r in rows if r["L"] >= 1}
        else:
            from arxiv_naming_features import gold_labels
            names = json.loads(ARXIV_NAMES.read_text())
            gold_all = gold_labels()
            labels = {(L, i): n for L in range(1, len(names))
                      for i, n in enumerate(names[L])}
            gold = {k: v for k, v in gold_all.items() if k[0] >= 1}

        cell = Cell(ds)
        print(f"\n[{ds}] replayed fit {cell.counts}")
        units = []
        for (L, i), lab in sorted(labels.items()):
            if not lab or lab == "Unlabelled" or (L, i) not in gold:
                continue
            if lab.strip() == gold[(L, i)].strip():
                continue                       # inherited/identical => identical pm, no call
            units.append(make_unit(cell, L, i, "abl_subtopics", lab, K_DEFAULT, "nn", run=1))
        rng = np.random.default_rng(717)
        if units:
            pick = rng.choice(len(units), min(N_REPEAT, len(units)), replace=False)
            units += [dict(units[j], run=2, uid=units[j]["uid"][:-1] + "2") for j in pick]
        print(f"[{ds}] {len(units)} units x {K_SAMPLES} samples")
        if units:
            asyncio.run(run_all(units, cell, MODELS["sonnet"], concurrency, LINEUP[ds]))


# ------------------------------------------------------------------ report

def _stats(d):
    d = np.asarray(d, dtype=float)
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    p = wilcoxon(d).pvalue if len(d) > 1 and np.any(d != 0) else float("nan")
    return d.mean(), se, p


def _fmt(name, d):
    if len(d) < 3:
        return f"  {name:<28} n={len(d):>3}   (too few)"
    m, se, p = _stats(d)
    return (f"  {name:<28} n={len(d):>3}   {m:+.3f}  SE {se:.3f}  "
            f"[{m-1.96*se:+.3f},{m+1.96*se:+.3f}]  p={p:.3g}")


def report():
    trees = fits()
    nkids = {ds: child_counts(t) for ds, t in trees.items()}

    print("\n===== SUBTOPICS (SUBCLUSTER NAMES): WHAT ARE THEY WORTH? =====")
    print("coarse layers only (L>=1); single-child clusters take the [!SKIP!] name-propagation")
    print("path, so they are a different mechanism and are reported separately.\n")

    # ---------------- fit ----------------
    print("--- FIT (grounded judge; positive = dropping subtopics HURT) ---")
    pooled = {"single": [], "multi": []}
    for ds in ["20ng", "arxiv_home"]:
        if ds == "20ng":
            rows = [r for r in json.loads((HERE / "data" / "ablation.json").read_text())["subtopics"]
                    if r["L"] >= 1]
            data = [((r["L"], r["i"]), r["j_full"] - r["j_abl"]) for r in rows]
        else:
            if not ARXIV_JUDGE.exists():
                print(f"  [{ds}] not judged yet"); continue
            from arxiv_naming_features import gold_ratings
            gr = gold_ratings()
            j = json.loads(ARXIV_JUDGE.read_text())
            data = []
            for s, v in j.items():
                L, i = map(int, s.split("."))
                if v["overall"] is not None and (L, i) in gr:
                    data.append(((L, i), gr[(L, i)] - v["overall"]))
        allv = [v for _, v in data]
        single = [v for k, v in data if nkids[ds].get(k, 0) == 1]
        multi = [v for k, v in data if nkids[ds].get(k, 0) > 1]
        pooled["single"] += single
        pooled["multi"] += multi
        print(f" [{ds}]")
        print(_fmt("all coarse", allv))
        print(_fmt("  single-child (propagation)", single))
        print(_fmt("  multi-child (real prompt)", multi))
    print(" [pooled across corpora]")
    print(_fmt("  single-child", pooled["single"]))
    print(_fmt("  multi-child", pooled["multi"]))

    # ---------------- identification ----------------
    print("\n--- IDENTIFICATION (wayfinding lineup; positive = dropping subtopics HURT) ---")
    pooled_id = {"single": [], "multi": []}
    for ds, battery in [("20ng", "wayfinding_20ng_battery.json"),
                        ("arxiv_home", "wayfinding_arxiv_home_battery.json")]:
        if not LINEUP[ds].exists():
            print(f" [{ds}] lineup not run yet"); continue
        gp = {(u["L"], u["i"]): u["pm"] for u in
              json.loads((HERE / "data" / battery).read_text())["units"].values()
              if u["kind"] == "gold" and u["mode"] == "nn" and u.get("pm") is not None}
        us = [u for u in json.loads(LINEUP[ds].read_text())["units"].values()
              if u.get("pm") is not None]
        r1 = {u["uid"][:-1]: u for u in us if u["run"] == 1}
        r2 = {u["uid"][:-1]: u for u in us if u["run"] == 2}
        rep = [abs(r1[x]["pm"] - r2[x]["pm"]) for x in r2 if x in r1]
        band = float(np.percentile(rep, 90)) if rep else float("nan")
        data = [((u["L"], u["i"]), gp[(u["L"], u["i"])] - u["pm"]) for u in us
                if u["run"] == 1 and (u["L"], u["i"]) in gp]
        single = [v for k, v in data if nkids[ds].get(k, 0) == 1]
        multi = [v for k, v in data if nkids[ds].get(k, 0) > 1]
        pooled_id["single"] += single
        pooled_id["multi"] += multi
        print(f" [{ds}]  repeat band p90 = {band:.3f}")
        print(_fmt("all coarse", [v for _, v in data]))
        print(_fmt("  single-child", single))
        print(_fmt("  multi-child", multi))
    print(" [pooled across corpora]")
    print(_fmt("  single-child", pooled_id["single"]))
    print(_fmt("  multi-child", pooled_id["multi"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["name", "judge", "lineup", "report"])
    ap.add_argument("--concurrency", type=int, default=24)
    args = ap.parse_args()
    {"name": stage_name, "judge": lambda: stage_judge(args.concurrency),
     "lineup": lambda: stage_lineup(args.concurrency), "report": report}[args.stage]()


if __name__ == "__main__":
    main()
