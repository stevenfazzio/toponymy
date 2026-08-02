"""
Phase 6g -- do the naming features reduce how much work the disambiguation pass has to do?

Toponymy names topics, then runs a per-layer disambiguation pass: it clusters topic-NAME embeddings
(cluster_topic_names_for_renaming) and issues one renaming call per name-group of size >= 2. A
feature that gives the namer more context about what sits nearby could plausibly produce more
distinct names up front and leave the pass less to fix. Subtopics are the natural candidate -- a
parent that can see its children's names has explicit knowledge of its own neighbourhood.

Nothing on disk can answer this. Final (post-pass) name sets are clean for every condition, so the
pressure the pass absorbed is not recoverable; the load is only visible during fit, in
`layer.dismbiguation_topic_indices`. So this instruments `ClusterLayer.disambiguate_topics` to
capture, per layer: the PRE-pass names, the groups the trigger formed, and the POST-pass names.

Two measures, deliberately:

  GROUP COUNTS are what the pass actually costs, but they are tiny integers (#177 saw 2-5 groups
  over 74 fine-layer clusters) and naming runs at temperature 0.4, so they are badly underpowered
  at coarse layers where only 24 and 9 clusters exist.

  PRE-PASS NAME SIMILARITY is the continuous version of the same pressure -- 276 pairs at L1 rather
  than a 0-2 count -- and is where any real signal should show up first.

4 conditions x 2 corpora x 3 draws = 24 naming runs (haiku).

  uv run python experiments/label_quality/disambiguation_load.py --stage run
  uv run python experiments/label_quality/disambiguation_load.py --stage report
Writes data/disamb_load.json (resumable; each cell keyed ds|condition|draw).
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
NIBLING = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(NIBLING))

OUT = HERE / "data" / "disamb_load.json"
CONDITIONS = ["stock", "abl_exemplars", "abl_keyphrases", "abl_subtopics"]
DATASETS = ["20ng", "arxiv_home"]
DRAWS = 3


def instrument(record: dict):
    """Wrap disambiguate_topics to capture pre-pass names + the groups the trigger formed."""
    import toponymy.cluster_layer as clm

    orig = clm.ClusterLayer.disambiguate_topics

    def patched(self, llm, detail_level, all_topic_names, object_description,
                corpus_description, cluster_tree=None, embedding_model=None):
        pre = list(self.topic_names)
        orig(self, llm, detail_level, all_topic_names, object_description,
             corpus_description, cluster_tree, embedding_model)
        groups = [list(map(int, g)) for g in getattr(self, "dismbiguation_topic_indices", [])]
        record.setdefault("layers", []).append(
            {"layer_id": int(self.layer_id), "pre": pre, "post": list(self.topic_names),
             "groups": groups, "n_groups": len(groups),
             "n_topics_in_groups": int(sum(len(g) for g in groups))})

    clm.ClusterLayer.disambiguate_topics = patched
    return lambda: setattr(clm.ClusterLayer, "disambiguate_topics", orig)


def run_one(ds: str, cond: str) -> dict:
    from ab_harness import load_dataset, make_embedder, make_namer
    from ablation import ablate

    from toponymy.cluster_layer import ClusterLayerText
    from toponymy.clustering import ToponymyClusterer
    from toponymy.toponymy import Toponymy

    if ds == "20ng":
        objects, emb, coords, meta = load_dataset("20ng", None)
        embedder = make_embedder(meta["emb_model"])
    else:
        from arxiv_naming_features import substrate
        objects, emb, coords, meta, embedder = substrate()

    clusterer = ToponymyClusterer(min_clusters=4, base_min_cluster_size=25, verbose=False)
    clusterer.fit_predict(coords, emb, ClusterLayerText)
    counts = [int(l.centroid_vectors.shape[0]) for l in clusterer.cluster_layers_]

    record = {"counts": counts}
    restore = instrument(record)
    try:
        model = Toponymy(make_namer("haiku"), embedder, clusterer=clusterer,
                         object_description=meta["obj"], corpus_description=meta["corpus"],
                         verbose=False)
        feat = cond[4:] if cond.startswith("abl_") else None
        with (ablate(feat) if feat else nullcontext()):
            model.fit(objects, emb, coords)
    finally:
        restore()
    record["final"] = [list(x) for x in model.topic_names_]
    return record


def stage_run():
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [(ds, c, d) for ds in DATASETS for c in CONDITIONS for d in range(DRAWS)
            if f"{ds}|{c}|{d}" not in done]
    print(f"{len(todo)} naming runs to do ({len(done)} cached)")
    for ds, cond, draw in todo:
        key = f"{ds}|{cond}|{draw}"
        print(f"\n=== {key} ===", flush=True)
        rec = run_one(ds, cond)
        done[key] = rec
        OUT.write_text(json.dumps(done, indent=2))
        per = [(l["layer_id"], l["n_groups"], l["n_topics_in_groups"]) for l in rec["layers"]]
        print(f"  counts {rec['counts']} | (layer, groups, topics): {per}")


def report():
    from sentence_transformers import SentenceTransformer

    data = json.loads(OUT.read_text())
    E = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    def sim(names):
        if len(names) < 3:
            return float("nan"), float("nan")
        v = E.encode(names, convert_to_numpy=True, show_progress_bar=False)
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        S = v @ v.T
        p = S[np.triu_indices(len(names), 1)]
        return float(p.mean()), float(np.percentile(p, 95))

    print("\n===== DISAMBIGUATION LOAD BY NAMING FEATURE =====")
    print("groups = renaming calls the pass had to issue; pre-sim = pairwise similarity of the")
    print("PRE-pass names (the continuous version of collision pressure). 3 draws per cell.\n")

    for ds in DATASETS:
        cells = {c: [data[k] for k in data if k.startswith(f"{ds}|{c}|")] for c in CONDITIONS}
        if not any(cells.values()):
            continue
        nlayers = max(len(r["layers"]) for rs in cells.values() for r in rs)
        print(f"--- {ds} ---")
        header = f"  {'condition':<16}"
        for L in range(nlayers):
            header += f" | L{L}: {'grp':>4} {'topics':>6} {'pre-sim':>8} {'p95':>6}"
        print(header)
        for c in CONDITIONS:
            rs = cells[c]
            if not rs:
                continue
            row = f"  {c:<16}"
            for L in range(nlayers):
                ls = [next((x for x in r["layers"] if x["layer_id"] == L), None) for r in rs]
                ls = [x for x in ls if x]
                if not ls:
                    row += f" | L{L}: {'--':>4} {'--':>6} {'--':>8} {'--':>6}"
                    continue
                g = np.mean([x["n_groups"] for x in ls])
                t = np.mean([x["n_topics_in_groups"] for x in ls])
                sims = [sim(x["pre"]) for x in ls]
                m = np.nanmean([s[0] for s in sims])
                p = np.nanmean([s[1] for s in sims])
                row += f" | L{L}: {g:>4.1f} {t:>6.1f} {m:>8.3f} {p:>6.3f}"
            print(row)

        # the hypothesis, stated as a contrast against stock
        print(f"\n  vs stock (positive = the ablation made the pass work HARDER):")
        base = {L: [x for r in cells["stock"] for x in r["layers"] if x["layer_id"] == L]
                for L in range(nlayers)}
        for c in CONDITIONS[1:]:
            if not cells[c]:
                continue
            parts = []
            for L in range(nlayers):
                cur = [x for r in cells[c] for x in r["layers"] if x["layer_id"] == L]
                if not cur or not base[L]:
                    continue
                dg = np.mean([x["n_groups"] for x in cur]) - np.mean([x["n_groups"] for x in base[L]])
                ds_ = (np.nanmean([sim(x["pre"])[0] for x in cur])
                       - np.nanmean([sim(x["pre"])[0] for x in base[L]]))
                parts.append(f"L{L} groups {dg:+.1f}, pre-sim {ds_:+.3f}")
            print(f"    {c:<16} " + " | ".join(parts))
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run", "report"])
    args = ap.parse_args()
    (stage_run if args.stage == "run" else report)()


if __name__ == "__main__":
    main()
