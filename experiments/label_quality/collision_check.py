"""
Does the length controller create more name-collision pressure than stock?

Toponymy's disambiguation pass triggers per layer by clustering topic-NAME embeddings
(cluster_topic_names_for_renaming: agglomerative, complete linkage, adaptive cosine threshold)
and issuing one large renaming LLM call per name-group of size >= 2. Shorter names are more
collision-prone, so the controller's picks might buy their length savings back in disambiguation
calls. This simulates the trigger with the library's own function on (a) the stock names and
(b) the controller-chosen names, per layer and corpus, plus strict duplicate counts and
within-lineup collisions (nn lineup-mates sharing a name -- the ones that directly hurt
wayfinding). Zero LLM calls.

  uv run python experiments/label_quality/collision_check.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
NIBLING = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(NIBLING))
sys.path.insert(0, str(HERE))

from toponymy.prompt_construction import cluster_topic_names_for_renaming  # noqa: E402


def trigger_stats(names_by_layer, embedder):
    """Per layer: (groups, topics_in_groups, strict_dupes) under the library's own trigger."""
    out = []
    for names in names_by_layer:
        emb = embedder.encode(names, convert_to_numpy=True, show_progress_bar=False)
        groups, labels = cluster_topic_names_for_renaming(names, emb)
        n_topics = int(sum(np.sum(labels == g) for g in groups))
        dupes = sum(c - 1 for c in Counter(names).values() if c > 1)
        out.append(dict(n=len(names), groups=len(groups), topics=n_topics, dupes=dupes))
    return out


def lineup_collisions(ds, names_map):
    """Pairs of nn-lineup mates sharing an identical name (what wayfinding directly suffers)."""
    from wayfinding import Cell, K_DEFAULT
    cell = Cell(ds)
    hits = 0
    for L in range(len(cell.counts)):
        for i in range(cell.counts[L]):
            for j in cell.lineup(L, i, K_DEFAULT, "nn")[1:]:
                if names_map[(L, i)] == names_map[(L, j)]:
                    hits += 1
    return hits // 2  # each pair counted from both sides


def count_disambiguation_calls(ds: str, rung: int):
    """Refit one rung and count the disambiguation prompts the pass actually issued per layer
    (the ladder cache is post-disambiguation, so this is the only way to see the pass working)."""
    from ab_harness import make_embedder, make_namer
    from toponymy.toponymy import Toponymy
    from wayfinding import Cell

    cell = Cell(ds)
    model = Toponymy(make_namer("haiku"), make_embedder(cell.meta["emb_model"]),
                     clusterer=cell.clusterer, object_description=cell.meta["obj"],
                     corpus_description=cell.meta["corpus"],
                     lowest_detail_level=rung / 6, highest_detail_level=rung / 6, verbose=False)
    model.fit(cell.objects, cell.emb, cell.coords)
    for L, layer in enumerate(model.cluster_layers_):
        prompts = getattr(layer, "disambiguation_prompts", []) or []
        idx = getattr(layer, "dismbiguation_topic_indices", []) or []
        n_topics = int(sum(len(x) for x in idx))
        print(f"  rung {rung} L{L}: {len(prompts)} disambiguation calls covering {n_topics} topics "
              f"(of {cell.counts[L]})")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--count-disambig", nargs=2, metavar=("DS", "RUNG"),
                    help="refit one rung and count actual disambiguation calls")
    args = ap.parse_args()
    if args.count_disambig:
        count_disambiguation_calls(args.count_disambig[0], int(args.count_disambig[1]))
        return

    from ab_harness import make_embedder
    embedder = make_embedder("all-MiniLM-L6-v2")

    for ds in ["20ng", "arxiv_home"]:
        summary = json.loads((HERE / "data" / f"length_controller_{ds}.json").read_text())
        rows = summary["rows"]
        n_layers = max(r["L"] for r in rows) + 1
        counts = [max(r["i"] for r in rows if r["L"] == L) + 1 for L in range(n_layers)]
        stock = [[next(r["stock"] for r in rows if r["L"] == L and r["i"] == i)
                  for i in range(counts[L])] for L in range(n_layers)]
        chosen = [[next(r["chosen"] for r in rows if r["L"] == L and r["i"] == i)
                   for i in range(counts[L])] for L in range(n_layers)]

        print(f"\n================ {ds} (layers {counts}) ================")
        print(f"{'':>8} {'groups':>7} {'topics-in-groups':>17} {'strict dupes':>13}")
        for arm, nm in [("stock", stock), ("chosen", chosen)]:
            st = trigger_stats(nm, embedder)
            g = "/".join(str(x["groups"]) for x in st)
            t = "/".join(str(x["topics"]) for x in st)
            d = "/".join(str(x["dupes"]) for x in st)
            print(f"{arm:>8} {g:>7} {t:>17} {d:>13}   (per layer L0/L1/...)")
        for arm, nm in [("stock", stock), ("chosen", chosen)]:
            nmap = {(L, i): nm[L][i] for L in range(n_layers) for i in range(counts[L])}
            print(f"{arm:>8} within-lineup identical-name pairs: {lineup_collisions(ds, nmap)}")


if __name__ == "__main__":
    main()
