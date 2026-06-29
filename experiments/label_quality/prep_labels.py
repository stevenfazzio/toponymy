"""
Phase 0 . gold labeling -- one Toponymy naming pass on a dataset, cached as
`topic_names_` (finest->coarsest) for the perturbation battery + downstream probes.

Fits the clusterer with the SAME params `perturbations.py` uses (base_min_cluster_size,
min_clusters, subsample) so cluster indices line up, then runs the real Toponymy.fit
once (no contrast) and saves `model.topic_names_`.

  uv run python experiments/label_quality/prep_labels.py --model haiku
  uv run python experiments/label_quality/prep_labels.py --model haiku --subsample 2000

(opus rejects temperature=0; the stock AnthropicNamer is fine at default temp for
haiku/sonnet/gpt4omini, which is all we need for a gold anchor.)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
NIBLING = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(NIBLING))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="20ng")
    ap.add_argument("--model", default="haiku",
                    choices=["mock", "haiku", "sonnet", "opus", "gpt4omini"])
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--base-min-cluster-size", type=int, default=25)
    ap.add_argument("--min-clusters", type=int, default=4)
    args = ap.parse_args()

    from ab_harness import load_dataset, make_embedder, make_namer

    from toponymy.cluster_layer import ClusterLayerText
    from toponymy.clustering import ToponymyClusterer
    from toponymy.toponymy import Toponymy

    t0 = time.time()
    objects, emb, coords, meta = load_dataset(args.dataset, args.subsample)
    clusterer = ToponymyClusterer(min_clusters=args.min_clusters,
                                  base_min_cluster_size=args.base_min_cluster_size, verbose=False)
    clusterer.fit_predict(coords, emb, ClusterLayerText)
    counts = [int(l.centroid_vectors.shape[0]) for l in clusterer.cluster_layers_]
    print(f"[{args.dataset}/{args.model}] {len(objects)} objects; "
          f"clusters/layer (finest->coarsest): {counts}")

    namer = make_namer(args.model)
    embedder = make_embedder(meta["emb_model"])
    model = Toponymy(namer, embedder, clusterer=clusterer,
                     object_description=meta["obj"], corpus_description=meta["corpus"],
                     verbose=True)
    model.fit(objects, emb, coords)

    names = [list(layer) for layer in model.topic_names_]
    out = HERE / "data" / f"labels_{args.dataset}_{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(names, indent=2))

    for L, layer in enumerate(names):
        print(f"  L{L}: {len(layer)} labels ({len(set(layer))} unique); e.g. "
              + ", ".join(repr(x) for x in layer[:4]))
    print(f"\nsaved gold labels -> {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
