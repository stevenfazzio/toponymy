"""
arXiv human-calibration seed for the grounded judge (the 20NG seed -> kappa 0.64 does NOT cover
arXiv, and 3 of the 4 robustness cells are arXiv and/or cohere). Samples ~20 BLINDED, stratified
candidates from the at-home arXiv cells, regenerates the EXACT grounding docs the judge saw by
deterministically replaying each cell's clustering (cheap, no LLM), and emits the same rating HTML.

  uv run python experiments/label_quality/make_calibration_home.py
Then serve + open (localhost so clipboard works):
  python3 -m http.server 8765 --bind 127.0.0.1 --directory experiments/label_quality/data
  -> http://127.0.0.1:8765/arxiv_calibration.html
Rate every item, "Copy results", paste back. Score with:
  uv run python experiments/label_quality/score_calibration.py \
      --key arxiv_calibration_key.json --human arxiv_calibration_human.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
NIBLING = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(NIBLING))
sys.path.insert(0, str(HERE))

from make_calibration import HTML_TEMPLATE  # noqa: E402  (reuse the exact rating form)
from metrics import unit  # noqa: E402


def build_docs_for(tag: str, n_near: int, n_rand: int, maxlen: int):
    """Replay home_pipeline.py's clustering + grounding-doc selection for one cell, EXACTLY.

    Deterministic: cached coords/emb on disk -> same ToponymyClusterer -> same labels/centroids ->
    same nearest+random docs (rng seeded 1000+i, as in home_pipeline). Returns (docs_for, counts).
    """
    from ab_harness import load_dataset
    from toponymy.cluster_layer import ClusterLayerText
    from toponymy.clustering import ToponymyClusterer

    dataset = tag.split("_")[0]
    emb = np.load(HERE / "data" / f"home_{tag}_emb.npy")
    coords = np.load(HERE / "data" / f"home_{tag}_coords.npy")
    objects, _, _, _ = load_dataset(dataset, emb.shape[0])

    clusterer = ToponymyClusterer(min_clusters=4, base_min_cluster_size=25, verbose=False)
    clusterer.fit_predict(coords, emb, ClusterLayerText)
    counts = [int(l.centroid_vectors.shape[0]) for l in clusterer.cluster_layers_]

    emb64 = emb.astype(np.float64)
    docs_for = {}
    for L in range(len(counts)):
        cls = clusterer.cluster_layers_[L].cluster_labels
        C = clusterer.cluster_layers_[L].centroid_vectors
        for i in range(counts[L]):
            mem = np.where(cls == i)[0]
            if mem.size == 0:
                docs_for[(L, i)] = []
                continue
            sims = unit(emb64[mem]) @ unit(C[i].astype(np.float64))
            near = mem[np.argsort(-sims)[:n_near]]
            rest = np.array([m for m in mem if m not in set(near.tolist())])
            rng = np.random.default_rng(1000 + i)
            rand = rng.choice(rest, size=min(n_rand, rest.size), replace=False) if rest.size else np.array([], int)
            docs_for[(L, i)] = [" ".join(str(objects[j]).split())[:maxlen] for j in list(near) + list(rand)]
    return docs_for, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["arxiv_minilm", "arxiv_cohere"])
    ap.add_argument("--n-gold", type=int, default=8)
    ap.add_argument("--n-each", type=int, default=3, help="per degraded type (verbose/ancestor/sibling/generic)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n-near", type=int, default=10)
    ap.add_argument("--n-rand", type=int, default=5)
    ap.add_argument("--maxlen", type=int, default=280)
    args = ap.parse_args()

    # pool judged rows across cells, regenerating each cell's exact grounding docs
    pool = defaultdict(list)  # kind -> list of (item_dict, key_dict)
    for tag in args.cells:
        rows = json.loads((HERE / "data" / f"home_{tag}.json").read_text())["rows"]
        docs_for, _ = build_docs_for(tag, args.n_near, args.n_rand, args.maxlen)
        for r in rows:
            if r["overall"] is None or r["kind"].startswith("abl:"):
                continue
            iid = f"{tag}_{r['L']}_{r['i']}_{r['kind']}"
            item = {"id": iid, "label": r["label"], "docs": docs_for.get((r["L"], r["i"]), [])}
            key = {"type": r["kind"], "judge_overall": r["overall"], "cell": tag,
                   "layer": r["L"], "idx": r["i"]}
            pool[r["kind"]].append((item, key))

    rng = np.random.default_rng(args.seed)
    plan = {"gold": args.n_gold, "verbose": args.n_each, "ancestor": args.n_each,
            "sibling": args.n_each, "generic": args.n_each}
    chosen = []
    for t, n in plan.items():
        cand = pool.get(t, [])
        take = rng.choice(len(cand), size=min(n, len(cand)), replace=False)
        chosen += [cand[i] for i in take]
    rng.shuffle(chosen)

    items = [it for it, _ in chosen]
    key = {it["id"]: k for it, k in chosen}

    data_json = json.dumps(items).replace("</", "<\\/")
    (HERE / "data" / "arxiv_calibration.html").write_text(HTML_TEMPLATE.replace("__DATA__", data_json))
    (HERE / "data" / "arxiv_calibration_key.json").write_text(json.dumps(key, indent=2))

    print(f"arXiv calibration set: {len(items)} blinded items "
          f"({dict(Counter(k['type'] for k in key.values()))}; "
          f"cells {dict(Counter(k['cell'] for k in key.values()))})")
    print("wrote  experiments/label_quality/data/arxiv_calibration.html  (+ arxiv_calibration_key.json)")


if __name__ == "__main__":
    main()
