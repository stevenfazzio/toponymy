"""
Reproduce the Movie Madness "movies supercluster named as a genre" case, and test whether
contrast fixes it.

Re-embed movie+TV synopses with ramify's embedder (BAAI/bge-small-en-v1.5), cluster with
EVoCClusterer (as ramify does), then LLM-name (haiku) the hierarchy three ways: baseline,
v1-contrast (original block, no gate), v2-contrast (concise block, max_dist=0.25). Identify
the movies/TV superclusters at the coarsest layer via the real `media_type` field, and report
each coarsest cluster's name per arm -- does the movie-dominated supercluster flip from a genre
to a medium-level name ("Movies") once it can see the TV cluster as contrast? Also prints the
contrast actually injected, to show whether the v2 gate filters the (semantically distant) TV
sibling out -- the predicted tension between average-harm-reduction and this specific case.

Run:  uv run python experiments/nibling_contrast/reproduce_movies.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(HERE))
from ab_harness import make_namer, name_once  # noqa: E402

from sentence_transformers import SentenceTransformer  # noqa: E402
from toponymy.clustering import EVoCClusterer  # noqa: E402
from toponymy.cluster_layer import ClusterLayerText  # noqa: E402

MM = Path("/Users/stevenfazzio/repos/movie-madness-map")
N = 16000
SEED = 42
EMB_MODEL = "BAAI/bge-small-en-v1.5"
NAMER = "haiku"
META = dict(obj="movie and TV titles", corpus="a video rental store catalog")


def load():
    f = pd.read_parquet(MM / "data/films.parquet",
                        columns=["media_type", "embed_text_synopsis", "title"])
    f = f[f["media_type"].isin(["movie", "tv"])].reset_index(drop=True)
    sub = f.sample(n=min(N, len(f)), random_state=SEED).reset_index(drop=True)
    return (sub["embed_text_synopsis"].astype(str).tolist(),
            sub["media_type"].to_numpy(), sub["title"].tolist())


def main():
    texts, media, titles = load()
    print(f"{len(texts)} docs ({(media=='movie').sum()} movie / {(media=='tv').sum()} tv)", flush=True)

    enc = SentenceTransformer(EMB_MODEL, device="cpu")
    emb = enc.encode(texts, batch_size=64, show_progress_bar=True,
                     convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    print("embedded", emb.shape, flush=True)

    clusterer = EVoCClusterer(min_clusters=2, base_min_cluster_size=25, verbose=False)
    clusterer.fit_predict(emb, emb, ClusterLayerText)
    counts = [int(l.centroid_vectors.shape[0]) for l in clusterer.cluster_layers_]
    print("EVoC clusters/layer (finest->coarsest):", counts, flush=True)

    top = len(clusterer.cluster_layers_) - 1
    top_labels = clusterer.cluster_layers_[top].cluster_labels
    tree = clusterer.cluster_tree_

    namer = make_namer(NAMER)
    common = dict(objects=texts, emb=emb, coords=emb, clusterer=clusterer,
                  namer=namer, embedder=enc, meta=META, k=5, disambig=False)
    print("naming baseline ...", flush=True)
    base, _ = name_once(use_contrast=False, max_dist=None, **common)
    print("naming v1-contrast (no gate) ...", flush=True)
    c1, s1 = name_once(use_contrast=True, max_dist=None, block_version="v1", **common)
    print("naming v2-contrast (gate 0.25) ...", flush=True)
    c2, s2 = name_once(use_contrast=True, max_dist=0.25, block_version="v2", **common)

    print(f"\n===== COARSEST LAYER (L{top}, {counts[top]} clusters) =====", flush=True)
    rows = []
    for c in range(counts[top]):
        m = top_labels == c
        n = int(m.sum())
        if n == 0:
            continue
        tvf = float((media[m] == "tv").mean())
        kind = "TV" if tvf > 0.6 else ("MOVIE" if tvf < 0.25 else "mixed")
        kids = [idx for (cl, idx) in tree.get((top, c), []) if cl == top - 1]
        genres = [base[top - 1][i] for i in kids if i < len(base[top - 1])]
        print(f"\n cluster {c}: n={n}  tv_frac={tvf:.2f}  [{kind}]")
        print(f"   baseline : {base[top][c]!r}")
        print(f"   v1       : {c1[top][c]!r}")
        print(f"   v2       : {c2[top][c]!r}")
        print(f"   v1 contrast injected: {s1['injected_names'].get(f'{top},{c}', [])}")
        print(f"   v2 contrast injected: {s2['injected_names'].get(f'{top},{c}', [])}")
        print(f"   its children (genres, baseline): {genres[:10]}")
        rows.append(dict(cluster=c, n=n, tv_frac=tvf, kind=kind,
                         baseline=base[top][c], v1=c1[top][c], v2=c2[top][c],
                         v1_contrast=s1["injected_names"].get(f"{top},{c}", []),
                         v2_contrast=s2["injected_names"].get(f"{top},{c}", []),
                         children=genres))

    (HERE / "data" / "movie_repro.json").write_text(json.dumps(dict(
        counts=counts, coarsest=rows,
        all_layers=dict(baseline=base, v1=c1, v2=c2)), indent=2))
    print("\nsaved -> data/movie_repro.json", flush=True)


if __name__ == "__main__":
    main()
