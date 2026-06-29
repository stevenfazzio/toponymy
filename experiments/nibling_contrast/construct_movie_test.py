"""
Constructed test of the motivating hypothesis.

Impose the movies-vs-TV top split from the real `media_type` field (what you observed), and
sub-cluster genres WITHIN each medium naturally (KMeans on the real embeddings). Build a real
2-layer Toponymy structure (fine = genres/TV-types, coarse = movies/TV) + tree, then LLM-name
(haiku) the coarse layer THREE ways: baseline, v1-contrast (original block), v2-contrast
(concise block) -- both contrast arms WITHOUT a max_dist gate, so the (distant) TV sibling
actually reaches the movies cluster. Question: does seeing "TV" nearby flip the movies
supercluster from a genre/film-y name to a medium-level "Movies"?

Run:  uv run python experiments/nibling_contrast/construct_movie_test.py
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(HERE))
from ab_harness import make_namer, name_once  # noqa: E402

from sentence_transformers import SentenceTransformer  # noqa: E402
from toponymy.clustering import centroids_from_labels  # noqa: E402
from toponymy.cluster_layer import ClusterLayerText  # noqa: E402

MM = Path("/Users/stevenfazzio/repos/movie-madness-map")
N, SEED = 16000, 42
EMB_MODEL = "BAAI/bge-small-en-v1.5"
K_MOVIE, K_TV = 12, 6
NAMER = "haiku"
META = dict(obj="movie and TV titles", corpus="a video rental store catalog")
EMB_CACHE = HERE / "data" / "movie_emb.npz"


def load_and_embed(enc):
    f = pd.read_parquet(MM / "data/films.parquet",
                        columns=["media_type", "embed_text_synopsis", "title"])
    f = f[f["media_type"].isin(["movie", "tv"])].reset_index(drop=True)
    sub = f.sample(n=min(N, len(f)), random_state=SEED).reset_index(drop=True)
    texts = sub["embed_text_synopsis"].astype(str).tolist()
    media = sub["media_type"].to_numpy()
    if EMB_CACHE.exists():
        emb = np.load(EMB_CACHE)["emb"]
    else:
        emb = enc.encode(texts, batch_size=64, show_progress_bar=True,
                         convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        np.savez(EMB_CACHE, emb=emb)
    return texts, media, emb


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--namer", default=NAMER)
    args = ap.parse_args()
    print(f"namer = {args.namer}", flush=True)
    enc = SentenceTransformer(EMB_MODEL, device="cpu")
    texts, media, emb = load_and_embed(enc)
    is_movie = media == "movie"
    mv, tv = np.where(is_movie)[0], np.where(~is_movie)[0]
    print(f"{len(texts)} docs ({is_movie.sum()} movie / {(~is_movie).sum()} tv), emb {emb.shape}", flush=True)

    # fine layer: genres within movies, types within TV (real KMeans on the embeddings)
    lm = KMeans(n_clusters=K_MOVIE, random_state=SEED, n_init=10).fit(emb[mv]).labels_
    lt = KMeans(n_clusters=K_TV, random_state=SEED, n_init=10).fit(emb[tv]).labels_
    layer0 = np.empty(len(texts), dtype=int)
    layer0[mv] = lm
    layer0[tv] = lt + K_MOVIE
    layer1 = np.where(is_movie, 0, 1).astype(int)  # 0 = movies, 1 = TV

    L0 = ClusterLayerText(layer0, centroids_from_labels(layer0, emb), layer_id=0,
                          verbose=False, show_progress_bar=False)
    L1 = ClusterLayerText(layer1, centroids_from_labels(layer1, emb), layer_id=1,
                          verbose=False, show_progress_bar=False)
    tree = {(1, 0): [(0, j) for j in range(K_MOVIE)],
            (1, 1): [(0, j) for j in range(K_MOVIE, K_MOVIE + K_TV)]}
    fake = types.SimpleNamespace(cluster_layers_=[L0, L1], cluster_tree_=tree)

    namer = make_namer(args.namer)
    temp = 1.0 if args.namer == "opus" else 0.0  # Opus 4.8 rejects temperature=0
    common = dict(objects=texts, emb=emb, coords=emb, clusterer=fake,
                  namer=namer, embedder=enc, meta=META, k=5, disambig=False, temperature=temp)
    print("naming baseline ...", flush=True)
    base, _ = name_once(use_contrast=False, max_dist=None, **common)
    print("naming v1-contrast ...", flush=True)
    cc1, s1 = name_once(use_contrast=True, max_dist=None, block_version="v1", **common)
    print("naming v2-contrast ...", flush=True)
    cc2, s2 = name_once(use_contrast=True, max_dist=None, block_version="v2", **common)

    print("\n===== FINE LAYER (L0) names =====", flush=True)
    for j in range(K_MOVIE + K_TV):
        print(f"  ({'movie-genre' if j < K_MOVIE else 'TV-type  '}) L0[{j:>2}] = {base[0][j]!r}")

    print("\n===== COARSEST LAYER (L1): the test =====", flush=True)
    for c, label in [(0, "MOVIES"), (1, "TV")]:
        print(f"\n {label} supercluster (1,{c}):")
        print(f"   baseline : {base[1][c]!r}")
        print(f"   v1       : {cc1[1][c]!r}")
        print(f"   v2       : {cc2[1][c]!r}")
        print(f"   contrast shown to it: {s1['injected_names'].get(f'1,{c}', [])}")

    (HERE / "data" / f"movie_construct_{args.namer}.json").write_text(json.dumps(dict(
        fine=base[0], coarse=dict(baseline=base[1], v1=cc1[1], v2=cc2[1]),
        contrast=s1["injected_names"]), indent=2))
    print("\nsaved -> data/movie_construct.json", flush=True)


if __name__ == "__main__":
    main()
