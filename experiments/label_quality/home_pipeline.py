"""
Run ONE (dataset x embedder) cell fully AT HOME: embed docs, UMAP, cluster, name (haiku), and score
with that embedder's OWN geometry -- so the metric assesses the partition its own embedder drew, with
no home/away confound. Measures the two headline things: gate-b (does the metric track the judge) and
ablation-detection (does metric-Δ track judge-Δ when a naming feature is dropped). Fine-discrimination
and the secondary findings are skipped (robust negatives we just cite).

  uv run --with cohere python experiments/label_quality/home_pipeline.py --dataset 20ng --embedder cohere
  uv run python              experiments/label_quality/home_pipeline.py --dataset arxiv --embedder minilm

Outputs data/home_<dataset>_<embedder>.json with per-candidate judge + metric, and prints the result.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.linalg import norm
from scipy.stats import spearmanr

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
NIBLING = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(NIBLING))

from ablation import ablate  # noqa: E402  (monkeypatch context manager: drop a naming feature)
from async_judge import rate_many  # noqa: E402
from metrics import fit_whitening, unit  # noqa: E402

EMB = {"minilm": "all-MiniLM-L6-v2", "cohere": "embed-v4.0"}


class CohereEmbedder:
    """Toponymy-compatible .encode() backed by Cohere embed-v4 (input_type=clustering, 1024d)."""
    def __init__(self, dim=1024, batch=96):
        import cohere
        import os
        self.co = cohere.ClientV2(api_key=os.environ["CO_API_KEY"])
        self.dim, self.batch = dim, batch

    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True, **kw):
        texts = [str(t)[:8000] for t in texts]
        out = []
        for i in range(0, len(texts), self.batch):
            r = self.co.embed(texts=texts[i:i + self.batch], model="embed-v4.0",
                              input_type="clustering", embedding_types=["float"], output_dimension=self.dim)
            out.extend(r.embeddings.float_)
        return np.asarray(out, dtype=np.float32)


def make_embedder(kind):
    if kind == "cohere":
        return CohereEmbedder()
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMB["minilm"], device="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["20ng", "arxiv"], required=True)
    ap.add_argument("--embedder", choices=["minilm", "cohere"], required=True)
    ap.add_argument("--subsample", type=int, default=7000)
    ap.add_argument("--judge", default="anthropic/claude-sonnet-4-6")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--features", nargs="+", default=["exemplars", "keyphrases"])
    ap.add_argument("--n-near", type=int, default=10)
    ap.add_argument("--n-rand", type=int, default=5)
    ap.add_argument("--maxlen", type=int, default=280)
    args = ap.parse_args()
    tag = f"{args.dataset}_{args.embedder}"
    t0 = time.time()

    from ab_harness import load_dataset, make_namer

    from toponymy.cluster_layer import ClusterLayerText
    from toponymy.clustering import ToponymyClusterer
    from toponymy.toponymy import Toponymy

    objects, _, _, meta = load_dataset(args.dataset, args.subsample)
    embedder = make_embedder(args.embedder)

    # --- doc embeddings + UMAP coords (cached per cell) ---
    epath, cpath = HERE / "data" / f"home_{tag}_emb.npy", HERE / "data" / f"home_{tag}_coords.npy"
    if epath.exists() and cpath.exists():
        emb, coords = np.load(epath), np.load(cpath)
    else:
        print(f"[{tag}] embedding {len(objects)} docs ...", flush=True)
        emb = embedder.encode(objects, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
        import umap
        coords = umap.UMAP(n_components=2, metric="cosine", n_neighbors=15,
                           random_state=42).fit_transform(emb).astype(np.float32)
        np.save(epath, emb)
        np.save(cpath, coords)
    print(f"[{tag}] emb {emb.shape}", flush=True)

    # --- cluster (this embedder's geometry) ---
    clusterer = ToponymyClusterer(min_clusters=4, base_min_cluster_size=25, verbose=False)
    clusterer.fit_predict(coords, emb, ClusterLayerText)
    counts = [int(l.centroid_vectors.shape[0]) for l in clusterer.cluster_layers_]
    print(f"[{tag}] clusters {counts}", flush=True)

    # --- name (full) ---
    model = Toponymy(make_namer("haiku"), embedder, clusterer=clusterer,
                     object_description=meta["obj"], corpus_description=meta["corpus"], verbose=False)
    model.fit(objects, emb, coords)
    full = [list(x) for x in model.topic_names_]

    # --- grounding docs per cluster (nearest-centroid + random spread), in this embedder's space ---
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
            near = mem[np.argsort(-sims)[:args.n_near]]
            rest = np.array([m for m in mem if m not in set(near.tolist())])
            rng = np.random.default_rng(1000 + i)
            rand = rng.choice(rest, size=min(args.n_rand, rest.size), replace=False) if rest.size else np.array([], int)
            docs_for[(L, i)] = [" ".join(str(objects[j]).split())[:args.maxlen]
                                for j in list(near) + list(rand)]

    # --- ablated namings (drop one feature each) ---
    ablated = {}
    for feat in args.features:
        with ablate(feat):
            m2 = Toponymy(make_namer("haiku"), embedder, clusterer=clusterer,
                          object_description=meta["obj"], corpus_description=meta["corpus"], verbose=False)
            m2.fit(objects, emb, coords)
        ablated[feat] = [list(x) for x in m2.topic_names_]
    print(f"[{tag}] named full + {list(ablated)} ({time.time()-t0:.0f}s)", flush=True)

    # --- battery (gold + perturbation variants) for gate-b ---
    from perturbations import build_battery
    battery = build_battery(clusterer.cluster_layers_, clusterer.cluster_tree_, full)

    # --- assemble everything to judge: (key, label, (L,i)) ---
    items = []  # (kind, L, i, label)
    for it in battery:
        for typ, lab in {"gold": it["gold"], **it["variants"]}.items():
            items.append((typ, it["layer"], it["idx"], lab))
    valid = {(L, i) for L in range(len(counts)) for i in range(counts[L])
             if full[L][i] and full[L][i] != "Unlabelled"}
    for feat in args.features:
        for (L, i) in valid:
            al = ablated[feat][L][i]
            if al and al != "Unlabelled":
                items.append((f"abl:{feat}", L, i, al))

    # --- judge all, concurrently ---
    tasks = [(lab, docs_for[(L, i)]) for (_, L, i, lab) in items]
    print(f"[{tag}] judging {len(tasks)} candidates (k={args.k}, conc={args.concurrency}) ...", flush=True)
    ratings = rate_many(tasks, meta["obj"], args.judge, k=args.k, concurrency=args.concurrency)

    # --- metric (raw + whitened centroid) in this embedder's space ---
    mu, W = fit_whitening(emb64)
    def whiten(x):
        return (np.atleast_2d(x) - mu) @ W
    uniq = sorted({lab for (_, _, _, lab) in items})
    LE = dict(zip(uniq, embedder.encode(uniq, convert_to_numpy=True).astype(np.float64)))
    rc, rw = {}, {}
    for L in range(len(counts)):
        for i in range(counts[L]):
            c = clusterer.cluster_layers_[L].centroid_vectors[i].astype(np.float64)
            rc[(L, i)], rw[(L, i)] = unit(c), unit(whiten(c))[0]

    rows = []
    for (kind, L, i, lab), rat in zip(items, ratings):
        rows.append(dict(kind=kind, L=L, i=i, label=lab, overall=rat["overall"],
                         centroid=float(unit(LE[lab]) @ rc[(L, i)]),
                         whitened=float(unit(whiten(LE[lab]))[0] @ rw[(L, i)])))
    out = HERE / "data" / f"home_{tag}.json"
    out.write_text(json.dumps(dict(dataset=args.dataset, embedder=args.embedder, counts=counts, rows=rows), indent=2))

    # --- report ---
    by = {}
    for r in rows:
        by.setdefault((r["L"], r["i"]), {})[r["kind"]] = r
    print(f"\n================ {tag} ================  ({time.time()-t0:.0f}s)")
    judged = [r for r in rows if r["overall"] is not None]
    for metric in ("centroid", "whitened"):
        gb = [(r[metric], r["overall"]) for r in judged]
        rho = spearmanr([a for a, _ in gb], [b for _, b in gb])[0]
        print(f"GATE-B  {metric:<9} Spearman(metric, judge) = {rho:+.3f}   (n={len(gb)})")
    for feat in args.features:
        print(f"ABLATION  {feat}:")
        for metric in ("centroid", "whitened"):
            md, jd = [], []
            for (L, i), d in by.items():
                if "gold" in d and f"abl:{feat}" in d and d["gold"]["overall"] is not None and d[f"abl:{feat}"]["overall"] is not None:
                    if d["gold"]["label"] != d[f"abl:{feat}"]["label"]:
                        md.append(d["gold"][metric] - d[f"abl:{feat}"][metric])
                        jd.append(d["gold"]["overall"] - d[f"abl:{feat}"]["overall"])
            if len(md) > 3:
                ag = np.mean([(m > 0) == (j > 0) for m, j in zip(md, jd) if j != 0]) * 100
                print(f"    {metric:<9} judge-Δ {np.mean(jd):+.2f}  metric-Δ {np.mean(md):+.3f}  "
                      f"Spearman {spearmanr(md, jd)[0]:+.2f}  sign-agree {ag:.0f}%  (n={len(md)})")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
