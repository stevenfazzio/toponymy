"""
Can the vectors a caller already passes stand in for what Toponymy's embedding model produces?

Zero API calls. Reads cached embeddings and saved labels, writes only to the scratchpad given as
argv[1]. Compares, per embedding-model job, the selections made with:
  minilm    the corpus's own embedder (what the library does today)
  mpnet     a different real embedder (reference: how much does model choice alone move this?)
  docmean   keyphrase vector = mean of the documents containing it (no text model)
  centroid  layer-0 name vector = that cluster's document centroid (no text model)
"""
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/stevenfazzio/repos/toponymy")
SP = Path(sys.argv[1])
CACHE = SP / "embedder_jobs_cache"
CACHE.mkdir(exist_ok=True)
BRANCH = "experiment/wayfinding-lineup:experiments/label_quality/data/"
N_KEYPHRASES, N_SUBTOPICS = 16, 16  # ClusterLayerText defaults
STOP = set("the a an of and or in on for to with by from at as its their is are about".split())


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def saved(name):
    out = subprocess.run(["git", "show", BRANCH + name], cwd=REPO, capture_output=True, text=True)
    out.check_returncode()
    return json.loads(out.stdout)


def load_corpus(cell):
    if cell == "20ng":
        d = REPO / "experiments/nibling_contrast/data"
        emb = np.load(d / "ng_emb.npy").astype(np.float32)
        coords = np.load(d / "ng_coords.npy").astype(np.float32)
        objects = json.loads((d / "ng_texts.json").read_text())
    else:
        d = REPO / "experiments/label_quality/data"
        emb = np.load(d / "home_arxiv_minilm_emb.npy").astype(np.float32)
        coords = np.load(d / "home_arxiv_minilm_coords.npy").astype(np.float32)
        with zipfile.ZipFile(REPO / "examples/ai_arxiv_papers.zip") as z:
            with z.open("ai_arxiv_papers") as fh:
                papers = pd.read_csv(fh)
        objects = (papers["title"].astype(str) + ". " + papers["abstract"].astype(str)).tolist()
        objects = objects[: emb.shape[0]]
    assert len(objects) == emb.shape[0] == coords.shape[0], "rows disagree"
    return objects, emb, coords


def encode_cached(model_name, strings, tag):
    path = CACHE / f"{tag}_{model_name}.npy"
    if path.exists():
        v = np.load(path)
        if v.shape[0] == len(strings):
            return v
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    model = SentenceTransformer(model_name)
    v = model.encode(strings, batch_size=256, show_progress_bar=False, convert_to_numpy=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, v)
    assert np.load(tmp, mmap_mode="r").shape == v.shape
    os.replace(tmp, path)
    log(f"  encoded {len(strings)} strings with {model_name} in {time.time() - t0:.0f}s")
    return v


def overlap(a, b, n):
    return [len(set(x) & set(y)) / max(min(n, len(x), len(y)), 1) for x, y in zip(a, b)]


def content_tokens(name):
    return {t for t in re.findall(r"[a-z0-9]+", name.lower()) if t not in STOP}


def run(cell):
    from toponymy.cluster_layer import ClusterLayerText
    from toponymy.clustering import ToponymyClusterer
    from toponymy.keyphrases import KeyphraseBuilder, information_weighted_keyphrases
    from toponymy.subtopics import central_subtopics

    log(f"=== {cell} ===")
    objects, emb, coords = load_corpus(cell)
    cl = ToponymyClusterer(min_clusters=4, base_min_cluster_size=25, verbose=False)
    cl.fit_predict(coords, emb, ClusterLayerText)
    layers = cl.cluster_layers_
    counts = [int(l.centroid_vectors.shape[0]) for l in layers]

    rows = saved(f"length_controller_{cell}.json")["rows"]
    want = [sum(r["L"] == L for r in rows) for L in range(len(counts))]
    log(f"clusters per layer {counts}, saved labels expect {want}")
    assert counts == want, "replayed fit does not match the saved labels"
    stock = [[None] * n for n in counts]
    for r in rows:
        stock[r["L"]][r["i"]] = r["stock"]

    # integrity floor: stock names should share vocabulary with their own cluster's documents
    lower = [o.lower() for o in objects]
    rng = np.random.default_rng(0)
    own, other = [], []
    for i, name in enumerate(stock[0]):
        toks = content_tokens(name)
        members = np.where(layers[0].cluster_labels == i)[0][:40]
        j = int(rng.integers(counts[0] - 1))
        j += j >= i
        others = np.where(layers[0].cluster_labels == j)[0][:40]
        hit = lambda ids: np.mean([any(t in lower[k] for t in toks) for k in ids])
        own.append(hit(members))
        other.append(hit(others))
    log(f"alignment floor: name tokens in own docs {np.mean(own):.2f}, other cluster {np.mean(other):.2f}")

    # ---- job 1: keyphrases -------------------------------------------------------------------
    t0 = time.time()
    X, vocab, _ = KeyphraseBuilder(verbose=False).fit_transform(objects)
    X = X.tocsr()
    log(f"vocabulary {len(vocab)} keyphrases over {X.shape[0]} docs in {time.time() - t0:.0f}s")
    B = X.copy().astype(np.float32)
    B.data = np.ones_like(B.data)
    df = np.asarray(B.sum(axis=0)).ravel()
    log(f"keyphrases absent from every doc: {(df == 0).sum()}")
    docmean = np.asarray(B.T @ emb) / np.maximum(df, 1)[:, None]
    assert not np.isnan(docmean).any()
    vectors = {
        "minilm": encode_cached("all-MiniLM-L6-v2", vocab, f"{cell}_vocab"),
        "mpnet": encode_cached("all-mpnet-base-v2", vocab, f"{cell}_vocab"),
        "docmean": docmean,
        "docmean_centered": docmean - emb.mean(axis=0),
    }

    def picks(vecs, **kw):
        out = []
        for layer in layers:
            out.append(
                information_weighted_keyphrases(
                    layer.cluster_labels, X, vocab, vecs.copy(), None,
                    n_keyphrases=N_KEYPHRASES, **kw,
                )
            )
        return out

    base = picks(vectors["minilm"], max_alpha=1.0)
    keyphrase_table = {}
    for name, vecs in vectors.items():
        if name == "minilm":
            continue
        got = picks(vecs, max_alpha=1.0)
        keyphrase_table[name] = [float(np.mean(overlap(a, b, N_KEYPHRASES))) for a, b in zip(base, got)]
    try:
        none = picks(vectors["minilm"], max_alpha=1e-6, min_alpha=0.0, alpha_tolerance=1e-8)
        keyphrase_table["no_vectors_top_scores"] = [
            float(np.mean(overlap(a, b, N_KEYPHRASES))) for a, b in zip(base, none)
        ]
    except Exception as e:
        log(f"no-diversify baseline failed: {type(e).__name__}: {e}")
    repeat = picks(vectors["minilm"], max_alpha=1.0)
    keyphrase_table["repeat_same_inputs"] = [
        float(np.mean(overlap(a, b, N_KEYPHRASES))) for a, b in zip(base, repeat)
    ]
    log(f"keyphrase overlap with today's picks, per layer: {keyphrase_table}")

    # ---- job 2: subtopics --------------------------------------------------------------------
    names0 = stock[0]
    name_vecs = {
        "minilm": encode_cached("all-MiniLM-L6-v2", names0, f"{cell}_names0"),
        "mpnet": encode_cached("all-mpnet-base-v2", names0, f"{cell}_names0"),
        "centroid": layers[0].centroid_vectors.astype(np.float32),
    }

    def subtopic_picks(vecs, n):
        return [
            central_subtopics(
                layer.cluster_labels, names0, layers[0].cluster_labels,
                subtopic_vectors=vecs, n_subtopics=n, diversify_alpha=1.0,
            )
            for layer in layers[1:]
        ]

    subtopic_table = {"parents": [], "parents_over_16_children": []}
    for L, layer in enumerate(layers[1:], 1):
        kids = [
            len(set(layers[0].cluster_labels[layer.cluster_labels == c]) - {-1})
            for c in range(counts[L])
        ]
        subtopic_table["parents"].append(counts[L])
        subtopic_table["parents_over_16_children"].append(int(sum(k > N_SUBTOPICS for k in kids)))
    for n in (N_SUBTOPICS, 4):
        base_s = subtopic_picks(name_vecs["minilm"], n)
        for name in ("mpnet", "centroid"):
            got = subtopic_picks(name_vecs[name], n)
            subtopic_table[f"{name}_top{n}"] = [
                float(np.mean(overlap(a, b, n))) for a, b in zip(base_s, got)
            ]
    log(f"subtopic overlap with today's picks: {subtopic_table}")

    # ---- job 3: disambiguation trigger ------------------------------------------------------
    ladder = saved(f"ladder_names_{cell}.json")
    cent = [l.centroid_vectors / np.linalg.norm(l.centroid_vectors, axis=1, keepdims=True) for l in layers]
    near = {0.2: {}, 0.35: {}}
    for rung, by_layer in ladder.items():
        for L, names in enumerate(by_layer):
            if counts[L] < 3:
                continue
            v = encode_cached("all-MiniLM-L6-v2", names, f"{cell}_ladder_r{rung}_L{L}")
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            dist = 1 - v @ v.T
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    for cap, store in near.items():
                        if dist[i, j] <= cap:
                            store[(L, i, j)] = (names[i], names[j])
    trigger_table = {"pairs_tested_if_all": [n * (n - 1) // 2 for n in counts]}
    for k in (4, 8, 16):
        nbrs = {}
        for L, c in enumerate(cent):
            sim = c @ c.T
            np.fill_diagonal(sim, -9)
            nbrs[L] = np.argsort(-sim, axis=1)[:, : min(k, len(c) - 1)]
        trigger_table[f"pairs_tested_k{k}"] = [
            len({tuple(sorted((i, int(j)))) for i in range(len(nbrs[L])) for j in nbrs[L][i]})
            for L in range(len(cent))
        ]
        for cap, store in near.items():
            covered = sum(j in nbrs[L][i] or i in nbrs[L][j] for (L, i, j) in store)
            trigger_table[f"near{cap}_covered_k{k}"] = f"{covered}/{len(store)}"
    for cap, store in near.items():
        lexical = sum(bool(content_tokens(a) & content_tokens(b)) for a, b in store.values())
        trigger_table[f"near{cap}_share_a_word"] = f"{lexical}/{len(store)}"
    sample = list(near[0.35].values())[:6]
    log(f"trigger coverage: {trigger_table}")
    log(f"sample near pairs: {sample}")
    return dict(
        counts=counts, vocab=len(vocab), keyphrases=keyphrase_table,
        subtopics=subtopic_table, trigger=trigger_table,
    )


if __name__ == "__main__":
    results = {c: run(c) for c in ("20ng", "arxiv_home")}
    tmp = SP / "embedder_jobs_results.tmp"
    tmp.write_text(json.dumps(results, indent=2))
    os.replace(tmp, SP / "embedder_jobs_results.json")
    log("done")
