"""
How many unused documents does each cluster have for extra scoring draws? No paid calls.

A document is unavailable if it is one of the namer's exemplars, in the frozen draw, or in the fresh
draw, which must stay unseen for the evaluation to mean anything. A lineup is only as deep as the
shallowest of its five groups. Run with the library at the frozen commit on PYTHONPATH.
"""
import json
import sys
from pathlib import Path

import numpy as np

from dump_lineups import load_corpus, md5i

SP = Path(sys.argv[1])


def run(cell, keep):
    from toponymy.cluster_layer import ClusterLayerText
    from toponymy.clustering import ToponymyClusterer

    objects, emb, coords = load_corpus(cell)
    d = json.loads((SP / f"lineups_{cell}.json").read_text())
    cl = ToponymyClusterer(min_clusters=4, base_min_cluster_size=25, verbose=False)
    cl.fit_predict(coords, emb, ClusterLayerText)
    layers = cl.cluster_layers_
    for layer in layers:
        layer.make_exemplar_texts(objects, emb)
    cent = [l.centroid_vectors / np.linalg.norm(l.centroid_vectors, axis=1, keepdims=True) for l in layers]
    bad = 0
    for u in d["units"]["ladder"]:
        order = np.argsort(-(cent[u["L"]] @ cent[u["L"]][u["i"]]))
        others = [int(j) for j in order if int(j) != u["i"]][: min(5, len(cent[u["L"]])) - 1]
        bad += [u["i"]] + others != list(u["lineup"])
    if bad:
        return None
    left, sizes = {}, {}
    for L, layer in enumerate(layers):
        for i in range(len(layer.centroid_vectors)):
            members = np.where(layer.cluster_labels == i)[0]
            ex = set(map(int, layer.exemplar_indices[i]))
            pool = np.array([m for m in members if int(m) not in ex])
            if pool.size == 0:
                pool = members
            frozen = np.random.default_rng(md5i(f"wayfind-docs|{L}|{i}")).choice(pool, size=min(5, pool.size), replace=False)
            rest = np.array([m for m in pool if m not in set(map(int, frozen))])
            fresh = np.random.default_rng(md5i(f"wayfind-docs-v3|{L}|{i}")).choice(rest, size=min(5, rest.size), replace=False)
            left[(L, i)] = rest.size - fresh.size
            sizes[(L, i)] = members.size
    line = {}
    for u in d["units"]["ladder"]:
        line.setdefault((u["L"], u["i"]), u["lineup"])
    rows = []
    for key in sorted(left):
        if (cell, *key) not in keep:
            continue
        depth = min(left[(key[0], j)] for j in line[key])
        rows.append((key[0], left[key], depth, sizes[key]))
    return rows


if __name__ == "__main__":
    keep = {tuple(k) for k in json.loads((SP / "haiku_subset.json").read_text())}
    out = {}
    for cell in ("20ng", "arxiv_home"):
        for attempt in range(8):
            rows = run(cell, keep)
            if rows is not None:
                break
            print(f"{cell}: fit variant did not match the ladder, refitting", flush=True)
        assert rows is not None
        a = np.array(rows)
        print(f"{cell}: {len(a)} clusters, cluster size min {a[:, 3].min()} median {int(np.median(a[:, 3]))}", flush=True)
        print(f"  unused documents per cluster: min {a[:, 1].min()}, median {int(np.median(a[:, 1]))}", flush=True)
        print(f"  deepest usable, shallowest group in lineup: min {a[:, 2].min()}, median {int(np.median(a[:, 2]))}", flush=True)
        for r in (1, 2, 3, 5, 7):
            n = int((a[:, 2] >= 5 * r).sum())
            print(f"  {r} extra draw(s), {5 + 5 * r} scoring documents per group: {n}/{len(a)} clusters fully supported", flush=True)
        out[cell] = [list(map(int, r)) for r in rows]
    (SP / "docs_available.json").write_text(json.dumps(out))
