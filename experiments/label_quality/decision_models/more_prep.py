"""
Draw three extra five-document sets per cluster for the more-documents test. No paid calls.

Run with the library at the frozen commit on PYTHONPATH. Documents come only from a cluster's unused
pool: not an exemplar, not in the frozen draw, and not in the fresh draw, which must stay unseen.
Each pool is shuffled once and walked in steps of five. A pool under 15 documents wraps around, so
small clusters reuse documents across draws but never within one. Refuses a fit that does not
reproduce the ladder's lineups. Writes only to argv[1].
"""
import json
import sys
from pathlib import Path

import numpy as np

from dump_lineups import load_corpus, md5i

SP = Path(sys.argv[1])
DRAWS = 3


def run(cell):
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
    for u in d["units"]["ladder"]:
        order = np.argsort(-(cent[u["L"]] @ cent[u["L"]][u["i"]]))
        others = [int(j) for j in order if int(j) != u["i"]][: min(5, len(cent[u["L"]])) - 1]
        if [u["i"]] + others != list(u["lineup"]):
            return None

    text = lambda j: " ".join(str(objects[j]).split())[:500]
    groups, unique, frozen_check, fresh_check = {}, {}, 0, 0
    for L, layer in enumerate(layers):
        for i in range(len(layer.centroid_vectors)):
            members = np.where(layer.cluster_labels == i)[0]
            ex = set(map(int, layer.exemplar_indices[i]))
            pool = np.array([m for m in members if int(m) not in ex])
            if pool.size == 0:
                pool = members
            frozen = np.random.default_rng(md5i(f"wayfind-docs|{L}|{i}")).choice(pool, size=min(5, pool.size), replace=False)
            frozen_set = set(map(int, frozen))
            rest = np.array([m for m in pool if int(m) not in frozen_set])
            fresh = np.random.default_rng(md5i(f"wayfind-docs-v3|{L}|{i}")).choice(rest, size=min(5, rest.size), replace=False)
            fresh_set = set(map(int, fresh))
            unused = np.array(sorted(int(m) for m in rest if int(m) not in fresh_set))
            assert unused.size >= 1, (L, i)
            np.random.default_rng(md5i(f"wayfind-docs-extra|{L}|{i}")).shuffle(unused)
            seq = [int(unused[n % unused.size]) for n in range(DRAWS * 5)]
            draws = [seq[n * 5 : (n + 1) * 5] for n in range(DRAWS)]
            for dr in draws:
                dr[:] = list(dict.fromkeys(dr))
            used = {j for dr in draws for j in dr}
            frozen_check += bool(used & frozen_set)
            fresh_check += bool(used & fresh_set)
            groups[f"{L}|{i}"] = [[text(j) for j in dr] for dr in draws]
            unique[f"{L}|{i}"] = len(used)
    assert frozen_check == 0 and fresh_check == 0, "extra draws overlap the frozen or fresh documents"
    return groups, unique


if __name__ == "__main__":
    keep = {tuple(k) for k in json.loads((SP / "haiku_subset.json").read_text())}
    out = {}
    for cell in ("20ng", "arxiv_home"):
        got = None
        for attempt in range(8):
            got = run(cell)
            if got is not None:
                print(f"{cell}: fit matched the ladder on attempt {attempt + 1}", flush=True)
                break
        assert got is not None, "no fit matched the ladder"
        groups, unique = got
        own = np.array([unique[f"{k[1]}|{k[2]}"] for k in keep if k[0] == cell])
        print(f"  subset clusters {len(own)}: distinct extra documents min {own.min()}, median {int(np.median(own))}, "
              f"full 15 in {int((own >= 15).sum())}", flush=True)
        out[cell] = dict(groups=groups, unique=unique)
    tmp = SP / "more_groups.tmp"
    tmp.write_text(json.dumps(out))
    tmp.replace(SP / "more_groups.json")  # holds corpus text, gitignored
    (SP / "more_unique.json").write_text(json.dumps({c: v["unique"] for c, v in out.items()}))
