"""
Rebuild the frozen wayfinding lineups, tolerating the run-to-run id flip at arXiv L1.

Cluster ids 18 and 19 at arXiv layer 1 swap between processes, so a fresh fit can come back in
either numbering. This maps the replay onto the numbering the saved ladder and battery used, then
requires an exact match on every battery and ladder lineup. Held-out units are compared as saved
and any disagreement is reported rather than dropped. Reads the repo, writes only to argv[1].
"""
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

from dump_lineups import load_corpus, md5i, saved, unit_rows

SP = Path(sys.argv[1])
KEEP = ("uid", "L", "i", "true", "kind", "label", "mode", "run", "lineup", "k", "pm", "top1", "mass")


def run(cell):
    from toponymy.cluster_layer import ClusterLayerText
    from toponymy.clustering import ToponymyClusterer

    print(f"=== {cell} ===", flush=True)
    objects, emb, coords = load_corpus(cell)
    cl = ToponymyClusterer(min_clusters=4, base_min_cluster_size=25, verbose=False)
    cl.fit_predict(coords, emb, ClusterLayerText)
    layers = cl.cluster_layers_
    for layer in layers:
        layer.make_exemplar_texts(objects, emb)
    counts = [int(l.centroid_vectors.shape[0]) for l in layers]
    cent = [unit_rows(l.centroid_vectors.astype(np.float64)) for l in layers]

    stages = {
        s: list(saved(f"wayfinding_{cell}_{s}.json")["units"].values())
        for s in ("battery", "ladder", "heldout")
    }

    def lineup_under(perm):
        """perm maps saved id -> replay id, per layer."""
        def fn(L, i, k, mode):
            inv = {v: s for s, v in perm[L].items()}
            r = perm[L].get(i, i)
            kk = min(k, counts[L])
            order = np.argsort(-(cent[L] @ cent[L][r]))
            others = [inv.get(int(j), int(j)) for j in order if int(j) != r]
            picks = others[: kk - 1] if mode == "nn" else others[::-1][: kk - 1]
            return [i] + picks
        return fn

    def mismatches(fn, stage_names):
        return sum(
            fn(u["L"], u["i"], u["k"], u["mode"]) != list(u["lineup"])
            for s in stage_names for u in stages[s]
        )

    identity = [{} for _ in counts]
    flipped = [{} for _ in counts]
    if cell == "arxiv_home":
        flipped[1] = {18: 19, 19: 18}
    base = ("battery", "ladder")
    id_bad, flip_bad = mismatches(lineup_under(identity), base), mismatches(lineup_under(flipped), base)
    perm = identity if id_bad <= flip_bad else flipped
    print(f"  battery+ladder mismatches: as-fitted {id_bad}, flipped {flip_bad}; using "
          f"{'as-fitted' if perm is identity else 'flipped'} numbering", flush=True)
    assert min(id_bad, flip_bad) == 0, "neither numbering reproduces the saved lineups"
    lineup = lineup_under(perm)

    def held(L, i, namespace):
        r = perm[L].get(i, i)
        members = np.where(layers[L].cluster_labels == r)[0]
        ex = set(map(int, layers[L].exemplar_indices[r]))
        pool = np.array([m for m in members if int(m) not in ex])
        if pool.size == 0:
            pool = members
        rng = np.random.default_rng(md5i(f"{namespace}|{L}|{i}"))
        take = rng.choice(pool, size=min(5, pool.size), replace=False)
        return [" ".join(str(objects[j]).split())[:500] for j in take]

    units = {}
    moved = collections.Counter()
    for s, rows in stages.items():
        out = []
        for u in rows:
            row = {k: u.get(k) for k in KEEP}
            again = lineup(u["L"], u["i"], u["k"], u["mode"])
            if again != list(u["lineup"]):
                assert s == "heldout", "battery or ladder lineup failed to reproduce"
                moved[(u["L"], u["i"])] += 1
                row["saved_lineup"], row["lineup"] = list(u["lineup"]), again
            out.append(row)
        units[s] = out
        print(f"  {s}: {len(out)} units", flush=True)

    if moved:
        ho = units["heldout"]
        print(f"  heldout units saved under a different numbering: {sum(moved.values())} "
              f"across {len(moved)} clusters", flush=True)
        for r in ho:
            if r["L"] == 1 and r["i"] in (18, 19):
                print(f"    {r['kind']} i={r['i']} pm={r['pm']:.3f} true-as-saved={r['saved_lineup'][0]} "
                      f"neighbours-saved={r['saved_lineup'][1:]} | {r['label'][:58]}", flush=True)
        for kind in ("ho-chosen", "ho-stock"):
            full = [r["pm"] for r in ho if r["kind"] == kind]
            clean = [r["pm"] for r in ho if r["kind"] == kind and not (r["L"] == 1 and r["i"] in (18, 19))]
            print(f"    {kind}: mean pm {np.mean(full):.4f} (n={len(full)}), without the two "
                  f"swapped clusters {np.mean(clean):.4f} (n={len(clean)})", flush=True)

    ids = [(L, i) for L in range(len(counts)) for i in range(counts[L])]
    groups = {f"{L}|{i}": held(L, i, "wayfind-docs") for L, i in ids}
    groups_v2 = {f"{L}|{i}": held(L, i, "wayfind-docs-v2") for L, i in ids}
    controller = saved(f"length_controller_{cell}.json")
    out = dict(counts=counts, groups=groups, groups_v2=groups_v2, units=units,
               band=controller["band"], rows=controller["rows"])
    path = SP / f"lineups_{cell}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    os.replace(tmp, path)
    print(f"  wrote {path.name}: {len(groups)} groups", flush=True)


if __name__ == "__main__":
    for c in ("20ng", "arxiv_home"):
        run(c)
