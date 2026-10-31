"""
Prepare the fresh-document test. Run with the library at the frozen commit on PYTHONPATH.

Fixes the test before any paid call:
  pick    the highest mean Sonnet lineup score on the frozen documents among all pooled candidates
          for a cluster (ties go to fewer words)
  stock   the label stock Toponymy gave that cluster
  fresh   five new documents per cluster, excluding the namer's exemplars and every document the
          frozen draw used, truncated exactly as the original instrument did

Clusters whose pick is the stock label contribute a difference of zero and need no calls. Also
checks every saved source against the ladder's cluster numbering, because ids can swap between
processes. Reads the repo, writes only to argv[1].
"""
import collections
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from dump_lineups import load_corpus, md5i, unit_rows

REPO = Path("/Users/stevenfazzio/repos/toponymy")
SP = Path(sys.argv[1])
B = "experiment/naming-features:experiments/label_quality/data/"
SWAP = {18: 19, 19: 18}


def saved(name):
    r = subprocess.run(["git", "show", B + name], cwd=REPO, capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 else None


def fracs(u):
    out = []
    for s in u.get("samples") or []:
        total = sum(s["scores"]) or 1
        out.append(s["scores"][s["order"].index(u["true"])] / total)
    return out


def usable(u):
    return u.get("mode") == "nn" and u.get("k") == 5 and u.get("run") == 1 and len(u.get("samples") or []) == 3


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

    sources = {
        "ladder": saved(f"wayfinding_{cell}_ladder.json"),
        "battery": saved(f"wayfinding_{cell}_battery.json"),
        "features": saved(f"wayfinding_{cell}_features.json"),
        "subtopics": saved(f"wayfinding_{cell}_subtopics.json"),
    }
    if cell == "20ng":
        sources["pairs"] = saved("wayfinding_20ng_pairs.json")
    sources = {k: list(v["units"].values()) for k, v in sources.items() if v}

    def lineup_under(flip):
        perm = [dict(SWAP) if flip and L == 1 else {} for L in range(len(counts))]

        def fn(L, i, k, mode="nn"):
            inv = {v: s for s, v in perm[L].items()}
            r = perm[L].get(i, i)
            kk = min(k, counts[L])
            order = np.argsort(-(cent[L] @ cent[L][r]))
            others = [inv.get(int(j), int(j)) for j in order if int(j) != r]
            picks = others[: kk - 1] if mode == "nn" else others[::-1][: kk - 1]
            return [i] + picks

        return fn, perm

    def mismatch(fn, name, swap_unit=False):
        n = 0
        for u in sources[name]:
            if u.get("mode") not in ("nn", "gimme"):
                continue
            L, i, line = u["L"], u["i"], list(u["lineup"])
            if swap_unit and L == 1:
                i, line = SWAP.get(i, i), [SWAP.get(x, x) for x in line]
            n += fn(L, i, u["k"], u["mode"]) != line
        return n

    base, perm = None, None
    flips = [False, True] if cell == "arxiv_home" else [False]
    for flip in flips:
        fn, p = lineup_under(flip)
        if mismatch(fn, "ladder") == 0:
            base, perm, chosen_flip = fn, p, flip
            break
    assert base is not None, "no numbering reproduces the saved ladder"
    print(f"  ladder numbering: {'flipped' if chosen_flip else 'as fitted'} relative to this process", flush=True)

    def near_miss(name):
        """After swapping, lineups may differ only in the last neighbour, a near-tie in centroid
        similarity between fits. The true cluster and the first three neighbours must still match."""
        bad = []
        for u in sources[name]:
            if u.get("mode") != "nn":
                continue
            L, i, line = u["L"], u["i"], list(u["lineup"])
            if L == 1:
                i, line = SWAP.get(i, i), [SWAP.get(x, x) for x in line]
            again = base(L, i, u["k"], "nn")
            if again != line:
                bad.append((again[:-1] == line[:-1], u["uid"]))
        return bad

    swap_source = {}
    for name in sources:
        same = mismatch(base, name)
        other = mismatch(base, name, swap_unit=True) if cell == "arxiv_home" else None
        flagged = bool(cell == "arxiv_home" and same > 0)
        swap_source[name] = flagged
        print(f"  {name}: {len(sources[name])} units; lineup mismatches as saved {same}"
              + (f", after swapping 18/19 {other}" if other is not None else "")
              + (" -> relabelled to ladder numbering" if flagged else ""), flush=True)
        if flagged:
            misses = near_miss(name)
            assert all(ok for ok, _ in misses), f"{name}: mismatch beyond the last neighbour: {misses}"
            print(f"    {len(misses)} of those differ only in the last neighbour: {[u for _, u in misses]}", flush=True)
        else:
            assert same == 0, f"{name} matches neither numbering"

    # pool candidates in ladder numbering, in the same order selection_value.py uses
    pool = collections.defaultdict(dict)
    def add(name, keep=lambda u: True):
        for u in sources[name]:
            if not usable(u) or not keep(u):
                continue
            i = u["i"]
            if swap_source[name] and u["L"] == 1:
                i = SWAP.get(i, i)
            pool[(u["L"], i)][u["label"]] = fracs(u)
    add("ladder")
    add("battery", lambda u: u["kind"] == "gold")
    add("features")
    add("subtopics")
    if "pairs" in sources:
        add("pairs")

    rows = saved(f"length_controller_{cell}.json")["rows"]
    stock = {(r["L"], r["i"]): r["stock"] for r in rows}
    picks = {}
    for key, cands in pool.items():
        assert stock[key] in cands, f"stock label missing from pool for {key}"
        scored = sorted(cands.items(), key=lambda kv: (-float(np.mean(kv[1])), len(kv[0].split()), kv[0]))
        picks[key] = scored[0][0]
    same = sum(picks[k] == stock[k] for k in picks)
    print(f"  clusters {len(picks)}; pick is the stock label in {same}, needing no calls", flush=True)

    # fresh documents: exclude exemplars and the whole frozen draw
    groups, short, overlap = {}, 0, 0
    frozen_all = {}
    for L in range(len(counts)):
        for i in range(counts[L]):
            r = perm[L].get(i, i)
            members = np.where(layers[L].cluster_labels == r)[0]
            ex = set(map(int, layers[L].exemplar_indices[r]))
            pl = np.array([m for m in members if int(m) not in ex])
            if pl.size == 0:
                pl = members
            take = np.random.default_rng(md5i(f"wayfind-docs|{L}|{i}")).choice(pl, size=min(5, pl.size), replace=False)
            frozen = set(map(int, take))
            frozen_all[(L, i)] = frozen
            rest = np.array([m for m in pl if int(m) not in frozen])
            fresh = np.random.default_rng(md5i(f"wayfind-docs-v3|{L}|{i}")).choice(rest, size=min(5, rest.size), replace=False)
            short += fresh.size < 5
            overlap += bool(set(map(int, fresh)) & frozen)
            groups[f"{L}|{i}"] = [" ".join(str(objects[j]).split())[:500] for j in fresh]
    assert overlap == 0
    print(f"  fresh draws: {len(groups)} clusters, {short} with fewer than 5 fresh documents, "
          f"smallest {min(len(g) for g in groups.values())}", flush=True)

    units = []
    pairs_out = []
    for key in sorted(picks):
        L, i = key
        words = lambda s: len(s.split())
        line = base(L, i, 5)
        meta = dict(L=L, i=i, stock=stock[key], pick=picks[key], stock_frozen=pool[key][stock[key]],
                    pick_frozen=pool[key][picks[key]], n_cands=len(pool[key]),
                    stock_words=words(stock[key]), pick_words=words(picks[key]))
        pairs_out.append(meta)
        if picks[key] == stock[key]:
            continue
        for kind, label in (("fresh-pick", picks[key]), ("fresh-stock", stock[key])):
            digest = __import__("hashlib").md5(label.encode()).hexdigest()[:10]
            units.append(dict(uid=f"fresh|L{L}.{i}|{kind}|{digest}", L=L, i=i, true=i, kind=kind,
                              label=label, lineup=line))
    print(f"  units needing calls: {len(units)} labels x 3 samples = {len(units) * 3} calls", flush=True)
    return dict(groups=groups, units=units, pairs=pairs_out, counts=counts)


if __name__ == "__main__":
    out = {c: run(c) for c in ("20ng", "arxiv_home")}
    (SP / "fresh_units.json").write_text(json.dumps(out))  # holds corpus text, gitignored
    stripped = {c: {k: x for k, x in v.items() if k != "groups"} for c, v in out.items()}
    (SP / "fresh_pairs.json").write_text(json.dumps(stripped))
    n = sum(len(v["units"]) for v in out.values())
    print(f"total calls {n * 3}; at 3,751 input tokens each = {n * 3 * 3751 / 1e6:.2f}M input tokens", flush=True)
