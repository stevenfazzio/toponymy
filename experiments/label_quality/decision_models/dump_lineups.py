"""
Rebuild the frozen wayfinding lineups so another model can replay them.

Runs against the library exactly as it stood at the frozen commit (PYTHONPATH=frozen_lib), replays
each saved fit, reproduces held-out document samples with the original seeds, and dumps everything
needed to the scratchpad. Reads repo data, writes nothing outside argv[1]. Zero API calls.
"""
import collections
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/stevenfazzio/repos/toponymy")
SP = Path(sys.argv[1])
BRANCH = "experiment/wayfinding-lineup:experiments/label_quality/data/"


def saved(name):
    out = subprocess.run(["git", "show", BRANCH + name], cwd=REPO, capture_output=True, text=True)
    out.check_returncode()
    return json.loads(out.stdout)


def md5i(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def unit_rows(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def load_corpus(cell):
    if cell == "20ng":
        d = REPO / "experiments/nibling_contrast/data"
        emb = np.load(d / "ng_emb.npy").astype(np.float32)
        coords = np.load(d / "ng_coords.npy").astype(np.float32)
        objects = json.loads((d / "ng_texts.json").read_text())
    else:
        d = REPO / "experiments/label_quality/data"
        emb = np.load(d / "home_arxiv_minilm_emb.npy")
        coords = np.load(d / "home_arxiv_minilm_coords.npy")
        with zipfile.ZipFile(REPO / "examples/ai_arxiv_papers.zip") as z:
            with z.open("ai_arxiv_papers") as fh:
                papers = pd.read_csv(fh)
        objects = (papers["title"].astype(str) + ". " + papers["abstract"].astype(str)).tolist()
        objects = objects[: emb.shape[0]]
    assert len(objects) == emb.shape[0] == coords.shape[0], "rows disagree"
    return objects, emb, coords


def run(cell):
    import toponymy
    from toponymy.cluster_layer import ClusterLayerText
    from toponymy.clustering import ToponymyClusterer

    print(f"=== {cell} (library at {Path(toponymy.__file__).parent}) ===", flush=True)
    objects, emb, coords = load_corpus(cell)
    cl = ToponymyClusterer(min_clusters=4, base_min_cluster_size=25, verbose=False)
    cl.fit_predict(coords, emb, ClusterLayerText)
    layers = cl.cluster_layers_
    for layer in layers:
        layer.make_exemplar_texts(objects, emb)
    counts = [int(l.centroid_vectors.shape[0]) for l in layers]
    cent = [unit_rows(l.centroid_vectors.astype(np.float64)) for l in layers]

    def held(L, i, namespace):
        members = np.where(layers[L].cluster_labels == i)[0]
        ex = set(map(int, layers[L].exemplar_indices[i]))
        pool = np.array([m for m in members if int(m) not in ex])
        if pool.size == 0:
            pool = members
        rng = np.random.default_rng(md5i(f"{namespace}|{L}|{i}"))
        take = rng.choice(pool, size=min(5, pool.size), replace=False)
        return [" ".join(str(objects[j]).split())[:500] for j in take]

    def lineup(L, i, k, mode):
        kk = min(k, counts[L])
        order = np.argsort(-(cent[L] @ cent[L][i]))
        others = [int(j) for j in order if j != i]
        picks = others[: kk - 1] if mode == "nn" else others[::-1][: kk - 1]
        return [i] + picks

    keep = ("uid", "L", "i", "true", "kind", "label", "mode", "run", "lineup", "k", "pm", "top1", "mass")
    units = {}
    bad = total = 0
    moved = collections.Counter()
    for stage, name in [
        ("battery", f"wayfinding_{cell}_battery.json"),
        ("ladder", f"wayfinding_{cell}_ladder.json"),
        ("heldout", f"wayfinding_{cell}_heldout.json"),
    ]:
        rows = []
        for u in saved(name)["units"].values():
            total += 1
            again = lineup(u["L"], u["i"], u["k"], u["mode"])
            row = {k: u.get(k) for k in keep}
            if again != list(u["lineup"]):
                if stage != "heldout":
                    bad += 1
                else:
                    moved[(u["L"], u["i"])] += 1
                    row["saved_lineup"] = list(u["lineup"])
                    row["lineup"] = again
            rows.append(row)
        units[stage] = rows
        print(f"  {stage}: {len(rows)} units", flush=True)
    print(f"  battery+ladder lineups differing from saved: {bad}; heldout units differing: {sum(moved.values())}", flush=True)
    assert bad == 0, "replayed fit does not reproduce the saved battery/ladder lineups"
    if moved:
        swap = lambda x: {18: 19, 19: 18}.get(x, x)
        ok = sum(
            [swap(x) for x in r["saved_lineup"][1:]] == r["lineup"][1:]
            for r in units["heldout"] if "saved_lineup" in r and r["i"] not in (18, 19)
        )
        n = sum(1 for r in units["heldout"] if "saved_lineup" in r and r["i"] not in (18, 19))
        print(f"  heldout: swapping 18<->19 explains {ok}/{n} non-18/19 units; clusters touched {sorted(moved)}", flush=True)
        for r in units["heldout"]:
            if r["L"] == 1 and r["i"] in (18, 19):
                print("   ", r["kind"], r["i"], "saved", r.get("saved_lineup", r["lineup"]), "replay", r["lineup"], "|", r["label"][:60], "| pm", round(r["pm"], 3), flush=True)

    groups = {f"{L}|{i}": held(L, i, "wayfind-docs") for L in range(len(counts)) for i in range(counts[L])}
    groups_v2 = {f"{L}|{i}": held(L, i, "wayfind-docs-v2") for L in range(len(counts)) for i in range(counts[L])}
    controller = saved(f"length_controller_{cell}.json")
    out = dict(counts=counts, groups=groups, groups_v2=groups_v2, units=units,
               band=controller["band"], rows=controller["rows"])
    path = SP / f"lineups_{cell}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    os.replace(tmp, path)
    words = [len(" ".join(d).split()) for d in groups.values()]
    print(f"  wrote {path.name}: {len(groups)} groups, mean {np.mean(words):.0f} words per group", flush=True)


if __name__ == "__main__":
    for c in ("20ng", "arxiv_home"):
        run(c)
