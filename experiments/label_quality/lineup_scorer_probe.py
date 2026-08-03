"""
Phase 7a -- softmax-cosine over the frozen lineups (no LLM calls).

The question: how much of the listener's behaviour is recoverable by scoring the label against
the SAME frozen candidate sets with a plain embedding cosine, normalized over the neighbourhood?

Centroid-cosine (#173) is pointwise, so it cannot express "true of the region but wrong given the
neighbours". A softmax over the k=5 lineup candidates can, because the normalizer is the
neighbourhood -- and the same normalization should cancel the padding bias for free (padding pulls
a label toward the anisotropic mean direction; centroids are document means in that same space, so
padding raises cosine to ALL candidates by a similar amount, and common-mode shifts cancel).

Two tests, different questions:
  coarse -- reproduce the battery's paired gold-beats-variant ordering (listener 20NG: verbose 79%,
            ancestor 88%, sibling 89%, generic 95%, distant 99%, shuffled 100%)
  fine   -- within-cluster correlation with measured pm across ladder rungs, read against the
            oracle ceiling implied by listener reliability (~0.75 / ~0.79), NOT against 1.0

Raw un-normalized cosine runs alongside as the control that isolates what the softmax buys.
tau is fitted on one corpus and evaluated on the other -- never in-sample.

  uv run python experiments/label_quality/lineup_scorer_probe.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
DATA = HERE / "data"
sys.path.insert(0, str(REPO / "experiments" / "nibling_contrast"))
sys.path.insert(0, str(HERE))

DATASETS = ["20ng", "arxiv_home"]
# every committed unit file per dataset (sonnet unless noted); floors are haiku, heldout gpt-4o-mini
SOURCES = {
    "20ng": ["battery", "ladder", "conjuncts", "features", "pairs", "subtopics"],
    "arxiv_home": ["battery", "ladder", "features", "subtopics"],
}
TAUS = np.concatenate([np.linspace(0.005, 0.2, 40), np.linspace(0.22, 1.0, 20)])


def unit_rows(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def load_units(ds):
    out = {}
    for stage in SOURCES[ds]:
        p = DATA / f"wayfinding_{ds}_{stage}.json"
        if not p.exists():
            print(f"  (missing {p.name})")
            continue
        for uid, u in json.loads(p.read_text())["units"].items():
            if u.get("pm") is None or not u.get("mass"):
                continue
            u["stage"] = stage
            out[uid] = u
    return out


def held_out_idx(cell, L, i, n_docs=5):
    """Mirror Cell.held_out's frozen sample, but return member indices (it returns strings)."""
    from wayfinding import md5i

    members = np.where(cell.layers[L].cluster_labels == i)[0]
    ex = set(map(int, cell.layers[L].exemplar_indices[i]))
    pool = np.array([m for m in members if int(m) not in ex])
    if pool.size == 0:
        pool = members
    rng = np.random.default_rng(md5i(f"wayfind-docs|{L}|{i}"))
    return rng.choice(pool, size=min(n_docs, pool.size), replace=False)


def build_scores(ds):
    """-> units, and for each unit the raw cosine vector over its lineup, under 2 representations."""
    from ab_harness import make_embedder
    from wayfinding import Cell

    print(f"[{ds}] replaying cell ...", flush=True)
    cell = Cell(ds)
    units = load_units(ds)
    print(f"[{ds}] {len(units)} committed lineup units over "
          f"{len({(u['L'], u['true']) for u in units.values()})} clusters", flush=True)

    labels = sorted({u["label"] for u in units.values()})
    emb = make_embedder(cell.meta["emb_model"])
    L = unit_rows(np.asarray(emb.encode(labels, batch_size=128, show_progress_bar=False),
                             dtype=np.float64))
    lab_ix = {s: i for i, s in enumerate(labels)}

    # candidate representation A: cluster centroid (what nn-distractor selection used)
    # candidate representation B: mean of the 5 held-out docs the listener actually saw
    docmean = {}
    for u in units.values():
        for j in u["lineup"]:
            key = (u["L"], j)
            if key not in docmean:
                idx = held_out_idx(cell, u["L"], j)
                docmean[key] = unit_rows(cell.emb[idx].mean(axis=0)[None, :])[0]

    for u in units.values():
        v = L[lab_ix[u["label"]]]
        u["_cos_cent"] = np.array([float(v @ cell.cent[u["L"]][j]) for j in u["lineup"]])
        u["_cos_docs"] = np.array([float(v @ docmean[(u["L"], j)]) for j in u["lineup"]])
    return units


def model_p(u, rep, tau):
    s = u[rep] / tau
    e = np.exp(s - s.max())
    return e / e.sum()


def fit_tau(units, rep):
    """tau minimizing mean KL(listener mass || model p) over the given units."""
    best, best_tau = np.inf, None
    for tau in TAUS:
        tot = 0.0
        for u in units:
            p = np.clip(model_p(u, rep, tau), 1e-9, 1)
            q = np.array([u["mass"].get(str(j), 0.0) for j in u["lineup"]])
            q = q / max(q.sum(), 1e-9)
            tot += float((q * np.log(np.clip(q, 1e-9, 1) / p)).sum())
        if tot < best:
            best, best_tau = tot, tau
    return best_tau, best / max(len(units), 1)


def coarse_test(units, rep, tau):
    """Paired gold-beats-variant, on the battery. Softmax vs raw pointwise cosine."""
    gold = {(u["L"], u["true"]): u for u in units.values()
            if u["stage"] == "battery" and u["kind"] == "gold"}
    rows = defaultdict(lambda: [0, 0, 0, 0])  # kind -> [sm_wins, sm_n, raw_wins, raw_n]
    for u in units.values():
        if u["stage"] != "battery" or u["kind"] in ("gold", "gimme"):
            continue
        g = gold.get((u["L"], u["true"]))
        if g is None or g["lineup"] != u["lineup"]:
            continue
        r = rows[u["kind"]]
        gi = g["lineup"].index(g["true"])
        ui = u["lineup"].index(u["true"])
        r[0] += model_p(g, rep, tau)[gi] > model_p(u, rep, tau)[ui]
        r[1] += 1
        r[2] += g[rep][gi] > u[rep][ui]   # raw pointwise cosine to the true centroid (#173)
        r[3] += 1
    return rows


def fine_test(units, rep, tau):
    """Within-cluster correlation across ladder rungs, softmax vs raw cosine."""
    from scipy.stats import pearsonr, spearmanr

    byc = defaultdict(list)
    for u in units.values():
        if u["stage"] == "ladder":
            byc[(u["L"], u["true"])].append(u)
    sm_r, sm_s, raw_r = [], [], []
    for us in byc.values():
        if len(us) < 4:
            continue
        y = [u["pm"] for u in us]
        a = [model_p(u, rep, tau)[u["lineup"].index(u["true"])] for u in us]
        b = [u[rep][u["lineup"].index(u["true"])] for u in us]
        if len(set(y)) < 2:
            continue
        if len(set(a)) > 1:
            sm_r.append(pearsonr(a, y).statistic)
            sm_s.append(spearmanr(a, y).statistic)
        if len(set(b)) > 1:
            raw_r.append(pearsonr(b, y).statistic)
    return (len(sm_r), float(np.mean(sm_r)), float(np.nanmean(sm_s)), float(np.mean(raw_r)))


def direction_test(units, rep, tau):
    """When the listener's mass went to a WRONG candidate, does the model point at the same one?
    (the anti-conjunct mechanism: a conjunct overlapping a neighbour's territory pulls mass there)"""
    hit = tot = 0
    for u in units.values():
        if u["stage"] not in ("conjuncts", "battery"):
            continue
        q = {int(k): v for k, v in u["mass"].items()}
        wrong = {j: v for j, v in q.items() if j != u["true"]}
        if not wrong or max(wrong.values()) <= q.get(u["true"], 0):
            continue  # listener was right; nothing to explain
        p = model_p(u, rep, tau)
        pm_ = {j: p[a] for a, j in enumerate(u["lineup"]) if j != u["true"]}
        hit += max(wrong, key=wrong.get) == max(pm_, key=pm_.get)
        tot += 1
    return hit, tot


def main():
    all_units = {ds: build_scores(ds) for ds in DATASETS}
    print()
    for rep in ["_cos_cent", "_cos_docs"]:
        print("=" * 78)
        print(f"CANDIDATE REPRESENTATION: {'cluster centroid' if 'cent' in rep else 'mean of the 5 held-out docs'}")
        print("=" * 78)
        taus = {}
        for ds in DATASETS:
            fit_on = [u for u in all_units[ds].values() if u["stage"] == "battery"]
            taus[ds], kl = fit_tau(fit_on, rep)
            print(f"  tau fitted on {ds} battery = {taus[ds]:.3f} (mean KL {kl:.3f})")
        for ds in DATASETS:
            other = [d for d in DATASETS if d != ds][0]
            tau = taus[other]                      # OUT-OF-CORPUS tau, never in-sample
            u = all_units[ds]
            print(f"\n--- {ds}   (tau {tau:.3f} fitted on {other})")
            rows = coarse_test(u, rep, tau)
            print("  COARSE (paired gold-beats-variant):")
            print(f"    {'kind':10s} {'softmax':>9s} {'raw cosine':>11s}   n")
            for kind in ["verbose", "ancestor", "sibling", "generic", "distant", "shuffled"]:
                if kind not in rows:
                    continue
                w, n, rw, rn = rows[kind]
                print(f"    {kind:10s} {100*w/n:8.0f}% {100*rw/rn:10.0f}%   {n}")
            n, r, s, rawr = fine_test(u, rep, tau)
            ceil = {"20ng": 0.754, "arxiv_home": 0.787}[ds]
            print(f"  FINE (within-cluster across ladder rungs, n={n} clusters):")
            print(f"    softmax    Pearson {r:+.3f}   Spearman {s:+.3f}")
            print(f"    raw cosine Pearson {rawr:+.3f}")
            print(f"    oracle ceiling given listener reliability: ~{ceil:.2f}")
            h, t = direction_test(u, rep, tau)
            print(f"  DIRECTION (listener's mass went to a wrong candidate; same one?): "
                  f"{h}/{t} = {100*h/max(t,1):.0f}%  (chance ~{100/4:.0f}%)")
    print()


if __name__ == "__main__":
    main()
