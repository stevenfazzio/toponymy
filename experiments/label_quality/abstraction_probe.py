"""
Phase 7b -- the abstraction probe on 20NG gold categories (no LLM calls).

Closes the long-open Phase 0 leg 2, and doubles as the kill criterion for the standalone
dual-encoder idea, whose premise is that "is a good label for" is asymmetric in a way cosine
cannot express: for a French recipe, "Italian cuisine" outscores "European cuisine" but is a far
worse label. 20NG ships a naturally-occurring two-level naming hierarchy (rec.sport.hockey
< rec.sport < rec) with per-document assignments, so the test needs no new data.

For a document aggregate drawn from one leaf category, score five candidate labels:
    exact leaf | parent | grandparent | sibling leaf | distant leaf
Headline metric is NOT top-1 but the ASYMMETRY FAILURE RATE: how often a wrong-but-specific
sibling outranks a correct-but-general ancestor. Contenders, all cheap:
    raw cosine | softmax over the candidate set (the 7a trick) | cosine - lambda*generality
    (Phase 2a's saved axis, i.e. the Renner Sim+SpecLoss shape) | label length (Phase 2b's winner)

Prior, and it is negative: Phase 2b measured that generality axis at 50% = chance direction
accuracy on real Toponymy parent->child label pairs, with length at 86%.

  uv run python experiments/label_quality/abstraction_probe.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
NIBLING = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(NIBLING))
sys.path.insert(0, str(HERE))

N_DOCS = 15          # documents per aggregate
N_DRAWS = 20         # independent aggregates per leaf category
SEED = 20260803

# Natural-language rendering of the newsgroup paths. Written once from the group names alone,
# before seeing any result, and NOT tuned. Level-1/level-2 nodes get a name only where the
# grouping is real (comp.sys, rec.sport, talk.politics, soc.religion, comp.os, comp.windows).
LEAF = {
    "alt.atheism": "Atheism",
    "comp.graphics": "Computer Graphics",
    "comp.os.ms-windows.misc": "Microsoft Windows Operating System",
    "comp.sys.ibm.pc.hardware": "IBM PC Hardware",
    "comp.sys.mac.hardware": "Macintosh Hardware",
    "comp.windows.x": "X Window System",
    "misc.forsale": "Items For Sale",
    "rec.autos": "Automobiles",
    "rec.motorcycles": "Motorcycles",
    "rec.sport.baseball": "Baseball",
    "rec.sport.hockey": "Hockey",
    "sci.crypt": "Cryptography",
    "sci.electronics": "Electronics",
    "sci.med": "Medicine",
    "sci.space": "Space and Astronomy",
    "soc.religion.christian": "Christianity",
    "talk.politics.guns": "Gun Politics",
    "talk.politics.mideast": "Middle East Politics",
    "talk.politics.misc": "Political Discussion",
    "talk.religion.misc": "Religious Discussion",
}
NODE = {
    "alt": "Alternative Discussion", "comp": "Computers", "misc": "Miscellaneous",
    "rec": "Recreation", "sci": "Science", "soc": "Society", "talk": "Debate",
    "comp.os": "Operating Systems", "comp.sys": "Computer Systems",
    "comp.windows": "Windowing Systems", "rec.sport": "Sports",
    "soc.religion": "Religion", "talk.politics": "Politics", "talk.religion": "Religion",
}


def unit_rows(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def ancestors(path):
    """Named ancestors of a leaf, nearest first."""
    parts = path.split(".")
    out = []
    for n in range(len(parts) - 1, 0, -1):
        key = ".".join(parts[:n])
        if key in NODE:
            out.append(key)
    return out


def main():
    from ab_harness import load_dataset, make_embedder

    names = json.loads((NIBLING / "data" / "ng_target_names.json").read_text())
    targets = np.load(NIBLING / "data" / "ng_targets.npy")
    objects, emb, coords, meta = load_dataset("20ng", 7000)
    emb = unit_rows(np.asarray(emb, dtype=np.float64))
    targets = np.asarray(targets)[: emb.shape[0]]
    print(f"20NG: {emb.shape[0]} docs, {len(set(targets.tolist()))} gold categories present")

    # candidate label strings
    label_of = dict(LEAF)
    label_of.update(NODE)
    strings = sorted(set(label_of.values()))
    embedder = make_embedder(meta["emb_model"])
    E = unit_rows(np.asarray(embedder.encode(strings, show_progress_bar=False), dtype=np.float64))
    ix = {s: i for i, s in enumerate(strings)}

    def vec(path):
        return E[ix[label_of[path]]]

    # Phase 2a's saved generality axis (higher projection = more general, per that fit)
    axis = np.load(HERE / "data" / "generality_axis.npy").astype(np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    gen = {p: float(vec(p) @ axis) for p in label_of}
    wlen = {p: len(label_of[p].split()) for p in label_of}

    rng = np.random.default_rng(SEED)
    trials = []
    for t, path in enumerate(names):
        anc = ancestors(path)
        if not anc:
            continue                                  # no named ancestor -> no abstraction test
        sibs = [q for q in names if q != path and ancestors(q) and ancestors(q)[0] == anc[0]]
        if not sibs:
            continue
        far = [q for q in names if q.split(".")[0] != path.split(".")[0]]
        pool = np.where(targets == t)[0]
        if pool.size < N_DOCS:
            continue
        for d in range(N_DRAWS):
            docs = rng.choice(pool, size=N_DOCS, replace=False)
            agg = unit_rows(emb[docs].mean(axis=0)[None, :])[0]
            sib = sibs[int(rng.integers(len(sibs)))]
            dist = far[int(rng.integers(len(far)))]
            cands = {"leaf": path, "parent": anc[0],
                     "grand": anc[1] if len(anc) > 1 else anc[0],
                     "sibling": sib, "distant": dist}
            trials.append((cands, agg, [emb[j] for j in docs]))
    print(f"{len(trials)} trials over {len({c['leaf'] for c, _, _ in trials})} leaf categories "
          f"({N_DRAWS} aggregates x {N_DOCS} docs each)\n")

    def evaluate(scorer, name, note=""):
        top1 = asym = anc_ok = n = 0
        for cands, agg, docs in trials:
            s = {r: scorer(cands[r], agg, docs) for r in cands}
            n += 1
            top1 += max(s, key=s.get) == "leaf"
            best_anc = max(s["parent"], s["grand"])
            asym += s["sibling"] > best_anc            # the asymmetry failure
            anc_ok += best_anc > s["distant"]
        print(f"  {name:34s} leaf-top1 {100*top1/n:5.1f}%   "
              f"ASYMMETRY FAILURE {100*asym/n:5.1f}%   ancestor>distant {100*anc_ok/n:5.1f}%{note}")
        return asym / n

    print("contender                            exact leaf ranked 1st   sibling ABOVE ancestor")
    print("-" * 96)
    evaluate(lambda p, a, d: float(vec(p) @ a), "raw cosine to aggregate")
    evaluate(lambda p, a, d: float(np.mean([vec(p) @ x for x in d])),
             "cosine to individual docs (mean)")
    # softmax over the candidate set is a monotone transform per-trial, so it cannot change any
    # within-trial ranking -- reported once, explicitly, rather than silently omitted
    print("  (softmax over the candidate set is order-preserving within a trial: identical ranks)")
    evaluate(lambda p, a, d: float(vec(p) @ a) - 0.02 * wlen[p], "cosine - 0.02*label length")

    # cosine - lambda*generality, lambda fitted on half the leaf categories, tested on the other
    leaves = sorted({c["leaf"] for c, _, _ in trials})
    fit_set = set(leaves[::2])
    best_lam, best_score = 0.0, 1.0
    for lam in np.linspace(0, 0.5, 51):
        bad = tot = 0
        for cands, agg, docs in trials:
            if cands["leaf"] not in fit_set:
                continue
            s = {r: float(vec(cands[r]) @ agg) - lam * gen[cands[r]] for r in cands}
            bad += s["sibling"] > max(s["parent"], s["grand"])
            tot += 1
        if bad / tot < best_score:
            best_score, best_lam = bad / tot, lam
    held = [(c, a, d) for c, a, d in trials if c["leaf"] not in fit_set]
    bad = sum(1 for c, a, d in held
              if float(vec(c["sibling"]) @ a) - best_lam * gen[c["sibling"]]
              > max(float(vec(c[r]) @ a) - best_lam * gen[c[r]] for r in ("parent", "grand")))
    t1 = sum(1 for c, a, d in held
             if max(c, key=lambda r: float(vec(c[r]) @ a) - best_lam * gen[c[r]]) == "leaf")
    print(f"  {'cosine - lam*generality (Phase 2a)':34s} leaf-top1 {100*t1/len(held):5.1f}%   "
          f"ASYMMETRY FAILURE {100*bad/len(held):5.1f}%   [lam={best_lam:.2f} fitted on held-out "
          f"category split, n={len(held)}]")

    print("\nper-category asymmetry failure (raw cosine to aggregate):")
    per = defaultdict(lambda: [0, 0])
    for cands, agg, docs in trials:
        s = {r: float(vec(cands[r]) @ agg) for r in cands}
        p = per[cands["leaf"]]
        p[0] += s["sibling"] > max(s["parent"], s["grand"])
        p[1] += 1
    for leaf, (b, t) in sorted(per.items(), key=lambda kv: -kv[1][0] / kv[1][1]):
        print(f"    {100*b/t:5.0f}%  {leaf:28s} leaf={LEAF[leaf]!r:36s} "
              f"anc={NODE[ancestors(leaf)[0]]!r}")


if __name__ == "__main__":
    main()
