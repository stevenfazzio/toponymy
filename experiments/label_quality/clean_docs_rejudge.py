"""
Phase 6d -- is the exemplar fit effect real, or is the judge reading the answer key?

The grounded judge scores a label against 15 cluster documents (10 nearest-centroid + 5 random,
judge_fair.sample_docs). It does NOT exclude the documents the namer saw. On 20NG layer 0 the
overlap between the namer's exemplars and the judge's documents grows with n_exemplars:

    k    1     2     4     8    16    32
  ov  7%   11%   16%   24%   38%   67%   (of the judge's 15 documents)

So every fit comparison that VARIES the exemplar count is confounded: the higher-k arm has read more
of what it is about to be graded on. That covers the exemplar dose-response (phase 6b/6c) and, more
importantly, the posted #173 headline that exemplars are worth ~+0.4-0.5 judge-points while
keyphrases are worth nothing -- the k=0 arm has ZERO overlap by construction while stock has 24%.

The confound is DIFFERENTIAL, so it only bites when exemplars vary. The keyphrase ablation holds
exemplars fixed at 8 on both sides and is unaffected; it is deliberately not re-judged here.

Fix: re-judge on documents drawn from cluster members EXCLUDING the union of every rung's exemplars
-- the same discipline the wayfinding lineup already applies to its held-out pool. Same 10-near +
5-random shape, same seed, same truncation, so the only thing that changes is that no arm can parrot.
Gold must be re-judged too: its stored ratings are on the leaky sample.

  uv run python experiments/label_quality/clean_docs_rejudge.py --stage check   # no LLM
  uv run python experiments/label_quality/clean_docs_rejudge.py --stage judge
  uv run python experiments/label_quality/clean_docs_rejudge.py --stage report
Writes data/clean_docs_20ng.json (resumable).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numpy.linalg import norm
from scipy.stats import wilcoxon

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
NIBLING = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(NIBLING))

OUT = HERE / "data" / "clean_docs_20ng.json"
EXCLUDE_KS = [1, 2, 4, 8]      # every rung whose labels get compared here
N_NEAR, N_RAND, MAXLEN = 10, 5, 280
K_SAMPLES = 3


def conditions() -> dict:
    """condition -> {(L,i): label}. gold = stock k=8; k0 = exemplars ablation; k2/k4 = dose rungs."""
    battery = json.loads((HERE / "data" / "battery_20ng.json").read_text())
    gold = {(it["layer"], it["idx"]): it["gold"] for it in battery}
    abl = json.loads((HERE / "data" / "ablation.json").read_text())["exemplars"]
    k0 = {(r["L"], r["i"]): r["abl"] for r in abl}
    dose = json.loads((HERE / "data" / "dose_names_20ng.json").read_text())
    k2 = {(L, i): n for L, names in enumerate(dose["2"]) for i, n in enumerate(names)}
    k4 = {(L, i): n for L, names in enumerate(dose["4"]) for i, n in enumerate(names)}
    return {"gold": gold, "k0": k0, "k2": k2, "k4": k4}


def build(exclude: bool):
    """Replay the canonical fit and return (docs_by_cluster, n_fallback, mean_pool_loss)."""
    from perturbations import load_fit

    cl, objects, emb, coords, meta = load_fit("20ng", None, 25, 4)
    counts = [int(l.centroid_vectors.shape[0]) for l in cl.cluster_layers_]

    # union of exemplars across every rung under comparison (nested, but take the union anyway)
    banned = {}
    for k in EXCLUDE_KS:
        for L, layer in enumerate(cl.cluster_layers_):
            layer.n_exemplars = k
            layer.make_exemplar_texts(objects, emb)
            for i in range(counts[L]):
                banned.setdefault((L, i), set()).update(map(int, layer.exemplar_indices[i]))

    emb64 = emb.astype(np.float64)
    docs, fallback, losses = {}, 0, []
    for L in range(len(counts)):
        cls = cl.cluster_layers_[L].cluster_labels
        C = cl.cluster_layers_[L].centroid_vectors
        for i in range(counts[L]):
            mem = np.where(cls == i)[0]
            pool = np.array([m for m in mem if int(m) not in banned[(L, i)]]) if exclude else mem
            losses.append(1 - (pool.size / max(mem.size, 1)))
            if pool.size < N_NEAR + N_RAND:
                fallback += 1
                if pool.size == 0:
                    pool = mem
            E = emb64[pool] / (norm(emb64[pool], axis=1, keepdims=True) + 1e-12)
            c = C[i].astype(np.float64) / (norm(C[i]) + 1e-12)
            near = pool[np.argsort(-(E @ c))[:N_NEAR]]
            rest = np.array([m for m in pool if m not in set(near.tolist())])
            rng = np.random.default_rng(1000 + i)          # same seed as judge_fair.sample_docs
            rand = (rng.choice(rest, size=min(N_RAND, rest.size), replace=False)
                    if rest.size else np.array([], int))
            docs[(L, i)] = [" ".join(str(objects[j]).split())[:MAXLEN]
                            for j in list(near) + list(rand)]
    return docs, fallback, float(np.mean(losses)), meta


def stage_check():
    docs, fallback, loss, _ = build(exclude=True)
    sizes = [len(v) for v in docs.values()]
    print(f"clean document sample built for {len(docs)} clusters")
    print(f"  mean pool shrinkage after excluding every rung's exemplars: {loss:.1%}")
    print(f"  clusters with fewer than {N_NEAR+N_RAND} eligible docs: {fallback}")
    print(f"  docs per cluster: min {min(sizes)} median {int(np.median(sizes))} max {max(sizes)}")
    conds = conditions()
    print(f"\nconditions to judge: {', '.join(f'{k} (n={len(v)})' for k, v in conds.items())}")
    print(f"total judge calls: {sum(len(v) for v in conds.values()) * K_SAMPLES}")


def stage_judge(concurrency: int):
    from async_judge import rate_many

    docs, fallback, loss, meta = build(exclude=True)
    print(f"clean docs: pool shrinkage {loss:.1%}, {fallback} clusters below the doc budget")
    conds = conditions()
    done = json.loads(OUT.read_text()) if OUT.exists() else {}

    for cond, labels in conds.items():
        done.setdefault(cond, {})
        todo = [(L, i, lab) for (L, i), lab in sorted(labels.items())
                if lab and lab != "Unlabelled" and docs.get((L, i))
                and done[cond].get(f"{L}.{i}", {}).get("label", "").strip() != lab.strip()]
        print(f"{cond}: {len(todo)} to judge ({len(done[cond])} cached)", flush=True)
        if not todo:
            continue
        tasks = [(lab, docs[(L, i)]) for L, i, lab in todo]
        res = rate_many(tasks, meta["obj"], "anthropic/claude-sonnet-4-6",
                        k=K_SAMPLES, concurrency=concurrency)
        for (L, i, lab), r in zip(todo, res):
            done[cond][f"{L}.{i}"] = {"label": lab, "overall": r["overall"]}
        OUT.write_text(json.dumps(done, indent=2))
        print(f"  done ({sum(1 for v in done[cond].values() if v['overall'] is not None)} rated)")


def report():
    clean = json.loads(OUT.read_text())
    # the leaky-sample ratings these are being compared against
    leaky_gold = {(x["layer"], x["idx"]): x["overall"] for x in
                  json.loads((HERE / "data" / "judge_ratings_20ng_sonnet.json").read_text())
                  if x["type"] == "gold" and x.get("overall") is not None}
    leaky_k0 = {(r["L"], r["i"]): r["j_abl"] for r in
                json.loads((HERE / "data" / "ablation.json").read_text())["exemplars"]}
    dj = json.loads((HERE / "data" / "dose_judge_20ng.json").read_text())
    leaky = {"gold": leaky_gold, "k0": leaky_k0,
             "k2": {tuple(map(int, s.split("."))): v["overall"] for s, v in dj["2"].items()
                    if v["overall"] is not None},
             "k4": {tuple(map(int, s.split("."))): v["overall"] for s, v in dj["4"].items()
                    if v["overall"] is not None}}

    keys = None
    for cond in ["gold", "k0", "k2", "k4"]:
        ks = {tuple(map(int, s.split("."))) for s, v in clean[cond].items()
              if v["overall"] is not None} & set(leaky[cond])
        keys = ks if keys is None else (keys & ks)
    keys = sorted(keys)

    def arr(src, cond):
        return np.array([src[cond][k] if cond in src and isinstance(src[cond].get(k), float)
                         else clean[cond][f"{k[0]}.{k[1]}"]["overall"] for k in keys])

    cl = {c: np.array([clean[c][f"{L}.{i}"]["overall"] for L, i in keys])
          for c in ["gold", "k0", "k2", "k4"]}
    lk = {c: np.array([leaky[c][k] for k in keys]) for c in ["gold", "k0", "k2", "k4"]}

    print(f"\n===== LEAKY vs CLEAN JUDGE DOCUMENTS (20NG, n={len(keys)}) =====")
    print("  the exemplar effect, measured both ways. clean = judge docs exclude every")
    print("  rung's exemplars, so no arm can parrot what it was shown.\n")
    print(f"  {'contrast':<24} {'leaky':>22} {'clean':>22}")

    def fmt(d):
        se = d.std(ddof=1) / np.sqrt(len(d))
        p = wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
        return f"{d.mean():+.3f} (SE {se:.3f}, p={p:.2g})"

    for cond, name in [("k0", "k=0 vs stock"), ("k2", "k=2 vs stock"), ("k4", "k=4 vs stock")]:
        print(f"  {name:<24} {fmt(lk[cond] - lk['gold']):>22} {fmt(cl[cond] - cl['gold']):>22}")

    print(f"\n  absolute judge means:")
    for c in ["gold", "k2", "k4", "k0"]:
        print(f"    {c:>5}   leaky {lk[c].mean():.3f}   clean {cl[c].mean():.3f}   "
              f"(shift {cl[c].mean()-lk[c].mean():+.3f})")

    d_leak = lk["k0"] - lk["gold"]
    d_clean = cl["k0"] - cl["gold"]
    shrink = 1 - (abs(d_clean.mean()) / abs(d_leak.mean())) if d_leak.mean() else float("nan")
    diff = d_clean - d_leak
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    print(f"\n  the #173 headline (k=0 vs stock): {d_leak.mean():+.3f} leaky -> "
          f"{d_clean.mean():+.3f} clean")
    print(f"  change in the effect: {diff.mean():+.3f} (SE {se:.3f}, "
          f"p={wilcoxon(diff).pvalue:.3g}); {shrink:.0%} of the measured effect was leakage")
    if abs(d_clean.mean()) > 2 * (d_clean.std(ddof=1) / np.sqrt(len(d_clean))):
        print("  -> exemplars still carry real fit value once leakage is controlled")
    else:
        print("  -> the exemplar fit effect does NOT survive leakage control")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["check", "judge", "report"])
    ap.add_argument("--concurrency", type=int, default=24)
    args = ap.parse_args()
    {"check": stage_check, "judge": lambda: stage_judge(args.concurrency),
     "report": report}[args.stage]()


if __name__ == "__main__":
    main()
