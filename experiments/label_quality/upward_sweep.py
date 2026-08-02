"""
Phase 6e -- does the exemplar curve turn over ABOVE the default?

Everything measured so far swept downward from n_exemplars=8, which only ever supports "you could
use fewer and save tokens" -- a weak recommendation, since tokens are the only established cost of
using more. The recommendation would only become interesting if MORE exemplars actively hurt
(context dilution producing worse names). That has never been tested here.

Design, and why it is shaped this way:

  k=4, 8, 16 -- fully leakage-controlled. Judge documents are drawn excluding the union of exemplars
  across k in {1,2,4,8,16}, so no arm has seen any document it is graded on. Excluding that union
  costs 27.6% of the member pool and leaves 12 of 107 clusters under the 15-document budget; those
  clusters are dropped rather than judged on a thinner sample.

  k=32 -- deliberately ADVANTAGED, and reported as such. Excluding 32 exemplars per cluster starves
  the pool (39/107 clusters under budget, 16 empty), so full control is impossible. Instead k=32 is
  judged on the same documents, where its exemplars beyond the excluded 16 CAN overlap what it is
  graded on. Leakage therefore favours k=32. That makes the inference asymmetric but still sound in
  the direction that matters: if k=32 fails to beat k=8 even with a parroting advantage, "more is
  better" is dead. (Measured leakage sensitivity is small anyway -- clean_docs_rejudge.py found that
  going from 0% to 24% overlap moved the k=0 effect by 2%.)

  uv run python experiments/label_quality/upward_sweep.py --stage check   # no LLM
  uv run python experiments/label_quality/upward_sweep.py --stage name
  uv run python experiments/label_quality/upward_sweep.py --stage judge
  uv run python experiments/label_quality/upward_sweep.py --stage report
Writes data/upward_names_20ng.json and data/upward_judge_20ng.json (resumable).
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

NEW_RUNGS = [16, 32]                  # 4 and 8 already have labels
EXCLUDE_KS = [1, 2, 4, 8, 16]         # 32 excluded would starve the pool -- see docstring
N_NEAR, N_RAND, MAXLEN = 10, 5, 280
K_SAMPLES = 3
NAMES = HERE / "data" / "upward_names_20ng.json"
JUDGE = HERE / "data" / "upward_judge_20ng.json"


def build_docs():
    """Judge docs excluding the union of exemplars over EXCLUDE_KS. Returns (docs, eligible)."""
    from perturbations import load_fit

    cl, objects, emb, coords, meta = load_fit("20ng", None, 25, 4)
    counts = [int(l.centroid_vectors.shape[0]) for l in cl.cluster_layers_]
    banned = {}
    for k in EXCLUDE_KS:
        for L, layer in enumerate(cl.cluster_layers_):
            layer.n_exemplars = k
            layer.make_exemplar_texts(objects, emb)
            for i in range(counts[L]):
                banned.setdefault((L, i), set()).update(map(int, layer.exemplar_indices[i]))

    emb64 = emb.astype(np.float64)
    docs, eligible = {}, []
    for L in range(len(counts)):
        cls = cl.cluster_layers_[L].cluster_labels
        C = cl.cluster_layers_[L].centroid_vectors
        for i in range(counts[L]):
            mem = np.where(cls == i)[0]
            pool = np.array([m for m in mem if int(m) not in banned[(L, i)]])
            if pool.size < N_NEAR + N_RAND:
                continue                       # drop, don't judge on a thinner sample
            eligible.append((L, i))
            E = emb64[pool] / (norm(emb64[pool], axis=1, keepdims=True) + 1e-12)
            c = C[i].astype(np.float64) / (norm(C[i]) + 1e-12)
            near = pool[np.argsort(-(E @ c))[:N_NEAR]]
            rest = np.array([m for m in pool if m not in set(near.tolist())])
            rng = np.random.default_rng(1000 + i)
            rand = (rng.choice(rest, size=min(N_RAND, rest.size), replace=False)
                    if rest.size else np.array([], int))
            docs[(L, i)] = [" ".join(str(objects[j]).split())[:MAXLEN]
                            for j in list(near) + list(rand)]
    return docs, eligible, meta


def labels_by_condition() -> dict:
    battery = json.loads((HERE / "data" / "battery_20ng.json").read_text())
    out = {"8": {(it["layer"], it["idx"]): it["gold"] for it in battery}}
    dose = json.loads((HERE / "data" / "dose_names_20ng.json").read_text())
    out["4"] = {(L, i): n for L, names in enumerate(dose["4"]) for i, n in enumerate(names)}
    if NAMES.exists():
        up = json.loads(NAMES.read_text())
        for k in NEW_RUNGS:
            if str(k) in up:
                out[str(k)] = {(L, i): n for L, names in enumerate(up[str(k)])
                               for i, n in enumerate(names)}
    return out


def stage_check():
    docs, eligible, _ = build_docs()
    print(f"leakage-controlled document sample (excluding exemplars for k in {EXCLUDE_KS}):")
    print(f"  eligible clusters: {len(eligible)}/107  (dropped {107-len(eligible)} under budget)")
    print(f"  docs per eligible cluster: {min(len(v) for v in docs.values())}"
          f"-{max(len(v) for v in docs.values())}")
    conds = labels_by_condition()
    have = [c for c in ["4", "8", "16", "32"] if c in conds]
    print(f"  conditions with labels: {have}")
    print(f"  judge calls once all named: {4 * len(eligible) * K_SAMPLES}")


def stage_name():
    from ab_harness import load_dataset, make_embedder, make_namer

    from toponymy.cluster_layer import ClusterLayerText
    from toponymy.clustering import ToponymyClusterer
    from toponymy.toponymy import Toponymy

    objects, emb, coords, meta = load_dataset("20ng", None)
    out = json.loads(NAMES.read_text()) if NAMES.exists() else {}
    for k in NEW_RUNGS:
        if str(k) in out:
            print(f"k={k}: cached")
            continue
        print(f"\n=== naming at k={k} exemplars ===", flush=True)
        clusterer = ToponymyClusterer(min_clusters=4, base_min_cluster_size=25, verbose=False)
        clusterer.fit_predict(coords, emb, ClusterLayerText)
        counts = [int(l.centroid_vectors.shape[0]) for l in clusterer.cluster_layers_]
        assert counts == [74, 24, 9], f"replay mismatch {counts}"
        for layer in clusterer.cluster_layers_:
            layer.n_exemplars = k
        model = Toponymy(make_namer("haiku"), make_embedder(meta["emb_model"]),
                         clusterer=clusterer, object_description=meta["obj"],
                         corpus_description=meta["corpus"], verbose=False)
        model.fit(objects, emb, coords)
        out[str(k)] = [list(x) for x in model.topic_names_]
        NAMES.write_text(json.dumps(out, indent=2))
        print(f"  layers {[len(l) for l in out[str(k)]]} written")


def stage_judge(concurrency: int):
    from async_judge import rate_many

    docs, eligible, meta = build_docs()
    conds = labels_by_condition()
    done = json.loads(JUDGE.read_text()) if JUDGE.exists() else {}
    for c in ["4", "8", "16", "32"]:
        if c not in conds:
            print(f"{c}: no labels yet, skipping")
            continue
        done.setdefault(c, {})
        todo = [(L, i, conds[c][(L, i)]) for (L, i) in eligible
                if conds[c].get((L, i)) and conds[c][(L, i)] != "Unlabelled"
                and done[c].get(f"{L}.{i}", {}).get("label", "").strip()
                != conds[c][(L, i)].strip()]
        print(f"k={c}: {len(todo)} to judge ({len(done[c])} cached)", flush=True)
        if not todo:
            continue
        res = rate_many([(lab, docs[(L, i)]) for L, i, lab in todo], meta["obj"],
                        "anthropic/claude-sonnet-4-6", k=K_SAMPLES, concurrency=concurrency)
        for (L, i, lab), r in zip(todo, res):
            done[c][f"{L}.{i}"] = {"label": lab, "overall": r["overall"]}
        JUDGE.write_text(json.dumps(done, indent=2))
        print(f"  done ({sum(1 for v in done[c].values() if v['overall'] is not None)} rated)")


def report():
    j = json.loads(JUDGE.read_text())
    rungs = [c for c in ["4", "8", "16", "32"] if c in j]
    keys = None
    for c in rungs:
        ks = {tuple(map(int, s.split("."))) for s, v in j[c].items() if v["overall"] is not None}
        keys = ks if keys is None else keys & ks
    keys = sorted(keys)
    A = {c: np.array([j[c][f"{L}.{i}"]["overall"] for L, i in keys]) for c in rungs}
    base = A["8"]

    print(f"\n===== EXEMPLAR SWEEP ABOVE THE DEFAULT (20NG, n={len(keys)}) =====")
    print(f"  judge docs exclude exemplars for k in {EXCLUDE_KS}")
    print(f"  k=4/8/16 fully controlled; k=32 ADVANTAGED (its extra exemplars can overlap)\n")
    print(f"  {'k':>4} {'judge':>8} {'vs k=8':>10} {'95% CI':>18} {'Wilcoxon':>10}")
    for c in rungs:
        d = A[c] - base
        m = d.mean()
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0
        p = wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
        ci = f"[{m-1.96*se:+.3f},{m+1.96*se:+.3f}]" if se else " " * 17
        tag = "  <- advantaged" if c == "32" else ""
        print(f"  {c:>4} {A[c].mean():>8.3f} {m:>+10.3f} {ci:>18} {p:>10.3g}{tag}")

    print("\n  verdict:")
    for c in [r for r in rungs if int(r) > 8]:
        d = A[c] - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        if d.mean() < -1.96 * se:
            print(f"   k={c}: SIGNIFICANTLY WORSE than the default -- the curve turns over,")
            print(f"         so over-provisioning exemplars has a real (non-cost) downside.")
        elif d.mean() > 1.96 * se:
            print(f"   k={c}: significantly BETTER than the default -- 8 is too low.")
        else:
            print(f"   k={c}: indistinguishable from the default ({d.mean():+.3f}, "
                  f"CI [{d.mean()-1.96*se:+.3f},{d.mean()+1.96*se:+.3f}]).")
    print("\n   -> a flat top means the only cost of more exemplars is tokens, and the only")
    print("      real risk is going too LOW. No per-corpus tuning is warranted.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["check", "name", "judge", "report"])
    ap.add_argument("--concurrency", type=int, default=24)
    args = ap.parse_args()
    {"check": stage_check, "name": stage_name,
     "judge": lambda: stage_judge(args.concurrency), "report": report}[args.stage]()


if __name__ == "__main__":
    main()
