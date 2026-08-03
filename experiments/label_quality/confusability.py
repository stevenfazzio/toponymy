"""
Phase 7 tranche 2 -- confusability: the one application tranche 1 supports.

Toponymy's disambiguation pass triggers on near-DUPLICATE name strings
(`cluster_topic_names_for_renaming`: agglomerative over name embeddings, complete linkage, 0.2
cosine-distance cap). `collision_check.py` found it fires on NOTHING in real name sets (closest
pair 0.305). Meanwhile Phase 4 showed that when a name is confusable the listener relocates 0.51
of its mass to the wrong region. So the pass is structurally blind to confusability that is not
string similarity -- it catches "same name twice", not "a reader cannot tell these two apart".

7a gave a free detector for exactly that: softmax(cos(label, cand)/tau) over the neighbourhood
predicts WHICH wrong region the listener drifts to 67-69% of the time (chance 25%).

  --stage score    offline: score every gold label's leaked mass, validate against the listener's
                   measured leak, compare against the library trigger, emit arbiter pair lists
  --stage arbiter  fresh PAIRWISE lineups (k=2, chance 0.50) over the flagged pairs and matched
                   controls, with fresh document draws and a held-out listener (gpt-4o-mini)
  --stage report   synthesis

tau = 0.170, taken from 7a where it was fitted on the OTHER corpus; not refitted here.

  uv run python experiments/label_quality/confusability.py --stage score
  uv run python experiments/label_quality/confusability.py --stage arbiter
  uv run python experiments/label_quality/confusability.py --stage report
"""
from __future__ import annotations

import argparse
import asyncio
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
TAU = 0.170                    # from 7a, fitted out-of-corpus
ARBITER_MODEL = "openai/gpt-4o-mini"   # held out from naming (haiku) and from 7a's targets (sonnet)
N_FLAG, N_CTRL, N_SIB, N_GIM = 40, 40, 15, 15
N_DOCS, MAXLEN, K_SAMPLES, TEMP = 5, 500, 3, 0.7
FRESH_SALT = "confusability-arbiter-v1"   # different from wayfinding's frozen doc seed


def unit_rows(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


# --------------------------------------------------------------------------- score

def score_dataset(ds):
    from ab_harness import make_embedder
    from wayfinding import Cell

    print(f"\n[{ds}] replaying cell ...", flush=True)
    cell = Cell(ds)
    bat = json.loads((DATA / f"wayfinding_{ds}_battery.json").read_text())["units"]
    gold = [u for u in bat.values()
            if u.get("kind") == "gold" and u.get("mode") == "nn" and u.get("pm") is not None]
    sib = [u for u in bat.values() if u.get("kind") == "sibling" and u.get("pm") is not None]
    print(f"[{ds}] {len(gold)} gold units, {len(sib)} sibling units")

    emb = make_embedder(cell.meta["emb_model"])
    labels = sorted({u["label"] for u in gold})
    E = unit_rows(np.asarray(emb.encode(labels, show_progress_bar=False), dtype=np.float64))
    ix = {s: i for i, s in enumerate(labels)}

    rows = []
    for u in gold:
        v = E[ix[u["label"]]]
        s = np.array([float(v @ cell.cent[u["L"]][j]) for j in u["lineup"]]) / TAU
        p = np.exp(s - s.max())
        p = p / p.sum()
        t = u["lineup"].index(u["true"])
        others = [(j, float(p[a])) for a, j in enumerate(u["lineup"]) if j != u["true"]]
        pred_j, pred_leak = max(others, key=lambda kv: kv[1])
        meas = {int(k): val for k, val in u["mass"].items()}
        meas_others = {j: meas.get(j, 0.0) for j in u["lineup"] if j != u["true"]}
        meas_j = max(meas_others, key=meas_others.get)
        rows.append(dict(L=u["L"], i=u["true"], label=u["label"], lineup=u["lineup"],
                         pred_true=float(p[t]), pred_leak_total=float(1 - p[t]),
                         pred_j=int(pred_j), pred_leak_top=pred_leak,
                         meas_true=u["pm"], meas_leak_total=float(1 - u["pm"]),
                         meas_j=int(meas_j), meas_leak_top=float(meas_others[meas_j])))
    return cell, rows, sib


def validate(ds, rows):
    from scipy.stats import pearsonr, spearmanr
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        roc_auc_score = None

    a = [r["pred_leak_total"] for r in rows]
    b = [r["meas_leak_total"] for r in rows]
    print(f"  predicted vs measured leaked mass: Pearson {pearsonr(a, b).statistic:+.3f}   "
          f"Spearman {spearmanr(a, b).statistic:+.3f}   (n={len(rows)})")
    drift = [r for r in rows if r["meas_leak_top"] > r["meas_true"]]
    hit = sum(r["pred_j"] == r["meas_j"] for r in drift)
    print(f"  top-confuser agreement where the listener actually drifted: "
          f"{hit}/{len(drift)} = {100*hit/max(len(drift),1):.0f}%  (chance ~25%)")
    hit_all = sum(r["pred_j"] == r["meas_j"] for r in rows)
    print(f"  top-confuser agreement over ALL gold labels: "
          f"{hit_all}/{len(rows)} = {100*hit_all/len(rows):.0f}%")
    if roc_auc_score:
        y = [1 if r["meas_true"] < 0.4 else 0 for r in rows]   # "a reader would struggle"
        if 0 < sum(y) < len(y):
            print(f"  detecting measurably-confusable labels (measured pm < 0.40, "
                  f"{sum(y)}/{len(y)} positives): AUC {roc_auc_score(y, a):.3f}")


def trigger_comparison(ds, cell, rows):
    """What does Toponymy's own renaming trigger see on these same name sets?"""
    from ab_harness import make_embedder
    from toponymy.prompt_construction import cluster_topic_names_for_renaming

    emb = make_embedder(cell.meta["emb_model"])
    by_layer = defaultdict(dict)
    for r in rows:
        by_layer[r["L"]][r["i"]] = r["label"]
    caught = set()
    for L, d in sorted(by_layer.items()):
        idxs = sorted(d)
        names = [d[i] for i in idxs]
        vec = emb.encode(names, convert_to_numpy=True, show_progress_bar=False)
        groups, lab = cluster_topic_names_for_renaming(names, vec)
        n_topics = int(sum(np.sum(lab == g) for g in groups))
        print(f"    L{L}: {len(names)} names -> trigger forms {len(groups)} group(s) "
              f"covering {n_topics} topic(s)")
        for g in groups:
            for pos in np.where(lab == g)[0]:
                caught.add((L, idxs[int(pos)]))
    return caught


def build_arbiter_units(ds, rows, sib_units):
    """flagged / control / sibling(positive control) / gimme(ceiling), as k=2 pairwise lineups."""
    ranked = sorted(rows, key=lambda r: -r["pred_leak_top"])
    flag = ranked[:N_FLAG]
    ctrl = ranked[-N_CTRL:]
    units = []
    for arm, rs in (("flag", flag), ("ctrl", ctrl)):
        for r in rs:
            units.append(dict(uid=f"pair|{ds}|{arm}|L{r['L']}|{r['i']}|{r['pred_j']}",
                              arm=arm, L=r["L"], true=r["i"], other=r["pred_j"],
                              label=r["label"], lineup=[r["i"], r["pred_j"]],
                              pred_leak_top=r["pred_leak_top"], meas_true_k5=r["meas_true"]))
    rng = np.random.default_rng(20260803)
    # positive control: cluster i carrying its SIBLING's label, vs that sibling
    sib_sorted = sorted(sib_units, key=lambda u: u["uid"])
    for u in sib_sorted[: N_SIB * 3]:
        meas = {int(k): v for k, v in u["mass"].items()}
        others = {j: meas.get(j, 0.0) for j in u["lineup"] if j != u["true"]}
        j = max(others, key=others.get)
        units.append(dict(uid=f"pair|{ds}|sib|L{u['L']}|{u['true']}|{j}", arm="sib",
                          L=u["L"], true=u["true"], other=int(j), label=u["label"],
                          lineup=[u["true"], int(j)], pred_leak_top=None,
                          meas_true_k5=u["pm"]))
    units = [u for u in units if u["arm"] != "sib"] + \
            [u for u in units if u["arm"] == "sib"][:N_SIB]
    # ceiling: gold label vs a FAR cluster (should be near 1.0 if the pairwise protocol works)
    for r in [rows[i] for i in rng.choice(len(rows), size=min(N_GIM, len(rows)), replace=False)]:
        units.append(dict(uid=f"pair|{ds}|gim|L{r['L']}|{r['i']}", arm="gim", L=r["L"],
                          true=r["i"], other=None, label=r["label"], lineup=None,
                          pred_leak_top=None, meas_true_k5=r["meas_true"]))
    return units


def stage_score():
    out = {}
    for ds in DATASETS:
        cell, rows, sib = score_dataset(ds)
        print(f"\n[{ds}] VALIDATION of the confusability score (tau={TAU}, from 7a):")
        validate(ds, rows)
        print(f"  Toponymy's own renaming trigger on these same names:")
        caught = trigger_comparison(ds, cell, rows)
        ranked = sorted(rows, key=lambda r: -r["pred_leak_top"])
        top = [(r["L"], r["i"]) for r in ranked[:N_FLAG]]
        print(f"    trigger catches {len(caught)} topic(s); of the {N_FLAG} most confusable "
              f"labels by score it catches {len(set(top) & caught)}")
        print(f"\n[{ds}] most confusable gold labels by score (top 6):")
        for r in ranked[:6]:
            print(f"    L{r['L']} c{r['i']:<3d} pred_leak {r['pred_leak_top']:.2f} -> c{r['pred_j']:<3d}"
                  f"  measured pm {r['meas_true']:.2f}  {r['label'][:58]!r}")
        units = build_arbiter_units(ds, rows, sib)
        n = sum(1 for u in units if u["arm"] == "flag")
        print(f"\n[{ds}] arbiter units built: {len(units)} "
              f"({n} flag / {sum(1 for u in units if u['arm']=='ctrl')} ctrl / "
              f"{sum(1 for u in units if u['arm']=='sib')} sib / "
              f"{sum(1 for u in units if u['arm']=='gim')} gim) "
              f"= {len(units)*K_SAMPLES} calls")
        (DATA / f"confusability_{ds}.json").write_text(json.dumps(
            {"tau": TAU, "rows": rows, "arbiter_units": units, "trigger_caught": sorted(caught)}))
        out[ds] = units
    total = sum(len(v) for v in out.values()) * K_SAMPLES
    print(f"\nTOTAL arbiter cost if run: {total} calls on {ARBITER_MODEL}")


# --------------------------------------------------------------------------- arbiter

def fresh_docs(cell, L, i):
    """Held-out member docs under a FRESH seed -- deliberately not wayfinding's frozen sample."""
    from wayfinding import md5i

    members = np.where(cell.layers[L].cluster_labels == i)[0]
    ex = set(map(int, cell.layers[L].exemplar_indices[i]))
    pool = np.array([m for m in members if int(m) not in ex])
    if pool.size == 0:
        pool = members
    rng = np.random.default_rng(md5i(f"{FRESH_SALT}|{L}|{i}"))
    take = rng.choice(pool, size=min(N_DOCS, pool.size), replace=False)
    return [" ".join(str(cell.objects[j]).split())[:MAXLEN] for j in take]


async def run_pair(u, cell, sem):
    import litellm
    from wayfinding import LETTERS, PROMPT, RESPONSE_FORMAT, md5i, parse_scores

    cids = u["lineup"]
    groups = {j: fresh_docs(cell, u["L"], j) for j in cids}

    async def one(s):
        rng = np.random.default_rng(md5i(f"{u['uid']}|s{s}"))
        disp = [cids[j] for j in rng.permutation(len(cids))]
        blocks = "\n\n".join(f"Candidate {LETTERS[a]}:\n" + "\n".join(f"- {d}" for d in groups[j])
                             for a, j in enumerate(disp))
        prompt = PROMPT.format(label=u["label"], k=len(cids),
                               span=f"{LETTERS[0]}-{LETTERS[len(cids)-1]}", blocks=blocks)
        async with sem:
            try:
                r = await litellm.acompletion(model=ARBITER_MODEL, temperature=TEMP, max_tokens=300,
                                              num_retries=5, response_format=RESPONSE_FORMAT,
                                              messages=[{"role": "user", "content": prompt}])
                sc = parse_scores(r.choices[0].message.content or "", len(cids))
            except Exception as e:
                print("  arbiter err:", type(e).__name__, str(e)[:70], flush=True)
                sc = None
        return (disp, sc) if sc else None

    got = [x for x in await asyncio.gather(*[one(s) for s in range(K_SAMPLES)]) if x]
    pms = []
    for disp, sc in got:
        tot = sum(sc) or 1
        pms.append(sc[disp.index(u["true"])] / tot)
    return dict(u, n_valid=len(got), pm=float(np.mean(pms)) if pms else None,
                samples=[{"order": d, "scores": s} for d, s in got])


async def arbiter_async(ds, concurrency):
    from wayfinding import Cell

    spec = json.loads((DATA / f"confusability_{ds}.json").read_text())
    cell = Cell(ds)
    units = spec["arbiter_units"]
    # the gimme arm needs a far cluster picked now (lineup was left null at score time)
    for u in units:
        if u["arm"] == "gim":
            far = cell.lineup(u["L"], u["true"], 5, "gimme")[1]
            u["other"], u["lineup"] = int(far), [u["true"], int(far)]
    path = DATA / f"confusability_arbiter_{ds}.json"
    done = {}
    if path.exists():
        done = {k: v for k, v in json.loads(path.read_text())["units"].items()
                if v.get("pm") is not None}
        units = [u for u in units if u["uid"] not in done]
        print(f"  resume: {len(done)} cached, {len(units)} to run")
    sem = asyncio.Semaphore(concurrency)
    res = await asyncio.gather(*[run_pair(u, cell, sem) for u in units])
    for r in res:
        done[r["uid"]] = r
    path.write_text(json.dumps({"model": ARBITER_MODEL, "config": dict(
        n_docs=N_DOCS, maxlen=MAXLEN, k_samples=K_SAMPLES, temp=TEMP, salt=FRESH_SALT),
        "units": done}))
    print(f"  wrote {path.name}: {len(done)} units")


def stage_arbiter(concurrency):
    for ds in DATASETS:
        print(f"\n[{ds}] arbiter (fresh docs, {ARBITER_MODEL}, k=2 pairwise, chance 0.50)")
        asyncio.run(arbiter_async(ds, concurrency))


# --------------------------------------------------------------------------- report

def stage_report():
    from scipy.stats import mannwhitneyu

    for ds in DATASETS:
        p = DATA / f"confusability_arbiter_{ds}.json"
        if not p.exists():
            print(f"[{ds}] no arbiter file yet")
            continue
        units = [u for u in json.loads(p.read_text())["units"].values() if u.get("pm") is not None]
        arms = defaultdict(list)
        for u in units:
            arms[u["arm"]].append(u["pm"])
        print(f"\n[{ds}] PAIRWISE ARBITER -- fresh docs, held-out listener, chance = 0.50")
        for arm, name in (("gim", "gimme (gold vs far)   ceiling"),
                          ("ctrl", "control (low score)   expect high"),
                          ("flag", "FLAGGED (high score)  expect low"),
                          ("sib", "sibling label         positive control")):
            v = arms.get(arm, [])
            if not v:
                continue
            below = sum(1 for x in v if x < 0.5)
            print(f"  {name:34s} n={len(v):3d}  mean pm {np.mean(v):.3f}   "
                  f"below chance {below}/{len(v)}")
        if arms.get("flag") and arms.get("ctrl"):
            U = mannwhitneyu(arms["flag"], arms["ctrl"], alternative="less")
            print(f"  flagged < control: Mann-Whitney p = {U.pvalue:.2g}   "
                  f"gap {np.mean(arms['ctrl']) - np.mean(arms['flag']):+.3f} pm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["score", "arbiter", "report"])
    ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()
    {"score": stage_score, "arbiter": lambda: stage_arbiter(a.concurrency),
     "report": stage_report}[a.stage]()


if __name__ == "__main__":
    main()
