"""
Phase 7 tranche 3b -- is the disambiguation pass net-harmful WHENEVER IT FIRES?

Tranche 3 showed the renaming machinery costs ~0.5 judge-pts when handed score-selected confusable
pairs. But those are not the pairs the pass actually catches: its trigger fires on near-DUPLICATE
name strings (0.2 cosine-distance cap). So the load-bearing question for the library -- and for
FEATURES.md's keyphrase recommendation, which priced the pass's 3x fine-layer load as a TOKEN cost
-- is what the pass does to the groups it forms on its own.

No renaming calls are needed: `disambiguation_load.py` already instrumented `ClusterLayer.
disambiguate_topics` across 24 real `fit()` runs (4 conditions x 2 corpora x 3 draws) and captured,
per layer, the pre-pass names, the post-pass names, and the groups the trigger formed. Those are
paired by construction. This measures both axes on them.

Structurally 20NG-only: arXiv produces ZERO renaming load under every condition (FEATURES.md), so
the two-corpus standard cannot be met here and the result is reported as single-corpus.

  uv run python experiments/label_quality/disamb_value.py --stage judge
  uv run python experiments/label_quality/disamb_value.py --stage lineup
  uv run python experiments/label_quality/disamb_value.py --stage report
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

JUDGE_MODEL = "anthropic/claude-sonnet-4-6"
LISTENER = "anthropic/claude-sonnet-4-6"     # the battery's listener, so pm is comparable to gold
JUDGE_OUT = DATA / "disamb_value_judge.json"
LINEUP_OUT = DATA / "disamb_value_lineup.json"


def instances():
    """(run, L, i, pre, post) for every topic the trigger's own groups actually renamed."""
    d = json.loads((DATA / "disamb_load.json").read_text())
    out = []
    for run, r in sorted(d.items()):
        if not run.startswith("20ng"):
            continue                      # arXiv never fires; nothing to measure
        for lay in r["layers"]:
            L, pre, post = lay["layer_id"], lay["pre"], lay["post"]
            for g in lay.get("groups", []):
                for i in map(int, g):
                    if pre[i] != post[i]:
                        out.append(dict(run=run, cond=run.split("|")[1], L=L, i=i,
                                        pre=pre[i], post=post[i]))
    return out


def units(insts):
    """Distinct (layer, cluster, label) measurement units, each tagged pre/post."""
    seen = {}
    for it in insts:
        for ver in ("pre", "post"):
            key = (it["L"], it["i"], it[ver])
            seen.setdefault(key, dict(L=it["L"], i=it["i"], label=it[ver], ver=ver))
    return list(seen.values())


def ukey(L, i, label):
    return f"{L}.{i}|{label}"


# --------------------------------------------------------------------------- judge

def stage_judge(concurrency):
    from ab_harness import load_dataset
    from async_judge import rate_many
    from judge_fair import sample_docs

    us = units(instances())
    done = json.loads(JUDGE_OUT.read_text()) if JUDGE_OUT.exists() else {}
    todo = [u for u in us if ukey(u["L"], u["i"], u["label"]) not in done]
    print(f"judging {len(todo)} distinct labels x3 = {len(todo)*3} calls "
          f"({len(us)-len(todo)} cached)")
    if not todo:
        return
    obj = load_dataset("20ng", None)[3]["obj"]
    tasks = [(u["label"], sample_docs("20ng", u["L"], u["i"], n_near=10, n_rand=5, maxlen=280))
             for u in todo]
    ratings = rate_many(tasks, obj, JUDGE_MODEL, k=3, temp=0.7, concurrency=concurrency)
    for u, r in zip(todo, ratings):
        done[ukey(u["L"], u["i"], u["label"])] = dict(u, overall=r.get("overall"))
    JUDGE_OUT.write_text(json.dumps(done, indent=1))
    print(f"  wrote {JUDGE_OUT.name}: {len(done)} labels")


# --------------------------------------------------------------------------- lineup

def stage_lineup(concurrency):
    from wayfinding import Cell, K_DEFAULT, run_all

    cell = Cell("20ng")
    us = units(instances())
    payload = []
    for u in us:
        payload.append(dict(uid=f"dv|{ukey(u['L'], u['i'], u['label'])}",
                            L=u["L"], i=u["i"], true=u["i"], kind=f"dv-{u['ver']}",
                            label=u["label"], mode="nn", run=1,
                            lineup=cell.lineup(u["L"], u["i"], K_DEFAULT, "nn")))
    print(f"lineups: {len(payload)} units x3 = {len(payload)*3} calls on {LISTENER} "
          f"(frozen battery lineups, so pm is comparable to gold 0.545)")
    asyncio.run(run_all(payload, cell, LISTENER, concurrency, LINEUP_OUT))


# --------------------------------------------------------------------------- report

STOP = {"and", "or", "of", "for", "in", "the", "a", "with", "to", "on", "from", "by", "using"}


def _toks(s):
    import re
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 2}


def stage_report():
    from scipy.stats import wilcoxon

    insts = instances()
    jd = json.loads(JUDGE_OUT.read_text()) if JUDGE_OUT.exists() else {}
    ld = json.loads(LINEUP_OUT.read_text())["units"] if LINEUP_OUT.exists() else {}
    lpm = {v["uid"].split("|", 1)[1]: v["pm"] for v in ld.values() if v.get("pm") is not None}

    print(f"\nWhat the trigger's OWN groups did, on {len(insts)} renamed topic instances "
          f"(20NG only -- arXiv never fires)\n")
    for axis, table, name, band in (("fit", jd, "grounded judge (0-4)", "judge sigma 0.168"),
                                    ("identification", lpm, "frozen k=5 lineup pm", "band p90 0.107")):
        rows = []
        for it in insts:
            a = table.get(ukey(it["L"], it["i"], it["pre"]))
            b = table.get(ukey(it["L"], it["i"], it["post"]))
            a = a.get("overall") if isinstance(a, dict) else a
            b = b.get("overall") if isinstance(b, dict) else b
            if a is not None and b is not None:
                rows.append((it, a, b))
        if not rows:
            print(f"{axis.upper()}: no data yet")
            continue
        pre = [a for _, a, _ in rows]
        post = [b for _, _, b in rows]
        print(f"{axis.upper()} -- {name}   (n={len(rows)} instances, {band})")
        print(f"  pre-pass {np.mean(pre):.3f}   ->   post-pass {np.mean(post):.3f}   "
              f"delta {np.mean(post)-np.mean(pre):+.3f}"
              + (f"   Wilcoxon p = {wilcoxon(post, pre).pvalue:.3g}" if len(rows) >= 6 else ""))
        print(f"  the pass improved {sum(1 for _,a,b in rows if b>a)}/{len(rows)}, "
              f"hurt {sum(1 for _,a,b in rows if b<a)}/{len(rows)}")
        by = defaultdict(list)
        for it, a, b in rows:
            by[it["cond"]].append(b - a)
        for c in sorted(by):
            print(f"    {c:16s} n={len(by[c]):3d}  delta {np.mean(by[c]):+.3f}")
        print()

    # mechanism: does the pass strip the vocabulary the colliding names shared?
    groups = defaultdict(list)
    for it in insts:
        groups[(it["run"], it["L"])].append(it)
    oj, nj, lost = [], [], []
    for gs in groups.values():
        if len(gs) < 2:
            continue
        for a in range(len(gs)):
            for b in range(a + 1, len(gs)):
                pa, pb = _toks(gs[a]["pre"]), _toks(gs[b]["pre"])
                qa, qb = _toks(gs[a]["post"]), _toks(gs[b]["post"])
                oj.append(len(pa & pb) / max(len(pa | pb), 1))
                nj.append(len(qa & qb) / max(len(qa | qb), 1))
                sh = pa & pb
                if sh:
                    lost.append(1 - len(sh & (qa | qb)) / len(sh))
    if oj:
        wo = np.mean([len(it["pre"].split()) for it in insts])
        wn = np.mean([len(it["post"].split()) for it in insts])
        print(f"MECHANISM (within the trigger's own groups, {len(oj)} colliding name pairs)")
        print(f"  token Jaccard {np.mean(oj):.3f} -> {np.mean(nj):.3f};  "
              f"{np.mean(lost):.0%} of shared vocabulary dropped from both;  "
              f"length {wo:.1f} -> {wn:.1f} words")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["judge", "lineup", "report"])
    ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()
    {"judge": lambda: stage_judge(a.concurrency),
     "lineup": lambda: stage_lineup(a.concurrency), "report": stage_report}[a.stage]()


if __name__ == "__main__":
    main()
