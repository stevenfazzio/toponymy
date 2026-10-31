"""
Rank the screened decision models, have Sonnet rate the best picks, and compare with Haiku and Sonnet.

  python dm_analyze.py <scratchpad> rank     agreement with Sonnet on the frozen documents
  python dm_analyze.py <scratchpad> rate     Sonnet rates picks from the top two models
  python dm_analyze.py <scratchpad> report   fresh-document gains on the same clusters

Ranking rule, fixed before the run: mean within-cluster Spearman with Sonnet's mean score, averaged
across the two corpora, per model and format. The top two models each go forward with their better
format. A pick is the top score for a cluster; ties go to fewer words, then alphabetical order.
"""
import asyncio
import collections
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from haiku_test import SONNET, Budget, call, digest, frac, haiku_scores, in_subset, picks_from, words

SP = Path(sys.argv[1])
RATE_CAP = 3_400_000
CELLS = ("20ng", "arxiv_home")
FMTS = ("lineup", "labels")


def boot(x, n=10000):
    rng = np.random.default_rng(0)
    x = np.asarray(x)
    b = [x[rng.integers(len(x), size=len(x))].mean() for _ in range(n)]
    return [float(v) for v in np.percentile(b, [2.5, 97.5])]


def within(rows):
    out = []
    for r in rows:
        a = np.array(r, float)
        if len(a) >= 4 and a[:, 0].std() > 0 and a[:, 1].std() > 0:
            out.append(spearmanr(a[:, 0], a[:, 1])[0])
    return float(np.mean(out)), len(out)


def load_scores():
    """(model, fmt, cell) -> uid -> score, requiring every document for the labels format."""
    if not (SP / "dm_results.jsonl").exists():  # the raw file is 12.8 MB; dm_scores.json is committed instead
        stored = json.loads((SP / "dm_scores.json").read_text())
        return ({tuple(k.split("|")): v for k, v in stored.items()}, collections.Counter(), collections.Counter())
    raw = collections.defaultdict(list)
    fails = collections.Counter()
    for r in map(json.loads, (SP / "dm_results.jsonl").read_text().splitlines()):
        if r["p"] is None:
            fails[(r["model"], r["fmt"], (r.get("error") or "")[:60])] += 1
            continue
        raw[(r["model"], r["fmt"], r["cell"], r["uid"])].append(r["p"])
    out = collections.defaultdict(dict)
    short = collections.Counter()
    for (m, f, c, uid), v in raw.items():
        need = 1 if f == "lineup" else 5
        if len(v) >= need:
            out[(m, f, c)][uid] = float(np.mean(v))
        else:
            short[(m, f, c)] += 1
    return out, fails, short


def clusters(cell, units, keep):
    by = collections.defaultdict(list)
    for u in units[cell]:
        if (cell, u["L"], u["i"]) in keep:
            by[(u["L"], u["i"])].append(u)
    return by


def choose(us, scores):
    ranked = sorted(us, key=lambda u: (-scores.get(u["uid"], -1), words(u["label"]), u["label"]))
    return ranked[0]["label"]


def ranked_pairs():
    scores, fails, short = load_scores()
    units = json.loads((SP / "haiku_units.json").read_text())
    keep = in_subset(SP)
    table = []
    models = sorted({k[0] for k in scores})
    for m in models:
        for f in FMTS:
            row = dict(model=m, fmt=f)
            vals, picks_mean, rand_mean = [], [], []
            for c in CELLS:
                by = clusters(c, units, keep)
                s = scores.get((m, f, c), {})
                rho, n = within([[(s[u["uid"]], np.mean(u["sonnet"])) for u in us if u["uid"] in s] for us in by.values()])
                p = [np.mean(next(u for u in us if u["label"] == choose(us, s))["sonnet"]) for us in by.values()]
                row[c] = rho
                row[f"{c}_pick"] = float(np.mean(p))
                row[f"{c}_scored"] = len(s)
                rand_mean.append(float(np.mean([np.mean([np.mean(u["sonnet"]) for u in us]) for us in by.values()])))
                vals.append(rho)
            row["mean"] = float(np.mean(vals))
            row["random"] = rand_mean
            table.append(row)
    table.sort(key=lambda r: -r["mean"])
    return table, fails, short, scores, units, keep


def rank():
    table, fails, short, *_ = ranked_pairs()
    print("reference: Haiku +.285 / +.287, Sonnet sample vs its other two +.412 / +.455")
    print(f"{'model':34s} {'format':7s} {'20ng':>7s} {'arxiv':>7s} {'mean':>7s}  pick quality 20ng/arxiv (random {table[0]['random'][0]:.3f}/{table[0]['random'][1]:.3f})  labels scored")
    for r in table:
        print(f"{r['model']:34s} {r['fmt']:7s} {r['20ng']:+.3f}  {r['arxiv_home']:+.3f}  {r['mean']:+.3f}  {r['20ng_pick']:.3f} / {r['arxiv_home_pick']:.3f}   {r['20ng_scored']} / {r['arxiv_home_scored']}")
    print("failures", dict(fails) if fails else "none", "| incomplete labels", dict(short) if short else "none")
    top, seen = [], set()
    for r in table:
        if r["model"] not in seen:
            top.append((r["model"], r["fmt"]))
            seen.add(r["model"])
        if len(top) == 2:
            break
    print("top two:", top)
    (SP / "dm_rank.json").write_text(json.dumps(dict(table=table, top=top), indent=2))


def fresh_scores():
    pairs = json.loads((SP / "fresh_pairs.json").read_text())
    labels = {u["uid"]: (c, u["L"], u["i"], u["label"]) for c, v in pairs.items() for u in v["units"]}
    samples = collections.defaultdict(list)
    for r in map(json.loads, (SP / "fresh_results.jsonl").read_text().splitlines()):
        c, L, i, lab = labels[r["uid"]]
        samples[(c, L, i, lab)].append(frac(r, i))
    for name in ("haiku_rate_results.jsonl", "dm_rate_results.jsonl"):
        path = SP / name
        if path.exists():
            for r in map(json.loads, path.read_text().splitlines()):
                if r["scores"] is not None:
                    samples[(r["cell"], r["L"], r["i"], r["label"])].append(frac(r, r["i"]))
    return pairs, samples


def rate():
    table, _, _, scores, units, keep = ranked_pairs()
    top = json.loads((SP / "dm_rank.json").read_text())["top"]
    pairs, samples = fresh_scores()
    fresh = json.loads((SP / "fresh_units.json").read_text())
    jobs, tally = collections.defaultdict(list), collections.Counter()
    queued = set()
    for m, f in top:
        for c in CELLS:
            by = clusters(c, units, keep)
            stock = {(p["L"], p["i"]): p["stock"] for p in pairs[c]["pairs"]}
            for key, us in by.items():
                pick = choose(us, scores[(m, f, c)])
                line = us[0]["lineup"]
                if pick == stock[key]:
                    tally["pick is stock"] += 1
                    continue
                for label in (pick, stock[key]):
                    k = (c, *key, label)
                    if len(samples.get(k, [])) >= 3:
                        tally["already rated"] += 1
                        continue
                    if k in queued:
                        tally["shared with the other model"] += 1
                        continue
                    queued.add(k)
                    tally["needs rating"] += 1
                    uid = f"fresh|L{key[0]}.{key[1]}|fresh-dm|{digest(label)}"
                    for s in range(3):
                        jobs[c].append((dict(uid=uid, L=key[0], i=key[1], true=key[1], label=label, lineup=line, cell=c), s))
    n = sum(len(v) for v in jobs.values())
    print(dict(tally), f"Sonnet calls {n}, about {n * 2813 / 1e6:.2f}M input tokens, cap {RATE_CAP / 1e6:.1f}M", flush=True)
    assert n * 2813 < RATE_CAP, "estimate exceeds the Sonnet cap"
    groups = {c: fresh[c]["groups"] for c in fresh}

    async def go():
        budget = Budget(0, RATE_CAP)
        sem = asyncio.Semaphore(12)
        stats = dict(done=0, failed=0, capped=0)
        with (SP / "dm_rate_results.jsonl").open("a") as fh:
            await asyncio.gather(*[call(SONNET, j, groups[c], sem, budget, fh, stats) for c, v in jobs.items() for j in v])
        print(f"finished {stats}; {budget.used:,} input tokens", flush=True)

    asyncio.run(go())


def report():
    table, _, _, scores, units, keep = ranked_pairs()
    top = json.loads((SP / "dm_rank.json").read_text())["top"]
    pairs, samples = fresh_scores()
    haiku = picks_from(SP)
    out = {}
    for c in CELLS:
        print(f"\n========== {c} ==========")
        by = clusters(c, units, keep)
        meta = {(p["L"], p["i"]): p for p in pairs[c]["pairs"]}
        f = lambda key, lab: float(np.mean(samples[(c, *key, lab)][:3]))
        methods = {"sonnet": {k: meta[k]["pick"] for k in by}, "haiku": haiku[c]}
        for m, fm in top:
            methods[f"{m} ({fm})"] = {k: choose(us, scores[(m, fm, c)]) for k, us in by.items()}
        base = None
        for name, picks in methods.items():
            d = np.array([0.0 if picks[k] == meta[k]["stock"] else f(k, picks[k]) - f(k, meta[k]["stock"]) for k in sorted(by)])
            if name == "sonnet":
                base = d
            nz = d[d != 0]
            ci = boot(d)
            w = np.mean([words(picks[k]) for k in by])
            same_s = sum(picks[k] == meta[k]["pick"] for k in by)
            gap = d - base
            gap_txt = "" if name == "sonnet" else f" | vs Sonnet {gap.mean():+.3f} ({boot(gap)[0]:+.3f} to {boot(gap)[1]:+.3f})"
            print(f"{name:44s} gain {d.mean():+.3f} ({ci[0]:+.3f} to {ci[1]:+.3f}) p={wilcoxon(nz).pvalue:.3g} wins {np.mean(nz > 0):.0%} words {w:.1f} same-as-Sonnet {same_s}{gap_txt}")
            out.setdefault(c, {})[name] = dict(gain=float(d.mean()), ci=ci, p=float(wilcoxon(nz).pvalue), wins=float(np.mean(nz > 0)), words=float(w), gap=float(gap.mean()), gap_ci=boot(gap))
        print(f"stock words {np.mean([meta[k]['stock_words'] for k in by]):.1f}, clusters {len(by)}")
    rows = [json.loads(l) for l in (SP / "dm_rate_results.jsonl").read_text().splitlines()]
    if (SP / "dm_results.jsonl").exists():
        screen = [json.loads(l) for l in (SP / "dm_results.jsonl").read_text().splitlines()]
        print(f"\nscreen: {len(screen)} requests, {sum(r['input_tokens'] for r in screen):,} input tokens, reported cost ${sum(r.get('cost') or 0 for r in screen):.2f}")
    print(f"sonnet rating: {len(rows)} calls, {sum(r['input_tokens'] for r in rows):,} input tokens, {sum(r['scores'] is None for r in rows)} failed")
    (SP / "dm_summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    {"rank": rank, "rate": rate, "report": report}[sys.argv[2]]()
