"""
Does scoring candidates on 20 documents per group beat scoring them on 5?

  python more_docs.py <scratchpad> score --cap-usd 1.50    three extra lineups per label, two models
  python more_docs.py <scratchpad> rate                    Sonnet rates new picks on the fresh draw
  python more_docs.py <scratchpad> report                  applies the success rule

A label's 20-document score is the mean of its original lineup score and its three new ones, one
call each with its own candidate order. Picks use the rule from the screen: top score, ties to fewer
words, then alphabetical order. Success rule, fixed before the run: for each model, 20-document
gain minus 5-document gain over both corpora, with a 95% bootstrap interval excluding zero.
"""
import argparse
import asyncio
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

SP = Path(sys.argv[1])

import dm_analyze as da  # noqa: E402
import dm_screen as ds  # noqa: E402
from haiku_test import SONNET, Budget, call, digest, frac, in_subset, words  # noqa: E402

TOP = [("typesafe/jev-1.14", "lineup"), ("liquid/d1-ultra", "lineup")]
RATE_CAP = 3_400_000
CELLS = ("20ng", "arxiv_home")
LETTERS = ds.LETTERS


def lineup_request(u, draw, groups):
    ids = u["lineup"]
    rng = np.random.default_rng(ds.md5i(f"{u['uid']}|d{draw}"))
    disp = [ids[j] for j in rng.permutation(len(ids))]
    state = {"label": u["label"],
             "candidate_groups": {LETTERS[a]: groups[f"{u['L']}|{j}"][draw] for a, j in enumerate(disp)}}
    q = {"q": {"type": "choice", "instructions": "Which candidate group of documents does the label name?",
               "criteria": {LETTERS[a]: f"Candidate group {LETTERS[a]}" for a in range(len(disp))}}}
    return state, q, LETTERS[disp.index(u["true"])]


def score(cap, concurrency):
    data = ds.load(SP)
    extra = json.loads((SP / "more_groups.json").read_text())
    path = SP / "more_results.jsonl"
    done, spent = set(), 0.0
    if path.exists():
        for r in map(json.loads, path.read_text().splitlines()):
            spent += r["input_tokens"] * ds.MODELS[r["model"]] / 1e6
            if r["p"] is not None:
                done.add((r["model"], r["uid"], r["doc"]))
    jobs = []
    for model, fmt in TOP:
        for draw in range(3):
            for cell in CELLS:
                for u in data[cell]["units"]:
                    if (model, u["uid"], draw) in done:
                        continue
                    state, q, true = lineup_request(u, draw, extra[cell]["groups"])
                    jobs.append((model, fmt, cell, u, draw, state, q, true))
    random.Random(0).shuffle(jobs)
    print(f"{len(done)} done, {len(jobs)} to run, ${spent:.3f} spent, cap ${cap:.2f}", flush=True)
    asyncio.run(ds.run(jobs, path, concurrency, dict(spent=spent, cap=cap)))


def twenty_doc_scores():
    """(model, cell) -> uid -> mean of the original score and the three new ones."""
    base, _, _ = da.load_scores()
    extra = collections.defaultdict(list)
    if (SP / "more_results.jsonl").exists():
        for r in map(json.loads, (SP / "more_results.jsonl").read_text().splitlines()):
            if r["p"] is not None:
                extra[(r["model"], r["cell"], r["uid"])].append(r["p"])
    else:  # the raw file is not committed; more_scores.json keeps each label's scores in order
        for key, labels in json.loads((SP / "more_scores.json").read_text()).items():
            model, cell = key.split("|")
            for uid, values in labels.items():
                extra[(model, cell, uid)] = values
    out = {}
    missing = 0
    for model, fmt in TOP:
        for cell in CELLS:
            five = base[(model, fmt, cell)]
            s = {}
            for uid, v in five.items():
                more = extra.get((model, cell, uid), [])
                missing += len(more) != 3
                s[uid] = float(np.mean([v] + more))
            out[(model, cell)] = (five, s)
    return out, missing


def fresh_scores():
    pairs, samples = da.fresh_scores()
    path = SP / "more_rate_results.jsonl"
    if path.exists():
        for r in map(json.loads, path.read_text().splitlines()):
            if r["scores"] is not None:
                samples[(r["cell"], r["L"], r["i"], r["label"])].append(frac(r, r["i"]))
    return pairs, samples


def rate():
    scores, missing = twenty_doc_scores()
    assert missing == 0, f"{missing} labels lack all three extra scores"
    units = json.loads((SP / "haiku_units.json").read_text())
    keep = in_subset(SP)
    pairs, samples = fresh_scores()
    fresh = json.loads((SP / "fresh_units.json").read_text())
    jobs, tally, queued = collections.defaultdict(list), collections.Counter(), set()
    for model, _ in TOP:
        for c in CELLS:
            by = da.clusters(c, units, keep)
            stock = {(p["L"], p["i"]): p["stock"] for p in pairs[c]["pairs"]}
            _, s20 = scores[(model, c)]
            for key, us in by.items():
                pick = da.choose(us, s20)
                if pick == stock[key]:
                    tally["pick is stock"] += 1
                    continue
                for label in (pick, stock[key]):
                    k = (c, *key, label)
                    if len(samples.get(k, [])) >= 3:
                        tally["already rated"] += 1
                    elif k in queued:
                        tally["shared with the other model"] += 1
                    else:
                        queued.add(k)
                        tally["needs rating"] += 1
                        uid = f"fresh|L{key[0]}.{key[1]}|fresh-more|{digest(label)}"
                        for s in range(3):
                            jobs[c].append((dict(uid=uid, L=key[0], i=key[1], true=key[1], label=label, lineup=us[0]["lineup"], cell=c), s))
    n = sum(len(v) for v in jobs.values())
    print(dict(tally), f"Sonnet calls {n}, about {n * 2813 / 1e6:.2f}M input tokens, cap {RATE_CAP / 1e6:.1f}M", flush=True)
    assert n * 2813 < RATE_CAP, "estimate exceeds the Sonnet cap"
    groups = {c: fresh[c]["groups"] for c in fresh}

    async def go():
        budget = Budget(0, RATE_CAP)
        sem = asyncio.Semaphore(12)
        stats = dict(done=0, failed=0, capped=0)
        with (SP / "more_rate_results.jsonl").open("a") as fh:
            await asyncio.gather(*[call(SONNET, j, groups[c], sem, budget, fh, stats) for c, v in jobs.items() for j in v])
        print(f"finished {stats}; {budget.used:,} input tokens", flush=True)

    asyncio.run(go())


def report():
    scores, missing = twenty_doc_scores()
    units = json.loads((SP / "haiku_units.json").read_text())
    keep = in_subset(SP)
    pairs, samples = fresh_scores()
    unique = json.loads((SP / "more_unique.json").read_text())
    out = {}
    print(f"labels missing an extra score: {missing}")
    for model, _ in TOP:
        print(f"\n========== {model} ==========")
        pooled = dict(g5=[], g20=[], d=[], own15=[], rich=[])
        per = {}
        for c in CELLS:
            by = da.clusters(c, units, keep)
            meta = {(p["L"], p["i"]): p for p in pairs[c]["pairs"]}
            five, twenty = scores[(model, c)]
            f = lambda key, lab: float(np.mean(samples[(c, *key, lab)][:3]))
            g5, g20, same, w5, w20, rho5, rho20, rich = [], [], 0, [], [], [], [], []
            for key in sorted(by):
                us = by[key]
                stock = meta[key]["stock"]
                p5, p20 = da.choose(us, five), da.choose(us, twenty)
                g5.append(0.0 if p5 == stock else f(key, p5) - f(key, stock))
                g20.append(0.0 if p20 == stock else f(key, p20) - f(key, stock))
                same += p5 == p20
                w5.append(words(p5))
                w20.append(words(p20))
                rho5.append([(five[u["uid"]], np.mean(u["sonnet"])) for u in us])
                rho20.append([(twenty[u["uid"]], np.mean(u["sonnet"])) for u in us])
                rich.append(unique[c][f"{key[0]}|{key[1]}"] >= 15)
            g5, g20 = np.array(g5), np.array(g20)
            d = g20 - g5
            nz = d[d != 0]
            r5, r20 = da.within(rho5)[0], da.within(rho20)[0]
            print(f"{c}: {len(d)} clusters, pick changed in {len(d) - same}; gain 5 docs {g5.mean():+.3f}, 20 docs {g20.mean():+.3f}, "
                  f"difference {d.mean():+.3f} ({da.boot(d)[0]:+.3f} to {da.boot(d)[1]:+.3f}), signed-rank p={wilcoxon(nz).pvalue:.3g}; "
                  f"words {np.mean(w5):.1f} to {np.mean(w20):.1f}; agreement with Sonnet {r5:+.3f} to {r20:+.3f}")
            per[c] = dict(clusters=len(d), changed=len(d) - same, gain5=float(g5.mean()), gain20=float(g20.mean()), diff=float(d.mean()),
                          ci=da.boot(d), words5=float(np.mean(w5)), words20=float(np.mean(w20)), agree5=r5, agree20=r20)
            pooled["g5"].extend(g5)
            pooled["g20"].extend(g20)
            pooled["d"].extend(d)
            pooled["rich"].extend(rich)
        d = np.array(pooled["d"])
        rich = np.array(pooled["rich"])
        ci = da.boot(d)
        verdict = "HELPS" if ci[0] > 0 else ("HURTS" if ci[1] < 0 else "NOT SHOWN")
        nz = d[d != 0]
        print(f"PRIMARY pooled: gain 5 docs {np.mean(pooled['g5']):+.3f}, 20 docs {np.mean(pooled['g20']):+.3f}, "
              f"difference {d.mean():+.3f} (95% CI {ci[0]:+.3f} to {ci[1]:+.3f}), signed-rank p={wilcoxon(nz).pvalue:.3g} -> {verdict}")
        print(f"clusters with the full 15 extra documents: {d[rich].mean():+.3f} (n={rich.sum()}), the rest {d[~rich].mean():+.3f} (n={(~rich).sum()})")
        out[model] = dict(per=per, gain5=float(np.mean(pooled["g5"])), gain20=float(np.mean(pooled["g20"])), diff=float(d.mean()), ci=ci,
                          p=float(wilcoxon(nz).pvalue), verdict=verdict, rich_diff=float(d[rich].mean()), other_diff=float(d[~rich].mean()))
    rate_rows = [json.loads(l) for l in (SP / "more_rate_results.jsonl").read_text().splitlines()]
    if (SP / "more_results.jsonl").exists():
        rows = [json.loads(l) for l in (SP / "more_results.jsonl").read_text().splitlines()]
        served = collections.Counter(str(r.get("served")) for r in rows)
        print(f"\nextra scoring: {len(rows)} requests, {sum(r['input_tokens'] for r in rows):,} input tokens, ${sum(r.get('cost') or 0 for r in rows):.3f}, "
              f"{sum(r['p'] is None for r in rows)} failed; served {dict(served)}")
    i, o = sum(r["input_tokens"] for r in rate_rows), sum(r["output_tokens"] for r in rate_rows)
    print(f"sonnet rating: {len(rate_rows)} calls, {i:,} input tokens, {o:,} output tokens, {sum(r['scores'] is None for r in rate_rows)} failed, ${i * 3 / 1e6 + o * 15 / 1e6:.2f}")
    (SP / "more_summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    if sys.argv[2] == "score":
        ap = argparse.ArgumentParser()
        ap.add_argument("scratchpad")
        ap.add_argument("stage")
        ap.add_argument("--cap-usd", type=float, required=True)
        ap.add_argument("--concurrency", type=int, default=16)
        a = ap.parse_args()
        score(a.cap_usd, a.concurrency)
    else:
        {"rate": rate, "report": report}[sys.argv[2]]()
