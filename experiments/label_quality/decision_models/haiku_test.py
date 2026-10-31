"""
Can Haiku stand in for Sonnet as the scorer that picks among candidates?

  prep      rebuild the pooled candidates, and check they reproduce every Sonnet pick
  score     Haiku scores every candidate once on the frozen documents
  rate      Sonnet rates Haiku's picks on the existing fresh draw, reusing any score already paid for
  analyze   compare the fresh-document gain with Sonnet's own picks

Same prompt, temperature, token limit and structured output as the July protocol. Haiku's pick is
its top score for the cluster; ties go to fewer words, then alphabetical order, so nothing depends
on call order. Results append to JSONL and resume. A hard cap covers all input tokens across both
models in this test. argv[1] is the scratchpad.
"""
import asyncio
import collections
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from fresh_run import LETTERS, MAX_TOKENS, PROMPT, RESPONSE_FORMAT, TEMP, md5i, parse_scores

REPO = Path("/Users/stevenfazzio/repos/toponymy")
B = "experiment/naming-features:experiments/label_quality/data/"
HAIKU = "anthropic/claude-haiku-4-5-20251001"
SONNET = "anthropic/claude-sonnet-4-6"
CAP = 9_000_000
RESERVE_FOR_RATING = 2_050_000  # worst case for the subset: every pick plus extra stock labels
EST_CALL = 4_700  # at least the largest call, so the cap cannot be overshot by in-flight work
HAIKU_TARGET = 6_780_000
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


def digest(label):
    return hashlib.md5(label.encode()).hexdigest()[:10]


def words(s):
    return len(s.split())


def pool_for(cell):
    pool = collections.defaultdict(dict)
    relabelled = {"features", "subtopics"} if cell == "arxiv_home" else set()

    def add(name, keep=lambda u: True):
        data = saved(f"wayfinding_{cell}_{name}.json")
        for u in (data or {"units": {}})["units"].values():
            if usable(u) and keep(u):
                i = SWAP.get(u["i"], u["i"]) if name in relabelled and u["L"] == 1 else u["i"]
                pool[(u["L"], i)][u["label"]] = fracs(u)

    add("ladder")
    add("battery", lambda u: u["kind"] == "gold")
    add("features")
    add("subtopics")
    if cell == "20ng":
        add("pairs")
    return pool


def prep(sp):
    pairs = json.loads((sp / "fresh_pairs.json").read_text())
    out = {}
    for cell, v in pairs.items():
        pool = pool_for(cell)
        data = json.loads((sp / f"lineups_{cell}.json").read_text())
        line = {}
        for u in data["units"]["ladder"]:
            line.setdefault((u["L"], u["i"]), u["lineup"])
        bad = 0
        for p in v["pairs"]:
            key = (p["L"], p["i"])
            ranked = sorted(pool[key].items(), key=lambda kv: (-float(np.mean(kv[1])), words(kv[0]), kv[0]))
            bad += ranked[0][0] != p["pick"] or len(pool[key]) != p["n_cands"]
        units = []
        for (L, i), cands in sorted(pool.items()):
            for label in sorted(cands):
                units.append(dict(uid=f"haiku|L{L}.{i}|{digest(label)}", L=L, i=i, true=i, label=label,
                                  lineup=line[(L, i)], sonnet=cands[label]))
        print(f"{cell}: {len(pool)} clusters, {len(units)} candidates, Sonnet picks not reproduced: {bad}", flush=True)
        assert bad == 0
        out[cell] = units
    (sp / "haiku_units.json").write_text(json.dumps(out))
    n = sum(len(u) for u in out.values())
    print(f"total Haiku calls {n}, about {n * 2950 / 1e6:.2f}M input tokens at Sonnet's rate", flush=True)


def subset(sp):
    """A fixed-seed, layer-stratified share of clusters that fits under the cap at Haiku's real rate.
    Calibrated by corpus from the pilot, in tokens per prompt character."""
    units = json.loads((sp / "haiku_units.json").read_text())
    rows = [json.loads(l) for l in (sp / "haiku_results.jsonl").read_text().splitlines()]
    groups = {c: json.loads((sp / f"lineups_{c}.json").read_text())["groups"] for c in units}

    def chars(c, u):
        return len(PROMPT) + len(u["label"]) + sum(len(d) + 3 for j in u["lineup"] for d in groups[c][f"{u['L']}|{j}"])

    rate_by_cell = {}
    for c, us in units.items():
        by_uid = {u["uid"]: u for u in us}
        seen = [r for r in rows if r["cell"] == c]
        rate_by_cell[c] = sum(r["input_tokens"] for r in seen) / sum(chars(c, by_uid[r["uid"]]) for r in seen)
    cost = collections.defaultdict(float)
    for c, us in units.items():
        for u in us:
            cost[(c, u["L"], u["i"])] += chars(c, u) * rate_by_cell[c]
    strata = collections.defaultdict(list)
    for key in sorted(cost):
        strata[key[:2]].append(key)
    for keys in strata.values():
        keys.sort(key=lambda k: md5i(f"subset|{k}"))

    def choose(f):
        return [k for keys in strata.values() for k in keys[: int(np.ceil(f * len(keys)))]]

    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if sum(cost[k] for k in choose(mid)) <= HAIKU_TARGET:
            lo = mid
        else:
            hi = mid
    keep = choose(lo)
    full = sum(cost.values())
    print(f"tokens per char {({c: round(v, 4) for c, v in rate_by_cell.items()})}; full Haiku estimate {full / 1e6:.2f}M", flush=True)
    print(f"keeping share {lo:.3f}: {len(keep)} of {len(cost)} clusters, Haiku estimate {sum(cost[k] for k in keep) / 1e6:.2f}M", flush=True)
    print({f"{c} L{L}": f"{sum(1 for k in keep if k[:2] == (c, L))}/{len(strata[(c, L)])}" for c, L in strata}, flush=True)
    (sp / "haiku_subset.json").write_text(json.dumps(sorted(keep)))


def in_subset(sp):
    return {tuple(k) for k in json.loads((sp / "haiku_subset.json").read_text())}


class Budget:
    def __init__(self, used, limit):
        self.used, self.limit, self.reserved = used, limit, 0

    def take(self):
        if self.used + self.reserved + EST_CALL > self.limit:
            return False
        self.reserved += EST_CALL
        return True

    def settle(self, actual):
        self.reserved -= EST_CALL
        self.used += actual


def spent(sp):
    total = 0
    for name in ("haiku_results.jsonl", "haiku_rate_results.jsonl"):
        path = sp / name
        if path.exists():
            total += sum(json.loads(l)["input_tokens"] for l in path.read_text().splitlines())
    return total


async def call(model, job, groups, sem, budget, fh, stats):
    import litellm

    u, s = job
    async with sem:
        if not budget.take():
            stats["capped"] += 1
            return
        cids = u["lineup"]
        k = len(cids)
        rng = np.random.default_rng(md5i(f"{u['uid']}|s{s}"))
        disp = [cids[j] for j in rng.permutation(k)]
        blocks = "\n\n".join(
            f"Candidate {LETTERS[a]}:\n" + "\n".join(f"- {d}" for d in groups[f"{u['L']}|{j}"])
            for a, j in enumerate(disp))
        span = f"{LETTERS[0]}-{LETTERS[k - 1]}"
        prompt = PROMPT.format(label=u["label"], k=k, span=span, blocks=blocks)
        rec = dict(cell=u["cell"], uid=u["uid"], L=u["L"], i=u["i"], label=u["label"], sample=s, order=disp,
                   scores=None, input_tokens=0, output_tokens=0)
        try:
            r = await litellm.acompletion(model=model, temperature=TEMP, max_tokens=MAX_TOKENS, num_retries=5,
                                          response_format=RESPONSE_FORMAT,
                                          messages=[{"role": "user", "content": prompt}])
            rec["input_tokens"] = int(r.usage.prompt_tokens)
            rec["output_tokens"] = int(r.usage.completion_tokens)
            rec["model"] = r.model
            rec["scores"] = parse_scores(r.choices[0].message.content or "", k)
            if rec["scores"] is None:
                rec["error"] = "unparseable"
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        budget.settle(rec["input_tokens"])
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        stats["done"] += 1
        stats["failed"] += rec["scores"] is None
        if stats["done"] % 250 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] {stats['done']} calls, {budget.used / 1e6:.2f}M input tokens so far, "
                  f"{stats['failed']} failed", flush=True)


async def run_calls(model, jobs_by_cell, groups, path, limit, concurrency=12):
    budget = Budget(spent(path.parent), limit)
    sem = asyncio.Semaphore(concurrency)
    stats = dict(done=0, failed=0, capped=0)
    n = sum(len(j) for j in jobs_by_cell.values())
    print(f"{model}: {n} calls to run, {budget.used / 1e6:.3f}M input tokens spent so far in this test, limit {limit / 1e6:.2f}M", flush=True)
    with path.open("a") as fh:
        await asyncio.gather(*[call(model, j, groups[c], sem, budget, fh, stats)
                               for c, jobs in jobs_by_cell.items() for j in jobs])
    print(f"finished: {stats}; {budget.used:,} input tokens spent in this test", flush=True)


def done_uids(path):
    if not path.exists():
        return set()
    return {(r["uid"], r["sample"]) for r in map(json.loads, path.read_text().splitlines()) if r["scores"] is not None}


def score(sp, limit=None):
    units = json.loads((sp / "haiku_units.json").read_text())
    groups = {c: json.loads((sp / f"lineups_{c}.json").read_text())["groups"] for c in units}
    path = sp / "haiku_results.jsonl"
    done = done_uids(path)
    keep = in_subset(sp) if (sp / "haiku_subset.json").exists() else None
    jobs = {c: [(dict(u, cell=c), 0) for u in us if (u["uid"], 0) not in done and (keep is None or (c, u["L"], u["i"]) in keep)]
            for c, us in units.items()}
    if limit:
        jobs = {c: j[: limit // 2] for c, j in jobs.items()}
    asyncio.run(run_calls(HAIKU, jobs, groups, path, CAP - RESERVE_FOR_RATING))


def frac(r, true):
    total = sum(r["scores"]) or 1
    return r["scores"][r["order"].index(true)] / total


def haiku_scores(sp):
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in map(json.loads, (sp / "haiku_results.jsonl").read_text().splitlines()):
        if r["scores"] is not None:
            out[r["cell"]][(r["L"], r["i"])][r["label"]] = frac(r, r["i"])
    return out


def picks_from(sp):
    keep = in_subset(sp)
    units = json.loads((sp / "haiku_units.json").read_text())
    expected = collections.Counter((c, u["L"], u["i"]) for c, us in units.items() for u in us)
    out = {}
    for c, cl in haiku_scores(sp).items():
        out[c] = {}
        for k, v in cl.items():
            if (c, *k) not in keep:
                continue
            assert len(v) == expected[(c, *k)], f"cluster {c} {k} is only partly scored"
            out[c][k] = sorted(v.items(), key=lambda kv: (-kv[1], words(kv[0]), kv[0]))[0][0]
    return out


def rate(sp):
    pairs = json.loads((sp / "fresh_pairs.json").read_text())
    fresh = json.loads((sp / "fresh_units.json").read_text())
    picks = picks_from(sp)
    have = {r["uid"] for r in map(json.loads, (sp / "fresh_results.jsonl").read_text().splitlines()) if r["scores"] is not None}
    path = sp / "haiku_rate_results.jsonl"
    done = done_uids(path)
    jobs, tally = {}, collections.Counter()
    for c, v in pairs.items():
        line = {(u["L"], u["i"]): u["lineup"] for u in fresh[c]["units"]}
        units = json.loads((sp / "haiku_units.json").read_text())[c]
        for u in units:
            line.setdefault((u["L"], u["i"]), u["lineup"])
        jobs[c] = []
        for p in v["pairs"]:
            key = (p["L"], p["i"])
            if key not in picks[c]:
                continue
            pick, stock = picks[c][key], p["stock"]
            if pick == stock:
                tally["pick is stock, no calls"] += 1
                continue
            if pick == p["pick"]:
                tally["same as Sonnet's pick, already rated"] += 1
            else:
                tally["new pick needs rating"] += 1
                uid = f"fresh|L{key[0]}.{key[1]}|fresh-haiku|{digest(pick)}"
                for s in range(3):
                    if (uid, s) not in done:
                        jobs[c].append((dict(uid=uid, L=key[0], i=key[1], true=key[1], label=pick, lineup=line[key], cell=c), s))
            uid = f"fresh|L{key[0]}.{key[1]}|fresh-stock|{digest(stock)}"
            if uid not in have:
                tally["stock not yet rated"] += 1
                for s in range(3):
                    if (uid, s) not in done:
                        jobs[c].append((dict(uid=uid, L=key[0], i=key[1], true=key[1], label=stock, lineup=line[key], cell=c), s))
    print(dict(tally), flush=True)
    groups = {c: fresh[c]["groups"] for c in fresh}
    asyncio.run(run_calls(SONNET, jobs, groups, path, CAP))


if __name__ == "__main__":
    sp = Path(sys.argv[1])
    stage = sys.argv[2]
    if stage == "prep":
        prep(sp)
    elif stage == "subset":
        subset(sp)
    elif stage == "score":
        score(sp, int(sys.argv[3]) if len(sys.argv) > 3 else None)
    elif stage == "rate":
        rate(sp)
