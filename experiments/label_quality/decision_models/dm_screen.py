"""
Screen decision models as label scorers through OpenRouter's Decisions API.

Two formats, each on the frozen documents for the same 197 clusters Haiku used:
  lineup   one Choice over the k document groups; score = P(true group)
  labels   each of the true cluster's documents in turn, Choice over the candidate and its
           neighbours' stock labels; score = mean P(candidate)

  python dm_screen.py <scratchpad> pilot
  python dm_screen.py <scratchpad> screen --cap-usd X

Results append to JSONL and resume. Throttling and server errors retry with backoff; other client
errors fail immediately and are recorded. The key is read from the environment and never logged.
"""
import argparse
import asyncio
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np

URL = "https://openrouter.ai/api/alpha/decisions"
MODELS = {
    "typesafe/jev-1.14": 0.042,
    "perplexity/pplx-decider-v1.2": 0.02,
    "liquid/d1-ultra": 0.042,
    "microsoft/microsoft-decision-1": 0.042,
}
FORMATS = ("lineup", "labels")
LETTERS = "ABCDEFG"
RETRY = {429, 500, 502, 503, 504}


def md5i(s):
    import hashlib

    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def load(sp):
    units = json.loads((sp / "haiku_units.json").read_text())
    keep = {tuple(k) for k in json.loads((sp / "haiku_subset.json").read_text())}
    data = {}
    for cell, us in units.items():
        d = json.loads((sp / f"lineups_{cell}.json").read_text())
        stock = {(r["L"], r["i"]): r["stock"] for r in d["rows"]}
        data[cell] = dict(units=[u for u in us if (cell, u["L"], u["i"]) in keep], groups=d["groups"], stock=stock)
    return data


def requests_for(cell, u, fmt, data):
    """Yield (doc_index, state, questions, true_option) for one label in one format."""
    groups = data[cell]["groups"]
    if fmt == "lineup":
        ids = u["lineup"]
        rng = np.random.default_rng(md5i(f"{u['uid']}|s0"))
        disp = [ids[j] for j in rng.permutation(len(ids))]
        state = {"label": u["label"],
                 "candidate_groups": {LETTERS[a]: groups[f"{u['L']}|{j}"] for a, j in enumerate(disp)}}
        q = {"q": {"type": "choice", "instructions": "Which candidate group of documents does the label name?",
                   "criteria": {LETTERS[a]: f"Candidate group {LETTERS[a]}" for a in range(len(disp))}}}
        yield 0, state, q, LETTERS[disp.index(u["true"])]
        return
    names = [u["label"]]
    for j in u["lineup"][1:]:
        s = data[cell]["stock"][(u["L"], j)]
        if s not in names:
            names.append(s)
    random.Random(md5i(f"{u['uid']}|o0")).shuffle(names)
    criteria = {LETTERS[a]: n for a, n in enumerate(names)}
    true = LETTERS[names.index(u["label"])]
    q = {"q": {"type": "choice", "instructions": "Which label best names the topic of this document?", "criteria": criteria}}
    for n, doc in enumerate(groups[f"{u['L']}|{u['true']}"]):
        yield n, {"document": doc}, q, true


async def post(client, body):
    delay = 2.0
    for attempt in range(6):
        try:
            r = await client.post(URL, json=body)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            if attempt == 5:
                return None, f"{type(e).__name__}"
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)
            continue
        if r.status_code in RETRY and attempt < 5:
            wait = r.headers.get("Retry-After", "")
            await asyncio.sleep(float(wait) if wait.isdigit() else delay + random.uniform(0, delay * 0.1))
            delay = min(delay * 2, 30)
            continue
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}: {r.text[:200]}"
        return r.json(), None
    return None, "exhausted"


async def one(client, sem, job, fh, stats, budget):
    model, fmt, cell, u, doc, state, q, true = job
    async with sem:
        if budget is not None and budget["spent"] >= budget["cap"]:
            stats["capped"] += 1
            return
        t0 = time.time()
        resp, err = await post(client, dict(model=model, state=state, questions=q))
        rec = dict(model=model, fmt=fmt, cell=cell, uid=u["uid"], doc=doc, p=None, input_tokens=0, secs=round(time.time() - t0, 3))
        if err:
            rec["error"] = err
        else:
            usage = resp.get("usage") or {}
            rec["input_tokens"] = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            rec["cost"] = usage.get("cost")
            rec["served"] = resp.get("model")
            try:
                a = resp["answers"]["q"]
                probs = a["probabilities"]
                rec["p"] = float(probs[true])
                rec["sum"] = round(float(sum(probs.values())), 4)
                rec["confidence"] = a.get("confidence")
            except Exception as e:
                rec["error"] = f"bad answer {type(e).__name__}: {json.dumps(resp)[:200]}"
        if budget is not None:
            budget["spent"] += rec["input_tokens"] * MODELS[model] / 1e6
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        stats["done"] += 1
        stats["failed"] += rec["p"] is None
        if stats["done"] % 2000 == 0:
            cost = f", ${budget['spent']:.3f} spent" if budget else ""
            print(f"[{time.strftime('%H:%M:%S')}] {stats['done']} requests, {stats['failed']} failed{cost}", flush=True)


async def run(jobs, path, concurrency, budget=None):
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "X-Title": "toponymy label eval"}
    sem = asyncio.Semaphore(concurrency)
    stats = dict(done=0, failed=0, capped=0)
    async with httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(90.0)) as client:
        with path.open("a") as fh:
            await asyncio.gather(*[one(client, sem, j, fh, stats, budget) for j in jobs])
    print(f"finished {stats}", flush=True)


def pilot(sp, data):
    jobs = []
    for model in MODELS:
        for fmt in FORMATS:
            for cell in data:
                for u in random.Random(3).sample(data[cell]["units"], 3):
                    for doc, state, q, true in requests_for(cell, u, fmt, data):
                        jobs.append((model, fmt, cell, u, doc, state, q, true))
            # one repeat, to see whether identical requests give identical answers
            u = data["20ng"]["units"][0]
            doc, state, q, true = next(iter(requests_for("20ng", u, fmt, data)))
            jobs += [(model, fmt, "20ng", dict(u, uid=u["uid"] + "|rep"), doc, state, q, true) for _ in range(2)]
    print(f"pilot requests {len(jobs)}", flush=True)
    asyncio.run(run(jobs, sp / "dm_pilot.jsonl", 8))
    rows = [json.loads(l) for l in (sp / "dm_pilot.jsonl").read_text().splitlines()]
    n_labels = {c: len(d["units"]) for c, d in data.items()}
    print(f"labels to score {n_labels}", flush=True)
    total, secs = 0.0, 0.0
    for model, price in MODELS.items():
        for fmt in FORMATS:
            r = [x for x in rows if x["model"] == model and x["fmt"] == fmt]
            ok = [x for x in r if x["p"] is not None]
            line = f"{model:34s} {fmt:7s} ok {len(ok)}/{len(r)}"
            if not ok:
                errs = {(x.get('error') or '')[:150] for x in r}
                print(line, "ERRORS", list(errs)[:2], flush=True)
                continue
            est = 0
            for cell, n in n_labels.items():
                sub = [x for x in ok if x["cell"] == cell]
                per = 1 if fmt == "lineup" else 5
                est += n * per * float(np.mean([x["input_tokens"] for x in sub]))
            usd = est * price / 1e6
            sec = float(np.median([x["secs"] for x in ok]))
            reqs = sum(n_labels.values()) * (1 if fmt == "lineup" else 5)
            rep = sorted({x["p"] for x in ok if x["uid"].endswith("|rep")})
            total += usd
            secs += reqs * sec
            print(f"{line} tokens/req {np.mean([x['input_tokens'] for x in ok]):7.0f} median {sec:.2f}s "
                  f"est {est / 1e6:.2f}M tok ${usd:.2f} sum {sorted({x.get('sum') for x in ok})[:3]} repeats {rep[:3]} "
                  f"served {sorted({str(x.get('served')) for x in ok})[:1]} reported_cost {[x.get('cost') for x in ok][:1]}", flush=True)
    print(f"screen estimate ${total:.2f}; {sum(n_labels.values())} labels; serial {secs / 3600:.1f}h, at 16 concurrent {secs / 16 / 60:.0f} min", flush=True)
    print(f"pilot spent ${sum(x['input_tokens'] * MODELS[x['model']] / 1e6 for x in rows):.4f}", flush=True)


def screen(sp, data, cap, concurrency):
    path = sp / "dm_results.jsonl"
    done = set()
    spent = 0.0
    if path.exists():
        for r in map(json.loads, path.read_text().splitlines()):
            spent += r["input_tokens"] * MODELS[r["model"]] / 1e6
            if r["p"] is not None:
                done.add((r["model"], r["fmt"], r["uid"], r["doc"]))
    jobs = []
    for model in MODELS:
        for fmt in FORMATS:
            for cell in data:
                for u in data[cell]["units"]:
                    for doc, state, q, true in requests_for(cell, u, fmt, data):
                        if (model, fmt, u["uid"], doc) not in done:
                            jobs.append((model, fmt, cell, u, doc, state, q, true))
    random.Random(0).shuffle(jobs)
    print(f"{len(done)} done, {len(jobs)} to run, ${spent:.3f} spent, cap ${cap:.2f}", flush=True)
    asyncio.run(run(jobs, path, concurrency, dict(spent=spent, cap=cap)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scratchpad")
    ap.add_argument("stage", choices=["pilot", "screen"])
    ap.add_argument("--cap-usd", type=float, default=None)
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()
    sp = Path(args.scratchpad)
    data = load(sp)
    if args.stage == "pilot":
        pilot(sp, data)
    else:
        assert args.cap_usd is not None, "screen needs --cap-usd"
        screen(sp, data, args.cap_usd, args.concurrency)
