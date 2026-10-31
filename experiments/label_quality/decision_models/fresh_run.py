"""
Run the fresh-document lineups through Sonnet with the July protocol unchanged.

Same prompt, model, temperature, token limit and structured output as wayfinding.py, three samples
per label with reseeded candidate order. Results append to fresh_results.jsonl, so a rerun resumes.
A hard cap stops new calls before total input tokens can pass CAP, counting tokens from any earlier
run in the file. Failed calls are recorded and not retried beyond litellm's own retries.

  python fresh_run.py <scratchpad> [--limit N]
"""
import argparse
import asyncio
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

MODEL = "anthropic/claude-sonnet-4-6"
TEMP, K_SAMPLES, MAX_TOKENS = 0.7, 3, 300
CAP = 5_000_000
EST_CALL = 3_900  # reserve per in-flight call when checking the cap
LETTERS = "ABCDEFG"

PROMPT = """You will identify which group of documents a LABEL names.

LABEL: "{label}"

Below are {k} candidate document groups ({span}). Each is shown as a few member documents \
(truncated). Exactly one of these groups is the one the label was written to name.

{blocks}

For EACH candidate group, rate from 0 to 100 how well the LABEL fits that group as a name, \
judging only from the documents shown. Be discriminating: reserve high scores for a group the \
label names genuinely and specifically; if the label fits several groups only in a generic way, \
give those groups similar middling scores.

Give your ratings as {k} integers in candidate order ({span})."""

SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {"scores": {"type": "array", "items": {"type": "integer"}}},
          "required": ["scores"]}
RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "scores", "schema": SCHEMA, "strict": True}}


def md5i(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def parse_scores(text, k):
    try:
        obj = json.loads(text)
        sc = obj["scores"] if isinstance(obj, dict) else obj
    except Exception:
        m = re.search(r"\[[\d\s,]+\]", text or "")
        if not m:
            return None
        try:
            sc = json.loads(m.group(0))
        except Exception:
            return None
    try:
        sc = [max(0, min(100, int(x))) for x in sc]
    except Exception:
        return None
    return (sc + [0] * k)[:k]


class Budget:
    def __init__(self, used):
        self.used, self.reserved = used, 0

    def take(self):
        if self.used + self.reserved + EST_CALL > CAP:
            return False
        self.reserved += EST_CALL
        return True

    def settle(self, actual):
        self.reserved -= EST_CALL
        self.used += actual


async def one(job, groups, sem, budget, fh, stats):
    import litellm

    cell, u, s = job
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
        rec = dict(cell=cell, uid=u["uid"], sample=s, order=disp, scores=None, input_tokens=0, output_tokens=0)
        try:
            r = await litellm.acompletion(model=MODEL, temperature=TEMP, max_tokens=MAX_TOKENS, num_retries=5,
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
        if stats["done"] % 100 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] {stats['done']} calls, {budget.used / 1e6:.2f}M input tokens, "
                  f"{stats['failed']} failed", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scratchpad")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=12)
    args = ap.parse_args()
    sp = Path(args.scratchpad)
    data = json.loads((sp / "fresh_units.json").read_text())
    path = sp / "fresh_results.jsonl"
    done, used = set(), 0
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            used += r["input_tokens"]
            if r["scores"] is not None:
                done.add((r["uid"], r["sample"]))
    jobs = [(c, u, s) for s in range(K_SAMPLES) for c, v in data.items() for u in v["units"]
            if (u["uid"], s) not in done]
    if args.limit:
        jobs = jobs[: args.limit]
    budget = Budget(used)
    print(f"{len(done)} calls already done, {used / 1e6:.3f}M input tokens used; running {len(jobs)}; cap {CAP / 1e6:.1f}M", flush=True)
    sem = asyncio.Semaphore(args.concurrency)
    stats = dict(done=0, failed=0, capped=0)
    with path.open("a") as fh:
        await asyncio.gather(*[one(j, data[j[0]]["groups"], sem, budget, fh, stats) for j in jobs])
    print(f"finished: {stats}, total input tokens {budget.used:,}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
