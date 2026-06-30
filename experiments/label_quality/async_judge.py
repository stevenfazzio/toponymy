"""
Concurrent grounded judging (litellm.acompletion) -- a drop-in faster replacement for the sequential
rate() loop in judge_quality. All candidate x k-sample calls are fired concurrently under a semaphore
(to respect rate limits), so a few-thousand-call judging run finishes in minutes, not hours. Same
rubric and parsing as judge_quality, so ratings are comparable.

  rate_many([(label, docs), ...], obj, "anthropic/claude-sonnet-4-6", k=3, concurrency=24)
    -> [{consistency, relevance, completeness, overall}, ...]  (aligned to the input tasks)
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
NIBLING = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(NIBLING))

import litellm  # noqa: E402

from judge_quality import FIELDS, RUBRIC, parse_scores  # noqa: E402  (reuse rubric/parser)


async def _one_call(prompt, model, temp, sem):
    async with sem:
        try:
            r = await litellm.acompletion(model=model, messages=[{"role": "user", "content": prompt}],
                                          temperature=temp, max_tokens=200, num_retries=5)
            return parse_scores(r.choices[0].message.content)
        except Exception as e:
            print("  judge err:", type(e).__name__, str(e)[:70], flush=True)
            return None


async def _rate(label, docs, obj, model, k, temp, sem):
    prompt = RUBRIC.format(obj=obj, docs="\n".join(f"- {d}" for d in docs), label=label)
    samples = await asyncio.gather(*[_one_call(prompt, model, temp, sem) for _ in range(k)])
    acc = {f: [s[f] for s in samples if s and f in s] for f in FIELDS}
    return {f: (float(np.mean(acc[f])) if acc[f] else None) for f in FIELDS}


async def _rate_all(tasks, obj, model, k, temp, concurrency):
    sem = asyncio.Semaphore(concurrency)
    return await asyncio.gather(*[_rate(lab, docs, obj, model, k, temp, sem) for (lab, docs) in tasks])


def rate_many(tasks, obj, model, k=3, temp=0.7, concurrency=24):
    """tasks: list of (label, docs). Returns a list of {field: mean or None} aligned to tasks."""
    return asyncio.run(_rate_all(tasks, obj, model, k, temp, concurrency))


if __name__ == "__main__":
    import json
    import time

    from ab_harness import load_dataset
    from judge_fair import sample_docs

    obj = load_dataset("20ng", None)[3]["obj"]
    battery = json.loads((HERE / "data" / "battery_20ng.json").read_text())[:20]
    tasks = [(it["gold"], sample_docs("20ng", it["layer"], it["idx"], n_near=10, n_rand=5)) for it in battery]
    n_calls = len(tasks) * 3
    print(f"concurrent judging test: {len(tasks)} gold labels, k=3 = {n_calls} calls, concurrency=24")
    t0 = time.time()
    ratings = rate_many(tasks, obj, "anthropic/claude-sonnet-4-6", k=3, temp=0.7, concurrency=24)
    dt = time.time() - t0
    ov = [r["overall"] for r in ratings if r["overall"] is not None]
    print(f"  done in {dt:.0f}s  ({dt/n_calls:.2f}s/call effective; sequential ~{n_calls*4/60:.0f} min)")
    print(f"  gold mean overall = {np.mean(ov):.2f}  (20NG sequential gold was 2.80)")
