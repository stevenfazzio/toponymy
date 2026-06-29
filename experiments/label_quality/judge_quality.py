"""
Phase 0 leg 3 / Phase 1 gate (b) -- grounded, self-consistent label-quality judge.

Rates a candidate label on the Preiss et al. (2024) rubric, GROUNDED in the cluster's
documents (Krumdick et al. 2025: never reference-free -- grounding lifts judge-human
agreement on the hard slice), with CoT + self-consistency (k samples, averaged). Used to
validate the cheap metrics: does whitened-centroid cosine track the judge's quality?

Grounding = ~15 sampled docs/cluster (nearest-centroid for relevance + a random spread for
coverage), reusing judge_fair.sample_docs so the fit/indices match the battery.
Rubric (Preiss): consistency (faithful), relevance (only important content), completeness
(all important content), overall -- each 0-4. The documents ARE the reference the judge
rates against (there is no single "correct" label, so we ground on docs, not a gold label).

Judge with a model INDEPENDENT of the namer where possible (naming was haiku) to avoid
self-preference -- default sonnet, per Preiss/judge_fair using the strong deterministic model.

  uv run python experiments/label_quality/judge_quality.py --judge sonnet --k 3 \
      --variants verbose sibling ancestor generic
  uv run python experiments/label_quality/judge_quality.py --judge haiku --k 3 --limit 30   # cheap pilot
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
NIBLING = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(NIBLING))

import litellm  # noqa: E402

from ab_harness import load_dataset  # noqa: E402
from judge_fair import sample_docs  # noqa: E402  (grounded doc sampling; cached fit, same params)

MODELS = {
    "haiku": "anthropic/claude-haiku-4-5-20251001",
    "sonnet": "anthropic/claude-sonnet-4-6",
    "gpt4omini": "openai/gpt-4o-mini",
}
FIELDS = ["consistency", "relevance", "completeness", "overall"]

RUBRIC = (
    "Below are representative documents from ONE group of {obj}:\n{docs}\n\n"
    'Candidate name for this group: "{label}"\n\n'
    "Rate how well the NAME describes this group, grounded ONLY in the documents above. "
    "Score each field 0-4 (0=very poor, 2=neutral, 4=very good):\n"
    "- consistency: faithful to the documents (asserts nothing the documents do not support)\n"
    "- relevance: reflects ONLY important content (not over-broad, not padded with filler)\n"
    "- completeness: reflects ALL the important content (not too narrow)\n"
    "- overall: overall name quality.\n"
    "Reason in one short sentence, then reply ONLY as JSON:\n"
    '{{"reason":"<short>","consistency":N,"relevance":N,"completeness":N,"overall":N}}'
)


def parse_scores(text: str):
    out = {}
    for f in FIELDS:
        m = re.search(rf'"{f}"\s*:\s*([0-4](?:\.\d+)?)', text)
        if m:
            out[f] = float(m.group(1))
    return out if "overall" in out else None


def rate(label: str, docs, obj: str, model: str, k: int, temp: float):
    """k self-consistency samples; returns (mean per field, n_valid)."""
    prompt = RUBRIC.format(obj=obj, docs="\n".join(f"- {d}" for d in docs), label=label)
    acc = {f: [] for f in FIELDS}
    for _ in range(k):
        try:
            r = litellm.completion(model=model, messages=[{"role": "user", "content": prompt}],
                                   temperature=temp, max_tokens=200)
            p = parse_scores(r.choices[0].message.content)
            if p:
                for f, v in p.items():
                    acc[f].append(v)
        except Exception as e:
            print("  judge err:", type(e).__name__, str(e)[:80], flush=True)
    return {f: (float(np.mean(acc[f])) if acc[f] else None) for f in FIELDS}, len(acc["overall"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="20ng")
    ap.add_argument("--battery", default="data/battery_20ng.json")
    ap.add_argument("--judge", default="sonnet", choices=list(MODELS))
    ap.add_argument("--k", type=int, default=3, help="self-consistency samples")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--variants", nargs="+", default=["verbose", "sibling", "ancestor", "generic"],
                    help="which degraded variants to judge alongside gold (distant omitted: trivially easy)")
    ap.add_argument("--n-near", type=int, default=10)
    ap.add_argument("--n-rand", type=int, default=5)
    ap.add_argument("--maxlen", type=int, default=280)
    ap.add_argument("--limit", type=int, default=None, help="judge only the first N clusters (cheap pilot)")
    args = ap.parse_args()

    obj = load_dataset(args.dataset, None)[3]["obj"]
    battery = json.loads((HERE / args.battery).read_text())
    if args.limit:
        battery = battery[: args.limit]
    model = MODELS[args.judge]
    n_cand = sum(1 + sum(v in it["variants"] for v in args.variants) for it in battery)
    print(f"[{args.dataset}] judge={args.judge} k={args.k} temp={args.temp} | "
          f"{len(battery)} clusters, ~{n_cand} candidates x {args.k} = ~{n_cand*args.k} calls")

    ratings, t0 = [], time.time()
    for n, it in enumerate(battery):
        docs = sample_docs(args.dataset, it["layer"], it["idx"],
                           n_near=args.n_near, n_rand=args.n_rand, maxlen=args.maxlen)
        cands = {"gold": it["gold"]}
        for v in args.variants:
            if v in it["variants"]:
                cands[v] = it["variants"][v]
        for ctype, lab in cands.items():
            scores, nval = rate(lab, docs, obj, model, args.k, args.temp)
            ratings.append(dict(layer=it["layer"], idx=it["idx"], type=ctype, label=lab,
                                n_valid=nval, **scores))
        if (n + 1) % 10 == 0:
            print(f"  judged {n+1}/{len(battery)} clusters ({time.time()-t0:.0f}s)", flush=True)

    out = HERE / "data" / f"judge_ratings_{args.dataset}_{args.judge}.json"
    out.write_text(json.dumps(ratings, indent=2))

    # sanity: mean overall by candidate type (gold should be highest)
    print("\nmean judge OVERALL by candidate type (grounded, self-consistent):")
    for ctype in ["gold"] + args.variants:
        vals = [r["overall"] for r in ratings if r["type"] == ctype and r["overall"] is not None]
        if vals:
            print(f"  {ctype:<9} {np.mean(vals):.2f}  (n={len(vals)})")
    print(f"\nsaved {len(ratings)} ratings -> {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
