"""
Phase 5a -- conjunct ablation: which parts of a compound label do the identification work?

Compound labels ("X, Y, and Z") are scored by the wayfinding lineup as a whole; this asks what
each conjunct contributes. Per compound gold label, two variants per conjunct run through the
cluster's FROZEN lineup (paired against the battery's already-measured gold pm, same docs, same
distractors, sonnet listener):
  drop:i -- the label with conjunct i removed  -> delta pm = marginal identification value
  only:i -- conjunct i alone                   -> is one conjunct sufficient (the MDL question)?
A conjunct is a FREE-RIDER if dropping it costs less than the repeat band (measured empirically:
a seeded subset of drop variants is re-run and the band is the p90 of |pm_r1 - pm_r2|), and a
label is REDUCIBLE if some single conjunct alone is within the band of the full label's pm.

Splits come from a sonnet call per label (temp 0, structured output), cached to
data/conjunct_splits_20ng.json -- run `--stage split` first and eyeball before `--stage run`.

  uv run python experiments/label_quality/conjunct_ablation.py --stage split
  uv run python experiments/label_quality/conjunct_ablation.py --stage run
  uv run python experiments/label_quality/conjunct_ablation.py --stage report
Writes data/wayfinding_20ng_conjuncts.json (resumable).
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
sys.path.insert(0, str(HERE))

from wayfinding import (Cell, K_DEFAULT, MODELS, load_battery, gold_by_cluster,  # noqa: E402
                        make_unit, run_all)

SPLITS = HERE / "data" / "conjunct_splits_20ng.json"
OUT = HERE / "data" / "wayfinding_20ng_conjuncts.json"
N_REPEAT = 30

SPLIT_PROMPT = """Split this topic label into its top-level conjuncts: the coordinated content \
units joined by commas or "and". Each conjunct must stand alone as a readable label fragment. If \
a modifier or head noun is shared across a coordination (e.g. "IDE and SCSI drives"), distribute \
it so each conjunct is self-contained ("IDE drives", "SCSI drives"). Keep the original wording \
otherwise. If the label has no top-level coordination, return the whole label as one conjunct.

Label: "{label}"

Return JSON: {{"conjuncts": ["...", ...]}}"""

SPLIT_SCHEMA = {"type": "object", "additionalProperties": False,
                "properties": {"conjuncts": {"type": "array", "items": {"type": "string"}}},
                "required": ["conjuncts"]}
SPLIT_FORMAT = {"type": "json_schema",
                "json_schema": {"name": "conjuncts", "schema": SPLIT_SCHEMA, "strict": True}}


def rejoin(conjs: list[str]) -> str:
    if len(conjs) == 1:
        return conjs[0]
    return ", ".join(conjs[:-1]) + (", and " if len(conjs) > 2 else " and ") + conjs[-1]


async def split_all(labels: dict, model: str, concurrency: int) -> dict:
    import litellm
    sem = asyncio.Semaphore(concurrency)

    async def one(key, label):
        async with sem:
            try:
                r = await litellm.acompletion(model=model, temperature=0.0, max_tokens=400,
                                              num_retries=5, response_format=SPLIT_FORMAT,
                                              messages=[{"role": "user",
                                                         "content": SPLIT_PROMPT.format(label=label)}])
                cs = [c.strip() for c in json.loads(r.choices[0].message.content)["conjuncts"]
                      if c.strip()]
                return key, (cs if cs else [label])
            except Exception as e:
                print("  split err:", key, type(e).__name__, str(e)[:60], flush=True)
                return key, [label]

    res = await asyncio.gather(*[one(k, v) for k, v in labels.items()])
    return dict(res)


def load_splits():
    return json.loads(SPLITS.read_text())


def build_units(cell, splits, gold):
    units = []
    for key, conjs in splits.items():
        if len(conjs) < 2:
            continue
        L, i = map(int, key.split("."))
        for ci, c in enumerate(conjs):
            drop = rejoin([x for j, x in enumerate(conjs) if j != ci])
            for kind, lab in [(f"drop{ci}", drop), (f"only{ci}", c)]:
                u = make_unit(cell, L, i, kind, lab, K_DEFAULT, "nn", run=1)
                u.update(conjunct=c, pos=ci, n_conj=len(conjs), full=gold[(L, i)])
                units.append(u)
    rng = np.random.default_rng(5151)
    drops = [u for u in units if u["kind"].startswith("drop")]
    for u in [drops[j] for j in rng.choice(len(drops), min(N_REPEAT, len(drops)), replace=False)]:
        u2 = dict(u, run=2, uid=u["uid"][:-1] + "2")
        units.append(u2)
    return units


def gold_pm():
    b = json.loads((HERE / "data" / "wayfinding_20ng_battery.json").read_text())["units"]
    return {(u["L"], u["i"]): u["pm"] for u in b.values()
            if u["kind"] == "gold" and u["mode"] == "nn" and u.get("pm") is not None}


def report(units):
    gp = gold_pm()
    us = [u for u in units.values() if u.get("pm") is not None]
    r1 = {u["uid"][:-1]: u for u in us if u["run"] == 1 and u["kind"].startswith("drop")}
    r2 = {u["uid"][:-1]: u for u in us if u["run"] == 2}
    rep = [abs(r1[x]["pm"] - r2[x]["pm"]) for x in r2 if x in r1]
    band = float(np.percentile(rep, 90)) if rep else float("nan")
    print(f"\n================ CONJUNCT ABLATION (5a, sonnet) ================")
    print(f"repeat subset n={len(rep)}: p90 |d pm| = {band:.3f} (mean {np.mean(rep):.3f})")

    drops = [dict(u, dpm=gp[(u["L"], u["i"])] - u["pm"]) for u in us
             if u["run"] == 1 and u["kind"].startswith("drop") and (u["L"], u["i"]) in gp]
    onlys = [dict(u, gap=gp[(u["L"], u["i"])] - u["pm"]) for u in us
             if u["kind"].startswith("only") and (u["L"], u["i"]) in gp]
    n_lab = len({(u["L"], u["i"]) for u in drops})
    print(f"compound labels: {n_lab}; conjuncts: {len(drops)} "
          f"(mean {len(drops)/max(n_lab,1):.1f}/label)")

    free = [u for u in drops if u["dpm"] < band]
    anti = [u for u in drops if u["dpm"] < -band]
    load = [u for u in drops if u["dpm"] >= band]
    print(f"\nmarginal value of a conjunct (drop:i, dpm = pm_full - pm_dropped):")
    print(f"  mean dpm {np.mean([u['dpm'] for u in drops]):+.3f}; "
          f"free-riders (dpm < band) {len(free)}/{len(drops)} ({len(free)/len(drops):.0%}); "
          f"load-bearing {len(load)} ({len(load)/len(drops):.0%}); "
          f"anti-conjuncts (dropping HELPS > band) {len(anti)} ({len(anti)/len(drops):.0%})")
    bypos = defaultdict(list)
    for u in drops:
        p = "first" if u["pos"] == 0 else ("last" if u["pos"] == u["n_conj"] - 1 else "mid")
        bypos[p].append(u["dpm"])
    print("  by position: " + "   ".join(f"{p} {np.mean(v):+.3f} (n={len(v)})"
                                         for p, v in [(x, bypos[x]) for x in ("first", "mid", "last") if x in bypos]))

    suff = defaultdict(list)
    for u in onlys:
        suff[(u["L"], u["i"])].append(u)
    reducible = {k: min(v, key=lambda u: u["gap"]) for k, v in suff.items()
                 if min(u["gap"] for u in v) <= band}
    print(f"\nsufficiency (only:i): labels where ONE conjunct alone is within band of full: "
          f"{len(reducible)}/{n_lab} ({len(reducible)/max(n_lab,1):.0%})")
    if reducible:
        cut = [1 - len(u['conjunct']) / len(u['full']) for u in reducible.values()]
        print(f"  mean length reduction if adopted: {np.mean(cut):.0%}")

    print("\nmost load-bearing conjuncts (biggest dpm):")
    for u in sorted(drops, key=lambda u: -u["dpm"])[:5]:
        print(f"  L{u['L']}#{u['i']:<3} dpm={u['dpm']:+.3f}  conj={u['conjunct'][:60]!r}")
    print("clearest anti-conjuncts (dropping helps):")
    for u in sorted(drops, key=lambda u: u["dpm"])[:5]:
        print(f"  L{u['L']}#{u['i']:<3} dpm={u['dpm']:+.3f}  conj={u['conjunct'][:60]!r}")
    print("cleanest sufficient-alone conjuncts:")
    for k, u in sorted(reducible.items(), key=lambda kv: kv[1]["gap"])[:5]:
        print(f"  L{u['L']}#{u['i']:<3} gap={u['gap']:+.3f}  only={u['conjunct'][:44]!r} "
              f"(full={u['full'][:44]!r})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["split", "run", "report"])
    ap.add_argument("--concurrency", type=int, default=24)
    args = ap.parse_args()

    battery = load_battery()
    gold = gold_by_cluster(battery)

    if args.stage == "split":
        labels = {f"{L}.{i}": g for (L, i), g in gold.items()}
        splits = asyncio.run(split_all(labels, MODELS["sonnet"], args.concurrency))
        SPLITS.write_text(json.dumps(splits, indent=2))
        ncomp = sum(1 for v in splits.values() if len(v) >= 2)
        nconj = sum(len(v) for v in splits.values() if len(v) >= 2)
        print(f"split {len(splits)} labels: {ncomp} compound ({nconj} conjuncts). Eyeball:")
        for k, v in list(splits.items()):
            if len(v) >= 2:
                print(f"  {k:<6} {gold[tuple(map(int, k.split('.')))][:64]!r}")
                for c in v:
                    print(f"         - {c}")
        return

    cell = Cell("20ng")
    if args.stage == "run":
        units = build_units(cell, load_splits(), gold)
        print(f"conjunct ablation: {len(units)} units x 3 samples on sonnet")
        units = asyncio.run(run_all(units, cell, MODELS["sonnet"], args.concurrency, OUT))
        report(units)
    else:
        report(json.loads(OUT.read_text())["units"])


if __name__ == "__main__":
    main()
