"""
Phase 6 -- the naming-feature ablation, re-scored by the wayfinding lineup.

`ablation.py` measured what each naming-prompt feature contributes using a FIT instrument (the
grounded judge): exemplars +0.47 judge-pts, keyphrases -0.17 (i.e. no measurable value, if
anything mildly harmful). Phase 4 then established that fit and identification are different axes
(rho 0.65-0.74; the judge ties on ~half of good-vs-good pairs, and is nearly blind to the
distinguishing specific that the lineup rewards). So a feature that is judge-neutral is NOT
automatically identification-neutral, and the keyphrase result -- which is currently being read
upstream as grounds for turning keyphrase extraction off by default -- has never been checked with
the discriminative instrument.

This closes that gap. The ablated labels are already cached in data/ablation.json, keyed by
(layer, cluster) on the SAME canonical 20NG fit the wayfinding battery replays ([74, 24, 9]), and
their `full` labels are byte-identical to the battery golds (verified at load). So each ablated
label runs through its cluster's FROZEN lineup -- same 4 distractors, same 5 held-out docs, same
sonnet listener -- and is paired against the battery's already-measured gold pm. No re-clustering,
no re-naming, nothing new to freeze.

  d_pm = pm(full label) - pm(ablated label)   > band => the feature carried identification value

Labels the ablation did not change contribute d_pm = 0 by construction and cost no calls. The band
is measured here, not assumed: a seeded subset is re-run (run=2) and the band is the p90 of
|pm_r1 - pm_r2|, the same protocol as the conjunct ablation.

EXEMPLARS IS THE POSITIVE CONTROL, not scope creep: it is the one feature the judge says carries a
large effect, so it is what tells us the lineup can resolve an ablation-sized change at all. A null
on keyphrases means nothing unless exemplars registers.

  uv run python experiments/label_quality/feature_ablation_lineup.py --stage check   # no LLM
  uv run python experiments/label_quality/feature_ablation_lineup.py --stage run
  uv run python experiments/label_quality/feature_ablation_lineup.py --stage report
Writes data/wayfinding_20ng_features.json (resumable; completed units are skipped on re-run).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
sys.path.insert(0, str(HERE))

from wayfinding import (Cell, K_DEFAULT, MODELS, load_battery, gold_by_cluster,  # noqa: E402
                        make_unit, run_all)

ABLATION = HERE / "data" / "ablation.json"
OUT = HERE / "data" / "wayfinding_20ng_features.json"
FEATURES = ["keyphrases", "exemplars"]  # keyphrases = the question; exemplars = positive control
N_REPEAT = 30


def load_ablation(gold: dict) -> dict:
    """feature -> list of rows, with the (L,i) -> gold pairing verified byte-for-byte."""
    raw = json.loads(ABLATION.read_text())
    out = {}
    for feat in FEATURES:
        rows = []
        for r in raw[feat]:
            key = (r["L"], r["i"])
            if key not in gold:
                raise SystemExit(f"{feat} L{r['L']}#{r['i']} absent from battery -- fit mismatch")
            if gold[key].strip() != r["full"].strip():
                raise SystemExit(f"{feat} L{r['L']}#{r['i']} label drift vs battery gold:\n"
                                 f"  ablation: {r['full']!r}\n  battery:  {gold[key]!r}")
            rows.append(r)
        out[feat] = rows
    return out


def gold_pm() -> dict:
    b = json.loads((HERE / "data" / "wayfinding_20ng_battery.json").read_text())["units"]
    return {(u["L"], u["i"]): u["pm"] for u in b.values()
            if u["kind"] == "gold" and u["mode"] == "nn" and u.get("pm") is not None}


def build_units(cell, ablation: dict) -> list[dict]:
    units = []
    for feat, rows in ablation.items():
        for r in rows:
            if not r["changed"]:
                continue  # identical label => identical lineup => d_pm = 0, no call needed
            u = make_unit(cell, r["L"], r["i"], f"abl_{feat}", r["abl"], K_DEFAULT, "nn", run=1)
            u.update(feature=feat, full=r["full"],
                     j_full=r["j_full"], j_abl=r["j_abl"])
            units.append(u)
    rng = np.random.default_rng(6161)
    pick = rng.choice(len(units), min(N_REPEAT, len(units)), replace=False)
    for u in [units[j] for j in pick]:
        units.append(dict(u, run=2, uid=u["uid"][:-1] + "2"))
    return units


def power_check():
    """The null below is only meaningful if the lineup resolves ablation-sized effects on THESE
    clusters. The battery's known-bad variants are the calibration: same clusters, same frozen
    lineups, same listener, so their paired d_pm is the instrument's measured dynamic range."""
    b = json.loads((HERE / "data" / "wayfinding_20ng_battery.json").read_text())["units"]
    us = [u for u in b.values() if u.get("pm") is not None]
    gold = {(u["L"], u["i"]): u["pm"] for u in us
            if u["kind"] == "gold" and u["mode"] == "nn"}
    byk = defaultdict(list)
    for u in us:
        if u["kind"] not in ("gold", "gimme") and u["mode"] == "nn":
            byk[u["kind"]].append(u)
    print("\n-- power check: paired d_pm of the battery's known-bad variants (same lineups) --")
    print(f"   {'variant':<10}{'n':>5}{'d pm':>9}{'SE':>8}{'p':>11}")
    rows = []
    for k, v in sorted(byk.items()):
        d = [gold[(u["L"], u["i"])] - u["pm"] for u in v if (u["L"], u["i"]) in gold]
        if len(d) < 5:
            continue
        m, se = float(np.mean(d)), float(np.std(d, ddof=1) / np.sqrt(len(d)))
        p = wilcoxon(d).pvalue
        rows.append((k, m, se))
        print(f"   {k:<10}{len(d):>5}{m:>+9.3f}{se:>8.3f}{p:>11.1g}")
    weakest = min(rows, key=lambda r: r[1])
    print(f"   -> the SMALLEST degradation the instrument resolves is {weakest[0]} at "
          f"{weakest[1]:+.3f} (SE {weakest[2]:.3f}).")
    print("   An ablation arm at comparable n and SE is therefore powered to see anything")
    print(f"   down to roughly that size. Read the nulls below against {weakest[1]:+.3f}.")
    return weakest


def report(units: dict, ablation: dict):
    gp = gold_pm()
    us = [u for u in units.values() if u.get("pm") is not None]

    r1 = {u["uid"][:-1]: u for u in us if u["run"] == 1}
    r2 = {u["uid"][:-1]: u for u in us if u["run"] == 2}
    rep = [abs(r1[x]["pm"] - r2[x]["pm"]) for x in r2 if x in r1]
    band = float(np.percentile(rep, 90)) if rep else float("nan")

    print("\n========== NAMING-FEATURE ABLATION, WAYFINDING LINEUP (20NG, sonnet) ==========")
    print(f"repeat subset n={len(rep)}: p90 |d pm| = {band:.3f} (mean {np.mean(rep):.3f})")
    print("  NB: the band is a PER-CLUSTER noise yardstick (use it to classify individual")
    print("  clusters). The aggregate mean is far better resolved than the band -- judge it")
    print("  against its own SE/CI, not against the band.")
    print(f"paired baseline: battery gold pm, mean {np.mean(list(gp.values())):.3f} "
          f"(chance = {1/K_DEFAULT:.2f})")
    weakest = power_check()

    for feat in FEATURES:
        rows = {(r["L"], r["i"]): r for r in ablation[feat]}
        scored = [u for u in us if u["run"] == 1 and u["feature"] == feat
                  and (u["L"], u["i"]) in gp]
        recs = [dict(u, dpm=gp[(u["L"], u["i"])] - u["pm"]) for u in scored]
        # unchanged labels are exact zeros: same string, same frozen lineup
        n_unchanged = sum(1 for r in rows.values() if not r["changed"])
        all_dpm = [r["dpm"] for r in recs] + [0.0] * n_unchanged

        print(f"\n---- {feat}  (n={len(all_dpm)} clusters: {len(recs)} changed, "
              f"{n_unchanged} unchanged @ d=0) ----")
        print(f"  judge d (fit, from ablation.json): "
              f"{np.mean([r['j_full'] - r['j_abl'] for r in rows.values()]):+.3f} pts")
        m = float(np.mean(all_dpm))
        se = float(np.std(all_dpm, ddof=1) / np.sqrt(len(all_dpm)))
        hi = max(abs(m - 1.96 * se), abs(m + 1.96 * se))
        print(f"  lineup d pm (identification):      {m:+.3f}  (SE {se:.3f}, "
              f"95% CI [{m-1.96*se:+.3f}, {m+1.96*se:+.3f}])")
        sig = "DETECTED" if abs(m) > 1.96 * se else "NULL"
        print(f"  -> {sig}: CI rules out any identification cost larger than {hi:.3f} pm, "
              f"i.e. < {hi/weakest[1]:.0%} of the")
        print(f"     weakest degradation the instrument resolves ({weakest[0]}, {weakest[1]:+.3f})")

        d = [r["dpm"] for r in recs]
        if d:
            try:
                stat, p = wilcoxon(d)
                print(f"  paired Wilcoxon on the {len(d)} changed labels: p = {p:.3g}")
            except ValueError:
                pass
            hurt = [r for r in recs if r["dpm"] > band]
            help_ = [r for r in recs if r["dpm"] < -band]
            print(f"  per-cluster: ablation HURT identification {len(hurt)} "
                  f"({len(hurt)/len(recs):.0%}), HELPED {len(help_)} ({len(help_)/len(recs):.0%}), "
                  f"within band {len(recs)-len(hurt)-len(help_)} "
                  f"({(len(recs)-len(hurt)-len(help_))/len(recs):.0%})")

            bylayer = defaultdict(list)
            for r in recs:
                bylayer[r["L"]].append(r["dpm"])
            print("  by layer: " + "   ".join(
                f"L{L} {np.mean(v):+.3f} (n={len(v)})" for L, v in sorted(bylayer.items())))

            jd = [r["j_full"] - r["j_abl"] for r in recs]
            rho, prho = spearmanr(jd, d)
            print(f"  cross-instrument: rho(judge d, lineup d) = {rho:+.3f} (p = {prho:.3g}) "
                  f"-- do the two instruments agree on WHICH clusters the ablation hurt?")

            print("  biggest identification losses from dropping this feature:")
            for r in sorted(recs, key=lambda r: -r["dpm"])[:3]:
                print(f"    L{r['L']}#{r['i']:<3} dpm={r['dpm']:+.3f}  full={r['full'][:52]!r}")
                print(f"    {'':16} abl ={r['label'][:52]!r}")
            print("  biggest identification gains from dropping this feature:")
            for r in sorted(recs, key=lambda r: r["dpm"])[:3]:
                print(f"    L{r['L']}#{r['i']:<3} dpm={r['dpm']:+.3f}  full={r['full'][:52]!r}")
                print(f"    {'':16} abl ={r['label'][:52]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["check", "run", "report"])
    ap.add_argument("--concurrency", type=int, default=24)
    args = ap.parse_args()

    battery = load_battery()
    gold = gold_by_cluster(battery)
    ablation = load_ablation(gold)
    print(f"ablation.json verified against battery golds: "
          + ", ".join(f"{f} n={len(r)}" for f, r in ablation.items()))

    cell = Cell("20ng")
    assert cell.counts == [74, 24, 9], f"canonical replay mismatch {cell.counts} -- do not proceed"
    print(f"replayed canonical fit: {cell.counts}")

    if args.stage == "check":
        units = build_units(cell, ablation)
        n1 = sum(1 for u in units if u["run"] == 1)
        print(f"\nwould run {len(units)} units ({n1} labels + {len(units)-n1} repeats) "
              f"x 3 samples = {len(units)*3} listener calls")
        for feat in FEATURES:
            fu = [u for u in units if u["feature"] == feat and u["run"] == 1]
            print(f"  {feat:<11} {len(fu)} changed labels")
            u = fu[0]
            print(f"    example L{u['L']}#{u['i']} lineup={u['lineup']}")
            print(f"      full: {u['full']!r}")
            print(f"      abl : {u['label']!r}")
        return

    if args.stage == "run":
        units = build_units(cell, ablation)
        print(f"\n{len(units)} units x 3 samples on sonnet")
        units = asyncio.run(run_all(units, cell, MODELS["sonnet"], args.concurrency, OUT))
        report(units, ablation)
    else:
        report(json.loads(OUT.read_text())["units"], ablation)


if __name__ == "__main__":
    main()
