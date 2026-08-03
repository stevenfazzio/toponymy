"""
Phase 7c -- headroom for "resample exemplars, name N times, pick the best" (no LLM calls).

Prices the idea on BOTH axes before anything is built, from data already committed.

FIT: the two independent naming draws at identical conditions (dose_names/judge + .draw1),
20NG layer 0, grounded sonnet judge.

IDENTIFICATION: FEATURES.md measured the Phase-6 ablation arms identification-INERT
(+0.011 / -0.006, well-powered nulls), so for identification purposes gold / abl_exemplars /
abl_keyphrases are up to three independent namings of the same cluster through IDENTICAL frozen
lineups -- exactly the best-of-N setup, already paid for.

The number that makes any of this interpretable is the WINNER'S-CURSE FLOOR: oracle best-of-N
inflates even when the draws are identical in expectation, purely by selecting on measurement
noise. Measured two ways rather than assumed:
  (a) empirical -- wayfinding_*_floors.json contains the SAME gold label on the SAME lineup run
      twice (run 1 / run 2), so oracle best-of-2 over a pure repeat IS the floor (haiku listener)
  (b) analytic  -- per-unit noise sd from its own 3 listener samples; for two independent draws
      with no true difference, E[max - mean] = sigma_draw / sqrt(pi)
Only headroom clearing that floor counts.

  uv run python experiments/label_quality/redraw_headroom.py
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA = Path("/Users/stevenfazzio/repos/toponymy/experiments/label_quality/data")


def units(name):
    p = DATA / name
    return json.loads(p.read_text())["units"] if p.exists() else {}


def sample_pms(u):
    """Per-sample prob-mass on the true cluster (recomputed from the stored raw samples)."""
    out = []
    for s in u.get("samples", []):
        order, sc = s["order"], s["scores"]
        tot = sum(sc)
        if len(order) != len(sc) or tot <= 0:
            continue
        out.append(sc[order.index(u["true"])] / tot)
    return out


def oracle(draws):
    """(mean of single draws, mean of best-of-N) over clusters with >=2 draws."""
    single = [st.mean(v) for v in draws if len(v) >= 2]
    best = [max(v) for v in draws if len(v) >= 2]
    return len(single), st.mean(single), st.mean(best), st.mean(best) - st.mean(single)


# E[max of N iid N(mu, sigma^2)] - mu, in units of sigma: the pure selection-on-noise inflation
EMAX = {2: 1 / np.sqrt(np.pi), 3: 3 / (2 * np.sqrt(np.pi))}


def judge_repeat_floor():
    """Judge winner's-curse floor from the ONE pure-ish repeat on disk: clean_docs_rejudge scored
    the SAME gold labels again on a fresh (leakage-free) document draw. The paired scatter
    therefore contains judge noise AND doc-sample variance, so this OVERSTATES the floor --
    which is the conservative direction for a headroom claim."""
    a = {r["label"]: r["overall"] for r in json.loads((DATA / "judge_ratings_20ng_sonnet.json")
                                                      .read_text()) if r.get("type") == "gold"}
    b = json.loads((DATA / "clean_docs_20ng.json").read_text())["gold"]
    d = [a[v["label"]] - v["overall"] for v in b.values()
         if v.get("label") in a and v.get("overall") is not None]
    if len(d) < 10:
        return None
    sd_diff = st.stdev(d)
    sigma = sd_diff / np.sqrt(2)          # per-measurement sd, if the two draws are exchangeable
    print(f"  JUDGE REPEAT (same label, fresh docs): n={len(d)}  mean shift {st.mean(d):+.3f} "
          f"(the known ~0.07 clean-doc penalty)  sd of paired diff {sd_diff:.3f}")
    print(f"     => sigma(one judge measurement) {sigma:.3f}   "
          f"WINNER'S-CURSE FLOOR for best-of-2 = +{sigma*EMAX[2]:.3f} judge-pts  "
          f"(upper bound: includes doc-sample variance)")
    return sigma


def fit_axis():
    print("=" * 84)
    print("FIT AXIS -- two independent naming draws, identical conditions (20NG L0, judge 0-4)")
    print("=" * 84)
    a = json.loads((DATA / "dose_judge_20ng.json").read_text())
    b = json.loads((DATA / "dose_judge_20ng.draw1.json").read_text())
    na = json.loads((DATA / "dose_names_20ng.json").read_text())
    nb = json.loads((DATA / "dose_names_20ng.draw1.json").read_text())
    for k in ["1", "2", "4"]:
        A, B, NA, NB = a[k], b[k], na[k][0], nb[k]
        pairs, same = [], 0
        for x in B:
            if "0." + x not in A:
                continue
            p, q = A["0." + x].get("overall"), B[x]
            if p is None or q is None:
                continue
            pairs.append([p, q])
            same += NA[int(x)] == NB[int(x)]
        n, mean, best, gain = oracle(pairs)
        ad = [abs(p - q) for p, q in pairs]
        print(f"  n_exemplars={k}: n={n}  identical string {same}/{len(pairs)}  "
              f"mean |d| {st.mean(ad):.3f}  >=1 pt apart {sum(v >= 1 for v in ad)}/{len(pairs)}")
        print(f"     single-draw {mean:.3f} -> ORACLE best-of-2 {best:.3f}   headroom +{gain:.3f} judge-pts")
    print("  scale: exemplars ablated = -0.49 judge-pts; k=2 vs stock = -0.18; "
          "controller vs stock fit = +0.04 (n.s.)")
    judge_repeat_floor()
    print()


def identification_axis():
    print("=" * 84)
    print("IDENTIFICATION AXIS -- gold vs the two ablation arms, IDENTICAL frozen lineups")
    print("=" * 84)
    for ds in ["20ng", "arxiv_home"]:
        bat = units(f"wayfinding_{ds}_battery.json")
        fea = units(f"wayfinding_{ds}_features.json")
        by = defaultdict(dict)
        for u in bat.values():
            if u.get("kind") == "gold" and u.get("pm") is not None:
                by[(u["L"], u["true"])]["gold"] = u
        for u in fea.values():
            if u.get("pm") is not None:
                by[(u["L"], u["true"])][u["kind"]] = u

        def draws_for(arms):
            out = []
            for key, d in by.items():
                got = [d[a] for a in arms if a in d]
                if len(got) < len(arms):
                    continue
                base = got[0]["lineup"]
                if any(g["lineup"] != base for g in got):
                    continue                      # only strictly paired lineups
                labels = {g["label"] for g in got}
                out.append(([g["pm"] for g in got], len(labels)))
            return out

        for arms in (["gold", "abl_exemplars"], ["gold", "abl_keyphrases"],
                     ["gold", "abl_exemplars", "abl_keyphrases"]):
            rows = draws_for(arms)
            if not rows:
                continue
            pms = [r[0] for r in rows]
            distinct = sum(1 for r in rows if r[1] > 1)
            n, mean, best, gain = oracle(pms)
            tag = "+".join(a.replace("abl_", "") for a in arms)
            print(f"  [{ds}] {tag:34s} n={n:3d} (distinct labels in {distinct})  "
                  f"single {mean:.3f} -> best-of-{len(arms)} {best:.3f}   headroom +{gain:.3f} pm")

        # ---- winner's-curse floor (a): identical label, identical lineup, run twice (haiku)
        fl = units(f"wayfinding_{ds}_floors.json")
        rep = defaultdict(dict)
        for u in fl.values():
            if u.get("kind") == "gold" and u.get("mode") == "nn" and u.get("k") == 5 \
                    and u.get("pm") is not None:
                rep[(u["L"], u["true"])][u.get("run")] = u
        pairs = [[d[1]["pm"], d[2]["pm"]] for d in rep.values() if 1 in d and 2 in d]
        if pairs:
            n, mean, best, gain = oracle(pairs)
            print(f"  [{ds}] {'FLOOR (a) same label, run twice':34s} n={n:3d}                     "
                  f"  single {mean:.3f} -> best-of-2 {best:.3f}   FLOOR  +{gain:.3f} pm  (haiku)")

        # ---- winner's-curse floor (b): analytic, from each unit's own 3 listener samples
        sds = []
        for u in list(bat.values()) + list(fea.values()):
            v = sample_pms(u)
            if len(v) >= 3:
                sds.append(st.stdev(v) / np.sqrt(len(v)))    # sd of a 3-sample pm estimate
        if sds:
            sigma = float(np.mean(sds))
            print(f"  [{ds}] {'FLOOR (b) analytic, sonnet samples':34s}                          "
                  f"  sigma(3-sample pm) {sigma:.3f}   FLOOR  best-of-2 +{sigma*EMAX[2]:.3f} / "
                  f"best-of-3 +{sigma*EMAX[3]:.3f} pm")
            print(f"         (floor (b) counts only within-unit sample scatter, so it UNDERSTATES; "
                  f"floor (a) is a real repeat but on the noisier haiku listener, so it OVERSTATES\n"
                  f"          for sonnet -- the true sonnet floor lies between them)")
        print()


def cross_model_check():
    """Sanity cross-check on the identification axis: haiku vs gpt-4o-mini namings of the same
    cluster (Phase 4's 104 fine pairs) are a genuine cross-model best-of-2."""
    print("=" * 84)
    print("CROSS-CHECK -- the 104 fine pairs (haiku vs gpt-4o-mini namings), same frozen lineups")
    print("=" * 84)
    u = units("wayfinding_20ng_pairs.json")
    by = defaultdict(dict)
    for v in u.values():
        if v.get("pm") is not None:
            by[(v["L"], v["true"])][v["kind"]] = v
    rows = [[d["pair-base"]["pm"], d["pair-alt"]["pm"]] for d in by.values()
            if "pair-base" in d and "pair-alt" in d
            and d["pair-base"]["lineup"] == d["pair-alt"]["lineup"]]
    n, mean, best, gain = oracle(rows)
    print(f"  n={n}  single {mean:.3f} -> ORACLE best-of-2 {best:.3f}   headroom +{gain:.3f} pm")
    print("  (#177 sonnet repeat band on these pairs: p90 |delta| = 0.107)\n")


if __name__ == "__main__":
    fit_axis()
    identification_axis()
    cross_model_check()
