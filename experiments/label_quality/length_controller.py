"""
Phase 5b -- the length controller: replace the detail_levels dial with a measured choice.

Toponymy sets label specificity open-loop: detail_levels = linspace(0,1,n_layers) indexes a
SUMMARY_KINDS ladder of word-count phrases (finest layer -> "domain expert (8 to 15 word)",
coarsest -> "simple (1 or 2 word)"); the templates ask for distinguishability, nothing checks it.
Here the wayfinding lineup is the checker:

  ladder   -- name EVERY cluster at EVERY rung r in 0..6 with stock Toponymy machinery
              (haiku namer, same fitted clusterer, lowest=highest_detail_level=r/6, so the
              disambiguation pass and all features behave exactly as the library would).
              Stock arm = the layer's linspace rung ([0,3,6] for 3 layers), same draw => paired.
  lineups  -- every distinct (cluster, name) through the cluster's FROZEN lineup (sonnet),
              + a 30-unit repeat subset for the band.
  select   -- per cluster, the MDL rule: the SHORTEST name whose pm is within the repeat band
              of the best rung's pm (minimize length s.t. no measurable identification loss).
              Report per-layer rate-distortion (mean pm vs mean words per rung), the selection
              distribution, and chosen-vs-stock deltas.
  heldout  -- Goodhart guard (Gao et al. 2023): chosen vs stock, ONLY where they differ, on a
              held-out lineup config: fresh doc draws (new seed namespace) + a different
              listener (gpt-4o-mini; falls back to haiku if OPENAI_API_KEY is absent) + fresh
              order seeds. Plus grounded judge fit (sonnet, async_judge) non-regression.

  uv run python experiments/label_quality/length_controller.py --stage ladder
  uv run python experiments/label_quality/length_controller.py --stage lineups
  uv run python experiments/label_quality/length_controller.py --stage select
  uv run python experiments/label_quality/length_controller.py --stage heldout
Writes data/ladder_names_20ng.json, data/wayfinding_20ng_ladder.json,
data/wayfinding_20ng_heldout.json, data/length_controller_20ng.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
NIBLING = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(NIBLING))
sys.path.insert(0, str(HERE))

from wayfinding import (Cell, K_DEFAULT, MODELS, make_unit, md5i, run_all,  # noqa: E402
                        unit_rows)

N_RUNGS = 7
N_REPEAT = 30
DS = "20ng"
LADDER = LINEUPS = HELDOUT = SUMMARY = None


def set_dataset(ds: str):
    global DS, LADDER, LINEUPS, HELDOUT, SUMMARY
    DS = ds
    LADDER = HERE / "data" / f"ladder_names_{ds}.json"
    LINEUPS = HERE / "data" / f"wayfinding_{ds}_ladder.json"
    HELDOUT = HERE / "data" / f"wayfinding_{ds}_heldout.json"
    SUMMARY = HERE / "data" / f"length_controller_{ds}.json"


set_dataset(DS)


def stock_rungs(n_layers: int) -> dict:
    """Toponymy's linspace dial: detail_levels = linspace(0,1,n_layers) -> SUMMARY_KINDS index."""
    return {L: int(round(L / (n_layers - 1) * (N_RUNGS - 1))) for L in range(n_layers)}


def words(s: str) -> int:
    return len(s.split())


# ------------------------------------------------------------------ ladder naming

def stage_ladder():
    from ab_harness import make_embedder, make_namer
    from toponymy.toponymy import Toponymy

    done = json.loads(LADDER.read_text()) if LADDER.exists() else {}
    cell = Cell(DS)
    embedder = make_embedder(cell.meta["emb_model"])
    for r in range(N_RUNGS):
        if str(r) in done:
            print(f"rung {r}: cached")
            continue
        t0 = time.time()
        model = Toponymy(make_namer("haiku"), embedder, clusterer=cell.clusterer,
                         object_description=cell.meta["obj"],
                         corpus_description=cell.meta["corpus"],
                         lowest_detail_level=r / (N_RUNGS - 1),
                         highest_detail_level=r / (N_RUNGS - 1), verbose=False)
        model.fit(cell.objects, cell.emb, cell.coords)
        done[str(r)] = [list(layer) for layer in model.topic_names_]
        LADDER.write_text(json.dumps(done, indent=2))
        ex = done[str(r)][0][0]
        print(f"rung {r}: named ({time.time()-t0:.0f}s)  e.g. L0#0 = {ex!r}", flush=True)
    counts = [len(x) for x in done["0"]]
    print(f"ladder complete: {N_RUNGS} rungs x layers {counts}")


# ------------------------------------------------------------------ lineups over the ladder

def ladder_names():
    d = json.loads(LADDER.read_text())
    assert all(str(r) in d for r in range(N_RUNGS)), "run --stage ladder first"
    return d


def distinct_candidates(names):
    """(L, i) -> {label: [rungs that produced it]} -- dedupe so each name runs once."""
    cands = defaultdict(lambda: defaultdict(list))
    for r in range(N_RUNGS):
        for L, layer in enumerate(names[str(r)]):
            for i, lab in enumerate(layer):
                cands[(L, i)][lab].append(r)
    return cands


def stage_lineups(concurrency):
    cell = Cell(DS)
    cands = distinct_candidates(ladder_names())
    units = []
    for (L, i), labs in cands.items():
        for lab in labs:
            units.append(make_unit(cell, L, i, "cand", lab, K_DEFAULT, "nn", run=1))
    rng = np.random.default_rng(6161)
    for u in [units[j] for j in rng.choice(len(units), N_REPEAT, replace=False)]:
        units.append(dict(u, run=2, uid=u["uid"][:-1] + "2"))
    n_all = sum(len(v) for v in cands.values())
    print(f"ladder lineups: {len(units)} units ({n_all} rung-names, "
          f"{len(units)-N_REPEAT} distinct) x 3 samples on sonnet")
    units = asyncio.run(run_all(units, cell, MODELS["sonnet"], concurrency, LINEUPS))
    return units


# ------------------------------------------------------------------ selection + report

def collect_pm(units):
    pm = {}
    for u in units.values():
        if u.get("pm") is None or u["run"] != 1:
            continue
        pm[(u["L"], u["i"], u["label"])] = u["pm"]
    return pm


def band_from(units):
    r1 = {u["uid"][:-1]: u for u in units.values() if u["run"] == 1 and u.get("pm") is not None}
    r2 = {u["uid"][:-1]: u for u in units.values() if u["run"] == 2 and u.get("pm") is not None}
    rep = [abs(r1[x]["pm"] - r2[x]["pm"]) for x in r2 if x in r1]
    return (float(np.percentile(rep, 90)) if rep else float("nan")), len(rep)


def stage_select():
    units = json.loads(LINEUPS.read_text())["units"]
    names = ladder_names()
    cands = distinct_candidates(names)
    pm = collect_pm(units)
    band, n_rep = band_from(units)
    print(f"band: p90 |d pm| = {band:.3f} (repeat n={n_rep})")

    stock = stock_rungs(len(names["0"]))
    rows = []
    for (L, i), labs in sorted(cands.items()):
        scored = [(lab, pm.get((L, i, lab)), min(rungs)) for lab, rungs in labs.items()]
        scored = [(lab, p, r) for lab, p, r in scored if p is not None]
        if not scored:
            continue
        best = max(p for _, p, _ in scored)
        ok = [(lab, p, r) for lab, p, r in scored if p >= best - band]
        lab_c, pm_c, rung_c = min(ok, key=lambda t: (words(t[0]), -t[2]))
        lab_s = names[str(stock[L])][L][i]
        rows.append(dict(L=L, i=i, chosen=lab_c, chosen_pm=pm_c, chosen_rung=rung_c,
                         chosen_words=words(lab_c), stock=lab_s,
                         stock_pm=pm.get((L, i, lab_s)), stock_words=words(lab_s),
                         best_pm=best, differs=lab_c != lab_s))

    print("\nrate-distortion by layer (mean pm / mean words per rung):")
    for L in sorted({r["L"] for r in rows}):
        cells = []
        for r in range(N_RUNGS):
            ps, ws = [], []
            for (LL, i), labs in cands.items():
                if LL != L:
                    continue
                for lab, rungs in labs.items():
                    if r in rungs and (LL, i, lab) in pm:
                        ps.append(pm[(LL, i, lab)])
                        ws.append(words(lab))
            mark = "*" if r == stock[L] else " "
            cells.append(f"r{r}{mark} {np.mean(ps):.2f}/{np.mean(ws):.0f}w" if ps else f"r{r}  -")
        print(f"  L{L}: " + "  ".join(cells) + "   (* = stock rung)")

    print("\ncontroller selection:")
    for L in sorted({r["L"] for r in rows}):
        rl = [r for r in rows if r["L"] == L]
        diff = [r for r in rl if r["differs"]]
        dw = np.mean([r["chosen_words"] - r["stock_words"] for r in rl])
        dp = np.mean([r["chosen_pm"] - (r["stock_pm"] or 0) for r in rl if r["stock_pm"] is not None])
        print(f"  L{L}: chosen rungs {dict(sorted(Counter(r['chosen_rung'] for r in rl).items()))}, "
              f"differs from stock {len(diff)}/{len(rl)}, d words {dw:+.1f}, d pm {dp:+.3f}")
    (SUMMARY).write_text(json.dumps(dict(band=band, rows=rows), indent=2))
    print(f"\nexamples (chosen != stock):")
    for r in [x for x in rows if x["differs"]][:8]:
        print(f"  L{r['L']}#{r['i']:<3} stock[{r['stock_words']}w pm={r['stock_pm']}]: {r['stock'][:56]!r}")
        print(f"        chosen[{r['chosen_words']}w pm={r['chosen_pm']:.2f} rung {r['chosen_rung']}]: {r['chosen'][:56]!r}")
    print(f"saved -> {SUMMARY}")
    return rows, band


# ------------------------------------------------------------------ held-out eval (Goodhart guard)

def judge_docs(cell, L, i, n_near=10, n_rand=5, maxlen=280):
    """Grounding docs by the judge convention (near + random, rng 1000+i), from the cell."""
    members = np.where(cell.layers[L].cluster_labels == i)[0]
    sims = unit_rows(cell.emb[members]) @ cell.cent[L][i]
    near = members[np.argsort(-sims)[:n_near]]
    rest = np.array([m for m in members if m not in set(near.tolist())])
    rng = np.random.default_rng(1000 + i)
    rand = rng.choice(rest, size=min(n_rand, rest.size), replace=False) if rest.size \
        else np.array([], int)
    return [" ".join(str(cell.objects[j]).split())[:maxlen] for j in list(near) + list(rand)]


def stage_heldout(concurrency):
    rows = json.loads(SUMMARY.read_text())["rows"]
    diff = [r for r in rows if r["differs"]]
    print(f"held-out eval on {len(diff)} clusters where chosen != stock")
    listener = "openai/gpt-4o-mini" if os.environ.get("OPENAI_API_KEY") else MODELS["haiku"]
    if "gpt" not in listener:
        print("  (OPENAI_API_KEY absent -- falling back to haiku as held-out listener)")
    cell = Cell(DS)
    cell._docs = {}
    _orig = Cell.held_out

    def held_out_v2(self, L, i):
        if (L, i) in self._docs:
            return self._docs[(L, i)]
        import numpy as _np
        members = _np.where(self.layers[L].cluster_labels == i)[0]
        ex = set(map(int, self.layers[L].exemplar_indices[i]))
        pool = _np.array([m for m in members if int(m) not in ex])
        if pool.size == 0:
            pool = members
        rng = _np.random.default_rng(md5i(f"wayfind-docs-v2|{L}|{i}"))
        take = rng.choice(pool, size=min(5, pool.size), replace=False)
        docs = [" ".join(str(self.objects[j]).split())[:500] for j in take]
        self._docs[(L, i)] = docs
        return docs

    Cell.held_out = held_out_v2
    try:
        units = []
        for r in diff:
            for kind, lab in [("ho-chosen", r["chosen"]), ("ho-stock", r["stock"])]:
                units.append(make_unit(cell, r["L"], r["i"], kind, lab, K_DEFAULT, "nn", run=1))
        units = asyncio.run(run_all(units, cell, listener, concurrency, HELDOUT))
    finally:
        Cell.held_out = _orig

    by = {(u["L"], u["i"], u["kind"]): u["pm"] for u in units.values()
          if u.get("pm") is not None}
    pairs = [(by.get((r["L"], r["i"], "ho-chosen")), by.get((r["L"], r["i"], "ho-stock")), r)
             for r in diff]
    pairs = [(c, s, r) for c, s, r in pairs if c is not None and s is not None]
    d = [c - s for c, s, _ in pairs]
    from scipy.stats import wilcoxon
    print(f"\nheld-out ({listener}): n={len(pairs)}  chosen pm {np.mean([c for c,_,_ in pairs]):.3f} "
          f"vs stock pm {np.mean([s for _,s,_ in pairs]):.3f}  (d {np.mean(d):+.3f}, "
          f"wilcoxon p={wilcoxon(d)[1]:.3f})" if pairs else "no pairs")
    dw = [words(r["chosen"]) - words(r["stock"]) for _, _, r in pairs]
    print(f"  words: chosen {np.mean([words(r['chosen']) for _,_,r in pairs]):.1f} "
          f"vs stock {np.mean([words(r['stock']) for _,_,r in pairs]):.1f} (d {np.mean(dw):+.1f})")

    # judge-fit non-regression (grounded sonnet judge; docs from the cell itself --
    # judge_fair.sample_docs would route arxiv through the misaligned examples loader)
    from async_judge import rate_many
    tasks, meta = [], []
    for _, _, r in pairs:
        docs = judge_docs(cell, r["L"], r["i"])
        tasks += [(r["chosen"], docs), (r["stock"], docs)]
        meta.append(r)
    ratings = rate_many(tasks, cell.meta["obj"], MODELS["sonnet"], k=3, concurrency=24)
    jd = [(ratings[2 * n]["overall"], ratings[2 * n + 1]["overall"]) for n in range(len(meta))]
    jd = [(c, s) for c, s in jd if c is not None and s is not None]
    dj = [c - s for c, s in jd]
    print(f"  judge fit: chosen {np.mean([c for c,_ in jd]):.2f} vs stock "
          f"{np.mean([s for _,s in jd]):.2f} (d {np.mean(dj):+.2f}, "
          f"wilcoxon p={wilcoxon(dj)[1] if any(dj) else float('nan'):.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["ladder", "lineups", "select", "heldout"])
    ap.add_argument("--dataset", default="20ng")
    ap.add_argument("--concurrency", type=int, default=24)
    args = ap.parse_args()
    set_dataset(args.dataset)
    if args.stage == "ladder":
        stage_ladder()
    elif args.stage == "lineups":
        stage_lineups(args.concurrency)
    elif args.stage == "select":
        stage_select()
    else:
        stage_heldout(args.concurrency)


if __name__ == "__main__":
    main()
