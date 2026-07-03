"""
Phase 4 gate (c) -- the 104 fine pairs through frozen lineups (the payoff question).

Each pair = two genuinely-good labels for the SAME cluster (base = haiku, alt = gpt-4o-mini);
the grounded judge tied on ~half and decided the rest (finepairs.json). Both labels run through
the cluster's frozen k=5 nn lineup -- identical distractors + held-out docs, only the label
differs -- with the sonnet listener. Per pair: delta = pm_alt - pm_base.

A pair is LINEUP-DECIDED when |delta| clears the sonnet repeat band, measured here directly on
the objects we interpret: a seeded subset of pairs is re-run in full (run=2) and the band is the
p90 of |delta_run1 - delta_run2|. (The floors stage's repeat band is haiku; deltas judged by the
sonnet listener need a sonnet band.)

Report answers gate (c): (1) does the lineup decide judge-tied pairs; (2) does it agree with the
judge on the judge-decided pairs; (3) --export-human writes a blinded A/B form over ~20
lineup-decided-judge-tied pairs (human preference vs the lineup's pick -- the Phase-0 calibration
bar applied to the new axis).

  uv run python experiments/label_quality/wayfinding_pairs.py                 # run + report
  uv run python experiments/label_quality/wayfinding_pairs.py --report-only
  uv run python experiments/label_quality/wayfinding_pairs.py --export-human
Writes data/wayfinding_20ng_pairs.json; human seed -> data/wayfinding_pair_calibration.html/.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
sys.path.insert(0, str(HERE))

from wayfinding import (Cell, K_DEFAULT, MODELS, make_unit, md5i, run_all)  # noqa: E402

N_REPEAT_PAIRS = 30   # seeded subset re-run for the sonnet repeat band
N_HUMAN = 20


def load_pairs():
    return json.loads((HERE / "data" / "finepairs.json").read_text())


def build_units(cell, pairs, repeat_idx):
    units = []
    for n, p in enumerate(pairs):
        runs = (1, 2) if n in repeat_idx else (1,)
        for r in runs:
            units.append(make_unit(cell, p["L"], p["i"], "pair-base", p["base"], K_DEFAULT, "nn", run=r))
            units.append(make_unit(cell, p["L"], p["i"], "pair-alt", p["alt"], K_DEFAULT, "nn", run=r))
    return units


def collect(units, pairs):
    """Per-pair rows: pm_base/pm_alt per run, delta, judge scores."""
    by = {(u["L"], u["i"], u["kind"], u["run"]): u for u in units.values() if u.get("pm") is not None}
    rows = []
    for n, p in enumerate(pairs):
        b1 = by.get((p["L"], p["i"], "pair-base", 1))
        a1 = by.get((p["L"], p["i"], "pair-alt", 1))
        if not (b1 and a1):
            continue
        row = dict(n=n, L=p["L"], i=p["i"], base=p["base"], alt=p["alt"],
                   j_base=p["j_base"], j_alt=p["j_alt"],
                   pm_base=b1["pm"], pm_alt=a1["pm"], delta=a1["pm"] - b1["pm"])
        b2 = by.get((p["L"], p["i"], "pair-base", 2))
        a2 = by.get((p["L"], p["i"], "pair-alt", 2))
        if b2 and a2:
            row["delta2"] = a2["pm"] - b2["pm"]
        rows.append(row)
    return rows


def report(rows):
    rep = [abs(r["delta"] - r["delta2"]) for r in rows if "delta2" in r]
    band = float(np.percentile(rep, 90)) if rep else float("nan")
    print(f"\n================ FINE PAIRS (gate c, sonnet) ================")
    print(f"pairs scored: {len(rows)};  repeat subset n={len(rep)}: "
          f"mean |d(delta)| {np.mean(rep):.3f}, p90 {band:.3f}, max {np.max(rep) if rep else float('nan'):.3f}")
    tied = [r for r in rows if r["j_base"] == r["j_alt"]]
    decided = [r for r in rows if r["j_base"] != r["j_alt"]]
    t_dec = [r for r in tied if abs(r["delta"]) > band]
    print(f"\n(1) judge-TIED pairs (n={len(tied)}): lineup decides {len(t_dec)} "
          f"({len(t_dec)/max(len(tied),1):.0%}) at |delta| > repeat band {band:.3f}; "
          f"mean |delta| {np.mean([abs(r['delta']) for r in tied]):.3f}")
    if decided:
        agree = [np.sign(r["delta"]) == np.sign(r["j_alt"] - r["j_base"]) for r in decided]
        d_dec = [r for r in decided if abs(r["delta"]) > band]
        agree_dec = [np.sign(r["delta"]) == np.sign(r["j_alt"] - r["j_base"]) for r in d_dec]
        print(f"(2) judge-DECIDED pairs (n={len(decided)}): sign agreement {np.mean(agree):.0%}; "
              f"on the {len(d_dec)} clearing the band: {np.mean(agree_dec) if agree_dec else float('nan'):.0%}")
    print("\nlargest lineup deltas on judge-tied pairs (human-seed candidates):")
    for r in sorted(tied, key=lambda r: -abs(r["delta"]))[:8]:
        pick = "alt" if r["delta"] > 0 else "base"
        print(f"  L{r['L']}#{r['i']:<3} d={r['delta']:+.3f} pick={pick:<4} "
              f"base={r['base'][:52]!r} alt={r['alt'][:52]!r}")
    return band, tied


HTML = """<!doctype html><meta charset="utf-8"><title>pair calibration</title>
<style>body{font-family:system-ui;max-width:900px;margin:2rem auto;padding:0 1rem;line-height:1.4}
.item{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1rem 0}
.docs{font-size:.85rem;background:#f6f6f6;padding:.6rem;border-radius:6px;max-height:260px;overflow-y:auto}
.lab{margin:.5rem 0;font-weight:600}button{padding:.5rem 1rem;font-size:1rem;margin:1rem 0}
textarea{width:100%;height:110px}</style>
<h1>Which label better identifies this group?</h1>
<p>Each item shows sample documents from ONE group of newsgroup posts, plus two candidate
labels. Pick the label that better <b>identifies</b> the group: the one you could more reliably
use to find this exact group among its neighbours. If truly equal, pick "can't decide".</p>
<div id="items"></div><button onclick="copyResults()">Copy results</button>
<textarea id="out" readonly></textarea>
<script>
const DATA = __DATA__;
const root = document.getElementById("items");
DATA.forEach((it, n) => {
  const d = document.createElement("div"); d.className = "item";
  d.innerHTML = `<b>Item ${n + 1} / ${DATA.length}</b>
    <div class="docs">${it.docs.map(x => `<p>${x}</p>`).join("")}</div>
    <div class="lab">A: ${it.A}</div><div class="lab">B: ${it.B}</div>
    <label><input type="radio" name="q${n}" value="A"> A</label>
    <label><input type="radio" name="q${n}" value="B"> B</label>
    <label><input type="radio" name="q${n}" value="tie"> can't decide</label>`;
  root.appendChild(d);
});
function copyResults(){
  const res = {};
  DATA.forEach((it, n) => {
    const v = document.querySelector(`input[name=q${n}]:checked`);
    if (v) res[it.id] = v.value;
  });
  const t = document.getElementById("out");
  t.value = JSON.stringify(res);
  t.select(); document.execCommand("copy");
}
</script>"""


def export_human(cell, rows, band):
    """Top-N judge-tied pairs by |delta|: the band-clearing ones are the test items; the
    sub-band tail rides along as blind controls (a meaningful band predicts human ties there)."""
    tied = sorted([r for r in rows if r["j_base"] == r["j_alt"]],
                  key=lambda r: -abs(r["delta"]))[:N_HUMAN]
    rng_order = np.random.default_rng(md5i("pairhuman-order"))
    rng_order.shuffle(tied)
    items, key = [], {}
    for r in tied:
        rng = np.random.default_rng(md5i(f"pairhuman|{r['L']}|{r['i']}"))
        flip = bool(rng.integers(2))
        A, B = (r["alt"], r["base"]) if flip else (r["base"], r["alt"])
        iid = f"pair_{r['L']}_{r['i']}"
        items.append(dict(id=iid, docs=cell.held_out(r["L"], r["i"]), A=A, B=B))
        key[iid] = dict(A="alt" if flip else "base", B="base" if flip else "alt",
                        lineup_pick="alt" if r["delta"] > 0 else "base",
                        clears_band=bool(abs(r["delta"]) > band),
                        delta=r["delta"], j_base=r["j_base"], j_alt=r["j_alt"])
    (HERE / "data" / "wayfinding_pair_calibration.html").write_text(
        HTML.replace("__DATA__", json.dumps(items).replace("</", "<\\/")))
    (HERE / "data" / "wayfinding_pair_calibration_key.json").write_text(json.dumps(key, indent=2))
    print(f"\nhuman seed: {len(items)} blinded A/B items -> data/wayfinding_pair_calibration.html "
          f"(+ key). Serve with: python3 -m http.server 8765 --bind 127.0.0.1 "
          f"--directory experiments/label_quality/data")


def score_human():
    """Gate (c)(3): human blinded A/B picks vs the lineup's picks, split by band status."""
    key = json.loads((HERE / "data" / "wayfinding_pair_calibration_key.json").read_text())
    H = json.loads((HERE / "data" / "wayfinding_pair_calibration_human.json").read_text())
    print("\n================ HUMAN SEED (gate c-3) ================")
    for grp, name in [(True, "band-clearing (test)"), (False, "sub-band (controls)")]:
        ids = [i for i in key if i in H and key[i]["clears_band"] == grp]
        ties = [i for i in ids if H[i] == "tie"]
        dec = [i for i in ids if H[i] in ("A", "B")]
        agree = [i for i in dec if key[i][H[i]] == key[i]["lineup_pick"]]
        print(f"{name:<22} n={len(ids)}  human ties {len(ties)} ({len(ties)/max(len(ids),1):.0%})"
              f"  |  of {len(dec)} decided: agree with lineup {len(agree)}"
              f" ({len(agree)/max(len(dec),1):.0%})")
    print("\nper item (sorted by |lineup delta|):")
    for i in sorted(key, key=lambda i: -abs(key[i]["delta"])):
        if i not in H:
            continue
        k = key[i]
        hp = "tie" if H[i] == "tie" else k[H[i]]
        mark = "=" if hp == "tie" else ("Y" if hp == k["lineup_pick"] else "n")
        print(f"  {i:<12} d={k['delta']:+.3f} {'BAND' if k['clears_band'] else '    '} "
              f"lineup={k['lineup_pick']:<4} human={hp:<4} {mark}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listener", default="sonnet")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--export-human", action="store_true")
    ap.add_argument("--score-human", action="store_true")
    args = ap.parse_args()

    if args.score_human:
        score_human()
        return

    cell = Cell("20ng")
    pairs = load_pairs()
    out = HERE / "data" / "wayfinding_20ng_pairs.json"
    repeat_idx = set(map(int, np.random.default_rng(4242).choice(len(pairs), N_REPEAT_PAIRS,
                                                                 replace=False)))
    if args.report_only or args.export_human:
        units = json.loads(out.read_text())["units"]
    else:
        us = build_units(cell, pairs, repeat_idx)
        print(f"fine pairs: {len(us)} units x 3 samples on {MODELS[args.listener]}")
        units = asyncio.run(run_all(us, cell, MODELS[args.listener], args.concurrency, out))

    rows = collect(units, pairs)
    band, _ = report(rows)
    if args.export_human:
        export_human(cell, rows, band)


if __name__ == "__main__":
    main()
