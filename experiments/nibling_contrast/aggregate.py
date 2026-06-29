"""
Aggregate result_*.json from the A/B matrix.

Key idea: a raw "change rate" conflates contrast-driven changes with model
nondeterminism. So for every layer we isolate the clusters that ACTUALLY got a
contrast block (injected_names) and report:
    effect  = change rate (contrast vs baseline A) on the injected subset
    floor   = change rate (baseline B vs baseline A) on the SAME subset
    NET     = effect - floor      <- the trustworthy signal
Layer 0 never gets contrast, so its row is a pure noise sanity check.

Run after the matrix completes:
  uv run python experiments/nibling_contrast/aggregate.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path("/Users/stevenfazzio/repos/toponymy/experiments/nibling_contrast/data")


def jacc(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    return len(ta & tb) / max(1, len(ta | tb))


def main():
    files = sorted(HERE.glob("result_*.json"))
    if not files:
        print("no result_*.json yet")
        return

    grid = {}
    for f in files:
        r = json.loads(f.read_text())
        ds, mdl = r["dataset"], r["model"]
        inj_by_layer = {}
        for key in r["injected_names"]:
            L, i = key.split(",")
            inj_by_layer.setdefault(int(L), set()).add(int(i))

        print(f"\n{'='*74}\n=== {ds} / {mdl}   (k={r['k']}, max_dist={r['max_dist']}, "
              f"disambig={r['disambig']}, {r.get('seconds','?')}s) ===")
        nd = r["injection"]["nearest_dist"]
        print(f"  clusters/layer {r['counts']}  injected={r['injection']['injected']}/"
              f"{r['injection']['calls']} (skip={r['injection']['skip']})  "
              f"nearest-dist {nd['min']:.2f}/{nd['med']:.2f}/{nd['max']:.2f}")
        print(f"  {'layer':>5} {'inj':>4} {'effect':>7} {'floor':>7} {'NET':>7}   (effect/floor on injected subset)")

        cell_nets = []
        for row, fr in zip(r["rows"], r["floor_rows"]):
            L = row["layer"]
            inj = inj_by_layer.get(L, set())
            cset = {i for i, _, _ in row["renames"]}
            fset = {i for i, _, _ in fr["renames"]}
            if inj:
                eff = len(cset & inj) / len(inj) * 100
                flr = len(fset & inj) / len(inj) * 100
                net = eff - flr
                cell_nets.append((L, net, len(inj)))
                print(f"  {('L'+str(L)):>5} {len(inj):>4} {eff:6.1f}% {flr:6.1f}% {net:+6.1f}%")
            else:
                # layer 0 (or no injections): show whole-layer noise as a sanity check
                print(f"  {('L'+str(L)):>5} {0:>4} {'--':>7} {'--':>7} {'--':>7}   "
                      f"(noise check: contrast {row['rate']*100:.1f}% / floor {fr['rate']*100:.1f}%)")
        grid[(ds, mdl)] = cell_nets

        # renames among INJECTED clusters only (the real candidates), tagged by edit size
        print("  renames on injected clusters (baseline -> contrast | contrast set):")
        shown = 0
        for row in r["rows"][1:]:
            L = row["layer"]
            inj = inj_by_layer.get(L, set())
            for (i, bn, cn) in row["renames"]:
                if i not in inj:
                    continue  # skip noise-only changes
                cs = r["injected_names"].get(f"{L},{i}", [])
                t = "[minor]     " if jacc(bn, cn) >= 0.5 else "[SUBSTANTIVE]"
                print(f"    {t} L{L} {bn!r} -> {cn!r}")
                print(f"                   vs: {', '.join(x[:26] for x in cs[:5])}")
                shown += 1
                if shown >= 12:
                    break
            if shown >= 12:
                break

    print(f"\n{'='*74}\nCROSS-CELL NET (%) ON INJECTED SUBSET, BY LAYER (effect minus floor)")
    for (ds, mdl), nets in sorted(grid.items()):
        s = "  ".join(f"L{L}:{net:+.0f}(n{n})" for L, net, n in nets)
        print(f"  {ds:>6}/{mdl:<10}  {s}")


if __name__ == "__main__":
    main()
