"""
Build the intrusion battery for an AT-HOME 2x2 cell -- the substrate where document/vector
alignment holds by construction (the bundled examples arXiv files are row-misaligned, issue
#176, so the original `perturbations.py --dataset arxiv` path grounds clusters with wrong
documents). Replays the cell's clustering via wayfinding.Cell, takes the gold labels the cell's
own haiku naming produced (home_<tag>.json), and runs the standard build_battery.

  uv run python experiments/label_quality/make_home_battery.py --dataset arxiv_home
Writes data/battery_<dataset>.json (the path wayfinding.py --dataset <dataset> expects).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
NIBLING = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(NIBLING))
sys.path.insert(0, str(HERE))

from perturbations import build_battery  # noqa: E402
from wayfinding import Cell, HOME_TAGS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="arxiv_home", choices=sorted(HOME_TAGS))
    args = ap.parse_args()

    tag = HOME_TAGS[args.dataset]
    home = json.loads((HERE / "data" / f"home_{tag}.json").read_text())
    cell = Cell(args.dataset)
    assert cell.counts == home["counts"], \
        f"replayed clustering {cell.counts} != at-home cell {home['counts']} -- do not proceed"

    gold = {(r["L"], r["i"]): r["label"] for r in home["rows"] if r["kind"] == "gold"}
    missing = [(L, i) for L in range(len(cell.counts)) for i in range(cell.counts[L])
               if (L, i) not in gold]
    assert not missing, f"home cell lacks gold labels for {missing[:5]}..."
    names = [[gold[(L, i)] for i in range(cell.counts[L])] for L in range(len(cell.counts))]

    battery = build_battery(cell.layers, cell.tree, names)
    cov = Counter(k for it in battery for k in it["variants"])
    print(f"[{args.dataset} <- home_{tag}] clusters/layer {cell.counts}; "
          f"battery {len(battery)} clusters; variant coverage {dict(cov)}")
    for it in battery[:3]:
        print(f"\n  (L{it['layer']},{it['idx']}) gold: {it['gold']!r}")
        for k, v in it["variants"].items():
            print(f"      {k:<9} {v[:88]!r}")

    out = HERE / "data" / f"battery_{args.dataset}.json"
    out.write_text(json.dumps(battery, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
