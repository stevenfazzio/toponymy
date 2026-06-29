"""
Judge the Opus contrast renames in isolation -- does the strongest model rescue contrast?

Reads only result_*_opus.json (so it doesn't re-judge / perturb the existing haiku/sonnet/
gpt-4o-mini verdicts), builds the changed-injected-cluster instances, and blind-judges each
in both A/B orders (reusing judge.ask / judge.reps). Reports the Opus win-rate overall and
per dataset, comparable to the matrix's gpt-4o-mini 30% / haiku 26% / sonnet 26%.

Run:  uv run python experiments/nibling_contrast/judge_opus.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path("/Users/stevenfazzio/repos/toponymy/experiments/nibling_contrast")
sys.path.insert(0, str(HERE))
from judge import ask, reps  # noqa: E402  (reuse the blind-judge call + doc sampler)


def main():
    files = sorted((HERE / "data").glob("result_*_opus.json"))
    if not files:
        print("no result_*_opus.json yet")
        return
    totals = {"contrast": 0, "baseline": 0, "tie": 0}
    for f in files:
        r = json.loads(f.read_text())
        ds = r["dataset"]
        inj = set(r["injected_names"])
        wins = {"contrast": 0, "baseline": 0, "tie": 0}
        for row in r["rows"][1:]:
            L = row["layer"]
            for (i, old, new) in row["renames"]:
                if f"{L},{i}" not in inj:
                    continue
                docs = reps(ds, L, i)
                w1 = ask(docs, old, new)   # A=baseline, B=contrast
                w2 = ask(docs, new, old)   # A=contrast, B=baseline
                c1 = {"B": "contrast", "A": "baseline", "tie": "tie"}[w1]
                c2 = {"A": "contrast", "B": "baseline", "tie": "tie"}[w2]
                wins[c1 if c1 == c2 else "tie"] += 1
        dec = wins["contrast"] + wins["baseline"]
        wr = 100 * wins["contrast"] / dec if dec else float("nan")
        print(f"{ds}/opus: contrast {wins['contrast']} | baseline {wins['baseline']} | "
              f"tie {wins['tie']}  -> win-rate {wr:.0f}% (n_dec={dec})")
        for k in totals:
            totals[k] += wins[k]
    dec = totals["contrast"] + totals["baseline"]
    wr = 100 * totals["contrast"] / dec if dec else float("nan")
    print(f"OPUS OVERALL: contrast {totals['contrast']} | baseline {totals['baseline']} | "
          f"tie {totals['tie']}  -> win-rate {wr:.0f}% (n_dec={dec})")
    (HERE / "data" / "judge_opus.json").write_text(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
