"""
Phase 7c follow-up -- the judge repeat band (~120 LLM calls).

7c's fit-axis headroom (+0.166 judge-pts from oracle best-of-2 over two naming draws) was compared
against a winner's-curse floor of +0.197 derived from `clean_docs_rejudge`, which re-scored the same
labels on a FRESH document draw -- so that floor carries doc-sample variance as well as judge noise
and is an upper bound. This runs the missing pure repeat: the SAME gold labels, the SAME documents
(judge_fair.sample_docs is deterministic, seed 1000+i), the same model/rubric/k, run again.

sigma from this repeat is the honest judge measurement noise, and
    floor(best-of-2) = sigma / sqrt(pi)
is the number 7c's +0.166 actually has to clear.

  uv run python experiments/label_quality/judge_repeat.py --n 40
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
NIBLING = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(NIBLING))
sys.path.insert(0, str(HERE))

OUT = HERE / "data" / "judge_repeat_20ng.json"
# exactly the committed run's recipe (judge_quality.py defaults)
N_NEAR, N_RAND, MAXLEN, K, TEMP = 10, 5, 280, 3, 0.7
MODEL = "anthropic/claude-sonnet-4-6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="gold labels to re-judge (x3 calls each)")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--report-only", action="store_true", help="re-derive from the saved file")
    args = ap.parse_args()

    if args.report_only:
        report(json.loads(OUT.read_text())["rows"])
        return

    from ab_harness import load_dataset
    from async_judge import rate_many
    from judge_fair import sample_docs

    battery = json.loads((HERE / "data" / "battery_20ng.json").read_text())
    prior = {(r["layer"], r["idx"]): r["overall"]
             for r in json.loads((HERE / "data" / "judge_ratings_20ng_sonnet.json").read_text())
             if r.get("type") == "gold" and r.get("overall") is not None}

    # deterministic subset spread across layers, so the band isn't a layer-0 artifact
    items = [it for it in battery if (it["layer"], it["idx"]) in prior]
    rng = np.random.default_rng(20260803)
    pick = sorted(rng.choice(len(items), size=min(args.n, len(items)), replace=False).tolist())
    items = [items[i] for i in pick]
    print(f"re-judging {len(items)} gold labels ({len(items)*K} calls), same docs, same recipe")

    obj = load_dataset("20ng", None)[3]["obj"]
    tasks = [(it["gold"], sample_docs("20ng", it["layer"], it["idx"],
                                      n_near=N_NEAR, n_rand=N_RAND, maxlen=MAXLEN))
             for it in items]
    ratings = rate_many(tasks, obj, MODEL, k=K, temp=TEMP, concurrency=args.concurrency)

    rows = []
    for it, r in zip(items, ratings):
        if r.get("overall") is None:
            continue
        rows.append(dict(layer=it["layer"], idx=it["idx"], label=it["gold"],
                         first=prior[(it["layer"], it["idx"])], second=r["overall"]))
    OUT.write_text(json.dumps({"model": MODEL, "config": dict(
        n_near=N_NEAR, n_rand=N_RAND, maxlen=MAXLEN, k=K, temp=TEMP), "rows": rows}, indent=1))

    report(rows)


def report(rows):
    OBSERVED = 0.166   # 7c oracle best-of-2 headroom, 20NG L0 k=4
    d = [r["second"] - r["first"] for r in rows]
    sd_diff = st.stdev(d)
    sigma = sd_diff / np.sqrt(2)
    floor = sigma / np.sqrt(np.pi)
    exact = sum(1 for r in rows if abs(r["second"] - r["first"]) < 1e-9)
    print(f"\nPURE JUDGE REPEAT (same label, same docs, same recipe): n={len(rows)}")
    print(f"  mean shift {st.mean(d):+.3f}   mean |d| {st.mean([abs(x) for x in d]):.3f}   "
          f"identical score {exact}/{len(rows)}   sd of paired diff {sd_diff:.3f}")
    print(f"  => sigma(one judge measurement) = {sigma:.3f}")
    print(f"  => WINNER'S-CURSE FLOOR for oracle best-of-2 = +{floor:.3f} judge-pts")

    # Which floor applies: exemplar_dose_response judges via sample_docs(ds, L, i), which is
    # deterministic per cluster -- so BOTH naming draws were scored on IDENTICAL documents.
    # The noise in the selection measurement is therefore pure judge noise, not the fresh-docs
    # quantity (+0.197) that clean_docs_rejudge implied.
    print(f"\n  DECOMPOSITION of 7c's fit headroom (both draws judged on identical docs):")
    print(f"    observed oracle best-of-2   +{OBSERVED:.3f}")
    print(f"    selection-on-noise          -{floor:.3f}   (sigma/sqrt(pi), sigma from this repeat)")
    print(f"    ------------------------------------")
    print(f"    TRUE oracle gain           ~+{OBSERVED - floor:.3f} judge-pts")
    print(f"\n  Reference: exemplars ablated -0.49; k=2 vs stock -0.18; controller vs stock +0.04 (n.s.)")
    print(f"  Caveat: this is an ORACLE gain (the judge itself selects). A deployed cheap selector")
    print(f"  recovers only a fraction of it, and 7a says cheap scorers are weak on fine distinctions.")


if __name__ == "__main__":
    main()
