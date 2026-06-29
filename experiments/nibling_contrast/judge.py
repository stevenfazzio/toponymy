"""
Phase 2: blind LLM-judge of the contrast-driven renames.

For each changed cluster, show a judge model 5 nearest-centroid documents and the two
candidate names, and ask which fits better. Each pair is judged in BOTH orders
(A=baseline/B=contrast and A=contrast/B=baseline); a winner only counts if it wins
both orders, otherwise it's a tie (position-bias + noise control).

Outputs:
  * contrast win-rate (the real Gate-3 signal), overall / per dataset / by edit size;
  * whether Δcentroid predicts the judge (mean Δcentroid for contrast-wins vs
    baseline-wins) -- i.e. is the cheap cosine metric usable after all.

Run:  uv run python experiments/nibling_contrast/judge.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from numpy.linalg import norm

HERE = Path("/Users/stevenfazzio/repos/toponymy/experiments/nibling_contrast")
sys.path.insert(0, str(HERE))
from ab_harness import load_dataset  # noqa: E402

import litellm  # noqa: E402
from toponymy.clustering import ToponymyClusterer  # noqa: E402
from toponymy.cluster_layer import ClusterLayerText  # noqa: E402

JUDGE = "anthropic/claude-haiku-4-5-20251001"
_cache = {}


def data_for(ds):
    if ds not in _cache:
        objects, emb, coords, _ = load_dataset(ds, None)
        cl = ToponymyClusterer(min_clusters=4, base_min_cluster_size=25, verbose=False)
        cl.fit_predict(coords, emb, ClusterLayerText)
        _cache[ds] = (cl, objects, emb)
    return _cache[ds]


def reps(ds, L, i, n=5, maxlen=300):
    cl, objects, emb = data_for(ds)
    labels = cl.cluster_layers_[L].cluster_labels
    cent = cl.cluster_layers_[L].centroid_vectors[i]
    members = np.where(labels == i)[0]
    c = cent / (norm(cent) + 1e-12)
    E = emb[members] / (norm(emb[members], axis=1, keepdims=True) + 1e-12)
    order = np.argsort(-(E @ c))[:n]
    return [" ".join(str(objects[members[o]]).split())[:maxlen] for o in order]


def ask(docs, A, B):
    block = "\n".join(f"- {d}" for d in docs)
    prompt = (
        f"Below are representative documents from a single group:\n{block}\n\n"
        f"Two candidate names for this group:\n  A: {A}\n  B: {B}\n\n"
        "Which name better describes the group as a whole? Weigh accuracy, the right "
        "level of specificity, and conciseness. Reply ONLY as JSON: "
        '{"winner": "A" | "B" | "tie", "reason": "<short>"}.'
    )
    try:
        r = litellm.completion(model=JUDGE, messages=[{"role": "user", "content": prompt}],
                               temperature=0.0, max_tokens=200)
        m = re.search(r'"winner"\s*:\s*"(A|B|tie)"', r.choices[0].message.content)
        return m.group(1) if m else "tie"
    except Exception as e:
        print("  judge error:", type(e).__name__, str(e)[:80])
        return "tie"


def main():
    insts = json.loads((HERE / "data" / "cosine_instances.json").read_text())
    out = []
    t0 = time.time()
    for k, x in enumerate(insts):
        docs = reps(x["ds"], x["L"], x["i"])
        w1 = ask(docs, x["old"], x["new"])      # A=baseline, B=contrast
        w2 = ask(docs, x["new"], x["old"])      # A=contrast, B=baseline
        c1 = {"B": "contrast", "A": "baseline", "tie": "tie"}[w1]
        c2 = {"A": "contrast", "B": "baseline", "tie": "tie"}[w2]
        winner = c1 if c1 == c2 else "tie"
        out.append({**{kk: x[kk] for kk in ("ds", "mdl", "L", "i", "old", "new", "dcent", "old_new")},
                    "winner": winner})
        if (k + 1) % 20 == 0:
            print(f"  judged {k+1}/{len(insts)} ({time.time()-t0:.0f}s)")

    (HERE / "data" / "judge_results.json").write_text(json.dumps(out, indent=2))

    def tally(rows):
        c = sum(r["winner"] == "contrast" for r in rows)
        b = sum(r["winner"] == "baseline" for r in rows)
        t = sum(r["winner"] == "tie" for r in rows)
        dec = c + b
        return c, b, t, (100 * c / dec if dec else float("nan"))

    print("\n================ BLIND JUDGE (contrast vs baseline) ================")
    c, b, t, wr = tally(out)
    print(f"OVERALL: contrast {c} | baseline {b} | tie {t}   -> contrast win-rate "
          f"{wr:.0f}% of decided ({c+b})")
    for ds in ("20ng", "arxiv"):
        rows = [r for r in out if r["ds"] == ds]
        c, b, t, wr = tally(rows)
        print(f"  {ds:>6}: contrast {c} | baseline {b} | tie {t}  -> {wr:.0f}% (n_dec={c+b})")
    for mdl in ("gpt4omini", "haiku", "sonnet"):
        rows = [r for r in out if r["mdl"] == mdl]
        c, b, t, wr = tally(rows)
        print(f"  {mdl:>9}: contrast {c} | baseline {b} | tie {t}  -> {wr:.0f}% (n_dec={c+b})")

    print("\n--- does Δcentroid predict the judge? ---")
    cw = [r["dcent"] for r in out if r["winner"] == "contrast"]
    bw = [r["dcent"] for r in out if r["winner"] == "baseline"]
    print(f"  mean Δcentroid | contrast-win = {np.mean(cw):+.3f} (n={len(cw)})  "
          f"baseline-win = {np.mean(bw):+.3f} (n={len(bw)})")
    decided = [r for r in out if r["winner"] != "tie"]
    agree = sum((r["dcent"] > 0) == (r["winner"] == "contrast") for r in decided)
    print(f"  sign(Δcentroid) agrees with judge on {agree}/{len(decided)} "
          f"= {100*agree/max(1,len(decided)):.0f}% (50% = no predictive value)")
    print(f"\n{len(out)} judged in {time.time()-t0:.0f}s -> data/judge_results.json")


if __name__ == "__main__":
    main()
