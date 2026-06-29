"""
Fair-judge pass -- addresses the blind judge's weaknesses (it couldn't see the neighbours,
saw fewer samples than the namer, and was a weaker model than some namers):

  * judge model = Sonnet (strongest model that runs deterministically at temp 0;
    >= every namer except Opus);
  * the contrast/neighbour names ARE shown, and the judge is asked which name fits the
    cluster AND sets it apart from the neighbours (distinctiveness is contrast's whole point);
  * ~20 documents sampled for COVERAGE (nearest-centroid + a random spread), raw docs rather
    than the namer's keyphrases (an independent check, not a re-grade of the same distillation);
  * both A/B orders, disagreement = tie.

Re-judges every model's contrast-driven renames (from result_*.json) so we get fair-vs-blind
win-rates side by side. Run: uv run python experiments/nibling_contrast/judge_fair.py
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
from judge import data_for  # noqa: E402  (fits clusterer + loads objects/emb, cached per dataset)

import litellm  # noqa: E402

JUDGE = "anthropic/claude-sonnet-4-6"


def sample_docs(ds, L, i, n_near=12, n_rand=8, maxlen=280):
    cl, objects, emb = data_for(ds)
    labels = cl.cluster_layers_[L].cluster_labels
    cent = cl.cluster_layers_[L].centroid_vectors[i]
    members = np.where(labels == i)[0]
    if members.size == 0:
        return []
    c = cent / (norm(cent) + 1e-12)
    E = emb[members] / (norm(emb[members], axis=1, keepdims=True) + 1e-12)
    near = members[np.argsort(-(E @ c))[:n_near]]
    rest = np.array([m for m in members if m not in set(near.tolist())])
    rng = np.random.default_rng(1000 + i)  # deterministic spread sample
    rand = rng.choice(rest, size=min(n_rand, rest.size), replace=False) if rest.size else np.array([], dtype=int)
    idx = list(near) + list(rand)
    return [" ".join(str(objects[j]).split())[:maxlen] for j in idx]


def ask_fair(docs, neighbors, A, B):
    docblock = "\n".join(f"- {d}" for d in docs)
    nb = "\n".join(f"- {n}" for n in neighbors) if neighbors else "(none provided)"
    prompt = (
        f"Below are representative documents from ONE group:\n{docblock}\n\n"
        f"For context, here are a few SEPARATE neighbouring groups (NOT part of this group):\n{nb}\n\n"
        f"Two candidate names for the group:\n  A: {A}\n  B: {B}\n\n"
        "Which name better describes this group as a whole AND sets it apart from the neighbouring "
        "groups? Weigh accuracy, the right level of specificity, and conciseness. "
        'Reply ONLY as JSON: {"winner": "A" | "B" | "tie", "reason": "<short>"}.'
    )
    try:
        r = litellm.completion(model=JUDGE, messages=[{"role": "user", "content": prompt}],
                               temperature=0.0, max_tokens=220)
        m = re.search(r'"winner"\s*:\s*"(A|B|tie)"', r.choices[0].message.content)
        return m.group(1) if m else "tie"
    except Exception as e:
        print("  err:", type(e).__name__, str(e)[:80])
        return "tie"


def main():
    files = sorted((HERE / "data").glob("result_*.json"))
    by_model, by_ds, total = {}, {}, {"contrast": 0, "baseline": 0, "tie": 0}
    t0 = time.time()
    n = 0
    for f in files:
        r = json.loads(f.read_text())
        ds, mdl = r["dataset"], r["model"]
        inj = r["injected_names"]
        for row in r["rows"][1:]:
            L = row["layer"]
            for (i, old, new) in row["renames"]:
                key = f"{L},{i}"
                if key not in inj:
                    continue
                docs = sample_docs(ds, L, i)
                neigh = inj[key]
                w1 = ask_fair(docs, neigh, old, new)   # A=baseline, B=contrast
                w2 = ask_fair(docs, neigh, new, old)   # A=contrast, B=baseline
                c1 = {"B": "contrast", "A": "baseline", "tie": "tie"}[w1]
                c2 = {"A": "contrast", "B": "baseline", "tie": "tie"}[w2]
                win = c1 if c1 == c2 else "tie"
                for bucket in (by_model.setdefault(mdl, {"contrast": 0, "baseline": 0, "tie": 0}),
                               by_ds.setdefault(ds, {"contrast": 0, "baseline": 0, "tie": 0}), total):
                    bucket[win] += 1
                n += 1
                if n % 25 == 0:
                    print(f"  judged {n} ({time.time()-t0:.0f}s)", flush=True)

    def wr(d):
        dec = d["contrast"] + d["baseline"]
        return 100 * d["contrast"] / dec if dec else float("nan"), dec

    print("\n================ FAIR JUDGE (Sonnet, neighbours shown, 20 docs) ================")
    print("blind-judge reference: gpt4omini 30% | haiku 26% | sonnet 26% | opus 43%")
    for mdl, d in sorted(by_model.items()):
        w, dec = wr(d)
        print(f"  {mdl:>9}: contrast {d['contrast']} | baseline {d['baseline']} | tie {d['tie']}  "
              f"-> fair win-rate {w:.0f}% (n_dec={dec})")
    for ds, d in sorted(by_ds.items()):
        w, dec = wr(d)
        print(f"  {ds:>6}: -> {w:.0f}% (n_dec={dec})")
    w, dec = wr(total)
    print(f"  OVERALL: contrast {total['contrast']} | baseline {total['baseline']} | "
          f"tie {total['tie']}  -> fair win-rate {w:.0f}% (n_dec={dec})")
    (HERE / "data" / "judge_fair.json").write_text(
        json.dumps({"by_model": by_model, "by_ds": by_ds, "total": total}, indent=2))


if __name__ == "__main__":
    main()
