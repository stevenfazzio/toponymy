"""
Phase 3 (gate) -- is the label<->document gap a low-rank LINEAR offset, or nonlinear?

Gates idea A (erase the label/doc difference). From the shared-datamap work: a corpus-identity gap
is low-rank-linear and erasable; a modality gap is nonlinear (linear erasure kills recoverability but
not interleaving). Which is the label/doc gap? If it's a clean low-rank offset, erasure (LEACE) is
worth trying; if nonlinear, it isn't (and whitening already handled the practical confound anyway).

Pool Toponymy label embeddings (class 1) and ALL document embeddings (class 0), MiniLM, unit-norm.
Using all 7000 docs (n >> d=384) with balanced class weights keeps linear separability meaningful
(not a d>=n artifact). Report:
  - raw linear separability (balanced-accuracy, 5-fold CV) -- expect high
  - after PER-CLASS CENTERING (remove each class mean) -- if this drops to ~0.5, the gap is just a
    rank-1 MEAN OFFSET (cleanly erasable); if separability survives, it is NOT only an offset
  - INLP rank sweep -- how many linear directions must be removed before label/doc are inseparable
    (low rank => linear/erasable; stays high => nonlinear)

  uv run python experiments/label_quality/phase3_rank_diagnostic.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from numpy.linalg import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
NIBLING = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(NIBLING))

EMB_MODEL = "all-MiniLM-L6-v2"


def bal_acc(X, y):
    return cross_val_score(LogisticRegression(max_iter=2000, class_weight="balanced"),
                           X, y, cv=5, scoring="balanced_accuracy").mean()


def main():
    from ab_harness import load_dataset

    _, emb, _, _ = load_dataset("20ng", None)
    docs = emb.astype(np.float64)

    labs = set()
    for m in ("haiku", "gpt4omini"):
        for layer in json.loads((HERE / "data" / f"labels_20ng_{m}.json").read_text()):
            labs |= {l for l in layer if l and l != "Unlabelled"}
    labs = sorted(labs)
    from sentence_transformers import SentenceTransformer
    LE = SentenceTransformer(EMB_MODEL, device="cpu").encode(labs, normalize_embeddings=False,
                                                             convert_to_numpy=True).astype(np.float64)
    print(f"{len(labs)} unique Toponymy labels vs {len(docs)} documents (d={docs.shape[1]})")

    X = np.vstack([LE, docs])
    X = X / (norm(X, axis=1, keepdims=True) + 1e-12)               # unit (cosine space)
    y = np.r_[np.ones(len(LE)), np.zeros(len(docs))]

    print("\nlabel-vs-doc linear separability (balanced-accuracy, 5-fold CV; chance = 0.50):")
    print(f"  raw                  {bal_acc(X, y):.3f}")
    Xc = X.copy()
    Xc[y == 1] -= X[y == 1].mean(0)
    Xc[y == 0] -= X[y == 0].mean(0)
    print(f"  per-class-centered   {bal_acc(Xc, y):.3f}   (~0.50 => the gap is a rank-1 mean offset)")

    print("\nINLP rank sweep (remove the top separating direction each step):")
    Xs = X.copy()
    for k in range(8):
        acc = bal_acc(Xs, y)
        print(f"  rank-{k:>2}: balanced-acc {acc:.3f}")
        if acc < 0.55:
            print(f"  -> linearly inseparable after removing {k} direction(s)")
            break
        w = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xs, y).coef_[0]
        w /= norm(w)
        Xs = Xs - np.outer(Xs @ w, w)
    else:
        print("  -> still separable after 8 directions => high-rank / nonlinear gap")


if __name__ == "__main__":
    main()
