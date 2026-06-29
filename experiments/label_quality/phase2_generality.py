"""
Phase 2a -- generality-proxy bake-off on HyperLex.

Does a linear GENERALITY axis exist in our (MiniLM) embedding space, and how does it compare to
cheap baselines and Renner's published WordNet-IC bar? HyperLex grades "to what degree is WORD1 a
type of WORD2" (0-6); a good generality proxy G should make the gap G(WORD2) - G(WORD1) track it
(WORD2 = the candidate hypernym, i.e. the more general term).

Proxies for G(word):
  emb-axis  : projection onto a generality direction LEARNED from HyperLex train-split hypernym
              pairs (the idea-D probe -- the only proxy that works directly on phrase-labels)
  frequency : log corpus frequency (wordfreq zipf) -- more frequent ~ more general
  emb-norm  : raw embedding norm -- is generality encoded radially? (hyperbolic-radius analog)
  length    : -char length -- naive floor
plus cosine (the symmetric relatedness baseline; Renner ~0.205) and a learned combined model.

Spearman is computed on the held-out lexical TEST split, vs Renner bars
(cosine 0.205 / LEAR 0.686 / WordNet-IC 0.744 / human ceiling 0.854).

  uv run --with wordfreq python experiments/label_quality/phase2_generality.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.linalg import norm
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression

HERE = Path("/Users/stevenfazzio/repos/toponymy/experiments/label_quality")
HL = HERE / "data" / "hyperlex" / "splits" / "lexical"
EMB_MODEL = "all-MiniLM-L6-v2"  # same embedder the 20NG docs/labels use


def load(path):
    rows = []
    for ln in path.read_text().splitlines():
        t = ln.split()
        if not t or t[0] == "WORD1":
            continue
        rows.append((t[0], t[1], t[3], float(t[4])))  # w1, w2, type, avg_score(0-6)
    return rows


def main():
    train = load(HL / "hyperlex_training_all_lexical.txt")
    test = load(HL / "hyperlex_test_all_lexical.txt")
    vocab = sorted({w for rows in (train, test) for (w1, w2, _, _) in rows for w in (w1, w2)})
    print(f"HyperLex lexical split: {len(train)} train / {len(test)} test pairs; {len(vocab)} words")

    from sentence_transformers import SentenceTransformer
    raw = SentenceTransformer(EMB_MODEL, device="cpu").encode(vocab, normalize_embeddings=False,
                                                              convert_to_numpy=True).astype(np.float64)
    U = {w: raw[i] / (norm(raw[i]) + 1e-12) for i, w in enumerate(vocab)}   # unit (cosine/axis space)
    NRM = {w: float(norm(raw[i])) for i, w in enumerate(vocab)}             # raw norm

    # --- learn the generality axis from clear train hypernym pairs (score >= 4 => W2 more general) ---
    pos = [(w1, w2) for (w1, w2, _, sc) in train if sc >= 4.0]
    D = np.array([U[w2] - U[w1] for w1, w2 in pos])
    X = np.vstack([D, -D])
    y = np.r_[np.ones(len(D)), np.zeros(len(D))]
    axis = LogisticRegression(fit_intercept=False, max_iter=2000, C=1.0).fit(X, y).coef_[0]
    axis /= norm(axis)
    np.save(HERE / "data" / "generality_axis.npy", axis)  # for phase2b (same EMB_MODEL)
    print(f"generality axis learned from {len(pos)} train hypernym pairs (score>=4); saved axis")

    from wordfreq import zipf_frequency
    zipf = {w: zipf_frequency(w, "en") for w in vocab}

    def feats(rows):
        cos = np.array([U[w1] @ U[w2] for (w1, w2, _, _) in rows])
        gap_axis = np.array([(U[w2] - U[w1]) @ axis for (w1, w2, _, _) in rows])
        gap_freq = np.array([zipf[w2] - zipf[w1] for (w1, w2, _, _) in rows])
        gap_norm = np.array([NRM[w2] - NRM[w1] for (w1, w2, _, _) in rows])
        gap_len = np.array([len(w1) - len(w2) for (w1, w2, _, _) in rows])  # W2 shorter => more general
        sc = np.array([sc for (_, _, _, sc) in rows])
        return dict(cosine=cos, **{"emb-axis": gap_axis, "frequency": gap_freq,
                                   "emb-norm": gap_norm, "length": gap_len}), sc

    ftr, sctr = feats(train)
    fte, scte = feats(test)

    print("\nSpearman( proxy , HyperLex graded score )  on the held-out TEST split")
    print("  Renner bars:  cosine 0.205 | LEAR 0.686 | WordNet-IC 0.744 | human 0.854")
    for name in ["cosine", "emb-axis", "frequency", "emb-norm", "length"]:
        rho = spearmanr(fte[name], scte)[0]
        tag = "  (symmetric relatedness)" if name == "cosine" else "  (directional generality gap)"
        print(f"  {name:<10} rho = {rho:+.3f}{tag}")

    # learned combined: cosine (relatedness) + emb-axis (directional generality), fit on train
    for combo in (["cosine", "emb-axis"], ["cosine", "emb-axis", "frequency", "emb-norm", "length"]):
        Xtr = np.column_stack([ftr[k] for k in combo])
        Xte = np.column_stack([fte[k] for k in combo])
        pred = LinearRegression().fit(Xtr, sctr).predict(Xte)
        print(f"  combined({'+'.join(combo)}) rho = {spearmanr(pred, scte)[0]:+.3f}")


if __name__ == "__main__":
    main()
