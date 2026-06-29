"""
Prepare a 20 Newsgroups testbed for the contrast A/B: document embeddings
(MiniLM, 384d) + a 2D UMAP map + the raw texts, cached to ./data/ so the harness
loads instantly on every (model x arm) run.

20NG is the multi-domain corpus where the contrast effect should be clearest:
genuinely distinct nearby categories (rec.sport.baseball vs rec.sport.hockey,
comp.sys.ibm vs comp.sys.mac, talk.religion vs alt.atheism, sci.space vs
sci.med, ...). Ground-truth category labels are saved too, for optional
"do clusters line up with real categories" sanity analysis later.

Run once (slow: fetch + embed + UMAP):
  uv run --project /Users/stevenfazzio/repos/toponymy \
      python experiments/nibling_contrast/prep_20ng.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups

OUT = Path("/Users/stevenfazzio/repos/toponymy/experiments/nibling_contrast/data")
N = 7000               # deterministic subsample size (keeps embedding fast)
SEED = 42
EMB_MODEL = "all-MiniLM-L6-v2"   # 384d; the harness must use the SAME model for 20NG


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ng = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    texts, targets = ng.data, ng.target

    # Drop near-empty docs (stripping headers/footers/quotes leaves some blank).
    keep = [i for i, t in enumerate(texts) if len(t.strip()) >= 40]
    texts = [texts[i] for i in keep]
    targets = targets[keep]

    rng = np.random.default_rng(SEED)
    sel = np.sort(rng.choice(len(texts), size=min(N, len(texts)), replace=False))
    texts = [texts[i] for i in sel]
    targets = targets[sel]
    print(f"20NG: {len(texts)} docs after filter+sample; {len(ng.target_names)} categories")

    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(EMB_MODEL, device="cpu")
    emb = enc.encode(
        texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    ).astype(np.float32)
    print("emb:", emb.shape)

    import umap

    coords = (
        umap.UMAP(n_components=2, metric="cosine", n_neighbors=15, random_state=SEED)
        .fit_transform(emb)
        .astype(np.float32)
    )
    print("coords:", coords.shape)

    np.save(OUT / "ng_emb.npy", emb)
    np.save(OUT / "ng_coords.npy", coords)
    np.save(OUT / "ng_targets.npy", targets)
    (OUT / "ng_texts.json").write_text(json.dumps(texts))
    (OUT / "ng_target_names.json").write_text(json.dumps(list(ng.target_names)))
    print("saved testbed to", OUT)


if __name__ == "__main__":
    main()
