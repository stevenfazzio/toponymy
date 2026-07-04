I was running an evaluation harness against the bundled example data and got results that only made sense if documents were attached to the wrong vectors. After some digging I'm fairly confident the three `examples/ai_arxiv_*` files (added in 909c599) are not in a consistent row order: `ai_arxiv_vectors.npy` and `ai_arxiv_coordinates.npz.npy` agree with each other, but `ai_arxiv_papers.zip` appears to be sorted differently, so pairing them row-wise attaches each paper to some other paper's embedding.

A reproduction that needs nothing but numpy and pandas: take a vector's nearest neighbour in embedding space and look up both rows' titles in the zip.

```python
import zipfile, numpy as np, pandas as pd
emb = np.load("examples/ai_arxiv_vectors.npy")
with zipfile.ZipFile("examples/ai_arxiv_papers.zip") as z:
    papers = pd.read_csv(z.open("ai_arxiv_papers"))
U = emb / np.linalg.norm(emb, axis=1, keepdims=True)
for i in [1411, 2224, 1430]:
    sims = U @ U[i]; sims[i] = -1
    j = int(np.argmax(sims))
    print(f"cos={sims[j]:.2f}  [{i}] {papers.title[i][:60]!r}  [{j}] {papers.title[j][:60]!r}")
```

```
cos=1.00  [1411] 'trading complexity for sparsity in random forest explanations'  [1410] 'towards ai enabled automated tracking of multiple boxers'
cos=0.89  [2224] 'addressing visual search in open and closed set settings'  [2225] 'initialization and regularization of factorized neural layers'
cos=1.00  [1430] 'a reinforcement learning approach to hybrid control design'  [1431] 'ai-driven mobile apps: an explorative study'
```

Near-identical vectors, unrelated papers. By contrast, the vectors and the 2D coordinates are mutually consistent: across 200 random probes, a vector's nearest neighbour in embedding space sits at median rank 2 out of 10,000 by map distance. So the zip (or the vectors) got re-sorted relative to the other at some point.

This matters because `doc/topic_summaries.ipynb` loads all three files and pairs them row-wise (the notebook asserts the lengths match, which passes at 10,000 = 10,000 but can't catch reordering). The downstream symptom is quiet and easy to miss: clustering still works (it runs on the vectors, which are fine), but exemplar texts and keyphrases come from the wrong documents, so every fine-layer topic name comes out as a vague composite. In my run, layer-0 names were almost uniformly of the form "Diverse Machine Learning Applications Spanning X, Y, and Z". Nothing errors, and an LLM judging label-vs-documents fit doesn't complain either, because a vague label fits a scrambled document sample about as well as anything.

The fix is presumably to regenerate the zip in the vectors' row order (or re-embed from the zip's order). Happy to help test a corrected version.
