# Can you judge Toponymy label quality from embedding geometry?

> **DRAFT — not for posting.** Holding for two more experiments (fine-grained discrimination;
> label↔document erasure) and a voice pass before sharing to TutteInstitute/toponymy Discussion #173.

## TL;DR

Scoring a Toponymy region label by the embedding cosine to its cluster is **usable as a coarse
guardrail** — it reliably catches bad labels (Spearman ≈ 0.74 against a human-validated LLM judge) —
but **not as a fine-grained ranker** among already-good labels. That reconciles an earlier "it's
basically chance" negative result: the metric works on *good-vs-bad*, not *good-vs-good*.

The classic failure mode — it rewards verbose / over-padded labels — is an **anisotropy** artifact,
and it's fixed by **whitening the embedding space before scoring**, *not* by swapping the centroid
for a medoid or exemplar. Separately, a geometric **hierarchy** check finds that Toponymy's coarse
labels **under-generalize ≈14%** of the time (a coarse region inheriting a child's exact name, plus
mis-altituded labels like *"Waco Siege"* sitting above broader political topics). The cheapest
reliable "is the parent more general than the child?" signal turns out to be plain **label length** —
a *learned* generality axis is at chance on multi-word labels.

## The question

A prior experiment (the contrast/nibling work, Discussion #173) noted in passing that **cosine
similarity between a label embedding and its cluster centroid does not track judged label quality** —
it rewarded verbosity, penalized specificity, and agreed with a judge at roughly chance. Two
hypotheses for why: (1) labels (short abstract phrases) and documents (long prose) are different
*classes* of text in embedding space; (2) a cluster's centroid ≠ its extent. This is an attempt to
rescue the embedding-space approach — or to characterize precisely why it can't be rescued.

## Setup

- **Corpus:** 20 Newsgroups (7,000 docs), MiniLM (`all-MiniLM-L6-v2`) embeddings, fit with Toponymy
  (haiku namer) → 3 layers of 74 / 24 / 9 named regions.
- **Intrusion battery:** for each gold label, Toponymy-native *known-bad* variants drawn from
  `cluster_tree_` — `ancestor` (the coarser parent label), `sibling` (a sibling region's label),
  `distant` (a far region's label), plus synthesized `generic` ("Various topics") and `verbose`
  (the gold over-qualified with parent/sibling material). A quality metric must rank gold above each.
- **Judge:** a grounded, self-consistent LLM judge (sonnet, k=3) scoring each label on a Preiss-style
  rubric (consistency / relevance / completeness / overall, 0–4), grounded in ~15 cluster documents
  (never reference-free — cf. Krumdick et al. 2025). **Human-calibrated** (below).

## Finding 1 — a coarse guardrail, not a fine ranker

Across all four reference points (raw centroid / whitened / medoid / top-exemplar), the label↔cluster
cosine correlates with the grounded judge at **Spearman ρ ≈ 0.71–0.74** (n = 487 candidates,
p ~ 1e-80), with 91–94% pairwise gold-vs-variant agreement. So embedding-similarity-to-cluster **is**
a usable quality signal.

The apparent contradiction with the earlier "≈ chance" result dissolves once you separate *coarse*
from *fine* discrimination: the prior test asked whether the centroid-similarity *delta* between two
*good* labels predicts which a judge prefers (fine, ≈ chance); this asks whether similarity separates
*good from degraded* (coarse, ρ = 0.74). **The metric is a guardrail (catch bad labels), not a
fine-grained ranker.**

## Finding 2 — the failure is anisotropy; whiten, don't switch reference point

The one degradation that actually discriminates between metrics is `verbose` (on-topic padding — the
documented failure mode). On the judge-free intrusion gate:

| reference point | strict top-1 (gold beats all variants) | `verbose` intrusion |
|---|---|---|
| raw centroid | 72.0% | 77.6% |
| **whitened centroid** | **89.7%** | **91.6%** |
| medoid | 74.8% | 84.1% |
| top-exemplar | 78.5% | 86.0% |

**Whitening the (document-fit) embedding space fixes the verbosity bias; switching to a real point
(medoid/exemplar) barely helps and is *worse* on ancestor/sibling.** So the centroid's averaging is
*helpful*; the problem was the embedding space's narrow-cone **anisotropy** (Mu & Viswanath 2018;
Su et al. 2021), which inflates similarity along a few dominant directions that padded labels exploit.

One honest wrinkle: on the *graded* judge correlation, raw centroid is marginally **best** (ρ = 0.740
vs whitened 0.708) — its anisotropic over-separation actually amplifies the big good-vs-bad gaps — but
whitening wins on the *pairwise* agreement that isolates the hard verbose cases (94.4% vs 91.1%
overall; **79.6% → 92.5%** on `verbose`). **Pick whitened for robustness to padding; raw centroid for
raw graded correlation. Both are usable; neither is a fine ranker.**

## Finding 3 — the hierarchy under-generalizes; length beats a learned generality axis

Can geometry check that a *parent* label is a true generalization of its *children*? We learned a
linear "generality axis" in MiniLM space from HyperLex (graded lexical entailment) and tested it.

- **On HyperLex** the axis is real but weak: lexical-split test Spearman 0.32 (vs cosine 0.34,
  combined 0.45) — far below structure-based WordNet-IC (Renner et al. 2023, 0.744). Embeddings encode
  hypernymy weakly and entangled with relatedness, as the literature predicts.
- **On Toponymy's own hierarchy** (86 real parent→child edges) the axis collapses to **chance (50.0%
  direction accuracy)** — single-word hypernymy geometry does not transfer to multi-word region
  labels. Meanwhile **label length is right 86%** of the time (the coarser label is simply shorter),
  and word frequency is actively *wrong* (27% — coarse labels use rarer, more formal words).
- **A real defect falls out:** **14% of parent→child edges are pure name-propagation** (the coarse
  region inherited a child's exact name → no generalization at all), and the worst-flagged edges are
  genuine mis-altitudes — e.g. *"Waco Siege"* sitting as a coarse **parent** of *"Second Amendment
  Rights and Gun Control"* (cosine 0.13 — barely related).

So a useful hierarchy-altitude check needs no learned geometry: **`parent shorter than child`
(generality) + high `parent–child cosine` (relatedness)** flags the under-generalized / mis-altituded
coarse labels. *(Caveat: length works partly because haiku names fine clusters verbosely and coarse
ones tersely — a terser namer would weaken it. It's the same verbosity through-line as Finding 2.)*

## Is the judge trustworthy? (calibration)

Everything above trusts the sonnet judge as ground truth, so we calibrated it against 28 blinded
human ratings: **Spearman ρ = 0.82, quadratic Cohen's κ = 0.64** (substantial agreement). Two riders:
the judge is uniformly **~0.85 harsher** than the human (fine for ranking, which is all we used it
for); and the human penalizes **verbosity even more** than the judge does — which *reinforces*
Finding 2, since whitening's whole edge is penalizing padded labels.

## What this means for Toponymy (actionable)

1. **Whiten before any embedding-based label scoring** — it removes the verbosity bias for free.
2. **There's a ~14% under-generalization rate in coarse labels** (name-propagation + mis-altitudes) —
   worth surfacing, and cheaply detectable (length + cosine).
3. **Don't expect embedding cosine to rank good-vs-good labels** — use an (grounded, calibrated) LLM
   judge for fine quality decisions; reserve the cheap metric as a guardrail / regression check.

## Caveats & open questions

Single corpus (20NG), single namer (haiku), single embedder (MiniLM) — the verbosity-driven results
in particular should be re-checked with a terser namer and a different embedder. Two experiments are
pending before this is complete:

## Coming: fine-grained discrimination

*(TBD)* — judge multiple genuinely-good real labels per cluster (a small alternative-labels run) to
test whether **any** reference point ranks good-vs-good, i.e. the exact regime the original negative
result lived in. Expectation: low, but we want the number.

## Coming: label↔document erasure

*(TBD)* — the original "labels and documents are different classes" hypothesis. A rank-diagnostic
(is the label↔document gap a low-rank linear offset, or nonlinear?) gates whether erasing it
(LEACE / INLP) could help. Lower priority now that whitening already neutralized the cross-class
confound, but the geometry question is interesting on its own.

## Reproduce

See `PLAN.md` (Files + Reproduce). Pipeline: `prep_labels.py` → `perturbations.py` → `metrics.py` →
`judge_quality.py` → `validate_gate_b.py` → `make_calibration.py`/`score_calibration.py` →
`phase2_generality.py` → `phase2b_hierarchy.py`.
