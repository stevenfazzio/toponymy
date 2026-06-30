# Can you judge Toponymy label quality from embedding geometry?

> **DRAFT.** All experiments complete. To share: a short teaser comment on TutteInstitute/toponymy
> Discussion #173 that links here (not the full text), pending a voice pass and sign-off.

## TL;DR

Scoring a Toponymy region label by the embedding cosine to its cluster is **usable as a coarse
guardrail** — it reliably catches bad labels (Spearman ≈ 0.74 against a human-validated LLM judge) —
but **not as a fine-grained ranker** among already-good labels. That reconciles an earlier "it's
basically chance" negative result: the metric works on *good-vs-bad*, not *good-vs-good*.

The classic failure mode — it rewards verbose / over-padded labels — is an **anisotropy** artifact,
and it's fixed by **whitening the embedding space before scoring**, *not* by swapping the centroid
for a medoid or exemplar. A separate geometric **hierarchy** check is mostly a negative: a *learned*
generality axis (from HyperLex) is at **chance** at ordering multi-word phrase-label hierarchies, and
the only cheap signal that tracks altitude is plain **label length**. (No hierarchy *defect* turned up
— an earlier draft misread a deliberate Toponymy behavior, single-child regions inheriting their
child's name, as one.)

And a reality check on whether any of this is *useful*: when we ablate Toponymy's own naming features
(exemplars / keyphrases / subtopics), the judge sees the resulting quality changes but the metric is
**blind to all of them** — so even whitened, it's a gross-failure guard, not a naming-quality monitor.
That ablation throws off a free Toponymy finding, though: **exemplars dominate naming quality (+0.49
judge-points), while keyphrases don't appear to help — and may slightly hurt.**

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
raw graded correlation. Both are usable as gross-failure guards; neither is a fine ranker — and
neither, per the feature-ablation test below, is sensitive enough to flag a real naming regression.**

## Finding 3 — a learned generality axis doesn't transfer; length is the only cheap signal

Can geometry check that a *parent* label is a true generalization of its *children*? We learned a
linear "generality axis" in MiniLM space from HyperLex (graded lexical entailment) and tested it.

- **On HyperLex** the axis is real but weak: lexical-split test Spearman 0.32 (vs cosine 0.34,
  combined 0.45) — far below structure-based WordNet-IC (Renner et al. 2023, 0.744). Embeddings encode
  hypernymy weakly and entangled with relatedness, as the literature predicts.
- **On Toponymy's own hierarchy** (86 real parent→child edges) the axis collapses to **chance (50.0%
  direction accuracy)** — single-word hypernymy geometry does not transfer to multi-word region
  labels. The only cheap signal that tracks altitude is **label length** (the coarser label is shorter
  86% of the time); word frequency is actively *wrong* (27% — coarse labels use rarer, more formal words).

**No hierarchy defect.** ~14% of parent→child edges have parent == child labels, but on inspection
*all* of them are **single-child regions**: Toponymy's deliberate `[!SKIP!]` inheritance, where a
coarse region with exactly one named child takes that child's name — correct, since there is nothing
to generalize over, and the (within-layer) disambiguation pass rightly leaves them alone. An earlier
draft mis-framed this as under-generalization; it isn't.

So Finding 3 is a negative plus an observation: a HyperLex-learned generality axis is no use for
ranking Toponymy's phrase-label hierarchy, and label length is the only cheap proxy for altitude
(partly an artifact of haiku naming fine clusters verbosely and coarse ones tersely — a terser namer
would weaken it).

## Is the judge trustworthy? (calibration)

Everything above trusts the sonnet judge as ground truth, so we calibrated it against 28 blinded
human ratings: **Spearman ρ = 0.82, quadratic Cohen's κ = 0.64** (substantial agreement). Two riders:
the judge is uniformly **~0.85 harsher** than the human (fine for ranking, which is all we used it
for); and the human penalizes **verbosity even more** than the judge does — which *reinforces*
Finding 2, since whitening's whole edge is penalizing padded labels.

## What this means for Toponymy (actionable)

1. **If you score labels with embedding cosine at all, whiten first** — it removes the verbosity bias
   for free — but treat it strictly as a **gross-failure guard** (off-topic / garbage), *not* a
   naming-quality monitor: it's blind to the realistic, on-topic quality changes a pipeline produces.
2. **Exemplars are the dominant naming feature** (+0.49 judge-points); **keyphrases may not be earning
   their place** (no measured benefit here, possibly a slight cost) — worth a closer look.
3. **For any fine quality decision, use a grounded, calibrated LLM judge** — embedding cosine can't
   rank good-vs-good labels and can't see feature-level regressions.

## Caveats & open questions

Single corpus (20NG), single namer (haiku), single embedder (MiniLM) — the verbosity-driven results
in particular should be re-checked with a terser namer and a different embedder. The follow-up
experiments below close out the metric (fine discrimination), the cross-class hypothesis (erasure),
and the metric's real-world usefulness (feature ablation):

## Fine-grained discrimination — the metric is a guardrail, not a ranker

To test the regime the original negative lived in, we gave each cluster two genuinely-good labels
from *different* namers (haiku vs gpt-4o-mini) and had the grounded sonnet judge (neither namer, so no
self-preference) pick the better one. Of 104 pairs, the judge **tied on nearly half** — the two good
labels are often indistinguishable — and on the 54 it decided, **no reference point tracks its
preference**: centroid 48%, whitened 37% (worst), medoid 50%, exemplar 57% (within noise), none
clearing 50% on *both* winner-splits. So definitively: **embedding similarity is a coarse guardrail
(good-vs-bad), not a fine ranker (good-vs-good)** — for subtle quality calls, use the LLM judge.

## Label↔document erasure — the gap is real, erasable, and irrelevant

The original "labels and documents are different classes" hypothesis turns out **correct but inert**.
A rank diagnostic shows label and document embeddings are highly linearly separable (balanced-acc
0.935) but become inseparable after removing just **1–2 linear directions** (per-class centering drops
it to 0.33) — so the gap is a **low-rank linear mean offset**, cleanly erasable, and *not* nonlinear
like the image/text modality gap (a small surprise — it behaves like a corpus-identity gap). **But
erasing it changes nothing:** coarse judge-correlation 0.737 (vs 0.740 raw), verbose intrusion 74.8%
(vs 77.6% raw, 91.6% whitened). The offset is **class-uniform** — every label is displaced the same
way — so removing it can't change which label is closer to a cluster, nor fix verbosity. The real
problem was never the cross-class offset; it was **anisotropy**, which whitening (a broader transform)
addresses and offset-erasure does not. So "subtract the difference between labels and documents" is
geometrically valid and practically useless.

## Is the metric actually *useful*? Feature ablation says: only for gross failures

Beating a weak baseline isn't the same as being useful, so we tested the whitened metric on
*realistic* quality variation: we ablated each of Toponymy's three naming-prompt features
(exemplars / keyphrases / subtopics) — re-naming with one dropped — and compared full vs ablated under
*both* the grounded judge and the metric. The judge is the load-bearing part here: without it, a null
metric result can't be told apart from "the feature didn't matter."

**The metric is blind.** Across all three features the metric-Δ is ≈0, its rank correlation with the
judge-Δ is ≈0 (−0.07 to +0.03), and its per-cluster sign agreement with the judge is at chance
(48–56%) — *including* exemplars, where the judge sees a large, unambiguous drop (full 2.82 → ablated
2.33, **judge-Δ +0.49**). Because the judge confirms the feature genuinely matters, the metric's null
is the metric's failure, not the feature's irrelevance. The reason is the familiar coarse/fine
boundary: ablated labels stay *on-topic* (the medical region drifts from "Alternative Medicine
Approaches" to a list of specific conditions, but stays medical), so the drop lives in the fine regime
where cosine-to-centroid has no signal. **So the whitened metric's useful range is narrow —
gross/off-topic failures, not naming-quality regressions.**

**Bonus — a Toponymy finding in its own right.** The judge-Δ measures each feature's contribution to
naming quality, independent of the metric: **exemplars dominate (+0.49)**; **keyphrases don't help —
and may slightly hurt (−0.17)** (dropping them nudged labels *up*); **subtopics are ≈neutral (−0.03)**
and act only at coarse layers. The keyphrases result is the surprise — it suggests the exemplar
grounding does the real work and the keyphrase channel is redundant or mildly distracting here.
(Single haiku draw on one corpus; the small effects are suggestive, but exemplars' +0.49 is robust.)

## Reproduce

See `PLAN.md` (Files + Reproduce). Pipeline: `prep_labels.py` → `perturbations.py` → `metrics.py` →
`judge_quality.py` → `validate_gate_b.py` → `make_calibration.py`/`score_calibration.py` →
`phase2_generality.py` → `phase2b_hierarchy.py`.
