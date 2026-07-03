# Can you judge Toponymy label quality from embedding geometry?

> **DRAFT.** All experiments complete. To share: a short teaser comment on TutteInstitute/toponymy
> Discussion #173 that links here (not the full text), pending a voice pass and sign-off.

## TL;DR

Scoring a Toponymy region label by the embedding cosine to its cluster is **usable as a coarse
guardrail** — it reliably catches bad labels (Spearman ≈ 0.74–0.84 against a human-validated LLM judge)
— but **not as a fine-grained ranker** among already-good labels. That reconciles an earlier "it's
basically chance" negative result: the metric works on *good-vs-bad*, not *good-vs-good*. The headline
results below replicate across a **2×2 of two embedders (MiniLM, Cohere embed-v4) × two corpora
(20 Newsgroups, arXiv)** — each embedder clustering, naming, and scoring in its *own* geometry.

The classic failure mode — it rewards verbose / over-padded labels — is an **anisotropy** artifact.
**Whitening the embedding space before scoring fixes it** (not swapping the centroid for a medoid or
exemplar) — but it's a *targeted patch*, not a blanket rule: the 2×2 shows the verbose blind spot only
appears in anisotropic *(embedder × corpus)* spaces (3 of 4 cells), and in the cleanest space (Cohere
on arXiv) raw cosine already handles padding and whitening slightly *costs* you. The honest default is
**raw centroid cosine** (the better graded ranker on all four cells); reach for whitening as cheap
insurance when your space has the padding bias.

A separate geometric **hierarchy** check is mostly a negative: a *learned* generality axis (from
HyperLex) is at **chance** at ordering multi-word phrase-label hierarchies, and the only cheap signal
that tracks altitude is plain **label length**. (No hierarchy *defect* turned up — an earlier draft
misread a deliberate Toponymy behavior, single-child regions inheriting their child's name, as one.)

And a reality check on whether any of this is *useful*: when we ablate Toponymy's own naming features
and re-name, the metric barely tracks the quality changes the judge sees — even raw centroid only
weakly catches the *largest* regression (dropping exemplars) and is blind to subtler ones; whitening
sees nothing. So it's a gross-failure guard, not a naming-quality monitor. That ablation throws off a
free Toponymy finding, and this one is **robust across all four cells**: **exemplars dominate naming
quality (≈ +0.4 judge-points), while keyphrases don't appear to help — and may slightly hurt.**

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
a usable quality signal. This is the most robust result in the study: across the full 2×2 (below),
raw-centroid ρ lands at **0.80–0.84** in every cell — the guardrail generalizes across embedder and
corpus.

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
overall; **79.6% → 92.5%** on `verbose`). So even on this single (20NG/MiniLM) cell it's a trade-off:
whitening for padding-robustness, raw centroid for graded correlation. **The 2×2 below sharpens this
into the study's main reframe** — raw centroid is the better default on *every* axis except verbose
robustness, and whether you even need the whitening patch depends on the *(embedder × corpus)*
anisotropy, not on a universal rule. Either way, both are gross-failure guards; neither is a fine
ranker, and neither (per the feature-ablation test) reliably flags a realistic naming regression.

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

## Does it hold up? Embedder × dataset robustness (the 2×2)

Everything above is one corpus, one embedder (20NG / MiniLM). Two earlier near-posts in this project
turned out to be artifacts, so before drawing conclusions we ran the headline tests across a **2×2**:
weak (MiniLM, 384-d) vs strong (Cohere embed-v4, 1024-d) embedders × 20 Newsgroups vs arXiv (7,000
docs each). Crucially each cell plays **at home** — it clusters, names (haiku), *and* scores in its own
embedder's geometry — so the metric is always judging the partition its own embedder actually drew (an
earlier quick-check that scored one embedder's labels against another's clusters was a home/away
confound and is superseded by this). We trimmed to the two load-bearing tests — gate-b (does the metric
track the judge) and the exemplars/keyphrases ablation — and cite the rest as robust negatives.

A reassuring control first: the at-home 20NG/MiniLM cell, re-clustered from a fresh UMAP, **reproduces
the canonical numbers** — verbose-intrusion 79% → 94% (vs 78 → 92 above) and exemplars +0.50 (vs +0.49)
— so the at-home recipe is faithful and the other three cells are trustworthy.

| cell | gate-b ρ (centroid / whitened) | `verbose` intrusion (centroid → whitened) | exemplars judge-Δ | keyphrases judge-Δ |
|---|---|---|---|---|
| 20ng / minilm | 0.81 / 0.75 | 79% → 94% (**+15**) | +0.50 | −0.09 |
| 20ng / cohere | 0.80 / 0.76 | 68% → 88% (**+20**) | +0.40 | −0.10 |
| arxiv / minilm | 0.82 / 0.78 | 53% → 84% (**+31**) | +0.48 | −0.07 |
| arxiv / cohere | 0.84 / 0.79 | **99% → 92% (−7)** | +0.39 | −0.00 |

**What's robust (two findings get *stronger*):**

1. **The coarse guardrail generalizes.** Gate-b ρ is 0.80–0.84 (centroid) in every cell, and **raw
   centroid beats whitened on graded correlation in all four** (~0.04 gap — the same ordering as the
   canonical cell). "Usable coarse guardrail" is not a 20NG/MiniLM accident.
2. **exemplars ≫ keyphrases replicates 4/4.** Dropping exemplars costs **≈ +0.4 judge-points
   everywhere**; dropping keyphrases costs ~0 or is marginally *helpful* everywhere. This was the
   shakiest "bonus" finding (a single haiku draw on one corpus); it's now the most robust thing here.

**What needs reframing (two findings get *more nuanced*):**

3. **Whitening is a targeted padding patch, not "whiten first."** It fixes the verbose blind spot in
   three cells (+15 to +31 pp) — *including 20ng/cohere, a strong embedder*, so this isn't simply "weak
   embedders need whitening." But in the cleanest space (arxiv/cohere) raw cosine already nails verbose
   (99%), and whitening *costs* 7 pp there (and ~0.05 on gate-b). The real variable is the
   *(embedder × corpus)* anisotropy of the specific space. Practically: **default to raw centroid; add
   whitening as cheap insurance when your space has the padding bias** (it gains a lot where needed and
   costs little where not).
4. **The "metric is blind to ablations" claim was the *whitened* metric.** Split by reference point:
   **raw centroid weakly but consistently tracks the *large* exemplars regression** — per-cluster
   sign-agreement with the judge is 62 / 73 / 68 / 72% (above the 50% chance line in all four cells) —
   while staying blind to the ~0 keyphrases change (correctly — there's nothing to detect). The whitened
   metric is blind/erratic on both (exemplars 54 / 69 / 32 / 45%). So the metric isn't stone-blind; it
   weakly catches the biggest regression (via raw centroid) and misses subtle ones — still not a precise
   regression guard, but the nuance, and raw centroid's edge, are real.

Net: **raw centroid is the better default on every axis except worst-case padding robustness**, and
whitening is a patch you reach for when the space needs it — not a blanket transform. (Reproduce:
`home_pipeline.py` per cell → `cross_cell.py`.)

## Is the judge trustworthy? (calibration)

Everything above trusts the sonnet judge as ground truth, so we calibrated it against 28 blinded
human ratings on 20NG: **Spearman ρ = 0.82, quadratic Cohen's κ = 0.64** (substantial agreement). Two
riders: the judge is uniformly **~0.85 harsher** than the human (fine for ranking, which is all we used
it for); and the human penalizes **verbosity even more** than the judge does — which *reinforces*
Finding 2, since whitening's whole edge is penalizing padded labels.

Because three of the four 2×2 cells are arXiv and/or Cohere — a domain the 20NG seed never covered — we
ran a second blinded seed of **20 arXiv items** (mixed across the MiniLM and Cohere arXiv cells):
<!-- ARXIV-CAL --> **Spearman ρ = 0.79, quadratic κ = 0.54** — essentially the same rank agreement as
20NG, with the same ~1-point judge-harsher offset (the offset is a stable judge property, not an
arXiv artifact; it costs κ but not ranking). Per-type ordering matches too: both human and judge put
gold > verbose > ancestor > generic ≈ sibling. One caveat cuts the other way: the human rater is
less at home in these subfields than in 20NG's everyday topics, so some disagreements (e.g. a
sibling label rated 3 by the human, 0 by the judge) plausibly reflect human, not judge, error. This
guards the arXiv numbers against an un-calibrated-on-arXiv judge (Krumdick et al. 2025).

## What this means for Toponymy (actionable)

1. **If you score labels with embedding cosine at all, default to the raw centroid** and treat it
   strictly as a **gross-failure guard** (off-topic / garbage), *not* a naming-quality monitor — it's
   blind to the realistic, on-topic quality changes a pipeline produces. **Add whitening only if your
   space has the padding bias** (verbose labels scoring too high): it's a big win in anisotropic spaces
   and a small cost in clean ones, so it's cheap insurance — but it's a patch, not a default.
2. **Exemplars are the dominant naming feature** (≈ +0.4 judge-points, robust across both embedders and
   both corpora); **keyphrases may not be earning their place** (no measured benefit in any of the four
   cells, occasionally a slight cost) — worth a closer look.
3. **For any fine quality decision, use a grounded, calibrated LLM judge** — embedding cosine can't
   rank good-vs-good labels and can't see feature-level regressions.

## Caveats & open questions

The verbosity- and ablation-driven results are now checked across **two embedders × two corpora** (the
2×2 above), which discharges the original single-embedder / single-corpus worry for those findings.
Still single-namer (haiku) — and label length as the altitude proxy (Finding 3) is partly a haiku
artifact, so a terser namer is the obvious next probe. The HyperLex hierarchy axis (Finding 3) and the
erasure diagnostic (below) were characterized only on 20NG/MiniLM and cited as robust negatives, not
re-run across the 2×2. The follow-up experiments below close out the metric (fine discrimination), the
cross-class hypothesis (erasure), and the metric's real-world usefulness (feature ablation):

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

Beating a weak baseline isn't the same as being useful, so we tested the metric on *realistic* quality
variation: we ablated each of Toponymy's three naming-prompt features (exemplars / keyphrases /
subtopics) — re-naming with one dropped — and compared full vs ablated under *both* the grounded judge
and the metric. The judge is the load-bearing part here: without it, a null metric result can't be told
apart from "the feature didn't matter." (This is the original 20NG/MiniLM deep-dive; the 2×2 section
above is the cross-embedder confirmation and adds the raw-vs-whitened nuance.)

**The whitened metric is blind.** Across all three features its metric-Δ is ≈0, its rank correlation
with the judge-Δ is ≈0 (−0.07 to +0.03), and its per-cluster sign agreement with the judge is at chance
(48–56%) — *including* exemplars, where the judge sees a large, unambiguous drop (full 2.82 → ablated
2.33, **judge-Δ +0.49**). Because the judge confirms the feature genuinely matters, the metric's null
is the metric's failure, not the feature's irrelevance. The reason is the familiar coarse/fine
boundary: ablated labels stay *on-topic* (the medical region drifts from "Alternative Medicine
Approaches" to a list of specific conditions, but stays medical), so the drop lives in the fine regime
where cosine-to-centroid has no signal. **The whitened metric's useful range is narrow —
gross/off-topic failures, not naming-quality regressions.** (The one wrinkle from the 2×2: *raw*
centroid does weakly track the large exemplars drop — sign-agreement 62–73% across cells — so the
floor isn't quite zero, but it's far from a reliable guard.)

**Bonus — a Toponymy finding in its own right.** The judge-Δ measures each feature's contribution to
naming quality, independent of the metric: **exemplars dominate (≈ +0.4)**; **keyphrases don't help —
and may slightly hurt (≈ −0.1)** (dropping them nudged labels *up*); **subtopics are ≈neutral (−0.03)**
and act only at coarse layers. The keyphrases result is the surprise — it suggests the exemplar
grounding does the real work and the keyphrase channel is redundant or mildly distracting here. Unlike
most of this write-up these effects **replicate across all four 2×2 cells** (exemplars +0.39…+0.50,
keyphrases −0.10…−0.00), so they're no longer a single-draw caveat — though still one namer (haiku).

## Reproduce

See `PLAN.md` (Files + Reproduce). 20NG/MiniLM deep-dive: `prep_labels.py` → `perturbations.py` →
`metrics.py` → `judge_quality.py` → `validate_gate_b.py` → `make_calibration.py`/`score_calibration.py`
→ `phase2_generality.py` → `phase2b_hierarchy.py`. Embedder × dataset 2×2: `home_pipeline.py --dataset
{20ng,arxiv} --embedder {minilm,cohere}` (one per cell) → `cross_cell.py`; arXiv judge calibration:
`make_calibration_home.py` → `score_calibration.py --key arxiv_calibration_key.json --human
arxiv_calibration_human.json`.
