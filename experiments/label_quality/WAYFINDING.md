# Wayfinding lineups: measuring whether topic labels identify their regions

> Shared as [TutteInstitute/toponymy Discussion #177](https://github.com/TutteInstitute/toponymy/discussions/177)
> (July 2026); this is the full write-up that thread links to. Companion to
> [WRITEUP.md](WRITEUP.md) (the fit-metric study, Discussion #173); this one covers the
> discrimination instrument and what it led to.

## TL;DR

Toponymy's framing is cartographic, and a map label's deployment function is identification: a
reader must find the right region among its neighbours given only the name. The **wayfinding
lineup** measures exactly that. A listener LLM sees a label plus k=5 candidate document groups
(the true cluster and its 4 nearest same-layer neighbours, each shown as held-out member
documents); the label's score is the probability mass the listener puts on the true cluster.
It's a grounded task with a verifiable answer, so it's scored by accuracy rather than
preference. Everything below replicates on 20 Newsgroups and an arXiv ML corpus.

Five results, in decreasing order of confidence:

1. **The instrument works.** Real labels score 0.55 to 0.57 against 0.20 chance; shuffled labels
   fall at or below chance; easy lineups hit ceiling; known-bad label variants order exactly as
   they should, with legible failure modes (a sibling's label physically moves the mass to the
   sibling's candidate).
2. **Fit and identifiability are different axes.** The degradation a grounded fit judge punishes
   hardest (on-topic padding: gold 2.80 vs verbose 1.70 on the 0 to 4 rubric) is the one the
   lineup barely notices (0.545 vs 0.444). Correlation with the judge is moderate (Spearman 0.65
   to 0.74): related, not redundant. A useful eval is (judge AND lineup), not either alone.
3. **Most of a compound label is identification-inert.** 87% of conjuncts can be dropped at no
   measurable cost; 96% of compound labels contain one conjunct that identifies as well as the
   whole label at 40% of the length; 12% of conjuncts actively *hurt* by pulling mass toward a
   neighbour that owns that content. (20 Newsgroups only for this leg.)
4. **The eval doubles as a pipeline integrity check.** Its easy-lineup floor caught the
   misaligned bundled example data (toponymy#176) in one run, after several fit-judged
   experiments had run on the same data without complaint.
5. **The `detail_levels` dial doesn't survive measurement.** Naming every cluster at every
   `SUMMARY_KINDS` rung and selecting the shortest name whose identification is within the noise
   band of the best rung produces names that match stock on identification *and* on judge-rated
   fit at **55% (20NG) / 64% (arXiv) shorter**, evaluated on a held-out lineup configuration as a
   Goodhart guard. The dial is wrong in both directions: fine layers get far more words than
   identification needs, and at the arXiv coarse layer the stock rung ("simple (1 or 2 word)")
   is the *worst* rung on the ladder. The measurement is not free (roughly 25 to 30 times the
   LLM traffic of stock naming, quantified below), so it is a calibration tool and an option,
   not a proposed default.

## Why identification

The previous study (WRITEUP.md) ended with a settled dichotomy: embedding cosine is a coarse
guardrail, and for fine quality calls you need a grounded LLM judge. But the judge measures
*fit* (is this label a true, well-formed description of this cluster), and fit saturates: on
pairs of genuinely good labels the judge ties about half the time. Fit also can't
distinguish the two degenerate ends of the label spectrum, "the set of documents consisting of
doc1, doc2, ..." and "topic", both of which are accurate. What separates points on that spectrum
is length spent versus identification bought.

The design comes from a side project (Atlantic Mirror, which pairs European and North American
cities by character and captions each pair); the only honest way I found to score those captions
was a police lineup, where a caption is good only if a reader can pick the right city out of a
group of look-alikes. A topic label has the same job. The idea has an older pedigree in NLG as
referring-expression generation (Krahmer & van Deemter 2012), where the canonical goal is the
*minimal distinguishing description*: say just enough to pick the referent out of the context
set. The frozen-listener/self-retrieval line in captioning (Liu et al. 2018; Ou et al. 2023) is
the same shape. Information-theoretically, a label is a lossy code for its region and the lineup
measures whether it decodes against the layer's neighbour distribution; label quality becomes a
rate-distortion trade-off (rate = label length, distortion = identification error).

## The instrument

One lineup per (layer, cluster, candidate label):

- **Candidates:** the true cluster plus its k−1 = 4 nearest same-layer clusters by centroid
  cosine in the high-D embedding space (the geometry naming actually used). A layer with fewer
  than k clusters uses the whole layer. `Unlabelled` never appears.
- **Candidate representation (load-bearing):** N = 5 member documents, sampled EXCLUDING the
  exemplars the namer saw (Toponymy's facility-location exemplars, replayed deterministically),
  truncated to 500 characters. Otherwise a label that parrots exemplar phrasing wins trivially
  and the test measures prompt memorization, not generalization to the region.
- **Listener:** scores every candidate 0 to 100 given only the label and the documents
  (structured output). The label's score is the probability mass on the true cluster,
  `s_true / sum(s)`. Three self-consistency samples at temperature 0.7, candidate order
  reshuffled per sample so position bias is marginalized. Sonnet is the listener for all
  headline numbers (it is neither of the namers, per the no-self-preference rule); haiku runs
  the floors and sweeps; gpt-4o-mini runs the held-out evaluations.
- **Pairing discipline:** distractor sets and document samples are seeded and frozen per
  cluster, so every label variant for a cluster faces an identical lineup and all comparisons
  are paired.

**Floors and ceilings, run before anything was interpreted** (haiku; sonnet catch trials in the
battery confirm the same picture):

| floor (k=5, chance 0.20) | 20NG | arXiv |
|---|---|---|
| gold labels | 0.47 / 0.55 / 0.68 by layer | 0.49 / 0.51 / 0.55 |
| shuffled labels (within-layer derangement) | 0.12 to 0.14 | 0.12 to 0.18 |
| gimme (distant distractors) | 0.80 to 0.88 | 0.50 to 0.75 |
| repeat band (p90 of pm change on identical re-run) | 0.185 | 0.112 |

The k-sweep (k = 3/5/7) gives gold 0.62/0.50/0.43 (20NG) and 0.64/0.50/0.41 (arXiv), above
chance and below ceiling everywhere, so neither kill criterion fires (a gold-at-chance floor
would mean broken grounding; gold-at-ceiling at k=7 would mean a saturated layer). The sonnet
battery includes same-listener shuffled and gimme trials so gate comparisons never cross
listeners: 20NG shuffled 0.038 / gimme 0.936, arXiv 0.121 / 0.814.

Two instrument-building notes worth keeping. First, claude-sonnet-4-6 deliberates per-candidate
in prose and truncates before emitting scores if you ask for a bare array (and it rejects
assistant prefill), which produced *silent, kind-correlated* parse failures in the first battery
run (verbose labels invite deliberation: 1 of 107 parsed). The fix is structured output
(`response_format` json_schema); every committed number is from the 100%-parse-valid protocol.
Second, top-1 rate inflates on garbage labels (an all-zeros score row ties everything at rank 1),
so probability mass is the primary metric throughout and top-1 is descriptive only.

## Does it order known-bad labels correctly? (the battery)

Every cluster's gold label plus its intrusion battery (ancestor / sibling / distant / generic /
verbose, from the earlier study) through frozen lineups, sonnet listener:

| kind | 20NG pm | arXiv pm | paired gold-beats (20NG / arXiv) |
|---|---|---|---|
| gold | 0.545 | 0.574 | — |
| verbose | 0.444 | 0.476 | **79% / 83% (weakest, both corpora)** |
| ancestor | 0.262 | 0.270 | 88% / 92% |
| generic | 0.195 | 0.182 | 95% / 98% |
| sibling | 0.154 | 0.163 | 89% / 95% |
| shuffled | 0.038 | 0.121 | 100% / 99% |
| distant | 0.004 | 0.062 | 99% / 100% |
| gimme | 0.936 | 0.814 | — |

The failure modes are legible rather than just low scores. A **sibling's** label sends the label
below chance on the true cluster, and where the sibling cluster is in the lineup the mass
physically relocates to it (0.51 on the sibling vs 0.21 on true at 20NG; 0.59 vs 0.16 at arXiv):
what the judge scored as a soft "partly fits" is a measured, directional confusion. **Generic**
("Various topics") spreads the mass nearly uniform (normalized entropy 0.97 to 0.98).
**Distant** labels are rejected outright (the listener scores essentially everything zero).

**The verbose row is the finding.** The grounded judge punishes on-topic padding harder than any
other degradation (gold 2.80 vs verbose 1.70 on 20NG; 2.58 vs 1.55 on arXiv), while the lineup
barely notices it (0.545 vs 0.444; 0.574 vs 0.476), and only 4 to 5% of verbose labels fall
below chance. Padding hurts fit and reading, but it does not make a label harder to find. Fit
and identifiability are different measurement axes, replicated on both corpora.

Correlation with the judge over all judged battery candidates: Spearman **+0.647** (20NG,
n=487) and **+0.741** (arXiv, n=637). Moderate and positive: the lineup is related to fit (a
broken instrument would sit near zero) but far from redundant with it (the embedding metric's
own correlation with the judge is 0.80 to 0.84).

## Can it rank good-vs-good? (the fine pairs)

The regime the whole program kept failing in: 104 pairs of genuinely good labels for the same
cluster (haiku vs gpt-4o-mini namings, 20NG), where the judge tied on 50 and no embedding
reference point tracked its preferences on the rest. Both labels of each pair through the
cluster's frozen lineup; the repeat band for deltas measured directly on a re-run 30-pair subset
(p90 = 0.107).

- On the judge-tied pairs the lineup mostly ties too: mean |Δpm| 0.073, below the band. It
  decides 11 of 50 (22%, against roughly 10% expected from band noise alone): a real but thin
  tail. The decided pairs are face-valid, and they are consistently the cases where one label
  carries a distinguishing specific ("Larson's Reciprocal System Theory" at Δ = 0.44 over
  "Debates on Unified Theories and Observations in Physics", for the crank-physics cluster).
  Identifiability differences live at the fine layer (mean tied |Δ| 0.088 / 0.044 / 0.014 by
  layer), and there is no systematic bias between the two namers (mean pm 0.537 vs 0.531,
  Wilcoxon p = 0.81).
- On the 54 judge-decided pairs, the lineup agrees with the judge's direction only **37%** of
  the time (Spearman −0.21): where fit and identifiability both express a preference, they point
  different ways. The two-axes result again, now on realistic label pairs.
- A small blinded human check (me, rating the top-20 tied pairs by |Δ| as A/B/can't-decide,
  without knowing the lineup's picks): where I could express a preference at all I agreed with
  the lineup 10 of 12 times, 5 of 5 on the band-clearing test items (coin-flip null p ≈ 0.03).
  My ties concentrated on the *largest*-delta items, including Larson, which I read as a form
  limitation I only understood afterwards: the form showed the true cluster's documents but not
  the neighbours, and the lineup's conviction comes precisely from neighbour contrast. The
  airtight version would make the human be the listener (label + five candidate doc groups, pick
  the group); I stopped short of that.

So: most good-vs-good pairs are functionally equivalent on the identification axis too, a thin
tail is genuinely more identifying, and the tail's picks are human-endorsed where the human can
see the difference.

## What the instrument caught: the misaligned example data (#176)

The first arXiv replication attempt used the bundled `examples/ai_arxiv_*` files, and the floors
failed in the most specific way possible: gold 0.219, shuffled 0.196, **gimme 0.217**, all at
chance. Gimme-at-chance means the listener cannot find the true cluster even with distant
distractors and a correct label: broken grounding, not bad labels. The diagnosis (full
reproduction in toponymy#176): `ai_arxiv_papers.zip` is not in the same row order as
`ai_arxiv_vectors.npy` (near-identical vectors at adjacent indices map to unrelated titles),
while the vectors and 2D coordinates are mutually consistent. Row-pairing the zip with the
vectors attaches every paper to some other paper's embedding. Clustering still "works" (it runs
on the vectors), naming produces plausible vague composites (every fine-layer name came out as
"Diverse Machine Learning Applications Spanning X, Y, and Z"), and the grounded fit judge does
not complain, because a vague label fits a scrambled document sample about as well as anything.
Several earlier fit-judged experiments ran on this substrate without tripping anything; the
lineup flagged it in one run.

The replication moved to a substrate that re-embeds the document text directly (the at-home
arXiv cell from the earlier 2×2 study, whose judge is human-calibrated), and the gimme ceiling
snapped back from 0.22 to 0.75, which doubles as the positive control for the diagnosis.

## Which parts of a label do the work? (conjunct ablation, 20NG)

94 of the 107 gold labels are compounds ("X, Y, and Z"). Splitting them into 225 standalone
conjuncts (LLM splitter distributing shared modifiers, splits cached and eyeballed) and running
two variants per conjunct through the frozen lineups, `drop:i` (label minus that conjunct) and
`only:i` (that conjunct alone), paired against the battery's gold runs; the repeat band for this
stage is p90 = 0.095.

- **87% of conjuncts are free-riders**: dropping them costs less than the band (mean marginal
  identification value +0.004, i.e. nothing).
- **12% are anti-conjuncts**: dropping them *helps* beyond the band. The mechanism is the
  interesting part: a conjunct that overlaps a neighbour's territory pulls mass toward that
  neighbour ("Used Motorcycle Sales Discussions" at −0.37, in a layer where a neighbour owns the
  sales-flavoured content). Part of a label can be true of the region and still be the wrong
  thing to say given the neighbours.
- **13% are load-bearing**, and they are the distinguishing specifics: "Personal Attacks on
  Racial and Intellectual Grounds" (+0.46), "Larson's Reciprocal System Theory" (+0.45, the same
  label element the fine pairs surfaced independently), "Biblical Interpretation of
  Homosexuality" (+0.43).
- **96% of compound labels contain a single conjunct that identifies as well as the whole label**
  (within band), at a mean 40% of the length. Position doesn't matter (first/mid/last conjuncts
  all ≈ 0 mean marginal value): padding is semantic, not structural, so no truncation heuristic
  recovers this; you need the measurement.

## Measuring the dial: the length controller

Toponymy sets label specificity open-loop. `detail_levels = linspace(lowest, highest, n_layers)`
indexes a seven-rung `SUMMARY_KINDS` ladder of word-count phrases, from "domain expert level
(8 to 15 word)" at the finest layer to "simple (1 or 2 word)" at the coarsest. The naming and
disambiguation templates ask for names "sufficiently detailed to be distinguished from other
topics", but nothing checks. The lineup is the checker.

**Method.** Name every cluster at every rung r (stock Toponymy machinery, same fitted clusterer,
`lowest_detail_level = highest_detail_level = r/6`, so the disambiguation pass and all features
behave exactly as the library would; the ladder subsumes the stock configuration, since stock is
rungs [0, 3, 6] for three layers, which makes the stock arm the same draw and perfectly paired).
Run every distinct name through the cluster's frozen lineup. Select per cluster the **shortest
name whose pm is within the repeat band of the best rung**: minimize length subject to no
measurable identification loss, with the band measured in-run (p90 = 0.047 on 20NG, 0.079 on
arXiv). No absolute threshold, so the criterion adapts per cluster.

**Cost, quantified, before the results.** Stock naming is one LLM call per cluster (plus the
disambiguation pass's occasional renaming calls). The controller replaces that with seven naming
calls plus the lineup evaluations: on these corpora, 6.8 and 6.3 distinct ladder names per
cluster (adjacent rungs often produce the same name, and duplicates run once) × 3 listener
samples = 20.3 and 18.8 listener calls per cluster at 3,751 input tokens each. Roughly 25 to 30
times the LLM traffic of stock naming, so everything below should be read as "what the
measurement buys", not as a proposed default.

**The identification-vs-length curve is nearly flat.** Mean pm / mean words per rung:

| layer | r0 | r1 | r2 | r3 | r4 | r5 | r6 |
|---|---|---|---|---|---|---|---|
| 20NG L0 (stock=r0) | **0.50**/10w | 0.49/9w | 0.50/6w | 0.48/5w | 0.48/4w | 0.49/3w | 0.47/2w |
| 20NG L1 (stock=r3) | 0.62/10w | 0.63/9w | 0.60/7w | **0.63**/5w | 0.60/4w | 0.59/3w | 0.59/2w |
| 20NG L2 (stock=r6) | 0.82/10w | 0.83/9w | 0.81/7w | 0.78/5w | 0.79/4w | 0.80/3w | **0.80**/2w |
| arXiv L0 (stock=r0) | **0.58**/10w | 0.58/9w | 0.56/6w | 0.57/5w | 0.57/4w | 0.56/3w | 0.55/2w |
| arXiv L1 (stock=r3) | 0.60/11w | 0.62/9w | 0.60/6w | **0.60**/5w | 0.61/4w | 0.61/4w | 0.59/2w |
| arXiv L2 (stock=r6) | 0.63/11w | 0.58/9w | 0.60/6w | 0.64/6w | 0.61/4w | 0.66/4w | **0.48**/2w |

Words 3 through 10 buy roughly nothing at the fine layers. And the arXiv coarse layer shows the
opposite failure: the stock rung is the *worst* rung on its ladder (0.48 vs 0.66 at four words).
The dial is wrong in both directions, which the controller's behaviour mirrors: it shortens hard
at L0 (differs from stock on 70/74 and 82/82 clusters, −5.7 and −6.9 words) and lengthens some
coarse-layer names (+1.4 and +1.0 words at L2).

**Held-out verdict (the Goodhart guard, Gao et al. 2023).** Selection optimizes against the
lineup, so all headline evaluation uses a held-out configuration: fresh document draws, fresh
order seeds, and a different listener (gpt-4o-mini), plus the grounded sonnet judge for fit,
on the clusters where chosen ≠ stock:

| | chosen pm | stock pm | | chosen words | stock words | judge fit (chosen / stock) |
|---|---|---|---|---|---|---|
| 20NG (n=91) | 0.516 | 0.524 | p = 1.0 | 3.8 | 8.5 | 2.79 / 2.75 (p = 0.57) |
| arXiv (n=111) | 0.448 | 0.436 | p = 0.99 | 3.2 | 8.8 | 2.49 / 2.45 (p = 0.68) |

Same identification, same fit, **55% / 64% shorter**. The in-sample "+pm gains" of the selection
stage are not quoted anywhere: picking within a band of a noisy max inflates apparent scores by
construction, and on held-out they evaporate, exactly as winner's-curse logic predicts.

**Interaction with the disambiguation pass.** Shorter names should collide more, and collisions
are what the disambiguation pass exists to fix, at the cost of large renaming calls (its prompts
carry exemplars and keyphrases for every topic in a name-group). Checked two ways. During
naming, the tersest rung does make the pass work slightly harder: 5 renaming calls covering 13
of 74 fine-layer topics, vs 2 calls covering 5 at the most verbose rung (a ~3-call marginal cost
against a ~107-call naming pass). On the final name sets, the pass's own trigger
(`cluster_topic_names_for_renaming`, simulated with the library function) finds *nothing* to
rename in either the stock or the controller-chosen names, on either corpus: no name pairs
within its similarity cap (closest chosen pair sits at cosine distance 0.305 against the 0.2
cap), no exact duplicates, no lineup-mates sharing a name. The shorter names do not buy their
savings back in disambiguation.

## What this means for Toponymy (actionable)

1. **Two separable suggestions about `detail_levels`.** The free one: the fine-layer
   `SUMMARY_KINDS` defaults could be much terser at, per these measurements, no cost to either
   fit or identification. The principled one: per-cluster specificity is measurable, and "the
   shortest name that still identifies the region among its neighbours" is a defensible
   definition of the right label length (REG's minimal distinguishing description, applied to
   regions). At 25 to 30 times the LLM traffic of stock naming, the controller is not a
   default; its natural uses are (a) **calibration**: run it once on a representative corpus to
   choose better static defaults, then ship the static result (the rate-distortion tables above
   are exactly that run for these two corpora); (b) **targeted use**: coarse layers only, where
   the clusters are few, the labels are most visible, and the dial is provably worst; (c)
   **high-value maps** that are fit once and read many times, where naming cost amortizes over
   readers.
2. **For naming-quality evaluation, fit alone is not enough.** The grounded judge and the lineup
   disagree exactly where labels get realistic (padding; good-vs-good pairs). A change that
   improves fit can leave identification untouched and vice versa; measure both.
3. **Run a gimme floor before trusting any labeling pipeline's outputs.** It is nearly free, and
   it catches broken document/vector grounding that fit-based evaluation is structurally blind
   to (that is how toponymy#176 was found).

## Caveats and open questions

Two corpora, one namer (haiku, which pads more than a terser model would; the free-rider and
length numbers are plausibly haiku-flavoured even if the mechanism is general). One listener
family for the headline numbers, with a measured but nonzero noise band; "no measurable loss"
means at the resolution of 3-sample self-consistency, not proven equality. The conjunct ablation
ran on 20NG only. The human seed is small (20 items, one rater) and its form under-tested the
discrimination component; the human-as-listener variant is the obvious upgrade. The arXiv coarse
layer's gimme ceiling is lower (0.50 at L2) because on a homogeneous corpus even distant regions
share vocabulary; per-layer chance and ceilings matter when comparing across corpora. And using
the lineup as a *selector* (rather than an evaluator) invites Goodhart pressure beyond what the
held-out configuration checks; anything stronger than the controller's shortest-within-band rule
should re-verify against a fresh listener and fresh draws, per Gao et al. 2023.

On cost, the precise accounting rather than a dollar figure: the committed result files record
**17,055 successful listener calls** across the floors, batteries, fine pairs, conjunct
ablation, ladder lineups, and held-out evaluations on both corpora (each unit stores its raw
samples, so this is counted, not estimated). A representative k=5 lineup prompt is **3,751 input
tokens** for claude-sonnet-4-6 (token-counting endpoint, 2026-07-04), with a small JSON object
back; three calls per lineup. The two seven-rung naming ladders and the grounded-judge calls
come on top of that. Multiply by your own prices.

## Reproduce

All under `experiments/label_quality/` on this branch; data files are committed except the
quarantined misaligned-substrate artifacts (`*.misaligned.bak.json`).

- `wayfinding.py --stage check|smoke|floors|battery [--dataset 20ng|arxiv_home]`: the
  instrument: frozen lineups, floors (shuffled/gimme/repeat/k-sweep), battery + gates a/b.
  `Cell("arxiv_home")` replays the at-home 2×2 arXiv cell (re-embedded text, aligned by
  construction); `make_home_battery.py` builds its intrusion battery.
- `wayfinding_pairs.py [--report-only|--export-human|--score-human]`: gate c: the 104 fine
  pairs, sonnet repeat band, blinded human A/B seed.
- `conjunct_ablation.py --stage split|run|report`: 5a: conjunct splits (cached), drop/only
  variants, marginal identification value.
- `length_controller.py --stage ladder|lineups|select|heldout [--dataset ...]`: 5b: the
  SUMMARY_KINDS ladder, MDL selection, rate-distortion table, held-out Goodhart check.
- `collision_check.py [--count-disambig DS RUNG]`: disambiguation-pass interaction: trigger
  simulation on stock vs chosen names; refit counting actual renaming calls per rung.

Key upstream mechanics this rests on: `detail_levels` → `SUMMARY_KINDS` via
`prompt_construction.py`; the disambiguation trigger `cluster_topic_names_for_renaming`
(agglomerative over name embeddings, complete linkage, 0.2 cosine-distance cap); facility-location
exemplars via `cluster_layer.make_exemplar_texts` (deterministic given a fit).
