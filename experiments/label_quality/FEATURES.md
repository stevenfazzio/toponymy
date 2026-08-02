# What are Toponymy's three naming-prompt features actually worth?

> Shared (August 2026) as comments on
> [TutteInstitute/toponymy Discussion #173](https://github.com/TutteInstitute/toponymy/discussions/173#discussioncomment-17874315)
> (the substance) and
> [Discussion #177](https://github.com/TutteInstitute/toponymy/discussions/177#discussioncomment-17874317)
> (a pointer, since the exemplars result sharpens that thread's central claim); this is the full
> write-up both link to. Companion to [WRITEUP.md](WRITEUP.md) (the fit-metric study, #173) and
> [WAYFINDING.md](WAYFINDING.md) (the identification instrument, #177).

## TL;DR

Toponymy builds every topic-naming prompt from three content channels: **exemplar documents**,
**keyphrases**, and **subtopics** (the names of a cluster's children). #173 measured them once, on
one corpus, with one instrument, and reported that exemplars carry essentially all the label
quality. This re-measures all three on **two corpora** (20 Newsgroups, arXiv), **three axes** (fit,
identification, disambiguation load), with the fit instrument's one plausible confound explicitly
tested and ruled out.

| feature | fit (grounded judge) | identification (lineup) | disambiguation load |
|---|---|---|---|
| **exemplars** | **large** — −0.49 judge-pts at zero, both corpora | inert, both corpora | none |
| **keyphrases** | none (−0.17 / +0.16 by corpus) | negligible (≤0.016 pm) | **3× on 20NG**, none on arXiv |
| **subtopics** | none (−0.13 pooled, p=0.08) | inert (tight null) | ~none |

Four things worth carrying away:

1. **Only exemplars measurably contribute to label quality, and only to fit — not to findability.**
   That is the sharpest form yet of the fit≠identification split from #177, and it replicates.
2. **The exemplar effect is real, not a parroting artifact.** The judge's documents overlap the
   namer's exemplars (24% at the default), and that overlap grows with `n_exemplars` — so every
   exemplar comparison is potentially confounded. Re-judging on leakage-free documents moves the
   #173 headline by +0.008 (p=0.93). **2% of the measured effect was leakage.** The posted claim
   stands.
3. **`n_exemplars` needs no tuning, and the useful advice is a floor, not a target.** Below 8 is
   corpus-dependent (harmless on 20NG, −0.19 on arXiv); above 8 the curve does not turn over.
4. **Two 20NG-only findings died on the second corpus** — a "knee at k=4" and a super-additive
   keyphrase×exemplar interaction — and are withdrawn here. Both would have been confidently wrong
   advice. This is the project's own standard doing its job for the third time.

## Why re-measure at all

Three reasons, in increasing order of force.

The #173 ablation ran on **one corpus with one instrument**. That study's own stated standard was
that only findings surviving all four cells of a 2×2 get called findings — a standard adopted
precisely because two earlier conclusions collapsed when the setup varied. The feature ablation was
never held to it. jc-healy also noted directly that these experiments "can be quite data dependent."

The **identification axis did not exist** when the ablation was run. #177 established that fit and
identification are different axes (ρ 0.65–0.74; the fit judge ties on about half of good-vs-good
pairs). A feature that is judge-neutral is not automatically identification-neutral, and the
keyphrase result — the one being acted on upstream — had only ever been measured on fit.

And the **subtopics arm was never properly scoped**. Its reported −0.033 averages over all layers,
but layer 0 has no children, so 73 of its 106 rows are structurally a no-op (see below).

## The instruments

Unchanged from the earlier phases; both are described in full in their own write-ups.

- **Fit** — grounded LLM judge (Sonnet), 0–4 rubric after Preiss et al. 2024, majority of 3 samples,
  scored against a deterministic per-cluster document sample. Calibrated against blinded human
  ratings at ρ ≈ 0.8 on both corpora ([WRITEUP.md](WRITEUP.md)).
- **Identification** — the wayfinding lineup. A listener sees a label plus 5 candidate document
  groups (the true cluster and its 4 nearest same-layer neighbours), each shown as 5 **held-out**
  member documents sampled to exclude the namer's exemplars. Score is the probability mass landing
  on the true cluster ([WAYFINDING.md](WAYFINDING.md)).
- **Disambiguation load** — Toponymy's own renaming trigger, instrumented during `fit()`
  (`disambiguation_load.py`). Not a quality measure: a count of the renaming LLM calls the pass had
  to issue.

Substrates are the canonical 20NG fit ([74, 24, 9] clusters) and the at-home arXiv/MiniLM cell
([82, 27, 7]). The arXiv cell re-embeds document text rather than using the bundled vectors, so it
is structurally immune to the row-misalignment in #176 — verified here by re-encoding sampled
documents and comparing against the cached matrix (cosine = 1.000 on every probe, both corpora).

## Exemplars

### Identification: inert, on both corpora

| corpus | Δ prob-mass when exemplars dropped | p | n |
|---|---|---|---|
| 20NG | +0.011 [−0.013, +0.035] | 0.23 | 106 |
| arXiv | −0.006 [−0.029, +0.018] | 0.75 | 116 |

Both nulls are well-powered rather than merely quiet. On the same clusters and the same frozen
lineups, the battery's known-bad variants resolve cleanly — the *weakest* of them, on-topic padding,
at +0.100 (SE 0.013, p=1e-11), and sibling-label swaps at +0.390. The ablation arms have comparable
n and SE, so the CIs rule out any identification cost above ~0.035 pm: roughly a third of the
smallest degradation the instrument demonstrably sees.

### Fit: large — and it survives leakage control

The grounded judge scores a label against 15 cluster documents. It does **not** exclude the
documents the namer saw, and the overlap grows with exactly the variable being swept:

| `n_exemplars` | 1 | 2 | 4 | **8** | 16 | 32 |
|---|---|---|---|---|---|---|
| judge docs the namer also read | 7% | 11% | 16% | **24%** | 38% | 67% |

That is a live alternative explanation for the whole effect: the k=0 arm has **zero** overlap by
construction while stock has 24%, so "exemplars help" could partly mean "the namer read the answer
key." Re-judging on documents drawn to exclude the union of every rung's exemplars (13.8% pool
shrinkage, all 107 clusters still get the full 15 documents, no fallbacks):

| contrast | leaky documents | leakage-free documents |
|---|---|---|
| k=0 vs stock | −0.498 (p=5e-07) | **−0.490** (p=6e-07) |
| k=2 vs stock | −0.183 (p=0.002) | **−0.196** (p=0.007) |
| k=4 vs stock | −0.015 (p=0.71) | **−0.012** (p=0.95) |

Change in the headline effect: **+0.008, SE 0.056, p=0.93 — 2% of it was leakage.** Every condition
drops ~0.07 in absolute score (removing exemplars removes a cluster's most central members, so the
clean documents are genuinely harder), but the shift is uniform across arms and every *contrast*
survives intact. The #173 exemplars result stands, now leakage-controlled.

The likely reason the confound doesn't bite: the judge grades whether a label describes a *set* of
15 documents against a rubric. Having read 3.6 of them does not by itself earn a higher rating.

### Dose: a floor, not a target

Sweeping downward from the default (exemplar selection is strictly nested — verified 74/74 — so
lower rungs are subsets of the stock 8):

| k | 20NG (all layers, n=103) | arXiv (n=90) |
|---|---|---|
| 0 | −0.479 (p=1e-06) | −0.485 (p=5e-06) |
| 1 | −0.387 (p=1e-05) | −0.506 (p=3e-08) |
| 2 | −0.147 (p=0.009) | −0.256 (p=2e-05) |
| **4** | **+0.015 (p=0.98)** | **−0.187 (p=0.002)** |

**On 20NG there is a knee at 4; on arXiv there is not.** The arXiv deficit reproduces across two
independent naming draws (−0.189 and −0.152, both p<0.005; the draws differ by +0.037, p=0.37), so
it is not draw noise. A "halve the default to 4" recommendation, which the 20NG numbers alone
strongly support, would have been **wrong advice on the second corpus we own**. It is withdrawn.

Sweeping *upward* is the direction that would actually justify tuning, since tokens are otherwise
the only established cost of over-provisioning. Leakage-controlled for k=4/8/16 (excluding
exemplars for k≤16 costs 27.6% of the pool and drops 12 of 107 clusters below the document budget),
with k=32 deliberately **advantaged** — full control there would starve 39 clusters, so its extra
exemplars *can* overlap what it is graded on:

| k | vs default | p |
|---|---|---|
| 4 | −0.063 [−0.200, +0.074] | 0.38 |
| **16** | **+0.131 [+0.011, +0.250]** | 0.035 |
| 32 *(advantaged)* | +0.063 [−0.069, +0.195] | 0.30 |

**The curve does not turn over.** Neither rung above the default is worse than it, so there is no
non-cost downside to over-provisioning. k=16 edging ahead is worth holding loosely — p=0.035 across
three contrasts does not survive multiplicity correction, and it is one corpus and one draw. The
k=32 null is the informative one: it had a parroting advantage and still didn't beat 16, so the
plateau is real rather than an artifact of the control.

**Net:** don't go below 8; above is safe; no per-corpus tuning is warranted. For reference, layer-0
naming input tokens scale 55K / 143K / 204K / 287K / 448K for k = 1 / 2 / 4 / 8 / 16.

## Keyphrases

**Fit: nothing.** −0.17 on 20NG (p=0.008 — i.e. dropping them slightly *helped*) and, on the fuller
all-layer sample, +0.162 (p=0.012) in the same direction. On arXiv: −0.009 (p=0.82), flatly neutral.
Dropping keyphrases never measurably hurts fit on either corpus.

**Identification: negligible, not quite zero.** 20NG −0.007 (p=0.15, null); arXiv **+0.016**
(p=0.022) — a statistically detectable cost, but one-sixth of the repeat band and one-sixth of the
weakest degradation the lineup resolves. "Inert" overstates it; "negligible" is right.

**Disambiguation load: the surprise.** Instrumenting the renaming pass over 3 draws per condition:

| 20NG, layer 0 | draws | mean groups |
|---|---|---|
| stock | 1, 1, 1 | 1.0 |
| exemplars ablated | 0, 1, 2 | 1.0 |
| **keyphrases ablated** | **3, 4, 3** | **3.3** |
| subtopics ablated | — | 1.3 |

Dropping keyphrases **triples** fine-layer disambiguation load, with draws that do not overlap a
stock arm that was perfectly stable at 1,1,1. On arXiv *no* condition produces any load at all — the
pass never fires — so the effect is corpus-dependent and arXiv can neither confirm nor refute it.

The mechanism is plausible: keyphrases supply corpus-contrastive vocabulary, which is exactly what
separates a cluster from its neighbours. They buy no fit and no findability, but they appear to buy
**distinctiveness**.

**This changes the recommendation.** Defaulting keyphrases off is quality-neutral on both axes and
both corpora, but on 20NG it costs ~2.3 extra renaming calls per run, each covering up to 12 topics
with full exemplar and keyphrase context. Against the prompt tokens saved across ~107 naming calls
that may be close to a wash. The honest summary is **quality-neutral, cost-ambiguous,
corpus-dependent** — not "free win."

## Subtopics

The least-examined feature, and the reported number is misleading as scoped.

| layer | n | judge Δ | note |
|---|---|---|---|
| L0 | 73 | +0.011 | **subtopics do not exist at layer 0** |
| L1 | 24 | −0.000 | |
| L2 | 9 | −0.481 (p=0.125) | large, badly underpowered |
| **all (as reported in #173)** | 106 | **−0.033** | two-thirds structurally null rows |

Layer 0 has no children, so most of the rows behind −0.033 cannot show an effect — and they are not
quiet: 54 of 73 layer-0 labels still "changed" under the ablation, which is pure temperature-0.4
redraw noise (below). Properly scoped to L≥1, 20NG alone gives n=33, −0.131, SE 0.121,
CI [−0.369, +0.106] — not a null, an absence of measurement.

There is also a **mechanism confound**. `topic_name_prompt` has a `[!SKIP!]` path: a cluster with
exactly one named child inherits that child's name outright, with no LLM call. Blanking subtopics
disables it, so single-child clusters take a different code path rather than the same prompt minus
a feature — 12 of 33 coarse clusters on 20NG. Reported separately below.

**Fit** (positive = dropping hurt):

| | n | Δ | p |
|---|---|---|---|
| pooled multi-child (the real prompt feature) | 43 | −0.132 [−0.292, +0.028] | 0.081 |
| pooled single-child (propagation path) | 24 | +0.028 | 0.62 |

**Identification** (repeat band ≈ 0.08):

| | n | Δ prob-mass | p |
|---|---|---|---|
| pooled multi-child | 35 | +0.005 [−0.026, +0.036] | 0.66 |
| pooled single-child | 24 | +0.002 [−0.047, +0.051] | 0.73 |

Subtopics are **identification-inert**, and that null is tight relative to the noise band — a real
measurement rather than an absence of one. On fit they contribute nothing positive; the direction is
consistently negative on both corpora (dropping them slightly helps), marginal at p=0.081. The
`[!SKIP!]` path is a no-op either way, which is itself useful: it means the multi-child rows are a
clean read of the prompt feature.

**Disambiguation load: ~none.** +0.3 groups at 20NG L0 (where subtopics don't exist — noise), +0.7
at L1, 0 at L2, nothing on arXiv. The largest pre-pass similarity shift anywhere is 20NG L2
(+0.069), consistent with subtopics reducing collision pressure at the coarsest layer, but with zero
groups triggered either way it is a hint rather than a result.

**Subtopics manufacture cross-layer duplicates.** Via `[!SKIP!]`, every single-child parent inherits
its child's name verbatim:

| condition | single-child parents | parent name == a child's |
|---|---|---|
| stock (both corpora) | 12 | **12** |
| subtopics ablated (both corpora) | 12 | **0** |

Because the disambiguation trigger is per-layer, these are invisible to it. Whether that is correct
behaviour is a design call — a single-child parent arguably *is* the same region at a coarser scale
— but it does mean `topic_names_` carries the same string at two levels, which a treemap or tree
consumer may not expect.

**Power ceiling, stated plainly:** pooling both corpora gives ~67 coarse clusters, and the
single/multi split cuts that to 43 and 24. This resolves an effect around 0.3 and cannot resolve
0.1. Toponymy hierarchies do not have many coarse clusters — that is the point of a hierarchy — so
this is structural, not fixable by spending more.

## The interaction that didn't replicate

Upstream is weighing defaulting keyphrases off; the dose work suggested exemplars could be halved.
Both were measured with the *other* feature at its default, so the combination needed its own cell.

On **20NG** it looked alarming — a paired 2×2 interaction contrast
`D = both − k4 − kp_off + stock` of **−0.269** (SE 0.095, p=0.008): each change alone free or
better, the two together cancelling. On **arXiv**: **D = +0.019** (SE 0.081, p=0.99). Nothing.

The reason is legible. The 20NG interaction was driven by keyphrases-off being a genuine improvement
there (+0.162, p=0.012), which itself doesn't replicate — on arXiv it is −0.009 (p=0.82). No effect,
nothing to interact with. **Withdrawn**, and a caution against generalising any combinatorial result
from a single corpus.

## Two incidental findings about the stock pipeline

**Naming is substantially nondeterministic by default.** Toponymy names at **temperature 0.4**
(`llm_wrappers` default), not 0. Two identical runs share only **16–19 of 74** layer-0 labels on
20NG (~76% churn) and 52% on arXiv. This is far above the 8–22% #173 measured for Haiku at
temperature 0, and it is a property of the stock pipeline rather than of any harness. Consequences:
per-cluster deltas are dominated by draw noise and only paired aggregates over 70+ clusters mean
anything; and any golden-test or regression-check design (#154) has to contend with it. Encouragingly,
*aggregate* quantities are far more stable than individual names — the disambiguation group count
came out 1, 1, 1 across three independent stock runs.

**A silent failure in the renaming pass.** One of 24 instrumented runs emitted
`All retries exhausted for generate_topic_cluster_names: IndexError: list index out of range.
Returning old names.` The disambiguation call fails and falls back to the un-disambiguated names
with only a warning. Rare (1/24) and it did not change any conclusion here — the affected cell's two
sibling draws bracket it — but it is silent in normal use and probably merits its own issue.

## What this means for Toponymy (actionable)

1. **Keyphrases can default off on label-quality grounds, but price the disambiguation cost first.**
   No measurable fit benefit on either corpus, negligible identification cost — but a 3× fine-layer
   renaming load on 20NG. Worth measuring on a representative corpus before flipping the default.
2. **Leave `n_exemplars` alone.** 8 is a sensible floor. Below it is corpus-dependent and risky;
   above it is safe and possibly marginally better. No per-corpus tuning warranted.
3. **Subtopics earn nothing measurable on any of the three axes.** The strongest statement the data
   supports is "no large value"; the coarse-layer sample cannot rule out a small one. If the goal is
   trimming the prompt, this is a better candidate than keyphrases — with the caveat that removing
   it also removes single-child name propagation, which is a behaviour change, not just a prompt one.
4. **Exemplar selection is where naming-quality effort pays off** — the only channel with a large,
   replicated, leakage-controlled effect. It is also the only one that buys nothing for
   findability, so effort there should be judged on fit, not on map navigability.
5. **Correct the #173 subtopics number.** As reported it averages a real coarse-layer effect against
   two-thirds structurally-null rows.

## Caveats

- **Two corpora, one namer (Haiku), one judge/listener family (Sonnet).** The namer caveat is the
  one most likely to matter: a terser model would pad less and might weight the channels differently.
- **Single naming draw per condition** except where noted (20NG dose k=1/2/4 and arXiv k=4 have two).
  At 76% label churn this is the dominant noise source; aggregates are what to read.
- **Disambiguation load is 20NG-only in effect** — arXiv generates no load for any condition, so it
  cannot corroborate.
- **Subtopics are underpowered by construction** (see the power ceiling above).
- **"No measurable difference" means at these resolutions**, not proven equality — the same standard
  used in #177.
- **Untested:** a keyphrase × subtopic interaction. Given that the keyphrase × exemplar interaction
  was 20NG-only, assuming anything about combinations is unwise.

## Reproduce

Everything below is resumable and caches to `data/`.

```bash
# identification: the two ablations through the frozen 20NG lineups, with a power check
uv run python experiments/label_quality/feature_ablation_lineup.py --stage run

# fit: exemplar dose-response (all layers) + the keyphrase x exemplar interaction cell
uv run python experiments/label_quality/exemplar_dose_response.py --stage name
uv run python experiments/label_quality/exemplar_dose_response.py --stage judge
uv run python experiments/label_quality/exemplar_dose_response.py --stage report

# the arXiv replication of all three claims (naming, judging, lineups)
uv run python experiments/label_quality/arxiv_naming_features.py --stage name
uv run python experiments/label_quality/arxiv_naming_features.py --stage judge
uv run python experiments/label_quality/arxiv_naming_features.py --stage lineup
uv run python experiments/label_quality/arxiv_naming_features.py --stage report

# is the exemplar fit effect a parroting artifact? (re-judge on leakage-free documents)
uv run python experiments/label_quality/clean_docs_rejudge.py --stage judge

# does the curve turn over above the default?
uv run python experiments/label_quality/upward_sweep.py --stage name
uv run python experiments/label_quality/upward_sweep.py --stage judge

# subtopics, properly scoped: coarse layers, single- vs multi-child, both axes, both corpora
uv run python experiments/label_quality/subtopics_value.py --stage name
uv run python experiments/label_quality/subtopics_value.py --stage judge
uv run python experiments/label_quality/subtopics_value.py --stage lineup

# disambiguation load: 4 features x 2 corpora x 3 draws, instrumented during fit()
uv run python experiments/label_quality/disambiguation_load.py --stage run

# exact call/token accounting for the dose-response arm (count_tokens, no generation)
uv run python experiments/label_quality/price_dose_response.py
```

## References

- Preiss et al. 2024 — the 0–4 topic-label rubric the grounded judge implements.
- Krahmer & van Deemter 2012, *Computational Generation of Referring Expressions* — the lineup as
  REG comprehension used as a metric ([WAYFINDING.md](WAYFINDING.md)).
- Toponymy issue #176 — the bundled arXiv example row-misalignment this work is structurally immune
  to (verified, not assumed).
- Toponymy issue #154 — golden tests / debuggability, which the temperature-0.4 nondeterminism bears on.
