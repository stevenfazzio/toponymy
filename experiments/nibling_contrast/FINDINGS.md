# Contrastive context for Toponymy's LLM naming — an evaluation

**Status: negative result, and it held up under each follow-up check** (four naming models, a
neighbour-aware judge, and a tuning pass). This is the full writeup; a condensed version is posted as
Discussion #173 (TutteInstitute/toponymy). It relates to the *Contrastive Extractor* item under
FeatureExtractor in #171 (the v0.6 refactor roadmap), which is a contrastive *feature* extractor — this
experiment is the naming-*prompt*-stage sibling of it.

## TL;DR

We tested a natural idea: when Toponymy's LLM names a cluster, also show it the names of a few
**nearby-but-separate** clusters ("niblings" — children of the cluster's tree-siblings, available for
free because layers are named finest→coarsest), so the model can pick a name that *distinguishes* this
region from its neighbours (e.g. "placental mammals" instead of "mammals").

- **It reliably changes names** (~50% of names at coarse layers move, above the noise floor) **but does
  not improve them.** A blind, position-controlled LLM judge prefers the **no-contrast baseline**;
  contrast win-rates run **26–43%** across **four models × two datasets**.
- **A fair judge gives the same answer.** The obvious objection is that the blind judge only sees the
  cluster it's naming, not the neighbours, so it can't credit the one thing contrast is *for*. Re-running
  with a stronger judge (Sonnet) that **is shown the neighbours and more documents**, and asked which name
  fits the cluster *while setting it apart from those neighbours*, gives **31% overall** — basically
  unchanged. The negative result is not an artifact of a blind judge.
- **No real model-strength rescue.** Opus looked best under the blind judge (43%) but washed out to 36%
  under the fair judge, in line with the others.
- **Tuning** (tight distance gate + concise anti-enumeration prompt) only reaches break-even by **firing
  rarely**, i.e. approaching "do nothing".
- **Mechanism flaw:** contrast makes the model want to *distinguish*, but it **doesn't control the *axis*
  of distinction**; its default axis (genre/era/sub-topic) is usually not the abstraction you want.
- **Two byproducts worth more than the feature:** (1) a reusable, deterministic A/B + noise-floor +
  blind/neighbour-aware **judge harness** for naming changes (relevant to #154); (2) **temperature-0
  naming is wildly non-deterministic and model-dependent** — contrast-free layer-0 names differ run-to-run
  by ~8–22% (Haiku, GPT-4o-mini), ~42–49% (Sonnet), and **Opus 4.8 rejects `temperature=0` entirely** (it
  can only be named at temp 1, where ~90% differ); and **centroid–label cosine does not track name
  quality** (it rewards verbosity, penalises specificity).

## The idea, precisely

Toponymy names layers finest→coarsest, so when naming layer *L*, the names of the finer layer *L-1*
already exist. A cluster's prompt already includes its own descendants' names (subtopics); it includes
**nothing about what lies outside the cluster**. The proposal adds that: a small "for contrast, these
nearby groups are SEPARATE" block listing the names of the nearest *L-1* clusters that are **not** this
cluster's own children. Selection is **geometric** — the *k* nearest *L-1* centroids (cosine) excluding
own descendants — which subsumes the genealogical "children of my siblings" notion and dissolves the
synthetic-root/top-layer edge case (we never look up a parent). Single-pass, no extra clustering or
naming calls.

## What was built (`experiments/nibling_contrast/`)

- `selector.py` — the geometric contrast-set selector (+ v1/v2 prompt blocks).
- `ab_harness.py` — names a fitted hierarchy three ways at a fixed temperature: baseline A, baseline B (a
  per-layer **noise-floor** control), and contrast — by monkeypatching only `topic_name_prompt`.
  Disambiguation is disabled so the raw effect is isolated. Per-model temperature (Opus forces temp 1).
- `aggregate.py` — isolates the effect to **injected clusters only**, netting out the per-subset noise.
- `analyze_cosine.py` — the centroid–label cosine study.
- `judge.py` / `judge_opus.py` — **blind** both-orders LLM-judge (sees only the focal cluster + two names).
- `judge_fair.py` — **fair** judge: stronger model (Sonnet), neighbour names shown, ~20 coverage-sampled
  docs, asks for fit *and* distinctiveness from the neighbours.
- `final_check.py` — tuned-contrast retest + disambiguation-count analysis.
- `construct_movie_test.py` — the Movie Madness motivating-case reproduction.

Datasets: 20 Newsgroups (diverse), an AI/arXiv corpus (flat single-domain), and a constructed Movie
Madness movies-vs-TV structure. Naming models: Claude Haiku 4.5, Sonnet 4.6, Opus 4.8, GPT-4o-mini.

## Findings

### 1. Contrast changes names, above the noise floor — but "change ≠ improvement"
On injected clusters the contrast change-rate exceeds the baseline-vs-baseline floor in nearly every
layer-cell, concentrated at coarse layers (e.g. 20ng/haiku L1: 87% changed vs 13% floor). Real
activation — it just turned out to be mostly *degradation* (next).

### 2. Blind judge: contrast loses, across four models
Each contrast-driven rename judged in both A/B orders (disagreement = tie), focal cluster + two names only:

| model | blind win-rate |
|---|---|
| GPT-4o-mini | 30% |
| Haiku | 26% |
| Sonnet | 26% |
| Opus | 43% |

The dominant failure is **over-qualification / verbose enumeration** (`Diverse ML Methods` → `Diverse AI
Methods Spanning Diffusion, RL, and Edge Systems`). Genuine sharpening exists (`Sports Analysis` →
`Hockey and Baseball`) but is the minority.

### 3. Fair judge (the robustness check): same answer
Re-judged with **Sonnet, neighbour names shown, ~20 coverage-sampled documents**, asking which name fits
*and* distinguishes from the neighbours:

| model | blind | fair |
|---|---|---|
| GPT-4o-mini | 30% | 28% |
| Haiku | 26% | 29% |
| Sonnet | 26% | 29% |
| Opus | 43% | 36% |
| **overall** | — | **31%** (n=95) |
| 20ng / arxiv (fair) | — | 48% / 23% |

Giving the judge exactly what it was "missing" moved the overall from ~27% to 31%. Contrast still loses
~2:1; the apparent Opus edge washed out; ties rose sharply (with the fuller picture the judge often finds
the two names equivalent). The only near-wash is the diverse corpus (20ng, 48%); the flat one loses clearly.

### 4. Centroid–label cosine is not a quality proxy
Contrast moves the label *toward* the centroid as often as away (48%/52%, mean Δ≈+0.011), and
`sign(Δcentroid)` agrees with the judge **56%** of the time (≈ chance). It **rewards verbosity**
(enumerations cover more of the cluster → higher cosine) and **penalises specificity** (a distinguishing
qualifier pulls away from the bland mean) — close to backwards for toponym quality.

### 5. Tuning rescues it to "neutral", by inaction
A tight `max_dist=0.25` gate + a concise anti-enumeration block lifts the win-rate (20ng/haiku 23%→50%,
20ng/gpt-4o-mini 38%→50%, arxiv/haiku 17%→25%) — but the decided-case count collapses alongside (the gate
filters most contrast out). The ceiling of "gate harder" is "become the baseline" = 50%. Where v2 still
fires enough to measure (arXiv), it still loses.

### 6. Disambiguations: suggestive but confounded
Using Toponymy's own trigger (`cluster_topic_names_for_renaming`), contrast *appears* to cut collision
groups (arxiv/haiku Σ groups baseline/v1/v2 = 11/3/6), but **layer 0, which gets no contrast, swings 6→2
on nondeterminism alone**, so the contrasted-layer reductions are within noise. Inconclusive without a
baseline-vs-baseline disambiguation control.

### 7. The motivating case (Movie Madness: movies vs TV)
Constructed the exact structure (movies/TV split from the real `media_type`, genres KMeans-clustered
within each) and named the movie supercluster with vs without the TV sibling as contrast:

| naming model | baseline | +contrast |
|---|---|---|
| Haiku | `Dramatic Cinema` | `Classic Drama` (wrong axis: genre/era) |
| Sonnet | `Mixed Genres` | `Diverse Cinema` (reached the medium) |
| Opus | `Film Anthology` | `Diverse Films` / `Feature Films` (already medium) |

The exact "Thrillers" failure didn't cleanly reproduce (with *diverse* genre children, the stronger models
abstract reasonably). Across the three models contrast's effect was worse / better / no-change — i.e.
inconsistent, which reads as noise (these are single runs, Opus at temp 1) rather than a clean
model-strength trend. TV was already medium-level without contrast in every case.

### 8. Byproduct: temperature-0 naming is very non-deterministic
Layer 0 never receives contrast, so contrast-arm-vs-baseline-arm there is pure noise: ~8–22% of names
differ for Haiku/GPT-4o-mini, ~42–49% for Sonnet, and **Opus 4.8 won't accept `temperature=0` at all**
(deprecated for that model) — at temp 1 ~90% differ. **Name quality can't be A/B-tested without a
per-model, per-layer noise floor**, and the newest models are the hardest to pin down.

## The core lesson

> **Contrast reliably makes the model want to distinguish, but it does not control the axis of
> distinction.** Given "Television Drama Series" next door, Haiku concluded "I'm *Classic Drama*" — right
> instinct, wrong dimension (genre/era, not medium). "Just show the neighbours" can't steer which way the
> model differentiates, and a judge that *can* see the neighbours still prefers the cleaner baseline.

## Recommendation

- **Don't ship naive contrast-in-the-naming-prompt.** It loses to the baseline across every model and both
  corpora (a wash on the diverse one, a clear loss on the flat one), a fair neighbour-aware judge doesn't
  rescue it, and tuning only reaches neutrality by switching itself off.
- **If the contrastive direction (the #171 Contrastive Extractor) is pursued, the open problem is *axis
  control*** — getting the model to distinguish on the *useful* dimension. Prompt-level steering ("are
  these a different *kind/medium*?") or doing the contrast at the **feature-extraction** stage (the way the
  Contrastive Extractor is headed) look more promising than dumping neighbour names into the prompt.
- **Reusable now:** the deterministic A/B + noise-floor + blind/neighbour-aware **judge harness** is useful
  for the golden-test/debug ask in #154, independent of this feature; and the temp-0 noise-floor +
  cosine-metric findings should inform any future naming-quality testing.

## Caveats

Single embedder/cluster settings per dataset; the blind judge is Haiku and the fair judge is Sonnet (LLMs,
not human — though both-orders + per-layer noise floors control position bias and run-to-run noise); the
Movie Madness reproduction is constructed (the top split is imposed from `media_type`) and uses a different
embedder/clusterer than the original ramify run; Opus could only be evaluated at temperature 1. The
aggregate verdict — a fair, neighbour-aware judge over ~95 decided renames across four models — is the
robust result; the motivating-case and disambiguation threads are suggestive, not conclusive.
