# Contrastive context for Toponymy's LLM naming — an evaluation

**Status: negative result (with a narrow, unproven exception).** Drafted to be adaptable into a
comment on TutteInstitute/toponymy #171 (the contrastive-feature thread).

## TL;DR

We tested a natural idea: when Toponymy's LLM names a cluster, also show it the names of a few
**nearby-but-separate** clusters ("niblings" — children of the cluster's tree-siblings, available
for free because layers are named finest→coarsest), so the model can pick a name that *distinguishes*
this region from its neighbours (e.g. "placental mammals" instead of "mammals").

- **It reliably changes names** (~50% of names at coarse layers move, above the noise floor) **but does
  not improve them.** A blind, position-controlled LLM judge prefers the **no-contrast baseline ~73% of
  the time** (contrast win-rate **27%** of decided cases), consistent across 3 models × 2 datasets.
- **Tuning** (tight distance gate + concise anti-enumeration prompt) reduces the harm to roughly
  break-even — but mostly by **firing rarely**, i.e. approaching "do nothing."
- **The mechanism's flaw is fundamental:** contrast makes the model want to *distinguish*, but it
  **doesn't control the *axis* of distinction**. The model's default axis (genre/era/sub-topic) is
  usually not the abstraction you want.
- **Two byproduct findings worth more than the feature:** (1) a reusable, deterministic A/B + blind-judge
  **evaluation harness** for naming changes (relevant to #154/#131); (2) **temperature-0 naming is
  highly non-deterministic and model-dependent** — layer-0 (contrast-free) names differ run-to-run by
  ~10–22% (haiku, gpt-4o-mini) up to **~42–49% (sonnet)**; and **centroid–label cosine does not track
  name quality** (it rewards verbosity, penalises specificity).
- **The narrow exception:** on the *coarsest layer*, where a cluster's children are all sub-types of one
  category (movie genres) and a sibling cluster is a different category (TV), a **strong** model (sonnet)
  used the contrast to reach for the medium ("Mixed Genres" → "Diverse Cinema"); a weak model (haiku)
  picked the wrong axis ("Dramatic Cinema" → "Classic Drama"). This is suggestive but **within sonnet's
  large noise** and not cleanly demonstrated.

## The idea, precisely

Toponymy names layers finest→coarsest, so when naming layer *L*, the names of the finer layer *L-1*
already exist. A cluster's prompt already includes its own descendants' names (subtopics); it includes
**nothing about what lies outside the cluster**. The proposal adds that: a small "for contrast, these
nearby groups are SEPARATE" block listing the names of the nearest *L-1* clusters that are **not** this
cluster's own children. Selection is **geometric** — the *k* nearest *L-1* centroids (cosine) excluding
own descendants — which subsumes the genealogical "children of my siblings" notion and dissolves the
synthetic-root/top-layer edge case (we never look up a parent). It's single-pass and adds no clustering
and no extra naming calls.

## What was built (`experiments/nibling_contrast/`)

- `selector.py` — the geometric contrast-set selector (+ v1/v2 prompt blocks).
- `ab_harness.py` — names a fitted hierarchy **three ways at temperature 0**: baseline A, baseline B
  (a per-layer **noise-floor** control), and contrast — by monkeypatching only `topic_name_prompt`.
  Disambiguation is disabled so the raw effect is isolated.
- `aggregate.py` — isolates the effect to **injected clusters only**, netting out the per-subset noise.
- `analyze_cosine.py` — the centroid–label cosine study.
- `judge.py` — blind, both-orders LLM-judge (ties absorb position bias / noise).
- `final_check.py` — tuned-contrast retest + disambiguation-count analysis.
- `construct_movie_test.py` — the Movie Madness motivating-case reproduction.

Datasets: 20 Newsgroups (diverse), an AI/arXiv corpus (flat single-domain), and a constructed
Movie Madness movies-vs-TV structure. Models: Claude Haiku 4.5, Claude Sonnet 4.6, GPT-4o-mini.

## Findings

### 1. Contrast changes names, above the noise floor — but "change ≠ improvement"
On injected clusters, the contrast change-rate exceeds the baseline-vs-baseline floor in 16/17
layer-cells, concentrated at coarse layers (e.g. 20ng/haiku L1: 87% changed vs 13% floor → +73 net).
This is real activation — it just turned out to be mostly *degradation* (see #2).

### 2. Blind judge: contrast loses
122 contrast-driven renames, each judged in both A/B orders (disagreement = tie):

| | contrast | baseline | tie | win-rate (decided) |
|---|---|---|---|---|
| overall | 23 | 62 | 37 | **27%** |
| 20ng | 10 | 19 | 12 | 34% |
| arxiv | 13 | 43 | 25 | 23% |
| gpt-4o-mini / haiku / sonnet | — | — | — | 30% / 26% / 26% |

The dominant failure is **over-qualification / verbose enumeration** (`Diverse ML Methods` → `Diverse AI
Methods Spanning Diffusion, RL, and Edge Systems`). Genuine sharpening exists (`Sports Analysis` →
`Hockey and Baseball`) but is the minority.

### 3. Centroid–label cosine is not a quality proxy
Across the 122 renames, contrast moves the label *toward* the centroid as often as away (48%/52%, mean
Δ≈+0.011), and `sign(Δcentroid)` agrees with the judge **56%** of the time (≈ chance). Mechanistically it
**rewards verbosity** (enumerations cover more of the cluster → higher cosine) and **penalises
specificity** (a distinguishing qualifier pulls away from the bland mean) — close to backwards for
toponym quality. Useful only as a weak off-topic filter, not a Gate-3 metric.

### 4. Tuning rescues it to "neutral", by inaction
A tight `max_dist=0.25` gate + a concise, anti-enumeration block lifts the win-rate (20ng/haiku 23%→50%,
20ng/gpt-4o-mini 38%→50%, arxiv/haiku 17%→25%) — but the number of decided cases collapses alongside
(the gate filters most contrast out). The structural ceiling of "gate harder" is "become the baseline" =
50%. Where v2 still fires enough to measure (arXiv), it still loses 3:1.

### 5. Disambiguations: suggestive but confounded
Using Toponymy's own trigger (`cluster_topic_names_for_renaming`), contrast *appears* to cut collision
groups (arxiv/haiku Σ groups baseline/v1/v2 = 11/3/6). But **layer 0, which gets no contrast, swings
6→2 on nondeterminism alone**, and the contrasted-layer reductions are within that noise. Inconclusive
without a baseline-vs-baseline disambiguation control. (20NG had ~no collisions to begin with.)

### 6. The motivating case (Movie Madness: movies vs TV)
Constructed the exact structure — movies/TV split from the real `media_type`, genres KMeans-clustered
within each — and named the movie supercluster with vs without the TV sibling as contrast:

| | baseline | +contrast (v1) | +contrast (v2) |
|---|---|---|---|
| **haiku** movies | `Dramatic Cinema` | `Classic Drama` | `Classic Drama` |
| **sonnet** movies | `Mixed Genres` | `Diverse Cinema` | `Classic Cinema` |
| (both) TV | `Television Seasons` / `TV Seasons` | (unchanged) | (unchanged) |

The exact "Thrillers" failure didn't cleanly reproduce (with *diverse* genre children, both models
abstracted reasonably — they don't collapse to one genre). On contrast: **haiku distinguished on the
wrong axis** (genre/era: `Classic Drama`), while **sonnet reached for the medium** (`Diverse Cinema` —
"Cinema" = movies). TV was already medium-level without contrast. So the intuition has a kernel of truth
a *strong* model can act on — but it's a single observation inside sonnet's ~45% noise, and v2 still
injected a spurious "Classic". Not a clean win.

### 7. Byproduct: temperature-0 naming is very non-deterministic
Layer 0 never receives contrast, so contrast-arm-vs-baseline-arm there is pure noise: ~8–22% of names
differ for haiku/gpt-4o-mini, and **~42–49% for sonnet** (likely MoE routing). This is the single most
practically-useful finding for the project: **name quality cannot be A/B-tested without a per-model,
per-layer noise floor**, and sonnet in particular is a poor instrument for it.

## The core lesson

> **Contrast reliably makes the model want to distinguish, but it does not control the axis of
> distinction.** Given "Television Drama Series" next door, haiku concluded "I'm *Classic Drama*" — right
> instinct, wrong dimension. Whether the model lands on the useful axis (medium, here) instead of a
> salient-but-unhelpful one (genre/era) depends on model strength and is noisy. "Just show the
> neighbours" can't steer that.

## Recommendation

- **Don't add naive contrast-in-the-naming-prompt** to Toponymy: it degrades quality on the average
  cluster and only reaches break-even by switching itself off.
- **If the #171 contrastive direction is pursued, the open problem is *axis control*** — getting the
  model to distinguish on the *useful* dimension. Prompt-level steering ("are these a different
  *kind/medium*?") or doing contrast at the **keyphrase/feature** stage (a contrastive keyphrase
  extractor) are more promising than dumping neighbour names into the prompt. The one place the naive
  version showed a pulse is **coarsest-layer + strong model + ungated** — worth a *replicated* (n>1 per
  arm, to beat sonnet's noise) follow-up before any investment, not a feature yet.
- **Reusable now:** the A/B + blind-judge + noise-floor **harness** here is directly useful for the
  "golden test / stage inspection" asks in #154 and #131, independent of this feature. And the temp-0
  noise-floor + cosine-metric findings should inform any future naming-quality testing.

## Caveats

Single embedder/cluster settings per dataset; LLM-judge (haiku) not human; the Movie Madness reproduction
is constructed (the top split is imposed from `media_type`) and uses a different embedder/clusterer than
the original ramify run; sonnet's high nondeterminism limits single-instance claims. The aggregate
verdict (blind judge over 122 instances, noise-controlled) is the robust result; the motivating-case and
disambiguation threads are suggestive, not conclusive.
