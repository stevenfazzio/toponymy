# Label-Quality Evaluation for Toponymy — Experiment Plan

**Goal.** Find a usable, cheap, deterministic metric for the quality of Toponymy's
region/cluster labels — and, along the way, settle a few open questions about embedding
geometry the metric depends on. Leaning **practical** (a usable metric is the win), with
enough **scientific** instrumentation that negative results are also keepers.

**Context.** Prior negative result (the `experiment/nibling-contrast` work; GitHub
Discussion #173): scoring a label by **cosine similarity between the label embedding and
its cluster centroid** does not track judge-rated quality — it rewards verbose labels,
penalizes specific ones, ~chance agreement with a judge. Two hypothesized causes:
1. labels (short, abstract phrases) and documents (long prose) are different *classes* of
   text in embedding space;
2. a cluster's **centroid ≠ its extent**.

This plan tests fixes for both, adds a hierarchy-consistency check, and gates an optional
"erase the label/document difference" approach.

---

## Two literature findings that shaped this plan

1. **The closest published analog (Preiss et al. 2024) does NOT validate the
   centroid-cosine negative.** It uses an LLM as *namer only*, a **human judge grounded in
   the cluster's documents**, and computes **no automatic metric** (rejecting ROUGE-style
   reference matching on principle: "a cluster can be named differently but equally well").
   So "centroid-cosine is weak" currently rests *only* on our own result — an open gap to
   nail, not a settled literature finding. What we borrow: their **5-domain rubric**
   (fluency / **consistency** = faithfulness / **relevance** ≈ precision, "only important
   content" / **completeness** ≈ recall, "all important content" / **overall**), 0–4
   Likert, blinded to name source, judge grounded in ≥10 sampled docs, + a **pick-the-best**
   adjudication.

2. **The SOTA graded-entailment method (Renner et al. 2023) is embedding-FREE and beats
   embeddings** (HyperLex Spearman ρ: WordNet-IC **0.744** vs LEAR 0.686 vs Word2Vec-cosine
   0.205), decomposing as `gle(X,Y) = Sim(symmetric) + SpecLoss(asymmetric)` (Sim-only
   0.393, SpecLoss-only 0.521, sum 0.744). Consequences:
   - **Cosine is not "wrong" for hierarchy — it's the necessary-but-insufficient
     *symmetric half*.** `cosine(parent,child)` is the `Sim` term; add an asymmetric
     **generality-gap** term to get direction.
   - **The generality signal is best obtained from structure** (information content
     `IC(s) = −log P(s)` ≈ denotation breadth / hyponym count), **not an assumed embedding
     axis.** Toponymy gives a native IC proxy for free: a region's size / its child-region
     count. So "find a generality axis" becomes a **bake-off of generality proxies validated
     on HyperLex**, with the embedding axis as one (expected-weak) contender.

---

## Phase 0 — The instrument (extend the nibling harness)

**Substrate:** 20 Newsgroups, already prepped in `experiments/nibling_contrast/data/`
(`ng_texts.json`, `ng_emb.npy`, `ng_coords.npy`, `ng_targets.npy`, `ng_target_names.json`)
— note the **gold categories**, which give a free objective leg. *(Decision: 20NG now;
defer a second in-the-wild substrate, e.g. a fitted datamap dataset.)* Re-fit Toponymy →
`topic_names_`, `cluster_tree_`, per-cluster exemplars/keyphrases.

**Three validation legs (increasing cost):**
1. **Perturbation / intrusion battery (judge-free, cheapest).** Per gold label, generate
   known-bad variants from `cluster_tree_`: *ancestor-swap* (coarse parent label =
   over-generalization), *sibling-swap* (a true sibling's label), *generic-ify* ("Various
   topics"), *verbose paraphrase*, *off-topic*. Any candidate metric must rank gold > each.
   (Chang et al. 2009 intrusion design, Toponymy-native.)
2. **Gold-category alignment (objective, cheap).** Does the label match the cluster's
   majority 20NG class? Coarse sanity anchor, complementary to the judge.
3. **Grounded judge (gold standard, costly).** Preiss rubric; LLM judge **grounded in N
   exemplar docs + keyphrases** (Krumdick et al. 2025: never reference-free — grounding
   lifts judge–human κ from ~0.14–0.41 to ~0.41–0.80 on the hard slice); CoT +
   **self-consistency** (majority over k samples — also tames the temp-0 nondeterminism the
   nibling noise-floor measured). **Calibrate against a small human seed (~25–30 blinded
   items on 0–4 overall)** *(decision: yes)* → report κ before trusting it.

**Deliverable:** for any labeling, a judge-free score + a grounded-judge score — the
yardstick for everything below.

---

## Phase 1 — The cheap fidelity metric *(practical payoff; ideas C / B1)*

Compute label↔reference similarity under four reference points:
**raw centroid vs whitened centroid vs medoid vs top-exemplar** (whitening per Su et al.
2021 / Mu & Viswanath 2018 = the published fix for the central/verbose bias that sank raw
cosine; medoid/exemplar = a real in-distribution point that keeps specificity).

**Gates:** (a) ranks gold > each perturbation; (b) Spearman-correlates with the grounded
judge, beating the raw-centroid baseline's ~chance.

**If a variant clears both → that's the usable cheap metric** (+ the post-mortem the field
lacks). Reasonable stop point for the practical goal.

---

## Phase 2 — The hierarchy check *(optional / scientific; ideas B2 / D)*

- **2a — Generality-proxy bake-off, validated on HyperLex** (2,616 pairs, 0–6 scale,
  Spearman ρ; targets: cosine 0.205 / LEAR 0.686 / WordNet-IC 0.744 / human ceiling 0.854).
  Contenders for a label's *linguistic* generality: WordNet-IC of the head term (Renner's
  winner), corpus frequency, an **embedding linear axis (the idea-D probe, expected weak)**,
  label length (naive floor). Keep the winner.
- **2b — Renner-style parent⊇child scorer** = `cosine(parent,child) + SpecLoss`, where
  `SpecLoss = 1 − G(parent)/G(child)` and `G` = the winning generality proxy. Test on real
  `cluster_tree_` edges (should read as proper entailment) vs ancestor/sibling perturbations
  (should break); correlate with the judge's relevance/completeness. Side-finding: run on
  Toponymy's *own* hierarchy to flag mis-altituded labels (specific region, generic name) —
  the evaluator the nibling experiment lacked.

---

## Phase 3 — Erasure *(optional / gated; idea A)*

- **Gate probe only — do not build erasure yet.** INLP/LEACE rank-sweep on
  `class ∈ {label, doc}`: is the label↔doc gap a low-rank linear offset? Plus a
  Gao & Metaxas 2026 check: is the separating axis **offset or dispersion** (don't assume
  "length")?
- **Low-rank offset →** erase (LEACE; LANCER, Huang et al. 2024, is the text precedent),
  re-run Phase-1 probes in the erased space, keep only if it beats raw. **Nonlinear /
  dispersion →** park and report (the shared-datamap multimodal result predicts this is
  live). Note the bar here is only *within-cluster rank preservation* — weaker than the
  cloud-merge the modality experiment failed, so don't pre-write it off.

---

## Phase 4 — The wayfinding lineup *(the candidate fine instrument; discriminative label eval)*

**Why.** Fine discrimination is the program's open cell: geometry is chance on good-vs-good and even
the grounded judge TIES ~half the time. Both existing instruments measure *fit* (is this label
true/well-formed for this cluster). A map label's deployment function is different: **a reader must
find the right region among its neighbours given only the name** — wayfinding, the cartographic frame
made operational. That's a *grounded* task (verifiable answer; accuracy, not preference), and the
plausible fine ranker: two labels that both "fit" (judge tie) can differ sharply in how *identifying*
they are. Ports the police-lineup metric from atlantic-mirror (`scripts/lineup_eval.py` there), where
the nn-hardened version separated systems the preference judge had scored as a tie — the same
tie-shape we're stuck on. Lineage: REG comprehension (Krahmer & van Deemter 2012), self-retrieval /
frozen-listener captioning (Liu et al. 2018; Ou et al. 2023). Relation to the Phase-0 battery: the
battery plants *bad labels on the true cluster* (detection); wayfinding tests the *real label against
real neighbours* (discrimination). Complementary — and its confusion structure is a behavioral map of
label overlap between regions: the cluster-edges question (#173, Leland) approached from behavior
rather than geometry.

**Unit.** One lineup per (layer, cluster, label-candidate): listener sees label L + k candidate
clusters = true cluster + its k−1 nearest same-layer clusters (centroid distance in the at-home
high-D embedding space — the geometry naming actually used; low-D clusterable-space distance is the
noted alternative), identities hidden (A…E). Listener picks which candidate L names. Per-layer only;
`Unlabelled` never appears; a layer with < k clusters uses the whole layer. Run all layers, report
per-layer (fine = the hard/interesting regime; coarse lineups are few and near-whole-layer).

**Candidate representation — the leads-only analog (load-bearing).** Each candidate = N **held-out**
member docs, sampled EXCLUDING the exemplars the namer saw (truncated as in judge grounding).
Otherwise a label that parrots exemplar phrasing wins trivially and we test prompt-memorization, not
generalization to the region. Defaults: k=5, N=5, ~500-char truncation; sweep k∈{3,5,7} once (chance
floor moves with k — always report top-1 against 1/k).

**Pairing discipline (what makes deltas interpretable).** Per cluster, the distractor set + held-out
doc sample are seeded/frozen ONCE and reused across every label variant → comparisons are paired,
lineups identical except the label. Candidate order reshuffles per self-consistency sample (k=3
samples → majority top-1 + prob-mass), so position bias is marginalized rather than frozen in.
Listener = sonnet (∉ {haiku, gpt-4o-mini} namers, per the fine-disc no-self-preference rule); haiku
for sweeps/floors.

**Metrics.** Per label: top-1, prob-mass on true, below-chance flag. Per layer/system: top-1 rate vs
1/k, mean prob-mass, below-chance fraction, and the **cluster×cluster confusion matrix** (where lost
mass went) — per-pair confusability as a deliverable, not a nuisance (sibling-swap's "2/5 genuinely
fit" becomes measured ambiguity instead of annotation noise).

**Floors & catch trials (instrument checks before use).**
- *Shuffled-label floor:* labels permuted within layer → expect ≈ chance (the noise-floor rung).
- *Gimme trials:* distractors = distant clusters → expect ≈ ceiling (listener sanity; the battery
  says distant is easy).
- *Repeat floor:* identical lineups re-run → listener nondeterminism band (the temp-0 finding);
  every claimed delta must clear it.

**Gates.**
- **(a) Battery ordering — sharp, falsifiable signatures.** Gold must beat the shuffled floor and
  every variant on prob-mass. Predicted signatures: `sibling` label → mass shifts to the sibling
  (BELOW chance on true — sharper than the judge's soft 0.62); `generic` → ~uniform; `ancestor` →
  diffuse over the parent's children; `distant` → off-true. `verbose` is the **predicted
  divergence**: judge/human punish padding hard, the lineup may not (padding can *aid*
  identification) — if confirmed, fit and identifiability are genuinely different axes and the
  combined instrument is (judge ∧ lineup): the two-graders result in a second domain.
- **(b) Judge relation on the battery set:** expect moderate positive ρ, NOT ~1 (≈1 ⇒ redundant with
  the judge; ≈0 while also failing gate (a) ⇒ broken instrument). The interesting zone: passes (a),
  moderate ρ, and —
- **(c) The payoff — the 104 fine pairs.** (1) Does the lineup *decide* judge-tied pairs (per-pair
  prob-mass gap clearing the repeat floor)? (2) Agreement with the judge on its 54 decided pairs?
  (3) Small blinded human seed (~20 lineup-decided-judge-tied pairs; reuse the calibration
  machinery): does human preference track the lineup's pick? — the calibration bar Phase 0 set,
  applied to the new axis.

**Outcomes (all keepers).** (i) Lineup ranks good-vs-good → the fine ranker exists; judge stays the
fit guard. (ii) Lineup ties too → judge-tie + lineup-tie ⇒ the pairs are *functionally equivalent*
and "fine ranking" was ill-posed — a strong close-out, not a failure. (iii) Lineup decides but
disagrees with the judge on decided pairs → two axes (fit vs identifiability); report both.
Downstream (separate experiment, only on (i)/(iii)): **lineup-as-selector** — propose k names
WITHOUT contrast context (nibling says the speaker shouldn't see neighbours), listener picks the
most identifying candidate that passes the judge; generalizes the disambiguation pass from exact
duplicates to confusables. Goodhart guard before believing any gain: a held-out lineup config
(different listener model + fresh distractor/doc draw), per Gao et al. 2023.

**Cost (20NG, order of magnitude).** Battery: 595 lineups (107 gold + 488 variants) × 3 samples ≈
1.8k listener calls at ~25 short docs each (~3.5k tokens in, tiny out) ≈ 6–7M input tokens. Fine
pairs: 104 × 2 × 3 ≈ 0.6k more; floors/sweeps on haiku. Same async_judge plumbing.

---

## Phase 5 — from measuring to controlling *(the constructive turn; conjunct ablation + length controller)*

**Why.** Phase 4 established identifiability as a second axis and left one library-shaped opening:
Toponymy sets label specificity with an open-loop dial (`detail_levels` → a `SUMMARY_KINDS` ladder of
word-count phrases indexed by layer altitude; the templates *ask* for distinguishability but nothing
checks it). The lineup is the checker. Framing (from the 2026-07-03 debrief): a label is a lossy code
for its region; the lineup ≈ an InfoNCE-style estimate of I(label; region) against the layer's
neighbours; label quality = rate (length) vs distortion (identification error). Post NOTHING until the
story completes (batch-by-story decision); 20NG first, ≥1 arXiv cell before any post.

**5a — conjunct ablation** *(cheap; validates 5b's core assumption)*. Split each compound gold label
into top-level conjuncts (haiku splitter, structured output, light distribution of shared modifiers so
conjuncts stand alone; splits cached + eyeballed). Two variants per conjunct through the cluster's
FROZEN lineup (paired with the already-run battery gold): `drop:i` (label minus that conjunct) and
`only:i` (that conjunct alone). Δpm(drop) = marginal identification value (free-rider if below the
repeat band); pm(only) vs pm(full) = could the label just *be* that conjunct (MDL question directly).
Report: free-rider fraction, position effects, % of labels with a single sufficient conjunct,
face-valid examples. ~1.2k sonnet calls ≈ $13.

**5b — the length controller** *(the payoff; possible library contribution)*. Per cluster: name at
every `SUMMARY_KINDS` rung with Toponymy's own templates (haiku namer, stock features), run each rung
through the frozen lineup (sonnet), select the **shortest rung whose pm is within the repeat band of
the best rung** (MDL: minimize length s.t. no measurable identification loss — adaptive per cluster,
no absolute threshold). Compare controller picks vs stock linspace names on judge fit, lineup pm, and
length. Byproduct figure: pm vs label length across rungs per layer = the empirical rate–distortion
curve of topic naming. **Goodhart guard (non-negotiable, Gao et al. 2023): all headline evaluation on
a HELD-OUT lineup config** — fresh doc/distractor draws + a different listener (gpt-4o-mini; haiku
names, sonnet selects, so no self-preference) + judge-fit non-regression. ~107×7×3 ≈ 2.2k sonnet
calls ≈ $25 + held-out eval.

**5c (optional, at post-assembly time)** — human-as-listener form (~20 lineups: label + k candidate
doc-groups, pick the group) to close c-3's fit-style-presentation gap.

---

## Phase 7 — can identification be scored without an LLM at all? *(the distillation question)*

**Why.** #177's controller works but costs 25–30× stock naming, almost all of it listener traffic.
The obvious move is to replace the listener with something cheap. The framing that makes this
different from #173's dead centroid-cosine: **the lineup normalizes over a neighbourhood.**
Centroid-cosine is a *pointwise* score, so it structurally cannot express "this label is true of
the region but wrong given the neighbours" — the mechanism behind Phase 5a's 12% anti-conjuncts.
A softmax over the k=5 frozen lineup candidates can, because the normalizer is the neighbourhood.
The same normalization is also predicted to cancel the padding bias for free: padding pulls a label
toward the anisotropic mean direction, centroids are document means in that same anisotropic space,
so padding raises cosine to *all five* candidates by a similar amount and a softmax cancels
common-mode shifts by construction. **So the cheap half of "train toward the lineup" may need no
training at all** — which is why the first tranche is free and the tower is the last rung, not the
first.

**Two conditionals, two axes, two corpora on disk.** The lineup is `p(region | label)`; naming is
`p(label | region)`. A CLIP-style dual encoder trains one scoring function against both. But what
decides *what to build* is the axis, and the two supervision sets already committed here map onto
the two axes cleanly:

| corpus | size (committed) | axis | selects among |
|---|---|---|---|
| lineup units (full k-way `mass`) | **6,272** over 223 clusters | identification | regions, given a label |
| grounded-judge records | **~3,192** | fit | labels, given a region |

Phase 4/6 established these axes disagree (ρ 0.65–0.74; opposite directions on 37% of decided
pairs), so this is two distillation problems, not one, and each application picks its own.

**The noise ceiling (measured 2026-08-03, before any probe).** Split-half over the three stored
listener samples, *within cluster across ladder rungs* — the exact regime rung selection operates
in — with a Spearman-Brown extrapolation:

| | 20NG | arXiv |
|---|---|---|
| within-cluster split-half (1 sample vs other 2), Pearson | 0.378 | 0.427 |
| implied reliability of the committed 3-sample `pm` | **≈0.57** | **≈0.62** |
| ⇒ correlation an *oracle* scorer would show against these targets | ~0.75 | ~0.79 |

This is an **upper bound** relative to the held-out condition, because the three samples share
frozen documents (only order and temperature vary) — document-draw variance is not in it. Three
consequences that shape everything below: (1) never report R²/ρ against measured pm as a headline,
it is capped near 0.75; (2) a smooth model fit over ~6k units can be a *lower-variance* estimator
of the same quantity than any single 3-sample listener call, so the student can legitimately beat
its teacher; (3) more samples is the competing lever — 3→9 samples takes reliability ≈0.57→≈0.80
for 3× cost, and any "cheap scorer" claim must be priced against that alternative.

**Capacity ladder** (report where the curve flattens; the `n_exemplars` house style):
r0 `softmax(cosine/τ)`, one parameter · r1 + cheap features (IDF-weighted lexical overlap
label↔candidate docs, length) under the same softmax · r2 learned diagonal/low-rank Mahalanobis
over frozen embeddings — i.e. #173's whitening, but *learned from identification data* instead of
by hand · r3 the label tower.

### Tranche 1 — three free probes (no LLM calls, no new data)

- **7a — softmax-cosine over the frozen lineups** (`lineup_scorer_probe.py`). Score every committed
  lineup unit with `softmax_j(cos(label, centroid_j)/τ)` on the *same* frozen candidate sets, τ
  fitted on one corpus and evaluated on the other (never in-sample). Two tests, and they are
  different questions: **coarse** = does it reproduce the battery's paired gold-beats-variant
  ordering (listener: verbose 79%, ancestor 88%, generic 95%, sibling 89%, distant 99%)?
  **fine** = within-cluster correlation with measured pm across ladder rungs, read against the
  ≈0.75 oracle ceiling above. Raw un-normalized cosine is run alongside as the control that
  isolates what the softmax buys — specifically on the gold-vs-verbose contrast, which is where
  #173's padding bias lives.
- **7b — the abstraction probe on 20NG gold categories** (`abstraction_probe.py`). Closes the
  long-open Phase 0 leg 2 and doubles as the kill criterion for the standalone dual-encoder idea,
  whose premise is that "is a good label for" is asymmetric in a way cosine cannot express (for a
  French recipe, "Italian cuisine" outscores "European cuisine" but is a far worse label).
  `ng_target_names.json` is a naturally-occurring two-level naming hierarchy
  (`rec.sport.hockey` ⊂ `rec.sport` ⊂ `rec`) with per-document assignments in `ng_targets.npy`, so
  the test needs no new data: score a document aggregate against {exact leaf, parent, grandparent,
  sibling leaf, distant leaf} rendered as natural-language phrases (mapping written once, not
  tuned). Headline metric is **not** top-1 but the **asymmetry failure rate**: how often a
  *wrong-but-specific* sibling outranks a *correct-but-general* ancestor. Contenders: raw cosine,
  aggregate-vs-single-document scoring, and Phase 2a's saved `generality_axis.npy` as the Renner
  `Sim + SpecLoss` fix. Relevant prior, and it is negative: **Phase 2b measured that axis at 50%
  = chance** direction accuracy on real Toponymy parent→child label pairs, with length at 86%. The
  dual-encoder's actual bet is that *set-level* supervision manufactures a generality signal that
  word-level geometry lacks — which the Phase 2 linear probe never tested.
- **7c — redraw headroom on both axes** (`redraw_headroom.py`). Prices the "resample exemplars,
  generate N labels, pick the best" idea before building anything. On **fit** this is already
  measured from the two committed independent naming draws (20NG L0, `dose_*` + `.draw1`): 16/74
  identical strings, mean |Δjudge| 0.331, 13/74 differ by ≥1 full point, **oracle best-of-2
  +0.166 judge-pts** — a third of the largest effect in the program (exemplars, −0.49), from an
  intervention that is embarrassingly parallel. On **identification** the same statistic is
  computable because the Phase-6 ablation arms are, for identification purposes, independent
  redraws: FEATURES.md measured them inert (+0.011 / −0.006), so gold + `abl_exemplars` +
  `abl_keyphrases` are up to 3 namings of the same cluster through *identical frozen lineups*.
  **The winner's-curse floor is measured, not assumed:** `wayfinding_*_floors.json` contains the
  same gold label on the same lineup run twice (`run` 1 and 2), so oracle best-of-2 over a pure
  repeat gives the selection-on-noise inflation directly, and only headroom above that floor counts.

### Kill criteria (Phase 7) — written before the probes run

- **7a:** if softmax-cosine clears the oracle ceiling on the *fine* test, the tower is unnecessary
  — report that and stop climbing the ladder. If it fails the *coarse* test (cannot even reproduce
  the battery ordering), the neighbourhood-normalization premise is wrong and the whole cheap-scorer
  branch is dead; do not proceed to r1 hoping features rescue it.
- **7b:** if raw cosine already ranks the correct level first at a high rate, there is no asymmetry
  problem to solve and the dual-encoder's stated premise is withdrawn. If cosine fails *and* both
  cheap fixes (aggregate scoring, SpecLoss) also fail, the premise is confirmed but unresolved by
  anything short of training — that is a real finding and the point at which a tower is justified.
- **7c:** if identification headroom does not clear the measured winner's-curse floor, best-of-N
  over redraws is a fit-axis lever only, and any selector for it must distil the **judge**, not the
  listener. If fit headroom also fails to survive a held-out judge, drop the idea entirely.
- **Global:** no scorer claim is believed on in-sample selection. Anything that survives tranche 1
  goes to a fresh listener run over the *student's own picks* (~700 calls) — the committed
  `wayfinding_*_heldout.json` files are contaminated for this purpose, since "chosen" was selected
  using the very ladder data a student would train on. Benchmark is **always-shortest**, not stock:
  in-sample the controller beats always-shortest by +0.097 (20NG) / +0.062 (arXiv) pm while
  picking the shortest available rung only 41/107 and 68/116 times, so per-cluster selection is
  doing real work — but always-shortest is unbiased and the controller is winner's-cursed, so the
  comparison is only settled on fresh draws (budget agreed: ~1,500 calls including that arm).
- **Generality:** 223 clusters over 2 corpora is enough to falsify a *measurement* but thin for a
  *trained artifact*, which can overfit corpus idiosyncrasy in a way a fixed instrument cannot. A
  third corpus in a different register (support tickets / code / chat, not a third collection of
  long-form English prose) is in scope before any trained rung is claimed to generalize. Deployment
  framing to prefer, since it dodges cross-corpus and cross-embedder transfer entirely: **fit the
  scorer on a small measured budget on *this* corpus** and amortize, rather than shipping a
  checkpoint — testable offline today by fitting on 30% of clusters and predicting the rest.

**Predictions, recorded now so they can be wrong.** 7a passes coarse and fails fine (the fine signal
is the distinguishing specific — "Larson's Reciprocal System Theory" — which cosine structurally
under-weights), which would point at confusability detection and away from rung selection. 7c shows
fit headroom and ~no identification headroom, per FEATURES.md's exemplars-are-identification-inert
result. 7b is the one I have no confident prior on.

**Downstream applications, ranked by how native they are to the library** (pick ONE to carry to a
held-out evaluation after tranche 1): (i) **confusability disambiguation** — the existing renaming
trigger is agglomerative over name embeddings with a 0.2 cosine-distance cap, i.e. symmetric,
pointwise and string-similarity-based; `collision_check.py` found it fires on *nothing* in real name
sets (closest pair 0.305), while Phase 4's sibling arm shows a confusable name relocates **0.51 of
the listener's mass to the wrong region**. That is a measured failure mode the component is
structurally blind to, not merely a suboptimal default — and Phase 4's own downstream note already
proposed "generalizes the disambiguation pass from exact duplicates to confusables". (ii) **rung
selection** (#177's controller, made cheap). (iii) **best-of-N over exemplar redraws** — fit axis,
needs the judge distilled instead; also reframable as a fix for #154, converting the measured 76%
naming churn from a liability into a budget.

## Kill criteria (so we don't fool ourselves)

- **Phase 0:** if grounded judge–human κ stays low, fix the instrument before trusting any
  probe.
- **Phase 1:** if no reference-point variant clears both gates, fidelity-via-similarity is
  dead in raw space → Phase 3 becomes necessary, not optional.
- **Phase 2a:** if nothing beats the cosine floor on HyperLex, there is no usable generality
  axis in our embedder → fall back to structural IC (region size) and say so.
- **Phase 4:** if gold can't clear the shuffled-label floor at the fine layer, the
  listener/grounding is broken — fix the instrument before interpreting anything. If gold sits at
  ceiling even at k=7 nn-hardening, the layer is saturated: wayfinding can't rank good-vs-good
  either — report saturation, don't torture the design until it confesses.
- **Phase 5a:** if the splitter's conjuncts are junk on inspection, fix the splitter before
  interpreting; if every |Δpm| sits inside the repeat band, conjuncts are behaviorally
  indistinguishable — report that and drop conjunct-level reasoning from 5b.
- **Phase 5b:** any claimed gain must survive the held-out lineup (fresh draws + gpt-4o-mini
  listener) AND not regress judge fit. If shortest-within-band collapses to always-longest (band too
  tight) or always-shortest (lineups saturated), the selection rule is broken — fix or report, don't
  tune τ until it flatters.

---

## Reuse (don't rebuild)

- **Harness:** `experiments/nibling_contrast/{judge,ab_harness,selector}.py` + its per-layer
  noise floor.
- **Datasets:** 20NG (in-tree); **HyperLex** (Phase 2a); WordNet (Renner IC).
- **Validation design:** Preiss rubric + Chang intrusion + Krumdick grounding.
- **Scorers:** Renner `Sim + SpecLoss`; whitening (Su 2021) / all-but-the-top (Mu &
  Viswanath 2018).

## Decisions locked

- Substrate: **20NG now**, defer second.
- Calibration: **small human seed (~25–30)**.
- Home: branch **`experiment/label-quality`** off `experiment/nibling-contrast`; this file
  at `experiments/label_quality/PLAN.md`.

## Key references

- Preiss et al. 2024, *Evaluation of Text Cluster Naming with Generative LLMs*, J. Data
  Science. (rubric + grounded human eval)
- Renner, Denis & Gilleron 2023, *WordNet Is All You Need: …Graded Lexical Entailment*,
  Findings of EMNLP. (`Sim + SpecLoss`, HyperLex targets)
- Krumdick et al. 2025, *No Free Labels: Limitations of LLM-as-a-Judge Without Human
  Grounding*, arXiv:2503.05061. (judge grounding recipe + κ lift)
- Hoyle et al. 2021, *Is Automated Topic Model Evaluation Broken?*, NeurIPS. (validate
  against humans + intrusion)
- Chang et al. 2009, *Reading Tea Leaves*, NeurIPS. (intrusion task)
- Vulić et al. 2017, *HyperLex*, Computational Linguistics. (graded-entailment dataset)
- Liang et al. 2022, *Mind the Gap*, NeurIPS; Schrodi et al. 2024 (modality gap = low-rank
  offset); Huang et al. 2024, *LANCER*, EMNLP (text-space erasure-to-align).
- Su et al. 2021 (whitening); Mu & Viswanath 2018 (all-but-the-top); Ethayarajh 2019
  (anisotropy); Gao & Metaxas 2026 (dispersion, not length, drives degradation).
- Krahmer & van Deemter 2012, *Computational Generation of Referring Expressions: A Survey*,
  Computational Linguistics. (Phase 4: the lineup = REG comprehension used as a metric)
- Liu et al. 2018, *Show, Tell and Discriminate*, ECCV; Ou, Krojer & Fried 2023, *Pragmatic
  Inference with a CLIP Listener for Contrastive Captioning*, ACL Findings. (Phase 4:
  self-retrieval / frozen-listener lineage)
- Gao, Schulman & Hilton 2023, *Scaling Laws for Reward Model Overoptimization*, ICML. (Phase 4:
  Goodhart guard if the lineup ever becomes a selector)

## Embedder + dataset robustness study (2×2 COMPLETE — in WRITEUP)

Two near-posts were caught as artifacts (under-generalization; whitening/blindness), so before posting
we checked embedder + dataset robustness. Design (after a home/away confound was caught: scoring labels
against a *different* embedder's clusters is a mismatch): each embedder plays AT HOME — cluster + name +
score with its OWN geometry (`home_pipeline.py` + `async_judge.py`, ~20× faster). The 2×2 =
weak (MiniLM) vs strong (Cohere embed-v4) × 2 datasets (20NG + arXiv @7k), trimmed to **gate-b +
exemplars/keyphrases ablation** (fine-disc + secondary findings cited as robust negatives, not re-run).
Synthesis: `cross_cell.py` → `data/cross_cell_summary.json`.

**The earlier confounded quick-check (`embedder_robustness.py`, cohere labels on MiniLM clusters) was
MISLEADING — the at-home 2×2 supersedes it.** Note especially: the at-home 20ng/minilm corner
**replicates the canonical numbers** (fresh UMAP → [72,22,8] clusters; verbose-intrusion 79%→94% vs
canonical 78→92; exemplars judge-Δ +0.50 vs +0.49), which validates the at-home recipe and makes the
other three cells trustworthy.

### 2×2 results (`cross_cell.py`)

| cell | gate-b ρ cen / wht | verbose-intrusion cen→wht | exemplars jΔ | keyphrases jΔ |
|---|---|---|---|---|
| 20ng/minilm | 0.81 / 0.75 | 79% → 94% (+15pp) | +0.50 | −0.09 |
| 20ng/cohere | 0.80 / 0.76 | 68% → 88% (+20pp) | +0.40 | −0.10 |
| arxiv/minilm| 0.82 / 0.78 | 53% → 84% (+31pp) | +0.48 | −0.07 |
| arxiv/cohere| 0.84 / 0.79 | **99% → 92% (−7pp)** | +0.39 | −0.00 |

**Robust across all 4 cells (strengthen WRITEUP):**
1. **Coarse guardrail** — gate-b ρ 0.80–0.84 (centroid), and **raw centroid > whitened on graded
   correlation in EVERY cell** (~0.04 gap, the canonical pattern). Generalizes across embedder×dataset.
2. **exemplars ≫ keyphrases** — exemplars worth ~+0.4 judge-pts everywhere; keyphrases ~0 or slightly
   negative everywhere. The shakiest "bonus" finding (1 draw) is now the **most** robust thing here.
   Promote from hedged aside to a real finding.

**Nuanced / reframe needed (don't just patch):**
3. **Whitening's verbose fix is an (embedder × corpus) thing, NOT a "weak-embedder patch."** Helps in
   3/4 cells (incl. 20ng/cohere, a *strong* embedder — so not embedder-only); only arxiv/cohere has raw
   centroid already solving verbose (99%), where whitening costs 7pp. Honest rule: **whiten as cheap
   insurance for the verbose/padding blind spot** (gains up to +31pp where the space is anisotropic;
   costs ≤7pp + ~0.05 gate-b where it isn't). It's a genuine trade-off: raw centroid = better graded
   ranker; whitened = padding-robust.
4. **Ablation guard — split by reference point (the canonical "blind" was the *whitened* metric).**
   **Raw centroid weakly but consistently tracks the LARGE exemplars regression** (sign-agree 62/73/68/72%
   = above chance in all 4; ρ +0.15…+0.47) yet is **blind to the ~0 keyphrases change** (correct — nothing
   to see). Whitened is blind/erratic on both (exemplars 54/69/32/45%). So: raw centroid wins again; the
   metric catches gross regressions weakly, subtle ones not at all. Still "not a precise regression guard."

**Net reframe for the WRITEUP:** raw centroid is the better default on *every* axis except verbose
robustness; whitening is a targeted padding patch, not a blanket "whiten first." Two desiderata (graded
correlation / ablation sensitivity vs verbose robustness) trade off.

arXiv judge-calibration spot-check DONE (20 blinded items, Steven rated 2026-07-03): **Spearman
ρ = 0.79, quadratic κ = 0.54** vs 20NG's 0.82/0.64 — same rank agreement, same ~1-pt judge-harsher
offset (stable judge property; costs κ, not ranking), same per-type ordering (gold > verbose >
ancestor > generic ≈ sibling). Caveat: Steven is less at home in these arXiv subfields, so some gaps
(a sibling rated human-3/judge-0) may be human error. Judge is trusted on arXiv; WRITEUP κ slot filled.

SHIPPED 2026-07-03: teaser posted to Discussion #173
(https://github.com/TutteInstitute/toponymy/discussions/173#discussioncomment-17526704);
`experiment/label-quality` frozen at the posted tip. Phase 4 proceeds on
`experiment/wayfinding-lineup`.

## Phase 4 results (gates a–c run 2026-07-03; sonnet listener, structured output)

**Instrument checks PASS** (both kill criteria clear). Same-listener sonnet floors: gold pm 0.545
vs shuffled 0.038 vs gimme ceiling 0.936 (k=5, chance 0.20). Haiku k-sweep: gold pm 0.62/0.50/0.43
at k=3/5/7 — above chance, below ceiling, no saturation. Repeat bands: haiku p90 |Δpm| 0.185;
sonnet p90 |Δdelta| 0.107 (measured on the pair lineups themselves, n=30 re-run).
*Protocol note:* sonnet deliberates in prose and truncates before emitting scores (silent
kind-correlated parse failures — verbose 1/107 valid!); claude-sonnet-4-6 rejects prefill; fix =
litellm `response_format` json_schema. First battery run (~$20) discarded as
`*.proto1.bak.json`; all v2 stages 100% parse-valid. Top-1 inflates on garbage labels
(all-zero ties count as rank 1) — prob-mass is the primary metric, as planned.

**Gate (a) PASS — every predicted signature confirmed** (n=107 clusters, paired on identical
lineups): gold pm 0.545; verbose 0.444; ancestor 0.262; generic 0.195 (entropy 0.97 ≈ uniform ✓);
sibling 0.154 (below chance ✓, and where the sibling cluster is in the lineup its candidate takes
0.513 vs true 0.213 — the mass-shift signature, sharper than the judge's soft 0.62); distant 0.004
(outright rejection). Paired gold-beats-variant: shuffled 100%, distant 99%, generic 95%,
sibling 89%, ancestor 88%, **verbose 79% (weakest — the predicted divergence CONFIRMED: the judge
punishes padding hard (1.89 vs gold 2.42), the lineup barely does; padding doesn't hurt
identification ⇒ fit and identifiability are different axes ⇒ combined instrument = judge ∧ lineup)**.

**Gate (b) PASS — the interesting zone**: Spearman(lineup pm, judge overall) = **+0.647** (n=487):
correlated with fit, not redundant with it (embedding metric's own gate-b vs judge is 0.80–0.84).

**Gate (c) — the payoff, mostly outcome (ii) with an (iii) tail:**
1. Judge-TIED pairs (n=50): mean |Δpm| 0.073 < band 0.107 — the lineup ties on the bulk too.
   It decides **11/50 (22%)** vs ~10% expected under the null band ⇒ a real but thin tail of
   genuinely-more-identifying labels. The decided list is face-valid: the lineup consistently
   prefers the distinguishing specific ("Larson's Reciprocal System Theory" d=0.44 over "Debates
   on Unified Theories in Physics") — rewarding exactly the qualifier centroid-cosine penalized.
2. Judge-DECIDED pairs (n=54): sign agreement **37%**, Spearman(Δlineup, Δjudge) = −0.21 — where
   the two instruments decide, they disagree ⇒ two axes again, now on realistic pairs.
3. No base/alt bias (mean pm 0.537 vs 0.531, Wilcoxon p=0.81). Fine-layer gradient: mean tied
   |Δ| 0.088 (L0) / 0.044 (L1) / 0.014 (L2) — identifiability differences live at the fine layer.
4. Human seed DONE (Steven, 2026-07-03; 20 blinded A/B items = 11 band-clearing + 9 sub-band
   controls): **when the human decided, agreement with the lineup was 10/12 overall and 5/5 on
   band-clearing items** (coin-flip null p≈0.03). Human ties concentrated on the LARGEST deltas
   (4 of top 5, incl. Larson d=0.44) — explained by a form asymmetry Steven independently
   flagged: the form showed only the true cluster's docs (fit-style presentation), while the
   lineup's conviction comes from contrast with neighbors it can see and the human couldn't.
   So c-3 validates the lineup's picks where the human could see the difference, and under-tests
   the discrimination component; the airtight version would make the human BE the listener
   (label + k candidate doc-groups, pick the group). Controls tied *less* than tests (22% vs
   55%, n tiny) — the p90 band is not validated by human tie-rates, though band-clearing picks
   were the more reliable ones (5/5 vs 5/7).

**Reading:** the wayfinding lineup is a valid, discriminating instrument (a+b), and it settles the
fine-discrimination question rather than winning it: most judge-tied good-vs-good pairs are
functionally equivalent on the identification axis too (outcome ii), a thin tail is genuinely more
identifying (outcome iii tail), and fit ≠ identifiability throughout (verbose divergence; −0.21 on
decided pairs). Venue call (new thread vs #173 append) after the human seed.

## Phase 5 results (run 2026-07-04, 20NG canonical cell)

**5a — conjunct ablation** (94 compound labels, 225 conjuncts; band p90 0.095): **87% of
conjuncts are free-riders** (mean marginal identification value +0.004 ≈ 0); **12% are
anti-conjuncts** (dropping HELPS — they pull mass toward a neighbour that owns that content: the
sub-label version of the Canada/US point); 13% load-bearing, and those are the distinguishing
specifics ("Larson's Reciprocal System Theory" +0.45 — independently reconfirms the Phase-4
fine-pair finding). **96% of compound labels have one conjunct within band of the full label**
(mean −40% length). No position effect (padding is semantic, not structural).

**5b — length controller** (7-rung ladder × 107 clusters, stock Toponymy machinery, shared
clusterer; 754 distinct names through frozen lineups; band p90 0.047):
- **The rate–distortion curve is nearly flat**: L0 mean pm 0.50→0.47 from 10-word to 2-word
  names; L1 0.63→0.59; L2 0.82→0.80. Words 3–10 buy ~no identification on average.
- **Controller vs stock (in-sample)**: differs on 91/107 clusters; L0 −5.7 words (70/74 differ),
  L1 −1.0, **L2 +1.4 (moves BOTH directions — "simple (1 or 2 word)" is sometimes too little)**.
  Selection did not collapse (L0 chosen rungs spread 0→6). In-sample +pm not quoted (winner's
  curse by construction).
- **Held-out Goodhart check (fresh doc draws + gpt-4o-mini listener, n=91 differing clusters):
  chosen pm 0.516 vs stock 0.524 (Δ −0.008, Wilcoxon p=1.0) at 3.8 vs 8.5 words; grounded judge
  fit 2.79 vs 2.75 (p=0.57).** The claim that survives: **same identification, same fit, 55%
  shorter labels.** detail_levels' open-loop word-count dial spends rate that buys no distortion.
- Deployment note for any library conversation: the full controller costs ~7 namings + ~7
  lineups per cluster; the flat curve also supports a cheap static fix (terser SUMMARY_KINDS
  defaults at fine layers) that captures much of the win, with the measurement reserved for the
  per-cluster corrections (anti-conjuncts; coarse clusters needing more words).

### arXiv replication COMPLETE (arxiv_home = the at-home 2×2 cell, 2026-07-04) — story confirmed

Everything replicates on the aligned at-home arXiv substrate ([82, 27, 7] clusters, 116 battery
clusters, judge-calibrated cell):
- **Floors:** gold 0.489–0.545, shuffled 0.121–0.182, gimme 0.68–0.75 (top1 98–100%) — the gimme
  ceiling returning (vs 0.217 misaligned) is the positive control for issue #176. No saturation.
- **Battery (gate a):** gold 0.574 / verbose 0.476 / ancestor 0.270 / generic 0.182 (H 0.98) /
  sibling 0.163 (mass→sibling 0.592 vs true 0.161) / distant 0.062 / shuffled 0.121 / gimme
  0.814. Paired gold-beats: 83–100%, **verbose again weakest (83%) — the fit≠identifiability
  divergence is now a two-corpus claim.** Gate (b) ρ = +0.741 (n=637).
- **Controller:** rate–distortion flat again (L0 0.58→0.55 over 10→2 words), and L2 sharpens
  "wrong in both directions": **the stock rung (simple 1–2 word) is the WORST rung at the coarse
  layer** (0.48 vs 0.66 at 4 words); controller lengthens 4/7 L2 labels. **Held-out Goodhart
  check (fresh docs + gpt-4o-mini, n=111 differing): chosen pm 0.448 vs stock 0.436 (p=0.99) at
  3.2 vs 8.8 words (−64%); judge fit 2.49 vs 2.45 (p=0.68).** Same identification, same fit,
  64% shorter (20NG: 55%).

**SHIPPED 2026-07-04 (Steven's explicit go):** posted as
https://github.com/TutteInstitute/toponymy/discussions/177 ("A wayfinding eval for topic labels:
identification is not fit (+ measuring the detail_levels dial)"), linking `WAYFINDING.md` at this
branch's tip — **branch frozen at the posted state; future work goes on a child branch.**
Post body preserved in `post_wayfinding_draft.md`. 5a-on-arXiv remains optional/unrun.

### ⚠ 2026-07-04: the examples arXiv substrate is MISALIGNED (kill criterion fired, correctly)

First arXiv attempt used the `ab_harness` loader (examples/ai_arxiv_{vectors,coordinates,papers}
paired row-wise, @7000). The haiku floors FAILED the gimme trial (gold 0.219 ≈ shuffled 0.196 ≈
**gimme 0.217** ≈ chance 0.20 — on 20NG gimme is 0.87–0.93): the listener can't find the true
cluster even with distant distractors, i.e. broken grounding, not bad labels. Diagnosis:
**`examples/ai_arxiv_papers.zip` row order ≠ `ai_arxiv_vectors.npy` row order** (10,000 rows
each, which invites row-pairing). Evidence: cosine(stored emb[j], mpnet(csv row j)) ≈ 0.12–0.34;
model-agnostic check: stored-space nearest neighbours have near-random text similarity (0.27 vs
0.15 random baseline) and sit at ADJACENT indices with unrelated texts (8496↔8497). Cluster
labels on this substrate are the tell: every L0 name is "Diverse Machine Learning Applications
Spanning …" (vs the at-home cell's specific names on the SAME 7000 docs, re-embedded).

**Blast radius:** (a) this chain's labels/battery/floors → quarantined as
`*.misaligned.bak.json`; (b) **the nibling-contrast arXiv arm (POSTED in #173) used this
loader** — both A/B arms equally scrambled so the comparison was internally consistent, but the
arXiv-arm numbers (e.g. "clear loss ~23% on the flatter arXiv set") were measured on
incoherent doc-groups; the headline negative still rests on clean 20NG + Leland's independent
confirmation; (c) **UNAFFECTED:** all 20NG work, the at-home 2×2 (re-embedded CSV text directly),
the arXiv judge calibration, yesterday's #173 comment. Files are TRACKED UPSTREAM
(TutteInstitute examples/) — possible upstream issue, Steven's call.

**Instrument note for the writeup:** the grounded fit judge ran on this scrambled substrate
without complaint (vague labels fit vague doc-sets); the gimme floor caught it in one run.
Identifiability testing doubles as substrate-integrity testing — fit testing does not.

**Reported (2026-07-04, Steven's go):** upstream issue
https://github.com/TutteInstitute/toponymy/issues/176 (body: `issue_arxiv_alignment.md`);
correction comment on the original post:
https://github.com/TutteInstitute/toponymy/discussions/173#discussioncomment-17532133
(body: `post_173_correction.md`).

**Path forward:** replicate on the **at-home arXiv substrate** (judge-calibrated, aligned by
construction: `home_arxiv_minilm_{emb,coords}.npy` + deterministic clusterer replay à la
`make_calibration_home.build_docs_for`, gold labels in `home_arxiv_minilm.json`) — implemented
as `Cell("arxiv_home")` (HOME_TAGS alias) + `make_home_battery.py` → `battery_arxiv_home.json`.

## Phase 6 results — what are the three naming-prompt features worth? (2026-08-02/03)

Branch `experiment/naming-features` (off the frozen `a49257c`). **Full write-up = `FEATURES.md`**;
this section is the log entry. Triggered by jc-healy's reply on #173 (2026-07-30) saying the
ablation made him lean toward defaulting keyphrase extraction OFF: the ablation behind that was one
corpus, one instrument, one draw, and the lineup did not exist when it ran.

Design: all three features × three axes (grounded judge / wayfinding lineup / the disambiguation
pass's own workload) × two corpora (20NG canonical, arxiv_home). **Headline: only exemplars
measurably contribute, and only to fit, not to findability** — the sharpest form of the Phase-4
fit≠identification split, and it replicates.

| feature | fit | identification | disambiguation load |
|---|---|---|---|
| exemplars | **−0.49 at k=0**, both corpora | inert (+0.011 / −0.006) | none |
| keyphrases | none (−0.17 / +0.16 by corpus) | ≤ +0.016 (negligible) | **3× on 20NG**, none on arXiv |
| subtopics | none (−0.132 pooled, p=0.081) | inert (tight null) | ~none |

- **Leakage control (`clean_docs_rejudge.py`) — the fit instrument had a real confound, and it did
  not bite.** `judge_fair.sample_docs` does NOT exclude the namer's exemplars, and the overlap grows
  with the swept variable (7/11/16/24/38/67% of the judge's 15 docs at k=1/2/4/8/16/32) — so the
  posted #173 headline had an alternative reading, since the k=0 arm has 0% overlap and stock has
  24%. Re-judging on documents excluding every rung's exemplars (13.8% pool shrinkage, all 107
  clusters keep 15 docs): effect moves **+0.008, p=0.93 → 2% was leakage, the claim STANDS**. Every
  condition drops ~0.07 absolute (clean docs are harder) but every *contrast* survives.
- **Dose-response, and a withdrawal.** Downward sweep (selection is nested, verified 74/74): 20NG
  has a clean knee at k=4 (+0.015, p=0.98) holding at every layer; **arXiv does not** (k=4 = −0.187,
  p=0.002, confirmed across two independent draws that differ by +0.037, p=0.37). A "halve
  `n_exemplars`" recommendation would have been wrong on the second corpus. **Withdrawn.**
- **Upward sweep (`upward_sweep.py`) — the curve does not turn over.** k=16 +0.131 (p=0.035), k=32
  +0.063 (n.s., and *deliberately advantaged* since full control would starve 39 clusters). So the
  only cost of over-provisioning is tokens, and the useful statement is a **floor, not a target**:
  don't go below 8, above is safe, no per-corpus tuning warranted. (This arm exists because Steven
  pointed out that a token-only downside makes "measure to save tokens" a weak recommendation.)
- **Keyphrase × exemplar interaction — the second withdrawal.** 20NG paired 2×2 contrast
  `D = both − k4 − kp_off + stock` = **−0.269** (SE 0.095, p=0.008): each change free alone,
  cancelling together. arXiv: **+0.019** (p=0.99). The 20NG effect was driven by keyphrases-off
  *helping* there (+0.162, p=0.012), which itself doesn't replicate (arXiv −0.009, p=0.82).
  **Withdrawn**, and a caution against generalising any combinatorial result from one corpus.
- **Disambiguation load (`disambiguation_load.py`, 4 conditions × 2 corpora × 3 draws,
  instrumented during `fit()`).** Final name sets are clean for every condition, so the absorbed
  pressure is only visible during the run. 20NG L0 renaming groups: stock 1,1,1 → **keyphrases
  ablated 3,4,3** (non-overlapping); exemplars 0,1,2; subtopics 1.3. arXiv produces **zero load for
  every condition**. Reading: keyphrases buy *distinctiveness* (corpus-contrastive vocabulary is
  what separates neighbours) rather than quality → defaulting them off is **quality-neutral but
  cost-ambiguous**, not a free win.
- **Subtopics, properly scoped (`subtopics_value.py`).** The −0.03 in WRITEUP.md averages over all
  layers, but layer 0 has no children: 73 of 106 rows are structurally null, and not quiet ones (54
  of 73 layer-0 labels still "changed" = temperature-0.4 redraw noise). Scoped to L≥1: 20NG n=33,
  −0.131, CI [−0.369, +0.106] — an absence of measurement, not a null. Measured properly (both
  corpora, both axes, single-child `[!SKIP!]` split out): identification-inert with a tight
  interval, nothing positive on fit. **Subtopics also manufacture cross-layer duplicates** — 12/12
  single-child parents inherit the child's name, 0/12 without, and the per-layer trigger never sees
  them. Power ceiling is structural (~67 coarse clusters pooled; resolves ~0.3, not ~0.1).
- **Two stock-pipeline observations.** (1) Naming runs at **temperature 0.4** by default, not 0:
  two identical runs share only 16–19 of 74 layer-0 labels on 20NG (~76% churn), 52% on arXiv. Read
  aggregates only; bears on #154. *Aggregate* quantities are much stabler (group count 1,1,1 across
  three stock runs). (2) 1 of 24 runs hit `All retries exhausted for generate_topic_cluster_names:
  IndexError ... Returning old names.` — this is #57's root cause, still live; arcrystal's
  `except IndexError: continue` did land in `cluster_layer._update_topic_names` and is **silent**,
  while the `llm_wrappers` tenacity callback at least warns.

**#176 does not touch any of this** (verified, not assumed): the arXiv-home cell re-embeds document
text, and re-encoding sampled documents against the cached matrix gives cosine = 1.000 on every
probe, both corpora. All arXiv judging uses home-geometry doc sampling (`home_docs`), never
`judge_fair.sample_docs`, which would replay the canonical 20NG fit.

**SHIPPED (2026-08-02, Steven's explicit go):** substance →
[#173 comment](https://github.com/TutteInstitute/toponymy/discussions/173#discussioncomment-17874315);
pointer → [#177 comment](https://github.com/TutteInstitute/toponymy/discussions/177#discussioncomment-17874317);
data point → [#57 comment](https://github.com/TutteInstitute/toponymy/issues/57#issuecomment-5161568456).
Branch pushed and FROZEN at the posted tip. Post drafts deliberately left untracked.

## Phase 7 results — tranche 1, the three free probes (2026-08-03)

Branch `experiment/lineup-scorer` (off the frozen `010f556`). No LLM calls, no new data. The
predictions recorded in the plan above came out **half right**: 7a split exactly as predicted,
7c did not.

**Headline: a free, LLM-free scorer detects confusability well and abstraction level not at all.**
That maps onto the three candidate applications with unusual sharpness — it supports (i)
confusability disambiguation, does not support (ii) rung selection, and (iii) best-of-N over
redraws is a null on both axes before it was ever built.

### 7a — softmax-cosine over the frozen lineups (`lineup_scorer_probe.py`)

All 6,272 committed units re-scored with `softmax_j(cos(label, cand_j)/τ)` on the same frozen
candidate sets; τ fitted by KL on one corpus and applied to the **other** (0.170 both ways, so the
parameter is not corpus-specific). Two candidate representations: cluster centroid, and the mean of
the 5 held-out documents the listener actually saw.

**COARSE — PASSES, and it repairs #173's padding blind spot** (paired gold-beats-variant, centroid
representation; "raw" = pointwise cosine to the true centroid, i.e. the #173 metric):

| | verbose | ancestor | sibling | generic | distant | shuffled |
|---|---|---|---|---|---|---|
| 20NG softmax | **93%** | 95% | 99% | 98% | 100% | 98% |
| 20NG raw cosine | 78% | 89% | 96% | 100% | 100% | 100% |
| arXiv softmax | **91%** | 91% | 97% | 97% | 99% | 99% |
| arXiv raw cosine | **51%** | 91% | 95% | 99% | 99% | 97% |
| *(listener, for reference)* | *79% / 83%* | *88% / 92%* | *89% / 95%* | *95% / 98%* | *99% / 100%* | *100% / 99%* |

The arXiv verbose column is the result: **raw pointwise cosine is at chance (51%) on gold-vs-verbose,
and normalizing over the neighbourhood takes it to 91%** — the predicted common-mode cancellation,
confirmed. Padding raises cosine to all five candidates alike, so a softmax removes it for free. Note
softmax also *exceeds* the listener on this contrast (93% vs 79%); that does not make it a better
instrument, it means gold-vs-verbose is easy geometrically once normalized and hard for an LLM reader.

**FINE — FAILS, as predicted.** Within-cluster correlation with measured pm across ladder rungs:

| | softmax Pearson | raw cosine | oracle ceiling |
|---|---|---|---|
| 20NG (n=107 clusters) | +0.238 (docs rep +0.261) | +0.075 | ~0.75 |
| arXiv (n=116) | +0.158 (docs rep +0.233) | +0.052 | ~0.79 |

Normalization roughly **triples** raw cosine, and still recovers only ~10–12% of the explainable
variance. Not enough for rung selection. The doc-mean representation is slightly better on the fine
test and slightly worse on the coarse one.

**DIRECTION — an unplanned positive, and the strongest signal in the tranche.** Restricted to units
where the listener put more mass on a *wrong* candidate than on the true one, does the model point at
**the same wrong candidate**? **20NG 277/399 = 69%, arXiv 239/357 = 67%, against ~25% chance.** This
is Phase 5a's anti-conjunct mechanism confirmed geometrically — a label that overlaps a neighbour's
territory is detectably nearer that neighbour — and it is exactly the signal confusability
disambiguation needs.

### 7b — the abstraction probe on 20NG gold categories (`abstraction_probe.py`)

Closes **Phase 0 leg 2**. 260 trials (13 leaf categories × 20 aggregates × 15 docs); candidates =
exact leaf / parent / grandparent / sibling leaf / distant leaf, rendered to natural language by a
mapping written once from the group names and not tuned.

| contender | exact leaf ranked 1st | **sibling ABOVE ancestor** |
|---|---|---|
| raw cosine to aggregate | 97.3% | **50.4%** |
| cosine to individual docs (mean) | 97.3% | 50.4% |
| cosine − 0.02·label length | 96.5% | 48.5% |
| cosine − λ·generality (Phase 2a axis) | — | 53.3% (**λ fitted to 0.00** — the correction was rejected) |

**The premise is confirmed and unfixed by anything cheap.** Cosine finds the right label easily when
it is present (97%), but has no notion of *correct-but-general* versus *wrong-but-specific*: half the
time a sibling outranks a true ancestor. Softmax over the candidate set cannot help here — within a
single trial it is order-preserving, so it changes no ranking (reported explicitly rather than
silently omitted; this is the precise limit of 7a's trick). Phase 2a's generality axis is rejected by
its own fit, replicating Phase 2b's chance-level negative in a third setting.

The failure is well-localised and tracks how semantically empty the ancestor term is: 100% for
Automobiles and Motorcycles (ancestor "Recreation"), 100% for Macintosh Hardware (ancestor "Computer
Systems"), 0% for Baseball and Medicine (ancestors "Sports", "Science"). *Caveat:* the rate is
sensitive to that rendering — "Recreation" is a genuinely weak natural-language name for `rec.*` —
so the mechanism is solid and the 50.4% number is not a corpus constant.

**7a and 7b are the same negative.** Choosing among ladder rungs *is* choosing an abstraction level,
so 7b explains why 7a's fine test fails: cosine has no principled handle on specificity, and that is
precisely what within-cluster rung ordering requires.

### 7c — redraw headroom (`redraw_headroom.py`) — NULL on both axes

**Identification** (gold + the two Phase-6 ablation arms as independent redraws through identical
frozen lineups; FEATURES.md measured those arms inert, which is what licenses the reuse):

| | best-of-2 headroom | best-of-3 | floor (a) real repeat, haiku | floor (b) analytic, sonnet |
|---|---|---|---|---|
| 20NG | +0.046 | +0.067 | +0.038 | +0.016 / +0.024 |
| arXiv | +0.039 | +0.057 | +0.028 | +0.015 / +0.022 |

Floor (a) is a genuine repeat but on the noisier haiku listener (overstates); floor (b) counts only
within-unit sample scatter (understates). The true sonnet floor lies between, and the headroom sits
at the top of that range rather than above it. Cross-check: haiku-vs-gpt-4o-mini namings of the same
cluster (Phase 4's 104 fine pairs) give **+0.041**, i.e. a genuine cross-model redraw buys no more
than a same-model one. **No demonstrated identification headroom.**

**Fit — and this is a correction to a number quoted mid-discussion before its floor was computed.**
The two independent naming draws give oracle best-of-2 **+0.166 judge-pts** (20NG L0, k=4; 16/74
identical strings, mean |Δ| 0.331, 13/74 a full point apart). The winner's-curse floor, from the one
pure-ish judge repeat on disk (`clean_docs_rejudge` scored the same gold labels again on a fresh
document draw): sd of paired difference 0.494 ⇒ σ ≈ 0.350 ⇒ **floor +0.197 judge-pts**. The headroom
**does not clear its floor**. The apparent "a third of the exemplars effect" was selection on judge
noise.

That floor is an upper bound (it contains doc-sample variance, and the clean-doc draw is genuinely
harder — a +0.061 systematic shift), so the fit side is *provisionally* null rather than settled. The
one measurement that would settle it is cheap: **re-judge ~40 already-scored labels on the same
documents (~120 calls) for a tight judge repeat band.**

### What tranche 1 decided

- **Rung selection via a cheap scorer: not supported.** 7a-fine at +0.24 against a ~0.75 ceiling,
  and 7b says why. Do not climb to r1/r2/r3 hoping features rescue the fine test — the kill criterion
  for that was written in advance and it did not fire in the scorer's favour.
- **Confusability disambiguation: supported, and it is the application to carry forward.** 7a-coarse
  at 91–99% and 7a-direction at 67–69% vs 25% chance, both replicated on two corpora, both free. It
  targets a component that `collision_check.py` showed fires on *nothing* in real name sets (closest
  pair 0.305 vs a 0.2 cap) while Phase 4 showed a confusable name relocates 0.51 of the listener's
  mass — a measured failure the trigger is structurally blind to.
- **Best-of-N over redraws: dead on identification, provisionally dead on fit.** Killed for ~$0.
- **The dual-encoder premise: confirmed as a real gap** (50% asymmetry failure, both cheap fixes
  rejected) **but Toponymy-facing value is unclear** — a tower would be justified by 7b, yet the
  application that needs abstraction ordering (rung selection) is the one 7a says is out of reach,
  and 223 clusters over 2 corpora is thin for a trained artifact.

## Phase 7 results — tranche 2: confusability (2026-08-03, 780 LLM calls)

**Headline: Toponymy's disambiguation pass is blind to the confusability it exists to prevent, and
a zero-cost geometric score finds it.** Validated on a fresh listener with fresh documents, two
corpora, with the protocol's own floor and ceiling run before anything was interpreted.

### The score, validated offline (`confusability.py --stage score`)

`C(i→j) = softmax_j(cos(label_i, cent_j)/τ)` over cluster i's k=5 neighbourhood, τ = 0.170 carried
over from 7a where it was fitted on the *other* corpus. Nothing is refitted here — no free
parameters were tuned on the data being reported.

| | 20NG (n=107) | arXiv (n=116) |
|---|---|---|
| predicted vs measured leaked mass (Pearson) | +0.534 | +0.661 |
| top-confuser agreement where the listener drifted | **16/20 = 80%** | **5/7 = 71%** |
| top-confuser agreement over all gold labels | 60% | 55% |
| **AUC, detecting measurably-confusable labels** (measured pm < 0.40) | **0.803** | **0.920** |

**What Toponymy's own trigger sees on the same name sets:** 20NG — 1 group covering 2 topics at
L0, nothing at L1/L2; arXiv — **nothing at any layer**. Of the 40 most confusable labels by score,
the trigger catches **2 (20NG) and 0 (arXiv)**.

The examples make the mechanism plain. arXiv L1: `Uncertainty Quantification and Anomaly Detection`
(c18, measured pm 0.20) and `Time Series Forecasting and Anomaly Detection` (c19, measured pm 0.09)
are adjacent clusters sharing a conjunct, and the listener essentially cannot tell them apart —
0.2-cosine-cap sees no problem. 20NG L1: `MLB Season Predictions and Player Analysis` (c2, pm 0.48)
against `Major League Baseball Player Statistics, Career Comparisons…` (c3, pm 0.46). These are not
duplicate strings; they are duplicate *referents*.

### The arbiter: fresh docs, held-out listener, k=2 pairwise (`--stage arbiter`)

660 calls on gpt-4o-mini (held out from the haiku namer and the sonnet listener), documents redrawn
under a fresh seed, chance = 0.50. Flagged = the 40 highest-scoring pairs; control = the 40
lowest-scoring; plus a ceiling and a positive control run **before** interpreting the arms:

| arm | 20NG | arXiv |
|---|---|---|
| gimme (gold vs far cluster) — *ceiling* | 0.949 | 0.887 |
| control (low score) | 0.767 | 0.746 |
| **FLAGGED (high score)** | **0.610** | **0.634** |
| sibling's label — *positive control* | 0.385 (10/15 below chance) | 0.347 (13/15 below chance) |
| flagged < control | **p = 2.2e-05**, gap 0.157 | **p = 1.6e-05**, gap 0.112 |

Instrument checks pass (gimme near ceiling, sibling below chance), so the arms are interpretable.
**11/40 (20NG) and 7/40 (arXiv) flagged pairs land BELOW CHANCE** — shown the label and the two
regions, a fresh reader picks the wrong one more often than a coin flip. Controls: 6/40 and 4/40.
Honest note: the control arm is not clean either, so the score ranks confusability rather than
partitioning it.

### The judge repeat (`judge_repeat.py`, 120 calls) — and a correction to the correction

The pure repeat 7c lacked: same 39 gold labels, **same documents** (`judge_fair.sample_docs` is
deterministic, seed 1000+i), same rubric/model/k. Result: 32/39 identical scores, mean |Δ| 0.090,
sd of paired difference 0.237 ⇒ **σ = 0.168 ⇒ best-of-2 floor +0.095**, far below the +0.197 that
`clean_docs_rejudge` implied — because that comparison changed the documents, and doc-sample
variance dominates judge noise.

Which floor applies is decidable, not a judgement call: `exemplar_dose_response` judges via
`sample_docs(ds, L, i)`, so **both naming draws were scored on identical documents**. The selection
noise is therefore pure judge noise. Decomposition:

| | judge-pts |
|---|---|
| observed oracle best-of-2 | +0.166 |
| selection-on-noise (σ/√π) | −0.095 |
| **true oracle gain** | **≈ +0.071** |

So 7c's fit side is **not** a null: real label-to-label quality variation exists across redraws.
It is also **small** — ~15% of the exemplars effect — and it is an *oracle* gain in which the
grounded judge does the selecting. Capturing it would need a selector approaching judge quality,
which is the expensive thing best-of-N was meant to avoid. Best-of-N is therefore unattractive on
engineering grounds rather than dead on measurement grounds, which is a different and more accurate
statement than either of the two made earlier in this phase.

### What tranche 2 establishes

1. **A measured failure mode Toponymy is structurally blind to.** The renaming trigger trips on
   near-duplicate *strings*; confusability is about *referents*, and on real name sets the trigger
   finds 0–2 of the 40 worst cases per corpus.
2. **A zero-cost detector for it**, AUC 0.80/0.92 offline, confirmed on a held-out listener with
   fresh documents at p ≈ 2e-05 on both corpora, with no parameter fitted on the reported data.
3. **The natural library shape:** widen the disambiguation trigger from name-embedding similarity
   to name-vs-neighbour-centroid confusability. Both quantities are already computed during `fit()`
   — the centroids exist, the name embeddings exist, the pass already runs per layer. No LLM cost,
   no new dependency.

**Open before any PR:** whether *renaming* a flagged pair actually fixes it (tranche 2 measures
detection, not repair) — the obvious tranche 3, and the point at which the disambiguation prompt
itself would need contrast context. Also untested on a third corpus, and the τ, though carried
across corpora, has only been asked to generalize once in each direction.

## Phase 7 results — tranche 3: does renaming repair it? (2026-08-03, 570 LLM calls)

**Headline: no — and the recommendation from tranche 2 inverts.** Toponymy's own disambiguation
machinery, handed exactly what a widened trigger would give it, buys a small and non-significant
identification gain at a large and highly significant fit cost, on both corpora. The blind spot
tranche 2 found is real; the repair already in the package is not the fix for it.

**This was the pre-registered prediction** (recorded in tranche 3's docstring before the run, on the
strength of #173's nibling-contrast negative and #177's fit≠identification split): *identification
improves and fit regresses*. It did.

**Method.** The 15 most confusable arbiter-**confirmed** pairs per corpus (lowest measured pairwise
pm). Layer state — exemplars, keyphrases, subtopics, gold names — rebuilt without re-naming
anything, so `distinguish_topic_names_prompt` is byte-for-byte what `fit()` would have constructed
for these names; then the library's own `generate_topic_cluster_names` (haiku, temp 0.4). Nothing
bespoke: this is the production repair path, invoked on the pairs the trigger currently misses.
30/30 pairs renamed successfully and all 30 changed. No plain-redraw control was re-run — 7c already
measured that (no identification headroom above the winner's-curse floor), so movement here is
attributable to the contrastive prompt rather than to drawing again.

| | 20NG | arXiv |
|---|---|---|
| identification (pairwise, fresh docs, gpt-4o-mini, chance 0.50) | 0.591 → **0.641** (+0.050) | 0.571 → **0.633** (+0.063) |
| | Wilcoxon p = 0.067, improved 22/30 sides | p = 0.36, improved 16/30 sides |
| below chance | 10/30 → 8/30 | 9/30 → 8/30 |
| **fit (grounded judge, paired vs committed gold)** | 2.722 → **2.139** (**−0.583**) | 2.605 → **2.068** (**−0.537**) |
| | **p = 0.0004** | **p = 0.002** |

The fit regression is **larger than the entire exemplars effect** (−0.49, the biggest number in the
program) and roughly 3.5σ of the judge repeat band measured in tranche 2 (σ = 0.168). The
identification gain does not clear significance on either corpus and barely moves the below-chance
count — the pairs that were unusable stay unusable.

**The mechanism is legible and measured.** Contrastive renaming names the *contrast*, not the
region: within-pair token Jaccard collapses (0.176 → 0.025 on 20NG, 0.143 → 0.037 on arXiv) and
**~41% of the vocabulary the pair shared is dropped from both new names**, at unchanged label length
(9.7 → 9.3 / 9.1 → 9.1 words). It is substitution, not compression. The shared vocabulary is often
the head noun that made each label a true description of its region, so dropping it trades
truthfulness for separability:

> `Motorcycle Riding Techniques, Safety, and Passenger Handling Advice` → `Riding Techniques,
> Safety, and Passenger Handling` — "Motorcycle" is gone, because the sibling cluster (used
> motorcycle sales) also had it.

That is Phase 5a's anti-conjunct result running in reverse: 5a found conjuncts that are true of the
region but wrong given the neighbours; this finds the prompt deleting content that is *right* for the
region because it is shared with a neighbour.

**Independent replication of #173.** The nibling-contrast experiment found that contrast context in
the *naming* prompt made labels worse (judge preferred baseline ~2:1 across 4 models). That result
had only ever been shown on a bespoke prompt built for that experiment. This reproduces it on
Toponymy's own production disambiguation path, on two corpora, on the fit axis, with a mechanism —
the confirmation that experiment never had.

### What tranche 3 changes

1. **Do not widen the trigger.** Tranche 2's blind spot stands, but routing more pairs into the
   existing repair would make label quality worse on net. The honest recommendation is now
   *diagnostic*: ship confusability as a **reported metric**, not as a trigger action.
2. **The disambiguation pass may be net-harmful whenever it fires.** Measured here on
   score-selected pairs rather than on the near-duplicate strings the trigger actually catches, so
   this is a hypothesis with a cheap test (rename the trigger's own groups, re-judge), not a result.
   If it holds it bears directly on **FEATURES.md's keyphrase recommendation**: dropping keyphrases
   triples fine-layer renaming load on 20NG, which was priced as a token cost — if renaming also
   costs ~0.5 judge-points per topic, that is a *quality* cost and the case against defaulting
   keyphrases off gets substantially stronger.
3. **A better repair would have to be constrained.** The failure is the prompt being free to delete
   shared content. "Distinguish these topics **without dropping the terms that make each name true
   of its own region**" is the obvious next prompt to test, and 5a's load-bearing/free-rider split
   is the tool for deciding which terms those are.

## Phase 7 results — tranche 3b: what the pass does on its OWN groups (2026-08-03, 444 LLM calls)

**Headline: the disambiguation pass succeeds at its stated goal, and the goal is expensive.** On the
groups its own trigger forms, it costs **−1.05 judge-points of fit** (p = 3e-07) and buys **+0.028
pm of identification** (p = 0.071, inside the repeat band). It de-duplicates effectively; that is
the only thing it demonstrably achieves.

**No renaming calls were needed.** `disambiguation_load.py` had already instrumented
`ClusterLayer.disambiguate_topics` across 24 real `fit()` runs and captured, per layer, the pre-pass
names, the post-pass names, and the groups the trigger formed — paired by construction. This
measures both axes on the 42 renamed topic instances (74 distinct labels). **Structurally 20NG-only:
arXiv produces zero renaming load under every condition**, so the two-corpus standard cannot be met
here and this is reported as single-corpus.

| axis | pre-pass | post-pass | Δ | p |
|---|---|---|---|---|
| fit (grounded judge 0–4, n=41) | 3.797 | 2.744 | **−1.053** | **3.4e-07** |
| identification (frozen k=5 lineup, n=42) | 0.333 | 0.361 | +0.028 | 0.071 (band 0.107) |

The pass improved fit on **0 of 41** instances and hurt 33. Per condition, all negative:
keyphrases-ablated −1.235, exemplars −1.167, subtopics −0.958, **stock −0.611**.

**Ceiling control, because 25/33 pre-pass names sit at the judge's 4.0 ceiling** — "improved 0/41"
is partly forced. Restricting to the 8 instances with ≥0.5 of headroom (pre ≤ 3.5, mean 3.000):
post 2.208, **Δ −0.792, improved 0/8**. The direction is not a ceiling artifact.

**These really are the confusable clusters**, which is the pass working as designed: their pre-pass
lineup pm is 0.333 against 0.545 for typical gold labels. But post-pass they sit at 0.361 — still
far below typical. The pass does not repair the identification problem it fires on.

**Same mechanism as tranche 3, one difference.** Within the trigger's own groups, token Jaccard
0.163 → 0.025 and **41% of shared vocabulary is dropped from both names** — identical to tranche 3's
41%, from a completely different selection rule. Here, though, labels get *longer* (9.1 → 10.6
words) rather than staying flat: the pass replaces shared head-nouns with additional distinguishing
detail, so it pays on both the truthfulness and the terseness axes at once.

### What this does to FEATURES.md's keyphrase recommendation

FEATURES.md priced the 3× fine-layer renaming load from dropping keyphrases as a **token** cost, and
concluded "quality-neutral, cost-ambiguous". The first half of that needs qualifying. Dropping
keyphrases fires the pass on ~6 layer-0 topics per draw instead of ~2, and each renamed topic loses
~1.24 judge-points. Spread over 74 layer-0 topics that is roughly −0.10 judge-points of layer
average — the same order as the ±0.17 aggregate effect FEATURES.md measured, which *already
contained* it.

So the corrected statement is not "keyphrases are quality-neutral" but: **dropping keyphrases is
roughly neutral on average while concentrating a large loss on a handful of topics** — and those
are precisely the fine-layer topics that collide, i.e. the crowded parts of the map where labels are
hardest to read. An average over 74 topics is the wrong summary for a cost shaped like that. This
strengthens the case against defaulting keyphrases off, on quality grounds rather than token
grounds.

### Caveats

- **One corpus, structurally.** arXiv never fires; this cannot be replicated on the second substrate.
- **Neither instrument scores the defect the pass exists to fix.** Duplicate names across two
  clusters are a real map-usability problem, and the grounded judge scores each label against its
  own cluster's documents while the lineup measures one cluster against neighbours. The pass's
  benefit is real and *unmeasured here*; what is measured is that it is not on either axis this
  program has instruments for.
- 42 instances over 12 distinct clusters, with clusters recurring across draws — the effective n is
  smaller than 42.

## Phase 7 results — tranche 3c: the obvious prompt fix does NOT work (2026-08-03, 390 calls)

**Headline: a minimal wording patch does not recover the damage, on either corpus.** This is a
negative, and it is the most useful thing to put in front of a maintainer, because it rules out the
first thing anyone would try.

**The cause is in the template, not in the model.** `templates.py` ends the disambiguation
instruction with:

> "**The primary goal** is to make each new topic name clearly distinguishable from the others in
> this list, based on the provided details."

Distinguishability is *declared* the primary goal, with nothing counterbalancing it about remaining
a true description of the region. So the mechanism found in 3 and 3b is not an emergent quirk — it
is compliance. The minimal patch appends two sentences supplying the counterweight ("...must also
remain an accurate, self-contained description of its own topic: distinguish by adding what
separates them, not by removing terms that are essential to what a topic is about, even when several
topics share those terms"), changes nothing else, and re-runs tranche 3's pairs so the stock arm is
already measured and the design is three-way paired on **both** corpora.

| | pre-disambiguation | stock | **constrained** | recovered |
|---|---|---|---|---|
| 20NG fit (n=28) | 2.702 | 2.185 | **2.244** | +0.060, p=0.81 — **11% of the damage** |
| arXiv fit (n=27) | 2.605 | 2.043 | **2.198** | +0.154, p=0.21 — **27% of the damage** |

Neither is significant, and both remain far below the pre-disambiguation names (−0.458 p=3e-04;
−0.407 p=0.012). **The patch fails.**

**And it fails for an informative reason: the constraint doesn't take.** Token Jaccard within the
pair is essentially unchanged (stock 0.025 → constrained 0.019 on 20NG; 0.037 → 0.044 on arXiv), and
of the shared terms the stock prompt deleted, the constrained prompt restored only **3/13 (20NG) and
1/12 (arXiv)**. The prompt did reach the model — 14/15 pairs on each corpus came back with different
names than stock — it simply kept stripping shared vocabulary anyway. The pull toward
differentiation is stronger than an added instruction.

**De-duplication is achieved by every arm, so it is not the differentiator.** Pairwise cosine
distance against the library's own 0.2 trigger cap: pre 0.400 / 0.500 (2/15 and 0/15 inside the
cap), stock 0.624 / 0.612, constrained 0.621 / 0.590 — 0/15 inside the cap everywhere. Identification
is flat across arms (constrained vs stock +0.022 p=0.44; +0.008 p=0.64).

**Reading.** The disambiguation pass buys de-duplication, pays 0.4–0.6 judge-points for it here
(1.05 on its own trigger groups, 3b), gains nothing measurable on identification, and **does not
respond to a wording-level counterweight**. That points at a design change rather than a prompt
tweak — rename only one member of a colliding pair; or constrain the edit to be additive by
construction rather than by instruction; or accept duplicates at the fine layer and disambiguate for
display instead. No further prompt variants were tried: this program's own standard is not to tune
until something flatters.

## Files (experiments/label_quality/)

- `PLAN.md` — this plan + running findings/status.
- `perturbations.py` — Phase 0 leg 1: intrusion battery from `cluster_tree_` (ancestor / sibling /
  distant / generic / verbose) → `data/battery_<ds>.json`.
- `prep_labels.py` — gold labeling: one Toponymy naming pass → `data/labels_<ds>_<model>.json`.
- `metrics.py` — Phase 1 intrusion gate: 4 reference points (centroid / whitened / medoid / exemplar)
  × cosine; intrusion table + per-candidate scores → `data/metric_scores_<ds>.json`.
- `judge_quality.py` — leg 3: grounded, self-consistent Preiss-rubric judge → `data/judge_ratings_*.json`.
- `validate_gate_b.py` — gate (b): Spearman + pairwise agreement of each metric vs the judge.
- `make_calibration.py` / `score_calibration.py` — human-calibration HTML form + judge–human κ.
- `wayfinding.py` — Phase 4: lineup construction (nn distractors, held-out non-exemplar docs,
  frozen per-cluster seeds), async listener (sonnet k=3, order reshuffled per sample), floors
  (shuffled / gimme / repeat), metrics + confusion matrix → `data/wayfinding_<ds>.json`.
- `wayfinding_pairs.py` — Phase 4 gate (c): the 104 fine pairs through frozen lineups + the
  human-seed export for lineup-decided-judge-tied pairs.
- `conjunct_ablation.py` — Phase 5a: split compound gold labels into standalone conjuncts (cached
  LLM splits), run drop:i / only:i variants through frozen lineups, marginal identification value
  per conjunct → `data/wayfinding_20ng_conjuncts.json`.
- `FEATURES.md` — Phase 6 write-up (the one both discussion comments link to).
- `feature_ablation_lineup.py` — Phase 6: the #173 exemplars/keyphrases ablation re-scored on the
  frozen 20NG lineups, with a power check against the battery's known-bad variants →
  `data/wayfinding_20ng_features.json`.
- `exemplar_dose_response.py` — Phase 6: name 20NG at k ∈ {1,2,4} plus the `4_nokp` interaction cell
  (all layers, stock machinery), judge against the gold doc sample →
  `data/dose_{names,judge}_20ng.json` (+ `.draw1.json`, an independent naming draw kept as a
  replication rather than a discard).
- `arxiv_naming_features.py` — Phase 6: the whole thing replicated on arxiv_home (naming, judging,
  lineups). ⚠ judge docs are sampled in the home cell's OWN geometry, matching `home_pipeline.py`;
  `judge_fair.sample_docs` would replay the canonical 20NG fit.
- `clean_docs_rejudge.py` — Phase 6: is the exemplar fit effect a parroting artifact? Re-judge on
  documents excluding every rung's exemplars → `data/clean_docs_20ng.json`.
- `upward_sweep.py` — Phase 6: does the curve turn over ABOVE the default? k ∈ {4,8,16} fully
  leakage-controlled, k=32 knowingly advantaged → `data/upward_{names,judge}_20ng.json`.
- `subtopics_value.py` — Phase 6: subtopics scoped to coarse layers, single- vs multi-child, both
  axes, both corpora → `data/subtopics_*_arxiv.json`, `data/wayfinding_*_subtopics.json`.
- `disambiguation_load.py` — Phase 6: instruments `ClusterLayer.disambiguate_topics` during `fit()`
  to capture pre-pass names + the groups the renaming trigger formed → `data/disamb_load.json`.
- `price_dose_response.py` — Phase 6: exact call/token accounting via `count_tokens` on the real
  prompts (no generation), in the #177 house style of costing arms before running them.
- `lineup_scorer_probe.py` — Phase 7a: re-scores every committed lineup unit with
  `softmax(cos/τ)` over the frozen candidate set (τ fitted out-of-corpus), against raw pointwise
  cosine as the #173 control; coarse/fine/direction tests. No LLM calls.
- `abstraction_probe.py` — Phase 7b: the 20NG gold-category abstraction test (Phase 0 leg 2) —
  asymmetry failure rate for cosine, length, and Phase 2a's generality axis. No LLM calls.
- `redraw_headroom.py` — Phase 7c: oracle best-of-N headroom for naming redraws on both axes,
  each against its own measured winner's-curse floor. No LLM calls.
- `confusability.py` — Phase 7 tranche 2: the confusability score (`--stage score`, free), the
  fresh-doc held-out-listener pairwise arbiter (`--stage arbiter`, 660 calls), and the synthesis
  (`--stage report`) → `data/confusability_{20ng,arxiv_home}.json`,
  `data/confusability_arbiter_*.json`.
- `judge_repeat.py` — Phase 7c follow-up: the pure judge repeat band (same labels, same documents,
  same recipe; 120 calls) → `data/judge_repeat_20ng.json`. `--report-only` re-derives the
  decomposition without spending calls.
- `disamb_value.py` — Phase 7 tranche 3b: measures what the disambiguation pass does on its OWN
  trigger groups, reusing the pre/post names `disambiguation_load.py` already captured (so zero
  renaming calls); fit (`--stage judge`) + identification through the frozen lineups
  (`--stage lineup`) + the shared-vocabulary mechanism (`--stage report`) →
  `data/disamb_value_{judge,lineup}.json`.
- `issue_disambiguation_cost.md` — DRAFT issue body for the tranche 3/3b/3c finding. **Held until
  the eval framework lands upstream** (it asks for a design decision, which needs the instruments
  in-tree to evaluate); the keyphrase repricing inside it is separable and goes to #173 first.
- `constrained_repair.py` — Phase 7 tranche 3c: appends a two-sentence counterweight to the stock
  disambiguation instruction (`--stage rename`), re-measures fit / identification / de-duplication
  three-way paired against the pre and stock arms (`--stage judge|arbiter|report`) →
  `data/constrained_{names,judge,arbiter}_*.json`. Fails loudly if `templates.py` moves the sentence
  it patches.
- `repair_check.py` — Phase 7 tranche 3: rebuilds layer state without re-naming, drives Toponymy's
  OWN `distinguish_topic_names_prompt` + `generate_topic_cluster_names` on the confirmed-confusable
  pairs (`--stage rename`), re-measures identification (`--stage arbiter`) and fit (`--stage judge`)
  paired against the committed gold, and reports the shared-vocabulary mechanism (`--stage report`)
  → `data/repair_{names,arbiter,judge}_*.json`.

Reuses the nibling harness (`../nibling_contrast/`): `ab_harness.{load_dataset,make_namer,make_embedder}`
and `judge_fair.sample_docs`. All fits use `ToponymyClusterer(min_clusters=4, base_min_cluster_size=25)`
on the full 20NG (`../nibling_contrast/data/ng_*`); cluster indices align across every script.

Reproduce: `prep_labels.py --model haiku` → `perturbations.py --labels data/labels_20ng_haiku.json`
→ `metrics.py --labels ...` → `judge_quality.py --judge sonnet --k 3` → `validate_gate_b.py --judge sonnet`.

## Status / next

- [x] Plan agreed (all three choices locked); branch `experiment/label-quality` created.
- [x] **Phase 0 · leg 1** — perturbation/intrusion battery (`perturbations.py`).
- [x] **Phase 0 · gold labeling** — haiku on full 7000 → `data/labels_20ng_haiku.json`; 3 layers
  (74/24/9). Real battery emitted: `data/battery_20ng.json`, **107 clusters**; variant coverage
  distant/generic/verbose=107, sibling=93, ancestor=74 (ancestor<107 is correct: coarsest clusters'
  only parent is the synthetic root, and name-propagation makes some parents share a child's label).
- [x] **Phase 1 · intrusion gate** — `metrics.py`. Winner: **whitened centroid** — strict top-1
  72%→90% vs raw centroid; the verbose-padding intrusion (the real discriminator) 78%→92%.
  Medoid/exemplar barely help and are *worse* on ancestor/sibling. distant/generic are easy for all
  (≥99%). Conclusion: the failure is **anisotropy**, not centroid-vs-extent.
- [x] **Phase 0 · leg 3** — grounded self-consistent judge (`judge_quality.py`, Preiss rubric +
  Krumdick grounding); sonnet k=3 over the battery → `judge_ratings_20ng_sonnet.json`. Judge means:
  gold 2.80 > ancestor 1.96 > verbose 1.70 > sibling 0.62 ≈ generic 0.59 (sensible, graded).
- [x] **Phase 1 · gate (b)** — `validate_gate_b.py`. RESULT: the metric family is **validated as a
  coarse quality signal** — all reference points ρ≈0.71–0.74 vs judge-overall (n=487, p~1e-80) and
  91–94% pairwise agreement. Whitening's win is **specific, not global**: best on the verbose blind
  spot (pairwise 80%→92%) and best overall pairwise (94.4%), but marginally LOWER global Spearman
  (centroid 0.740 > exemplar 0.730 > medoid 0.722 > whitened 0.708) and worse on ancestor (85.7→82.1).
- [x] **Phase 0 · human seed** — 28 blinded ratings (`make_calibration.py` + `score_calibration.py`).
  Judge VALIDATED: Spearman ρ=0.82, quadratic κ=0.64 (substantial), MAE 0.89/4. Caveats: (1) judge is
  uniformly ~0.85 HARSHER than the human (fine for the relative/rank use gate (b) made of it); (2)
  human penalizes VERBOSITY > over-generality, judge the reverse → human prefs favor whitening even
  more; (3) sibling-swap is a soft "known-bad" (2/5 siblings rated good — a related cluster's label can
  genuinely fit). → **Phase 0 instrument COMPLETE + calibrated.**
- [ ] **Phase 1 · fine discrimination** — judge multiple *good* real labels/cluster (the prior
  negative's regime): can any metric rank good-vs-good? (the open question gate (b) doesn't settle)
- [x] **Phase 2a · generality bake-off** — `phase2_generality.py` on HyperLex. A linear generality
  axis EXISTS in MiniLM space but is WEAK: lexical-split test Spearman emb-axis 0.32 / cosine 0.34 /
  combined 0.45; frequency 0.19, length ~0, emb-norm −0.12. Confirms idea D + Renner (structure ≫
  embeddings). Embedding axis is the phrase-label-applicable proxy → used in 2b. (axis saved.)
- [x] **Phase 2b · hierarchy check** — `phase2b_hierarchy.py` on 86 real parent→child edges. The
  embedding generality axis FAILS to transfer to Toponymy phrase labels: **50.0% direction accuracy
  (= chance)**. But **LENGTH works (86%)** — coarse labels are shorter (partly an artifact of haiku's
  verbose-fine/terse-coarse style); frequency is anti-correlated (27%, coarse labels use rarer words).
  **CORRECTION — NOT a defect:** the 14% parent==child edges are ALL single-child regions = Toponymy's
  deliberate `[!SKIP!]` inheritance (a coarse region with one named child takes its name; correct).
  Disambiguation is within-layer and rightly leaves them alone. No genuine hierarchy defect found; the
  earlier "under-generalization" framing is retracted (caught in review before posting).
- [x] **Feature-ablation characterization** — `ablation.py`. Re-name 20NG (haiku) dropping each naming
  feature; judge (grounded sonnet) + whitened-metric the ablated labels vs full. **Metric verdict: NOT a
  useful regression guard** — blind across all 3 features (metric-Δ≈0, Spearman −0.07…+0.03, sign-agree
  48–56% = chance) even where the judge sees a large drop. Confirms: "beats a weak baseline" ≠ useful;
  the metric only catches gross/off-topic failures, not realistic (on-topic) naming regressions. **Bonus
  feature-contribution finding (judge-measured):** exemplars +0.49 (clear help) ≫ keyphrases −0.17 (no
  help, maybe slight harm — suggestive) ≈ subtopics −0.03 (neutral, coarse-only). Single-draw / 20NG /
  haiku caveats on the small effects.
- [ ] **Phase 0 · leg 2** — gold-category alignment.
- [x] **Phase 1 · fine discrimination (DONE, rigorous)** — `fine_discrimination_grounded.py`: 104
  haiku-vs-gpt4o-mini pairs judged by grounded sonnet, 54 decided (judge TIED on 48% — the two good
  labels are often indistinguishable). ALL reference points ~chance (centroid 48%, whitened 37%
  [worst], medoid 50%, exemplar 57% but within noise; none clears 50% on both winner-splits).
  **DEFINITIVE: the metric is a coarse guardrail, NOT a fine ranker.**
- [x] **Phase 3 · erasure (DONE)** — `phase3_rank_diagnostic.py` + `phase3b_erasure.py`. SURPRISE:
  the label↔doc gap is a low-rank LINEAR mean offset (separability 0.935 → 0.325 after per-class
  centering; INLP inseparable after 2 directions) — erasable, NOT nonlinear like the modality gap;
  the gate PASSES. BUT erasing it does **NOTHING** (coarse ρ 0.737 vs 0.740 raw; verbose 74.8% vs
  77.6%) — the offset is class-uniform, hence irrelevant to scoring. **Idea A is dead: the gap is
  erasable but irrelevant; the real fix was anisotropy (whitening), a broader property than the offset.**
- [ ] **Phase 4 · instrument** — `wayfinding.py` on 20NG/minilm: build frozen lineups; run floors
  (shuffled / gimme / repeat) + gate (a) battery signatures + gate (b) judge relation. Decision
  point: proceed only if gold clears the shuffled floor and the repeat floor is tight.
- [ ] **Phase 4 · gate (c)** — `wayfinding_pairs.py`: do lineups decide the judge-tied half of the
  104 fine pairs? agreement on the 54 decided? ~20-item blinded human seed on the new axis.
- [ ] **Phase 4 · (conditional) robustness spot-check** — if gates pass, one away cell
  (arxiv/cohere, the cleanest space) through the same lineups to check the instrument isn't a
  20NG/minilm artifact.
- **Named tree has skip-level edges** (matters for Phase 2's parent⊇child pairs): a fine cluster's
  nearest *named* ancestor can sit 1–2 layers up or at the synthetic root (L0 parents on 20NG/haiku:
  50@L1, 14@L2, 10@root). The parent⊇child scorer must accept variable layer gaps, not assume L+1.
- **Name propagation = a free under-generalization signal**: coarse regions often inherit a child's
  exact label (e.g. "Nutritional and Alternative Medicine Debate" appears at both L1 and L2). Phase 2
  can flag these as coarse labels that failed to generalize.
- **Gold fine-layer labels are verbose/over-specified** (haiku) — a fair, hard testbed: the metric
  must not simply reward the wordiest candidate (the exact failure mode of raw centroid-cosine).
- **The centroid-cosine failure is ANISOTROPY, not centroid-vs-extent.** Whitening the (doc-fit)
  embedding space fixes the verbosity bias (verbose-intrusion 78%→92%, strict top-1 72%→90%);
  switching to a real point (medoid/exemplar) does *not* (and is worse on ancestor/sibling — the
  centroid's averaging is helpful). Keep the averaged centroid, whiten the space.
- **Gate (b) reconciles the prior "centroid-cosine ≈ chance" negative.** The metric family WORKS as a
  *coarse* quality signal (ρ≈0.74 with a grounded sonnet judge, 91–94% pairwise). The prior negative
  was *fine* discrimination (two good contrast labels, ~chance); this is *coarse* (good vs degraded).
  So embedding-similarity is a usable **guardrail** (catch bad labels), not yet shown to be a
  fine-grained ranker.
- **A generality axis exists in embedding space but is WEAK (Phase 2a).** On HyperLex it adds ~0.10
  over cosine (combined ρ~0.45) but is far below structural WordNet-IC (0.744) — embeddings encode
  hypernymy weakly/entangled, as the lit predicted. It's the only proxy that applies directly to
  multi-word labels, so 2b uses it but should expect noise. Faint hyperbolic hint: general words sit
  at slightly smaller embedding norm (ρ=−0.12).
- **Phase 2b: idea D is DEAD for Toponymy phrase labels.** The HyperLex-learned axis gives chance
  (50%) direction accuracy on real parent→child label pairs — single-word hypernymy geometry does not
  transfer to phrase-label generality. The working proxy is dead-simple **LENGTH** (86%: coarse labels
  are shorter, partly because haiku names fine verbosely). **NO hierarchy defect** — the 14%
  parent==child edges are all single-child regions = deliberate `[!SKIP!]` inheritance (correct), not
  under-generalization (earlier framing retracted in review).
- **Whitening is a trade-off, not a free win.** It fixes the verbose blind spot (the motivating
  failure) but compresses the anisotropic over-separation that helps track big good-vs-bad gaps
  (lower global Spearman). Pick whitened for robustness-to-padding; raw centroid for raw graded
  correlation — both usable. A combined metric (raw centroid + verbosity correction) is the obvious
  refinement.

## Phase 6 status (naming features) — COMPLETE, SHIPPED

- [x] **Lineup re-scoring** of the #173 exemplars/keyphrases ablation (20NG) — both inert; power
  check makes the nulls meaningful.
- [x] **Exemplar dose-response**, all layers, plus the keyphrase × exemplar interaction cell.
- [x] **arXiv replication** of all three claims — two of them died here (the k=4 knee, the
  interaction), which is the 2-corpus standard working as intended for the third time.
- [x] **Leakage control** of the fit instrument — confound was real, effect was 2%, #173 stands.
- [x] **Upward sweep** through k=32 — no turnover; `n_exemplars` is a floor, not a target.
- [x] **Subtopics scoped properly** (coarse layers, single- vs multi-child, both axes/corpora).
- [x] **Disambiguation load per feature** — the keyphrase cost nobody had priced.
- [x] **Posted** to #173 / #177 / #57; branch frozen at the posted tip.
- [ ] **Keyphrase × subtopic interaction** — untested. The keyphrase × exemplar one was 20NG-only,
  so assume nothing about combinations.
- [ ] **Terser-namer probe** — everything here is one namer (haiku), the caveat most likely to matter.
- [ ] **Is the mild negative direction of keyphrases/subtopics real?** Needs more coarse clusters
  than this substrate has.

## Phase 7 status (cheap scorer) — tranche 1 COMPLETE, not posted

- [x] **Noise ceiling measured first** — within-cluster reliability of the committed 3-sample `pm`
  is ≈0.57 / 0.62, so any predictor of it is capped near ρ 0.75. Everything below is read against
  that, not against 1.0.
- [x] **7a softmax-cosine** — passes coarse (and fixes the padding blind spot: arXiv verbose
  51%→91%), fails fine (+0.24 vs a ~0.75 ceiling), and the unplanned direction test is the
  tranche's strongest signal (67–69% vs 25% chance).
- [x] **7b abstraction probe** — Phase 0 leg 2 closed. Premise confirmed (50.4% asymmetry failure),
  both cheap fixes rejected, generality axis fitted to λ=0 (Phase 2b's negative, third replication).
- [x] **7c redraw headroom** — null on identification (headroom inside the floor range); fit
  headroom +0.166 does **not** clear its +0.197 winner's-curse floor. Best-of-N killed for ~$0.
- [x] **Correction recorded** — the +0.166 fit headroom was quoted mid-discussion before its floor
  was computed; the floor removes it.
- [x] **Tranche 2 — confusability** — score validated offline (AUC 0.803 / 0.920; top-confuser
  agreement 80% / 71%), Toponymy's trigger catches 2/40 and 0/40, and the fresh-doc held-out-listener
  pairwise arbiter separates flagged from control at p ≈ 2e-05 on both corpora. 660 calls.
- [x] **Judge repeat band** (120 calls) — σ = 0.168, floor +0.095. 7c's fit headroom **clears**;
  true oracle gain ≈ +0.071 judge-pts. Best-of-N is unattractive, not null — earlier statement
  corrected.
- [x] **Tranche 3 — does renaming repair it? NO.** Toponymy's own disambiguation path gives
  +0.050 / +0.063 pm identification (n.s.) for **−0.583 / −0.537 judge-pts fit** (p=4e-04 / 2e-03),
  both corpora. Mechanism measured: ~41% of the pair's shared vocabulary is deleted from both new
  names at unchanged length — it names the contrast, not the region. Pre-registered prediction,
  confirmed. **Recommendation inverts: report confusability, do not act on it.** 570 calls.
- [x] **Is the disambiguation pass net-harmful whenever it fires? YES on fit.** On its own trigger's
  groups: **−1.053 judge-pts** (p=3e-07, improved 0/41; −0.792 and 0/8 restricted to instances with
  real headroom, so not a ceiling artifact) for **+0.028 pm** identification (n.s., inside band).
  Same 41% shared-vocabulary deletion as tranche 3, from a different selection rule. Needed **zero**
  renaming calls — `disamb_load.json` already held pre/post names and the groups. 20NG-only,
  structurally. **Reprices FEATURES.md**: dropping keyphrases is neutral *on average* while
  concentrating ~1.24 judge-pts of loss on the handful of colliding fine-layer topics.
- [x] **A constrained repair prompt — TRIED, FAILED.** Two-sentence counterweight appended to the
  stock instruction: recovers 11% (p=0.81) / 27% (p=0.21) of the damage, both corpora, and the
  constraint does not take — only 3/13 and 1/12 of the deleted shared terms come back, Jaccard
  unchanged. Rules out the first fix anyone would try. Points at a design change, not wording.
- [ ] **Design-level alternatives, untested:** rename only one member of a colliding pair; make the
  edit additive by construction rather than by instruction; or disambiguate for display only.
- [x] **Issue drafted** (`issue_disambiguation_cost.md`), deliberately **not posted**: it asks for a
  design decision, which a maintainer can't evaluate while every number in it rests on instruments
  living in `experiments/` on a fork. Sequencing decided — eval-framework PR first, issue second,
  pointing at in-tree calls.
- [ ] **#173 comment: the keyphrase repricing.** Separable from the issue, no dependency on the PR,
  and it has a clock on it — jc-healy was leaning toward defaulting keyphrases off, and this is a
  correction to my own posted analysis. Send before the default flips.
- [ ] **Eval-framework PR.** Scope decision to make first: the findings rest on *both* instruments
  (lineup for identification, judge for the −1.05), so a lineup-only PR won't carry the issue.
  Suggested surface: score a labeling, not a framework — battery / calibration / human-seed
  machinery stays on the fork.
- [ ] **Write-up + venue call** for Phase 7 as a whole, if it warrants one. The arc completes as
  blind spot → free detector → held-out confirmation → the obvious fix makes it worse → the
  wording-level repair fails. Current read: the disambiguation finding is issue-shaped rather than
  discussion-shaped, and tranche 1/2 are program-internal.

**Phase 7 spend to date: 2,184 LLM calls** (660 tranche-2 arbiter + 120 judge repeat + 570 tranche
3 + 444 tranche 3b + 390 tranche 3c), against the ~1,500 discussed — the 3b overrun bought both axes rather than
fit alone, which the pass's purpose required. Tranche 1 was free, and 3b needed no renaming calls.
- [ ] **Third corpus in a different register** — required before any *trained* rung is claimed to
  generalize; not needed for tranche 2, which uses no trained parameters.
- [ ] **Not attempted, deliberately:** r1–r3 of the capacity ladder. 7a-fine + 7b together say the
  fine/abstraction signal is not in this geometry, so climbing would be tuning against a known wall.
