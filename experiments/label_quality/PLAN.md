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

## Kill criteria (so we don't fool ourselves)

- **Phase 0:** if grounded judge–human κ stays low, fix the instrument before trusting any
  probe.
- **Phase 1:** if no reference-point variant clears both gates, fidelity-via-similarity is
  dead in raw space → Phase 3 becomes necessary, not optional.
- **Phase 2a:** if nothing beats the cosine floor on HyperLex, there is no usable generality
  axis in our embedder → fall back to structural IC (region size) and say so.

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

## Embedder + dataset robustness study (IN PROGRESS — findings not yet in WRITEUP)

Two near-posts were caught as artifacts (under-generalization; whitening/blindness), so before posting
we're checking embedder + dataset robustness. A quick recompute with Cohere **embed-v4**
(`embedder_robustness.py`, scoring on fixed MiniLM clusters) already PARTIALLY REVERSED two findings:
the *raw centroid* metric DETECTS the exemplars/subtopics ablation (ρ +0.41 / +0.26) where MiniLM was
blind, and whitening HURTS in embed-v4's less-anisotropic space (gate-b 0.66 vs 0.74). So metric claims
are **embedder-dependent**.

Confound caught in review: scoring labels against clusters defined by a *different* (inferior) embedder
is a home/away mismatch. Fix = each embedder plays AT HOME (cluster + name + score with its own
geometry): `home_pipeline.py` (+ `async_judge.py`, ~20× faster concurrent judging). Study =
weak (MiniLM) vs strong (embed-v4) × 2 datasets (20NG + arXiv @7k), trimmed to **gate-b +
exemplars/keyphrases ablation** (fine-disc + secondary findings cited as robust negatives, not re-run).
20ng-minilm = the existing cell (already at-home); 3 new cells (`home_*.json`) running in background.
Update WRITEUP §2 (whitening → weak-embedder patch), the ablation §ablation (blindness is
MiniLM-specific), TL;DR, and the teaser once the 2×2 lands.

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

### Findings to carry forward
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
