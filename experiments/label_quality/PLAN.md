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
