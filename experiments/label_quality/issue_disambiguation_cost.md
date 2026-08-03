> **STATUS: DRAFT — NOT POSTED.** Deliberately held until the evaluation framework lands upstream
> (jc-healy's "I'd also very much hope to be able to integrate some of this evaluation framework
> back into Toponymy itself", [#173](https://github.com/TutteInstitute/toponymy/discussions/173)).
> The issue asks for a design decision rather than proposing a patch, and a design decision can't be
> evaluated without the instruments in-tree — every number below is currently unverifiable by the
> person receiving it. Post once the eval PR is merged, replacing the "Reproducing" section with
> in-tree calls. The keyphrase repricing is *separable* and goes to #173 as a comment now.
> Body follows.

---

## The disambiguation pass trades a lot of label fit for no measurable identification gain

While measuring naming quality I ended up instrumenting `ClusterLayer.disambiguate_topics`, and the
renaming pass looks like it costs more than it buys. Summary up front, including the limits, because
two of the three measurements are single-corpus:

- On the groups its **own trigger** forms, the pass costs **−1.05** on a 0–4 grounded-judge fit
  rubric (p=3e-07; it improved 0 of 41 renamed topics) and returns **+0.028** on a discrimination
  metric (p=0.07, inside that metric's repeat band). *20 Newsgroups only — see limits.*
- The mechanism is that renaming deletes **~41% of the vocabulary the colliding names shared**,
  including head nouns that make each name true of its own region.
- A minimal wording fix to the prompt **does not work** (recovers 11% / 27% of the damage, both
  n.s., on two corpora), and it fails because the constraint doesn't take.

I'm not proposing a patch — I tried the obvious one and it failed. I'm reporting it because the
first thing anyone would do on reading the above is try exactly that patch, and because the choice
among the real alternatives is yours rather than mine.

### The mechanism is in the template, not the model

`templates.py` ends the disambiguation instruction with:

> "**The primary goal** is to make each new topic name clearly distinguishable from the others in
> this list, based on the provided details."

Distinguishability is declared the primary goal with nothing counterbalancing it about each name
remaining a true description of its own region. So the model complies, and what it deletes is
whatever the colliding names have in common. A representative case from 20NG, where two adjacent
fine-layer clusters are motorcycle riding advice and used-motorcycle sales:

```
before:  "Motorcycle Riding Techniques, Safety, and Passenger Handling Advice"
after:   "Riding Techniques, Safety, and Passenger Handling"
```

"Motorcycle" is gone because the neighbour also had it. The name is now more distinguishable and
less true.

Measured across all renamed pairs, token Jaccard within a pair drops from 0.163 to 0.025, and 41% of
the shared vocabulary disappears from both new names. I got the same ~41% from a completely
different way of selecting pairs, so it doesn't look like an artifact of how I picked cases.

### What it costs

Two instruments: a grounded LLM judge on the Preiss et al. 2024 0–4 rubric (majority of 3 samples,
scored against a deterministic per-cluster document sample, calibrated against my blinded human
ratings at ρ ≈ 0.8), and a discrimination metric where a listener LLM sees a label plus five
candidate document groups — the true cluster and its four nearest same-layer neighbours, shown as
held-out member documents — and the score is the probability mass on the true cluster.

**(a) On the trigger's own groups.** 42 topics that `cluster_topic_names_for_renaming` grouped and
the pass renamed, captured across 24 real `fit()` runs (four prompt-feature conditions × three
draws), so pre- and post-pass names are paired by construction:

| | pre-pass | post-pass | Δ | p |
|---|---|---|---|---|
| fit (judge 0–4, n=41) | 3.797 | 2.744 | **−1.053** | 3.4e-07 |
| identification (n=42) | 0.333 | 0.361 | +0.028 | 0.071 |

All four conditions negative, including stock (−0.611). 25 of 33 pre-pass names sit at the judge's
4.0 ceiling, so "improved 0/41" is partly forced — restricting to the 8 instances with ≥0.5 of
headroom gives Δ −0.792 and still 0/8 improved, so the direction isn't a ceiling artifact.

Worth noting the pass is firing on genuinely confusable regions: their pre-pass identification score
is 0.333 against ~0.545 for typical labels. But post-pass they're at 0.361 — the pass doesn't repair
what it fires on.

**(b) On pairs I selected as confusable** (a geometric score, both corpora, 15 pairs each), driving
the same `distinguish_topic_names_prompt` → `generate_topic_cluster_names` path:

| | fit Δ | identification Δ |
|---|---|---|
| 20NG (n=30) | **−0.583** (p=4e-04) | +0.050 (p=0.07) |
| arXiv (n=30) | **−0.537** (p=2e-03) | +0.063 (p=0.36) |

For scale: dropping *all* exemplars from the naming prompt costs −0.49 on the same rubric, and the
judge's own repeat band is σ=0.168 (same labels, same documents, same recipe, re-run).

### The obvious fix doesn't work

I appended two sentences to the instruction above, changing nothing else — that each name must also
remain an accurate, self-contained description of its own topic, and that topics should be
distinguished by adding what separates them rather than removing what they share. Re-ran the same
pairs, three-way paired against the pre- and stock-disambiguated arms:

| | pre | stock | constrained | recovered |
|---|---|---|---|---|
| 20NG (n=28) | 2.702 | 2.185 | 2.244 | +0.060, p=0.81 — 11% |
| arXiv (n=27) | 2.605 | 2.043 | 2.198 | +0.154, p=0.21 — 27% |

Not significant on either corpus. And it fails informatively: **the constraint doesn't take.** Token
Jaccard is essentially unchanged (0.025 → 0.019 and 0.037 → 0.044), and of the shared terms the
stock prompt deleted, the constrained prompt restores only 3/13 and 1/12. The patch did reach the
model — 14 of 15 pairs per corpus came back with different names — it was simply outweighed. The
pull toward differentiation seems stronger than an added instruction.

De-duplication, meanwhile, is achieved by every arm: pairwise cosine distance between a pair's two
names ends up at 0.62 / 0.59 against the trigger's own 0.2 cap, with 0/15 inside the cap everywhere.
So the pass does do its job. That's the tension — it works, and working is expensive.

### Limits, stated plainly

- **(a) is 20 Newsgroups only, structurally.** On my arXiv substrate the trigger produces *zero*
  renaming load under every condition, so there's nothing to measure. (b) and the failed fix are
  two-corpus.
- **In the default configuration the aggregate footprint is small** — stock fires on ~2 of 107
  topics per run, so this is roughly −0.01 of corpus average, not a claim that Toponymy degrades
  labels generally. It matters as a prompt-level defect, and as a reason to be careful about changes
  that make the pass fire *more* (see below).
- **Neither instrument scores what the pass is for.** Two clusters carrying the same name is a real
  map-usability problem; the judge scores each label against its own cluster's documents and the
  discrimination metric compares one cluster against neighbours. The pass's actual benefit is real
  and unmeasured here.
- 42 instances over 12 distinct clusters, with clusters recurring across draws, so the effective n
  is smaller than 42. One namer (Haiku), one judge family (Sonnet).

### Possible directions, with no evidence behind any of them

Since the wording-level fix is out, the remaining options look structural: rename only *one* member
of a colliding pair, so the other keeps its region-true name; make the edit additive by construction
rather than by instruction (append a distinguishing qualifier to the existing name rather than
regenerating it); or accept duplicates at the fine layer and disambiguate for display only. I have
no data on any of these and don't want to guess which you'd prefer.

### A related note for #173

This bears on defaulting keyphrase extraction off. I measured that dropping keyphrases triples
fine-layer renaming load on 20NG, and priced it there as a token cost. Given the above it's also a
quality cost: it fires the pass on ~6 layer-0 topics per draw instead of ~2, at ~1.24 judge-points
each. That's about −0.10 of layer average — the same order as the aggregate effect I reported, which
already contained it. So "quality-neutral" was the wrong summary: it's neutral on average while
concentrating a large loss on precisely the colliding fine-layer topics, which are the crowded parts
of the map where labels are hardest to read.

### Reproducing

*(To be replaced with in-tree calls once the evaluation framework lands.)*

Happy to run any variant you'd like measured — the harness is paired and resumable, and the
expensive part (the listener and judge traffic) is already cached.
