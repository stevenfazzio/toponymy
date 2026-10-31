# Can a decision model do per-cluster label selection?

> Not shared anywhere yet (October 2026). Companion to [WAYFINDING.md](WAYFINDING.md), whose length
> controller this builds on, and [FEATURES.md](FEATURES.md). Scripts and results are in
> [decision_models/](decision_models/).

## TL;DR

The idea has three separable parts: generate several candidate labels per cluster, score them
against neighbouring clusters with a wayfinding lineup, and select one. Any preference about size
is optional. Cost was the main reason #177 stopped short of proposing its length controller as a
default, so the hope is that a cheap decision model such as Jev could do the scoring.

That splits into two questions, tested separately. Is selecting among candidates worth anything on
documents the selection never saw? That used Sonnet as the scorer. Can a cheap model do the scoring?
That used **Laya**, an open Jev-like model, on local hardware. **Jev itself was not tested.**

1. **Selection survives fresh documents.** The top-scoring candidate on one document draw beat stock
   on a fresh draw by +.059 (20NG) and +.018 (arXiv), while also being shorter. The arXiv gain only
   just clears zero, and most of the apparent gain on the original draw did not carry over.
2. **Size is not what separates candidates.** Length explains under 6% of within-cluster score
   differences for every candidate source, and about 2% across the full length ladder.
3. **Laya sorts bad labels from good but cannot rank one cluster's candidates.** Agreement with
   Sonnet across label kinds is 0.80 and 0.74. Among a single cluster's candidates it is +0.06 and
   −0.04, where a Sonnet sample agrees with its own other samples at 0.34 and 0.42. So selection has
   value, and Laya cannot capture it.
4. **Among labels that differ only in length, a fixed terse rung is the baseline to beat.** Rung 5
   at every layer matched or beat Laya's selections, about 60% shorter than stock and rated 0.020
   lower on 20NG and 0.001 lower on arXiv. This says nothing about choosing between labels with
   different content, which a fixed setting cannot do.
5. **Question format matters more than checkpoint.** The literal lineup and per-group questions sat
   near chance on Laya. Only documents as state with a Choice over labels carried signal.
6. **Cluster ids are not stable between processes** (see below), which touched 2 of 111 clusters in
   the posted arXiv held-out result without changing its conclusion.

## Selection is not about size

Saved Sonnet lineups keep three listener samples per label, so a label can be picked on one sample
and scored on the other two. That removes winner's curse from listener noise at no API cost. Real
spread is the within-cluster SD of score after removing listener noise. Gain is against a random
candidate from the same pool. Results are 20NG / arXiv.

| candidate source | candidates per cluster | real spread | length explains | gain |
|---|---|---|---|---|
| length ladder | 6.8 / 6.3 | .068 / .057 | 2.1% / 2.3% | +.044 / +.037 |
| same prompt, two draws | 2 / 2 | .047 / .091 | 1.1% / 1.2% | +.015 / +.029 |
| different naming evidence (a feature dropped) | 3.2 / 3.1 | .072 / .074 | 0.0% / 0.1% | +.044 / +.040 |
| different namers (Haiku, gpt-4o-mini) | 2 / n/a | .061 / n/a | 1.4% / n/a | +.019 / n/a |
| all pooled | 10.6 / 8.8 | .069 / .069 | 0.8% / 0.8% | +.062 / +.045 |

Even on the ladder, candidates differ mainly in what they say. Pooled picks were no longer than a
random candidate (6.8 / 7.0 words against 6.6 / 6.8). The number of candidates appears to matter
more than their source, since every source shows a similar spread. All samples here share one
five-document draw, though, which is what the next test addresses.

## Fresh-document test

Everything was fixed before any paid call. A cluster's **pick** is its highest mean Sonnet score on
the frozen documents among all pooled candidates, using all three samples. **Stock** is what stock
Toponymy named it. The **fresh draw** is five new documents per cluster, excluding the namer's
exemplars and every document in the frozen draw, and neighbouring groups were redrawn the same way.
Prompt, model, temperature, and three reseeded samples match the July protocol. Clusters whose pick
is the stock label count as zero difference and needed no calls.

| | 20NG | arXiv |
|---|---|---|
| clusters / pick differs from stock | 107 / 95 | 116 / 103 |
| **fresh gain over all clusters** | **+.059** | **+.018** |
| 95% bootstrap interval | +.037 to +.082 | +.000 to +.036 |
| Wilcoxon p, differing clusters | 5.4e-07 | .024 |
| pick / stock on fresh documents | .617 / .550 | .571 / .551 |
| pick beats stock, differing clusters | 73% | 62% |
| gain on the frozen draw (in-sample) | +.100 | +.088 |
| share of that gain kept | 59% | 20% |
| frozen vs fresh difference, Spearman | +.31 | +.32 |
| words, pick / stock | 6.7 / 8.0 | 7.1 / 8.5 |

The cost was 1,188 calls, 3,342,162 input tokens, and 21,461 output tokens, with no failures.

The cross-fit gains above were +.048 and +.045 against stock. They use a pick from one sample, so
they are not directly comparable with a pick from all three. By layer, the fresh gain is +.056,
+.071, +.050 on 20NG and +.007, +.035, +.077 on arXiv, with few clusters at the coarsest layers.

This is the value of selection with a strong scorer and about ten candidates per cluster, at roughly
30 listener calls per cluster. It is not yet a cheap way to get that value. Selection and evaluation
also shared one listener and only one fresh draw was used. The earlier check against the #177
held-out data looked much weaker (correlations of +.01 and +.37), but those picks were chosen for
brevity, which restricts their range.

## Format pilot

Laya's English checkpoint reads 512 tokens and a k=5 lineup prompt is 3,751, so the lineup could
not be replayed as written. Seven combinations of format and checkpoint ran on a stratified battery
sample (12 units per kind, 96 per corpus). Scores below are gold / shuffled / gimme, then pooled
Spearman with Sonnet. Sonnet on the same sample is 0.551 / 0.029 / 0.930 on 20NG and
0.558 / 0.117 / 0.836 on arXiv.

| format and checkpoint | 20NG | arXiv |
|---|---|---|
| one Noul per group, share on true (typed) | .247 / .177 / .358, ρ +.49 | .206 / .206 / .224, ρ +.15 |
| one Noul per group, share on true (multilingual) | .198 / .223 / .501, ρ +.17 | .208 / .193 / .205, ρ +.06 |
| one Score per group, share on true (typed) | .206 / .198 / .201, ρ +.13 | .202 / .200 / .209, ρ +.38 |
| one Score per group, share on true (multilingual) | .195 / .214 / .227, ρ −.12 | .204 / .192 / .195, ρ −.05 |
| Choice over document groups, the literal lineup (multilingual, 8,192 tokens) | .151 / .273 / .217, ρ +.04 | .309 / .157 / .149, ρ −.01 |
| documents as state, Choice over labels (typed) | .300 / .127 / .579, ρ +.82 | .201 / .182 / .248, ρ +.58 |
| documents as state, Choice over labels (multilingual) | .371 / .125 / .573, ρ +.60 | .216 / .138 / .207, ρ +.27 |

The last format compares the candidate with its four nearest neighbours' labels. Before a second
pilot, a bar was fixed for both corpora: gold minus shuffled at least 0.10, and gimme at least 0.50.
No variant passed on both.

| variant of documents-as-state (gold−shuffled / gimme) | 20NG | arXiv |
|---|---|---|
| typed, each document then averaged | .153 / .542, ρ +.84 | .075 / .342, ρ +.67 |
| **English, each document then averaged** | **.224 / .625, ρ +.80** | **.134 / .471, ρ +.71** |
| English, whole group, 512 tokens | .266 / .642, ρ +.79 | .079 / .320, ρ +.61 |
| English, whole group, 1,024 tokens | .244 / .644, ρ +.81 | .071 / .311, ρ +.62 |

The bold row came closest and went to the full replay despite missing the bar, because the replay
is the question being asked. A small check on 25 clusters per corpus then compared neighbour names
at stock length against neighbour names at the candidate's own rung. Stock distractors tracked
Sonnet's slope of score on log words most closely, so they were used, chosen on length bias and not
on agreement with Sonnet.

## Full replay

Every ladder unit ran with three option orders, matching the Sonnet protocol, and every k=5 battery
unit once: 6,215 units, 58 minutes on an Apple M3, no failures. Lineup replay reproduced every saved
battery and ladder lineup exactly.

| check | 20NG | arXiv |
|---|---|---|
| gold / shuffled / gimme | .371 / .118 / .635 | .290 / .134 / .463 |
| fixed bar | pass | **fail** (gimme) |
| pooled ρ with Sonnet across kinds | +.800 | +.744 |
| within-cluster ρ with Sonnet | +.062 (p = .17) | −.041 (p = .25) |
| within-cluster ρ, one Sonnet sample against its other two | +.342 | +.416 |
| precision of "safe to pick" / base rate | .416 / .403 | .603 / .613 |
| repeat band, Laya / Sonnet | .066 / .047 | .036 / .079 |

Padding behaves as it does for Sonnet's lineup and more so: verbose scores at or above gold (.377 vs
.371, .312 vs .290).

**Selection, rated by Sonnet's saved lineup scores.** Sonnet did not choose these picks, so their
scores carry no winner's curse. The documents are the same ones the ladder used, though, so this is
not a fresh-draw, fresh-listener held-out test. Sonnet's own rule is left out because its 0.619 and
0.621 are in-sample and not comparable.

| method | 20NG score / words | arXiv score / words |
|---|---|---|
| stock | .556 / 8.0 | .576 / 8.5 |
| Laya, shortest within its band | .536 / 3.6 | .569 / 5.1 |
| Laya, highest score | .539 / 5.2 | .576 / 6.6 |
| fixed rung 4 at every layer | .535 / 4.0 | .578 / 3.9 |
| fixed rung 5 at every layer | .536 / 3.3 | .575 / 3.4 |
| always the shortest candidate | .522 / 2.0 | .557 / 2.1 |
| random candidate | .536 / 5.9 | .584 / 6.0 |

Laya's rule against always-shortest is +.014 (p = .08) on 20NG and +.012 (p = .71) on arXiv. No
significance test was run on fixed rung against stock. A fixed rung also repairs the arXiv coarse
layer, from .483 at 2.0 words to .656 at 3.6 for rung 5, though that layer has only 7 clusters.

These comparisons cover the ladder only. The candidates differ mostly in content, as shown above,
so Laya's failure is not specific to length. The fixed-rung comparison is, because a fixed rung can
only stand in for a choice of length.

## What the `embedding_model` object does

A separate question: could Toponymy stop requiring an embedding model object, while still taking
the embeddings the caller passes? It has three jobs on `main`. Stand-ins were compared with today's
MiniLM picks, using mpnet as the reference for "a different real embedder". No model was asked
anything here, so the Jev side of this table is a proposal, not a result.

| job | stand-in | overlap with today | mpnet reference |
|---|---|---|---|
| subtopic selection | layer-0 `centroid_vectors`, no text model | 0.79 to 1.00 at top 4 | 0.89 to 1.00 |
| keyphrase diversification | mean of documents containing the phrase | **0.49 to 0.59** | 0.86 to 0.91 |
| keyphrase diversification | skip the step, take top scores | 0.64 to 0.73 | same |
| disambiguation trigger | 8 nearest centroid neighbours, then a yes/no | 46 of 46 near pairs covered | n/a |

Document-mean vectors failed as a stand-in, agreeing less than using no vectors at all. Overlap
measures agreement, not quality. At the default `n_subtopics=16` subtopic selection barely matters
here, because only 1 parent has more than 16 children, and that should change on larger corpora.
The trigger row is a proxy: saved names are post-disambiguation, so no pair reached the library's
0.2 cutoff and 0.35 was used. Neighbours test 549 and 629 pairs instead of 3,013 and 3,693.

## Cluster ids are not stable between processes

Refitting arXiv with identical inputs gives two variants of the fit. They number clusters 18 and 19
at layer 1 the opposite way, and they also differ slightly beyond that: cluster 9's last neighbour
changes on a near-tie (similarities of .559 and .558). The battery and ladder ran under one variant.
The #177 held-out stage and the phase 6 feature and subtopic runs ran under the other.

With labels keyed by (layer, index), the held-out stage scored those two clusters' labels against
each other's regions. Without them, held-out chosen and stock move from .4475 and .4361 to .4517 and
.4390, so the conclusion stands. Candidates from the feature and subtopic runs are relabelled to
ladder numbering here, which moved the pooled gain against stock from +.051 to +.045. Phase 6's
arXiv comparisons against gold labels may have mispaired the same two clusters and were not
recomputed. 20NG reproduced in every fit tried. The cause was not investigated.

## Caveats

- The scoring results test Laya, not Jev. Jev may do better, and a Jev run should use documents as
  state with a Choice over labels. It needs a key and is estimated at under a dollar. A pass would
  mean within-cluster agreement near 0.34 to 0.42 and picks that beat stock on fresh documents.
- If gains are partly document-specific, scoring on more documents may help, and that is where a
  cheap scorer could add the most. Five documents was a cost choice, and more were not tried.
- Laya ran zero-shot with its shipped checkpoints. Its docs say fine-tuning helps, and that was not
  tried. Laya version 0.4.18; checkpoints `laya` at `7b928d8`, `laya-typed-decisions` at `e929ae5`,
  `laya-multilingual` at `1720e3e`.
- The format was chosen on 192 pilot units, so battery numbers for that format are mildly selected.
  Ladder numbers are not.
- Two 7,000-document corpora, three layers, one namer. Sonnet's within-cluster signal is itself
  noisy, so the 0.34 and 0.42 are a reference point, not a ceiling.

## Reproduce

Needs the local embedding caches under `experiments/`, which are untracked. Run from the repo root.

```bash
D=experiments/label_quality/decision_models
FROZEN=$(mktemp -d); git archive a49257c toponymy | tar -x -C "$FROZEN"
PYTHONPATH="$FROZEN:$D" .venv/bin/python $D/dump_lineups2.py $D
for c in 20ng arxiv_home; do
  git show experiment/wayfinding-lineup:experiments/label_quality/data/ladder_names_$c.json > $D/ladder_$c.json
done
.venv/bin/python $D/embedder_jobs.py $D

LAYA=$(mktemp -d); uv venv --python 3.12 "$LAYA" && uv pip install --python "$LAYA/bin/python" laya==0.4.18
"$LAYA/bin/python" -I $D/laya_pilot.py $D
"$LAYA/bin/python" -I $D/laya_pilot2.py $D
"$LAYA/bin/python" -I $D/laya_replay.py $D design
"$LAYA/bin/python" -I $D/laya_replay.py $D full stock
.venv/bin/python $D/laya_analyze.py $D

.venv/bin/python $D/selection_value.py > $D/selection_value_report.txt
PYTHONPATH="$FROZEN:$D" .venv/bin/python $D/fresh_prep.py $D
.venv/bin/python $D/fresh_run.py $D   # paid, about 3.3M input tokens, hard cap 5.0M
.venv/bin/python $D/fresh_analyze.py $D
```

`laya_analyze.py` and `fresh_analyze.py` regenerate the committed summaries exactly from the
committed results. `dump_lineups.py` supplies helpers to `dump_lineups2.py`; running it directly
fails whenever the arXiv fit lands on the other variant. `fresh_prep.py` stops for the same reason,
so rerun it until it passes. Rebuilt lineup files and `fresh_units.json` contain corpus text and are
gitignored. `fresh_pairs.json` is the same data without documents.
