# Can a decision model do per-cluster label selection?

> Not shared anywhere yet (October 2026). Companion to [WAYFINDING.md](WAYFINDING.md), whose length
> controller this builds on, and [FEATURES.md](FEATURES.md). Scripts and results are in
> [decision_models/](decision_models/).

## TL;DR

The idea: generate several candidate labels per cluster, score each with a cheap decision model
such as Jev using a wayfinding lineup, and keep the shortest label that still identifies its
region. That is the #177 length controller with a much cheaper listener. Cost was the main reason
#177 stopped short of proposing it as a default.

This replays the frozen lineups through **Laya**, an open Jev-like model, on local hardware.
**Jev itself was not tested.** No API calls were made.

1. **Laya sorts bad labels from good but cannot rank one cluster's candidates.** Agreement with
   Sonnet across label kinds is 0.80 (20NG) and 0.74 (arXiv). Among a single cluster's candidates
   it is +0.06 and −0.04, where a Sonnet sample agrees with its own other samples at 0.34 and 0.42.
2. **Its picks did no better than a fixed rung.** Rung 5 at every layer matched or beat Laya's
   selections on both score and length, on both corpora.
3. **The bar for any per-cluster scorer is therefore a fixed terse rung, not stock.** Rung 5 is
   about 60% shorter than stock, and Sonnet rates it 0.020 lower on 20NG and 0.001 lower on arXiv.
4. **Question format matters more than checkpoint.** The literal lineup and per-group questions sat
   near chance. Only documents as state with a Choice over labels carried signal.
5. **Cluster ids are not stable between processes** (see below), which touched 2 of 111 clusters in
   the posted arXiv held-out result without changing its conclusion.

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

Refitting arXiv with identical inputs gave two numberings: clusters 18 and 19 at layer 1 swap ids.
The battery and ladder ran under one numbering and the #177 held-out stage under the other, with
labels keyed by (layer, index), so those two clusters' labels were scored against each other's
regions. Without them, held-out chosen and stock move from .4475 and .4361 to .4517 and .4390, so
the conclusion stands. 20NG reproduced exactly. The cause was not investigated.

## Caveats

- This tests Laya, not Jev. Jev may do better, and a Jev run should use documents as state with a
  Choice over labels. It needs a key and is estimated at under a dollar. A pass would mean
  within-cluster agreement near 0.34 to 0.42 and picks that beat a fixed rung.
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
```

`laya_analyze.py` regenerates the committed `laya_replay_summary.json` exactly from the committed
JSONL. `dump_lineups.py` supplies helpers to `dump_lineups2.py`; running it directly fails whenever
the arXiv numbering flips. The rebuilt lineup files contain corpus text and are gitignored.
