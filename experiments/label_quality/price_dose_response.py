"""
Pricing for the exemplar dose-response arm (no LLM generation; count_tokens only).

The single-feature ablations came out null on identification, but "drop both" is not a runnable
experiment: at layer 0 the naming prompt has only two content channels (subtopics come from
cluster_tree children, which layer 0 has none of), so blanking both leaves a byte-identical
contentless prompt for all 74 clusters. The answerable version sweeps the DOSE instead of the
presence: name layer 0 with k exemplars for k in {1,2,4,16} (k=8 = stock = the battery gold), run
every name through the cluster's frozen lineup, and find the cheapest rung inside the noise band.

This script prices that arm exactly, in the style of the WAYFINDING cost accounting: it builds the
real prompts at every rung and counts their tokens with the Anthropic count_tokens endpoint rather
than estimating. Output tokens are the one estimated quantity (a name is a short JSON object) and
are labelled as such.

  uv run python experiments/label_quality/price_dose_response.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
NIBLING = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(NIBLING))

RUNGS = [1, 2, 4, 16]          # k=8 is stock: its names + lineups are already the battery gold
STOCK_K = 8
K_SAMPLES = 3                   # listener self-consistency, matches wayfinding.py
N_REPEAT = 30                   # repeat subset for the noise band
NAME_OUT_TOKENS = 40            # est: {"topic_name": "...", "topic_specificity": 0.8}

PRICE = {  # $ per 1M tokens
    "haiku": {"in": 1.00, "out": 5.00},    # claude-haiku-4-5
    "sonnet": {"in": 3.00, "out": 15.00},  # claude-sonnet-4-6
}


async def count_many(prompts: list[str], model: str, concurrency: int = 16) -> list[int]:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic()
    sem = asyncio.Semaphore(concurrency)

    async def one(p: str) -> int:
        async with sem:
            r = await client.messages.count_tokens(
                model=model, messages=[{"role": "user", "content": p}])
            return int(r.input_tokens)

    return list(await asyncio.gather(*[one(p) for p in prompts]))


def main():
    from perturbations import load_fit
    from toponymy.prompt_construction import topic_name_prompt
    from wayfinding import Cell, K_DEFAULT, LETTERS, PROMPT, N_DOCS

    print("replaying canonical 20NG fit...")
    cl, objects, emb, coords, meta = load_fit("20ng", None, 25, 4)
    layer = cl.cluster_layers_[0]
    n_clusters = int(layer.centroid_vectors.shape[0])
    print(f"layer 0: {n_clusters} clusters")

    # keyphrases are built ONCE globally and are identical at every rung (they don't depend on
    # n_exemplars) -- but they're a real part of the prompt, so build them for an honest count
    print("building the global keyphrase matrix (this is the slow part)...")
    from ab_harness import make_embedder
    from toponymy.keyphrases import KeyphraseBuilder

    kb = KeyphraseBuilder(verbose=False)
    okm, klist, kvecs = kb.fit_transform(objects)
    if kvecs is None:
        kvecs = make_embedder(meta["emb_model"]).encode(klist, show_progress_bar=False)
    layer.make_exemplar_texts(objects, emb)
    layer.make_keyphrases(klist, okm, kvecs)
    keyphrases = layer.keyphrases
    print(f"  {len(klist):,} keyphrases; per-cluster median "
          f"{int(np.median([len(k) for k in keyphrases]))}\n")

    # ---- naming prompts at each rung (real prompts, real token counts) ----
    print("building naming prompts at each rung and counting tokens (haiku)...")
    naming = {}
    for k in RUNGS + [STOCK_K]:
        layer.n_exemplars = k
        layer.make_exemplar_texts(objects, emb)
        ex = layer.exemplars
        prompts = [
            topic_name_prompt(
                topic_index=i, layer_id=0, all_topic_names=[[""] * n_clusters],
                exemplar_texts=ex, keyphrases=keyphrases,
                subtopics=None, cluster_tree={},
                object_description=meta["obj"], corpus_description=meta["corpus"],
                summary_kind="very specific (8 to 15 word)", prompt_format="combined")
            for i in range(n_clusters)
        ]
        toks = asyncio.run(count_many(prompts, "claude-haiku-4-5"))
        naming[k] = toks
        print(f"  k={k:>2}: mean {np.mean(toks):>7.0f} tok  median {np.median(toks):>7.0f}  "
              f"total {sum(toks):>9,}")

    # ---- listener prompts (frozen lineups, layer 0 only) ----
    print("\nbuilding listener prompts for the frozen layer-0 lineups (sonnet)...")
    cell = Cell("20ng")
    lp = []
    for i in range(n_clusters):
        cids = cell.lineup(0, i, K_DEFAULT, "nn")
        blocks = "\n\n".join(
            f"Candidate {LETTERS[a]}:\n" + "\n".join(f"- {d}" for d in cell.held_out(0, j))
            for a, j in enumerate(cids))
        lp.append(PROMPT.format(label="X" * 60, k=len(cids),
                                span=f"{LETTERS[0]}-{LETTERS[len(cids)-1]}", blocks=blocks))
    ltoks = asyncio.run(count_many(lp, "claude-sonnet-4-6"))
    print(f"  mean {np.mean(ltoks):.0f} tok  median {np.median(ltoks):.0f}  "
          f"(WAYFINDING reported 3,751)")

    # ---- cost ----
    name_in = sum(sum(naming[k]) for k in RUNGS)
    name_calls = n_clusters * len(RUNGS)
    name_out = name_calls * NAME_OUT_TOKENS
    name_cost = name_in / 1e6 * PRICE["haiku"]["in"] + name_out / 1e6 * PRICE["haiku"]["out"]

    # every rung's name goes through its cluster's frozen lineup, x3 samples, + repeat subset
    lineup_units = n_clusters * len(RUNGS) + N_REPEAT
    lineup_calls = lineup_units * K_SAMPLES
    lineup_in = lineup_calls * float(np.mean(ltoks))
    lineup_out = lineup_calls * 30          # {"scores":[..,..,..,..,..]}
    lineup_cost = (lineup_in / 1e6 * PRICE["sonnet"]["in"]
                   + lineup_out / 1e6 * PRICE["sonnet"]["out"])

    print("\n" + "=" * 68)
    print("EXEMPLAR DOSE-RESPONSE ARM — COSTED (20NG, layer 0 only)")
    print("=" * 68)
    print(f"rungs: k = {RUNGS}  (k={STOCK_K} is stock; its names + lineups already exist)")
    print(f"\nnaming (haiku 4.5, $1.00/$5.00 per MTok)")
    print(f"  {name_calls:>6,} calls   {name_in:>10,} in tok   {name_out:>8,} out tok (est)"
          f"   ${name_cost:>6.2f}")
    print(f"lineups (sonnet 4.6, $3.00/$15.00 per MTok)")
    print(f"  {lineup_calls:>6,} calls   {int(lineup_in):>10,} in tok   {lineup_out:>8,} out tok"
          f"   ${lineup_cost:>6.2f}")
    print(f"\n  TOTAL: {name_calls + lineup_calls:,} calls, ${name_cost + lineup_cost:.2f}")
    print(f"\n  for reference, the keyphrase+exemplar lineup arm just run was 708 calls.")
    print(f"  the naming leg is cheap; {lineup_cost/(name_cost+lineup_cost):.0%} of the cost is "
          f"the sonnet listener.")

    # marginal cost of adding arXiv
    print(f"\n  adding the at-home arXiv cell would roughly double this AND require re-running")
    print(f"  the naming ablation there first (cross_cell_summary.json stores aggregates only).")


if __name__ == "__main__":
    main()
