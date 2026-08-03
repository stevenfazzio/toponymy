"""
Phase 7 tranche 3 -- does renaming actually REPAIR a confusable pair?

Tranche 2 measured detection: a free geometric score finds confusable label pairs that Toponymy's
0.2-cosine-cap trigger misses, confirmed on a held-out listener. It did NOT show that flagging them
helps. The library question is narrower than "can we write a better prompt": *if we widen the
trigger, does the repair machinery already in the package fix what it now catches?* So this uses
Toponymy's OWN `distinguish_topic_names_prompt` + `generate_topic_cluster_names`, invoked on the
flagged pair directly -- exactly what a widened trigger would hand it -- with nothing re-named that
the trigger would not have re-named.

Two axes, because there is a live prior in both directions:
  - identification: the pairwise arbiter again (fresh docs, held-out listener, chance 0.50)
  - fit:            the grounded judge, paired against the committed gold ratings
#173's nibling-contrast result (contrast context in the naming prompt made labels WORSE by judge,
~2:1 across 4 models) and #177's fit != identification split together predict the interesting
outcome: identification improves and fit regresses. That would make widening the trigger a
trade-off rather than a free win, which is worth knowing before proposing it.

Control: a plain redraw is NOT re-run here -- 7c already measured it (no identification headroom
above the winner's-curse floor on either corpus), so any movement here is attributable to the
contrastive prompt rather than to drawing again.

  uv run python experiments/label_quality/repair_check.py --stage rename
  uv run python experiments/label_quality/repair_check.py --stage arbiter
  uv run python experiments/label_quality/repair_check.py --stage judge
  uv run python experiments/label_quality/repair_check.py --stage report
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
DATA = HERE / "data"
sys.path.insert(0, str(REPO / "experiments" / "nibling_contrast"))
sys.path.insert(0, str(HERE))

DATASETS = ["20ng", "arxiv_home"]
N_PAIRS = 15                  # most-confusable arbiter-confirmed pairs per corpus
ARBITER_MODEL = "openai/gpt-4o-mini"
JUDGE_MODEL = "anthropic/claude-sonnet-4-6"
NAMER = "haiku"               # the model that produced the gold labels
N_DOCS, MAXLEN, K_SAMPLES, TEMP = 5, 500, 3, 0.7
SALT = "repair-arbiter-v1"    # fresh again: not tranche 2's draw, not wayfinding's frozen one


def battery_file(ds):
    return DATA / ("battery_20ng.json" if ds == "20ng" else "battery_arxiv_home.json")


def gold_names_by_layer(ds, counts):
    bat = json.loads(battery_file(ds).read_text())
    names = [["Unlabelled"] * n for n in counts]
    for it in bat:
        names[it["layer"]][it["idx"]] = it["gold"]
    return names


def pick_pairs(ds):
    """The most confusable arbiter-CONFIRMED flagged pairs (lowest measured pairwise pm)."""
    arb = json.loads((DATA / f"confusability_arbiter_{ds}.json").read_text())["units"]
    flag = [u for u in arb.values() if u["arm"] == "flag" and u.get("pm") is not None]
    flag.sort(key=lambda u: u["pm"])
    return flag[:N_PAIRS]


# --------------------------------------------------------------------------- rename

def build_layer_state(ds):
    """Populate exemplars / keyphrases / subtopics / topic_names WITHOUT re-naming anything,
    so the disambiguation prompt is byte-for-byte what fit() would have built for these names."""
    from ab_harness import make_embedder
    from toponymy.keyphrases import KeyphraseBuilder
    from wayfinding import Cell

    cell = Cell(ds)
    embedder = make_embedder(cell.meta["emb_model"])
    names = gold_names_by_layer(ds, cell.counts)

    print(f"  building keyphrase matrix over {len(cell.objects)} objects ...", flush=True)
    kb = KeyphraseBuilder()
    matrix, klist, kvecs = kb.fit_transform(list(cell.objects))
    if kvecs is None:
        kvecs = embedder.encode(klist, show_progress_bar=False)

    for L, layer in enumerate(cell.layers):
        layer.topic_names = list(names[L])
        layer.make_keyphrases(klist, matrix, kvecs, embedder)
    cell.layers[0].embed_topic_names(embedder)
    for L, layer in enumerate(cell.layers):
        if L > 0:
            layer.make_subtopics(names[0], cell.layers[0].cluster_labels,
                                 cell.layers[0].topic_name_embeddings, embedder)
    return cell, names


def stage_rename():
    from ab_harness import make_namer
    from toponymy.prompt_construction import distinguish_topic_names_prompt
    from toponymy.templates import SUMMARY_KINDS

    out = {}
    for ds in DATASETS:
        print(f"\n[{ds}] replaying cell + layer state (0 naming calls) ...", flush=True)
        cell, names = build_layer_state(ds)
        pairs = pick_pairs(ds)
        n_layers = len(cell.layers)
        detail_levels = np.linspace(0.0, 1.0, n_layers)   # Toponymy defaults
        namer = make_namer(NAMER)
        print(f"[{ds}] renaming {len(pairs)} flagged pairs = {len(pairs)} LLM calls")

        rows = []
        for u in pairs:
            L, i, j = u["L"], u["true"], u["other"]
            layer = cell.layers[L]
            rung = int(round(detail_levels[L] * (len(SUMMARY_KINDS) - 1)))
            prompt = distinguish_topic_names_prompt(
                np.array([i, j]), L, names,
                exemplar_texts=layer.exemplars, keyphrases=layer.keyphrases,
                subtopics=layer.subtopics if len(getattr(layer, "subtopics", [])) > 0 else None,
                cluster_tree=cell.tree,
                object_description=cell.meta["obj"], corpus_description=cell.meta["corpus"],
                summary_kind=SUMMARY_KINDS[rung],
                max_num_exemplars=layer.n_exemplars, max_num_keyphrases=layer.n_keyphrases,
                max_num_subtopics=layer.n_subtopics,
                exemplar_start_delimiter=layer.exemplar_delimiters[0],
                exemplar_end_delimiter=layer.exemplar_delimiters[1],
                prompt_format=layer.prompt_format, prompt_template=layer.prompt_template)
            old = [names[L][i], names[L][j]]
            try:
                new = namer.generate_topic_cluster_names(prompt, old, temperature=0.4)
            except Exception as e:
                print(f"    rename err L{L} ({i},{j}): {type(e).__name__} {str(e)[:60]}")
                new = None
            ok = isinstance(new, list) and len(new) == 2 and all(isinstance(x, str) for x in new)
            if not ok:
                # the #57 failure shape: the pass silently keeps the old names
                print(f"    L{L} ({i},{j}): renaming returned {new!r} -> keeping old names")
                new = old
            rows.append(dict(L=L, i=i, j=j, old_i=old[0], old_j=old[1],
                             new_i=new[0], new_j=new[1], renamed=ok,
                             changed=(new[0] != old[0] or new[1] != old[1]),
                             arbiter_pm_old=u["pm"], uid=u["uid"]))
            print(f"    L{L} c{i}->c{j} pm {u['pm']:.2f}\n"
                  f"       old: {old[0][:66]!r}\n            {old[1][:66]!r}\n"
                  f"       new: {new[0][:66]!r}\n            {new[1][:66]!r}")
        (DATA / f"repair_names_{ds}.json").write_text(json.dumps(rows, indent=1))
        out[ds] = rows
        ch = sum(r["changed"] for r in rows)
        print(f"[{ds}] {ch}/{len(rows)} pairs actually changed name")


# --------------------------------------------------------------------------- arbiter

def fresh_docs(cell, L, i):
    from wayfinding import md5i

    members = np.where(cell.layers[L].cluster_labels == i)[0]
    ex = set(map(int, cell.layers[L].exemplar_indices[i]))
    pool = np.array([m for m in members if int(m) not in ex])
    if pool.size == 0:
        pool = members
    rng = np.random.default_rng(md5i(f"{SALT}|{L}|{i}"))
    take = rng.choice(pool, size=min(N_DOCS, pool.size), replace=False)
    return [" ".join(str(cell.objects[j]).split())[:MAXLEN] for j in take]


async def run_pair(u, cell, sem):
    import litellm
    from wayfinding import LETTERS, PROMPT, RESPONSE_FORMAT, md5i, parse_scores

    cids = u["lineup"]
    groups = {j: fresh_docs(cell, u["L"], j) for j in cids}

    async def one(s):
        rng = np.random.default_rng(md5i(f"{u['uid']}|s{s}"))
        disp = [cids[j] for j in rng.permutation(len(cids))]
        blocks = "\n\n".join(f"Candidate {LETTERS[a]}:\n" + "\n".join(f"- {d}" for d in groups[j])
                             for a, j in enumerate(disp))
        prompt = PROMPT.format(label=u["label"], k=len(cids),
                               span=f"{LETTERS[0]}-{LETTERS[len(cids)-1]}", blocks=blocks)
        async with sem:
            try:
                r = await litellm.acompletion(model=ARBITER_MODEL, temperature=TEMP, max_tokens=300,
                                              num_retries=5, response_format=RESPONSE_FORMAT,
                                              messages=[{"role": "user", "content": prompt}])
                sc = parse_scores(r.choices[0].message.content or "", len(cids))
            except Exception as e:
                print("  arbiter err:", type(e).__name__, str(e)[:70], flush=True)
                sc = None
        return (disp, sc) if sc else None

    got = [x for x in await asyncio.gather(*[one(s) for s in range(K_SAMPLES)]) if x]
    pms = [sc[disp.index(u["true"])] / (sum(sc) or 1) for disp, sc in got]
    return dict(u, n_valid=len(got), pm=float(np.mean(pms)) if pms else None)


def stage_arbiter(concurrency):
    from wayfinding import Cell

    for ds in DATASETS:
        rows = json.loads((DATA / f"repair_names_{ds}.json").read_text())
        cell = Cell(ds)
        units = []
        for r in rows:
            # both members of the pair, both label versions, on the SAME fresh pairwise lineup
            for side, true, other in (("i", r["i"], r["j"]), ("j", r["j"], r["i"])):
                for ver in ("old", "new"):
                    units.append(dict(
                        uid=f"rep|{ds}|L{r['L']}|{r['i']}_{r['j']}|{side}|{ver}",
                        L=r["L"], true=true, other=other, side=side, ver=ver,
                        pair=f"{r['i']}_{r['j']}", changed=r["changed"],
                        label=r[f"{ver}_{side}"], lineup=[true, other]))
        path = DATA / f"repair_arbiter_{ds}.json"
        done = {}
        if path.exists():
            done = {k: v for k, v in json.loads(path.read_text())["units"].items()
                    if v.get("pm") is not None}
            units = [u for u in units if u["uid"] not in done]
        print(f"[{ds}] arbiter: {len(units)} units x {K_SAMPLES} = {len(units)*K_SAMPLES} calls")
        if units:
            for r in asyncio.run(_gather(units, cell, concurrency)):
                done[r["uid"]] = r
        path.write_text(json.dumps({"model": ARBITER_MODEL, "salt": SALT, "units": done}))
        print(f"  wrote {path.name}: {len(done)} units")


async def _gather(units, cell, concurrency):
    sem = asyncio.Semaphore(concurrency)
    return await asyncio.gather(*[run_pair(u, cell, sem) for u in units])


# --------------------------------------------------------------------------- judge

def stage_judge(concurrency):
    from ab_harness import load_dataset
    from async_judge import rate_many

    for ds in DATASETS:
        rows = json.loads((DATA / f"repair_names_{ds}.json").read_text())
        if ds == "20ng":
            from judge_fair import sample_docs
            docs_for = lambda L, i: sample_docs("20ng", L, i, n_near=10, n_rand=5, maxlen=280)
            obj = load_dataset("20ng", None)[3]["obj"]
        else:
            from arxiv_naming_features import home_docs
            from wayfinding import Cell
            cell = Cell(ds)
            table = home_docs(cell.clusterer, cell.objects, cell.emb)
            docs_for = lambda L, i: table.get((L, i), [])
            obj = cell.meta["obj"]

        tasks, keys = [], []
        for r in rows:
            if not r["changed"]:
                continue
            for side in ("i", "j"):
                L, i = r["L"], r[side]
                d = docs_for(L, i)
                if not d:
                    continue
                tasks.append((r[f"new_{side}"], d))
                keys.append((L, i, r[f"new_{side}"]))
        print(f"[{ds}] judging {len(tasks)} NEW labels x3 = {len(tasks)*3} calls "
              f"(old labels' fit is already committed)")
        if not tasks:
            continue
        ratings = rate_many(tasks, obj, JUDGE_MODEL, k=3, temp=0.7, concurrency=concurrency)
        out = [dict(L=L, i=i, label=lab, overall=r.get("overall"))
               for (L, i, lab), r in zip(keys, ratings)]
        (DATA / f"repair_judge_{ds}.json").write_text(json.dumps(out, indent=1))
        print(f"  wrote repair_judge_{ds}.json")


# --------------------------------------------------------------------------- report

def old_fit(ds):
    f = "judge_ratings_20ng_sonnet.json" if ds == "20ng" else "judge_ratings_arxiv_home_sonnet.json"
    return {(r["layer"], r["idx"]): r["overall"]
            for r in json.loads((DATA / f).read_text())
            if r.get("type") == "gold" and r.get("overall") is not None}


STOP = {"and", "or", "of", "for", "in", "the", "a", "with", "to", "on", "from", "by", "using"}


def _toks(s):
    import re
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 2}


def vocabulary_shift(rows):
    """Does contrastive renaming name the CONTRAST rather than the region? If so the pair's
    SHARED vocabulary -- often the head noun that makes each label a true description -- should
    be what gets dropped, at roughly constant length."""
    oj, nj, lost, lo, ln = [], [], [], [], []
    for r in rows:
        oi, ojj = _toks(r["old_i"]), _toks(r["old_j"])
        ni, njj = _toks(r["new_i"]), _toks(r["new_j"])
        oj.append(len(oi & ojj) / max(len(oi | ojj), 1))
        nj.append(len(ni & njj) / max(len(ni | njj), 1))
        shared = oi & ojj
        if shared:
            lost.append(1 - len(shared & (ni | njj)) / len(shared))
        lo.append((len(r["old_i"].split()) + len(r["old_j"].split())) / 2)
        ln.append((len(r["new_i"].split()) + len(r["new_j"].split())) / 2)
    print(f"  MECHANISM -- within-pair token Jaccard {np.mean(oj):.3f} -> {np.mean(nj):.3f}; "
          f"{np.mean(lost):.0%} of the pair's shared vocabulary dropped from BOTH new names; "
          f"length {np.mean(lo):.1f} -> {np.mean(ln):.1f} words (unchanged)")


def stage_report():
    from scipy.stats import wilcoxon

    for ds in DATASETS:
        p = DATA / f"repair_arbiter_{ds}.json"
        if not p.exists():
            print(f"[{ds}] no arbiter file yet")
            continue
        units = [u for u in json.loads(p.read_text())["units"].values() if u.get("pm") is not None]
        rows = json.loads((DATA / f"repair_names_{ds}.json").read_text())
        changed = {f"{r['i']}_{r['j']}" for r in rows if r["changed"]}
        by = defaultdict(dict)
        for u in units:
            by[(u["pair"], u["side"])][u["ver"]] = u["pm"]
        pairs = [(k, v) for k, v in by.items() if "old" in v and "new" in v and k[0] in changed]
        old = [v["old"] for _, v in pairs]
        new = [v["new"] for _, v in pairs]
        print(f"\n[{ds}] IDENTIFICATION -- pairwise, fresh docs, {ARBITER_MODEL}, chance 0.50")
        print(f"  n={len(pairs)} cluster-sides from {len(changed)} renamed pairs")
        print(f"  old labels {np.mean(old):.3f}   ->   new labels {np.mean(new):.3f}   "
              f"delta {np.mean(new)-np.mean(old):+.3f}")
        if len(pairs) >= 6:
            print(f"  Wilcoxon p = {wilcoxon(new, old).pvalue:.3g};  "
                  f"below chance: old {sum(1 for x in old if x<0.5)}/{len(old)} -> "
                  f"new {sum(1 for x in new if x<0.5)}/{len(new)}")
            print(f"  improved on {sum(1 for a,b in zip(old,new) if b>a)}/{len(pairs)} sides")

        jf = DATA / f"repair_judge_{ds}.json"
        if jf.exists():
            of = old_fit(ds)
            jn = json.loads(jf.read_text())
            a = [(of[(r["L"], r["i"])], r["overall"]) for r in jn
                 if r.get("overall") is not None and (r["L"], r["i"]) in of]
            if a:
                o = [x for x, _ in a]
                n = [y for _, y in a]
                print(f"  FIT -- grounded judge, paired against committed gold ratings (n={len(a)})")
                print(f"  old {np.mean(o):.3f}   ->   new {np.mean(n):.3f}   "
                      f"delta {np.mean(n)-np.mean(o):+.3f} judge-pts"
                      + (f";  Wilcoxon p = {wilcoxon(n, o).pvalue:.3g}" if len(a) >= 6 else ""))
                print(f"  (judge repeat band from tranche 2: sigma = 0.168)")
        vocabulary_shift(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["rename", "arbiter", "judge", "report"])
    ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()
    {"rename": stage_rename, "arbiter": lambda: stage_arbiter(a.concurrency),
     "judge": lambda: stage_judge(a.concurrency), "report": stage_report}[a.stage]()


if __name__ == "__main__":
    main()
