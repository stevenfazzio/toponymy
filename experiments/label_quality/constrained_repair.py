"""
Phase 7 tranche 3c -- does a two-sentence constraint on the disambiguation prompt fix it?

Tranches 3 and 3b found the same mechanism from two independent selection rules: disambiguation
deletes ~41% of the vocabulary a colliding pair shared, including the head nouns that made each
name true of its own region, costing 0.54-1.05 judge-points. The cause is not an emergent LLM
quirk -- it is what the prompt asks for. `templates.py` ends the disambiguation instruction with:

    "The primary goal is to make each new topic name clearly distinguishable from the others in
     this list, based on the provided details."

Distinguishability is declared the PRIMARY goal, with nothing counterbalancing it about remaining a
true description of the region. This tests the minimal patch: append two sentences that supply the
counterweight, change nothing else, and re-measure.

Run on tranche 3's pairs, so the stock arm is already measured and the comparison is three-way and
paired -- and, unlike 3b, on BOTH corpora.

Three things must hold for the patch to count as a fix:
  1. fit recovers toward the pre-disambiguation names
  2. de-duplication is still achieved -- the pair's names must stay outside the library's own 0.2
     cosine-distance trigger cap, or the "fix" is just "don't disambiguate"
  3. identification does not regress below the stock-disambiguated arm

  uv run python experiments/label_quality/constrained_repair.py --stage rename
  uv run python experiments/label_quality/constrained_repair.py --stage judge
  uv run python experiments/label_quality/constrained_repair.py --stage arbiter
  uv run python experiments/label_quality/constrained_repair.py --stage report
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
DATA = HERE / "data"
sys.path.insert(0, str(REPO / "experiments" / "nibling_contrast"))
sys.path.insert(0, str(HERE))

from repair_check import (ARBITER_MODEL, DATASETS, JUDGE_MODEL, K_SAMPLES,  # noqa: E402
                          _gather, _toks, build_layer_state, old_fit)

# the exact sentence the stock template ends its disambiguation instruction with
STOCK_SENTENCE = ("The primary goal is to make each new topic name clearly distinguishable from "
                  "the others in this list, based on the provided details.")
# the proposed patch: the same sentence plus a counterweight. Nothing else changes.
CONSTRAINT = (" Each new name must also remain an accurate, self-contained description of its own "
              "topic: distinguish the topics by adding what separates them, not by removing terms "
              "that are essential to what a topic is about, even when several topics share those "
              "terms.")


def patch(prompt):
    """Append the constraint to the stock prompt. Fails loudly if the template moved."""
    if isinstance(prompt, dict):
        out = dict(prompt)
        for k, v in prompt.items():
            if isinstance(v, str) and STOCK_SENTENCE in v:
                out[k] = v.replace(STOCK_SENTENCE, STOCK_SENTENCE + CONSTRAINT)
                return out
        raise RuntimeError("stock sentence not found in any prompt field")
    if STOCK_SENTENCE not in prompt:
        raise RuntimeError("stock sentence not found -- templates.py changed?")
    return prompt.replace(STOCK_SENTENCE, STOCK_SENTENCE + CONSTRAINT)


def stage_rename():
    from ab_harness import make_namer
    from toponymy.prompt_construction import distinguish_topic_names_prompt
    from toponymy.templates import SUMMARY_KINDS

    for ds in DATASETS:
        print(f"\n[{ds}] rebuilding layer state (0 naming calls) ...", flush=True)
        cell, names = build_layer_state(ds)
        rows = json.loads((DATA / f"repair_names_{ds}.json").read_text())
        detail_levels = np.linspace(0.0, 1.0, len(cell.layers))
        namer = make_namer("haiku")
        print(f"[{ds}] re-running {len(rows)} pairs with the CONSTRAINED prompt = {len(rows)} calls")

        out = []
        for r in rows:
            L, i, j = r["L"], r["i"], r["j"]
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
            old = [r["old_i"], r["old_j"]]
            try:
                new = namer.generate_topic_cluster_names(patch(prompt), old, temperature=0.4)
            except Exception as e:
                print(f"    err L{L} ({i},{j}): {type(e).__name__} {str(e)[:60]}")
                new = None
            ok = isinstance(new, list) and len(new) == 2 and all(isinstance(x, str) for x in new)
            if not ok:
                new = old
            out.append(dict(r, fix_i=new[0], fix_j=new[1], fixed=ok))
            print(f"    L{L} c{i}/c{j}\n       stock: {r['new_i'][:64]!r}\n"
                  f"              {r['new_j'][:64]!r}\n"
                  f"       fixed: {new[0][:64]!r}\n              {new[1][:64]!r}")
        (DATA / f"constrained_names_{ds}.json").write_text(json.dumps(out, indent=1))


def stage_judge(concurrency):
    from ab_harness import load_dataset
    from async_judge import rate_many

    for ds in DATASETS:
        rows = json.loads((DATA / f"constrained_names_{ds}.json").read_text())
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
            for side in ("i", "j"):
                d = docs_for(r["L"], r[side])
                if d:
                    tasks.append((r[f"fix_{side}"], d))
                    keys.append((r["L"], r[side], r[f"fix_{side}"]))
        print(f"[{ds}] judging {len(tasks)} constrained labels x3 = {len(tasks)*3} calls")
        ratings = rate_many(tasks, obj, JUDGE_MODEL, k=3, temp=0.7, concurrency=concurrency)
        (DATA / f"constrained_judge_{ds}.json").write_text(json.dumps(
            [dict(L=L, i=i, label=lab, overall=r.get("overall"))
             for (L, i, lab), r in zip(keys, ratings)], indent=1))


def stage_arbiter(concurrency):
    from wayfinding import Cell

    for ds in DATASETS:
        rows = json.loads((DATA / f"constrained_names_{ds}.json").read_text())
        cell = Cell(ds)
        units = []
        for r in rows:
            for side, true, other in (("i", r["i"], r["j"]), ("j", r["j"], r["i"])):
                units.append(dict(uid=f"fix|{ds}|L{r['L']}|{r['i']}_{r['j']}|{side}",
                                  L=r["L"], true=true, other=other, side=side, ver="fix",
                                  pair=f"{r['i']}_{r['j']}", label=r[f"fix_{side}"],
                                  lineup=[true, other]))
        path = DATA / f"constrained_arbiter_{ds}.json"
        done = {}
        if path.exists():
            done = {k: v for k, v in json.loads(path.read_text())["units"].items()
                    if v.get("pm") is not None}
            units = [u for u in units if u["uid"] not in done]
        print(f"[{ds}] arbiter: {len(units)} x{K_SAMPLES} = {len(units)*K_SAMPLES} calls")
        if units:
            for r in asyncio.run(_gather(units, cell, concurrency)):
                done[r["uid"]] = r
        path.write_text(json.dumps({"model": ARBITER_MODEL, "units": done}))


def stage_report():
    from ab_harness import make_embedder
    from scipy.stats import wilcoxon

    for ds in DATASETS:
        rows = json.loads((DATA / f"constrained_names_{ds}.json").read_text())
        of = old_fit(ds)
        stock_j = {(r["L"], r["i"]): r["overall"]
                   for r in json.loads((DATA / f"repair_judge_{ds}.json").read_text())
                   if r.get("overall") is not None}
        fix_j = {(r["L"], r["i"]): r["overall"]
                 for r in json.loads((DATA / f"constrained_judge_{ds}.json").read_text())
                 if r.get("overall") is not None}
        arb = json.loads((DATA / f"repair_arbiter_{ds}.json").read_text())["units"]
        fix_arb = json.loads((DATA / f"constrained_arbiter_{ds}.json").read_text())["units"]
        a_old, a_stock = {}, {}
        for u in arb.values():
            if u.get("pm") is not None:
                (a_old if u["ver"] == "old" else a_stock)[(u["pair"], u["side"])] = u["pm"]
        a_fix = {(u["pair"], u["side"]): u["pm"] for u in fix_arb.values()
                 if u.get("pm") is not None}

        print(f"\n=== {ds} ===")
        tri = []
        for r in rows:
            for side in ("i", "j"):
                k = (r["L"], r[side])
                if k in of and k in stock_j and k in fix_j:
                    tri.append((of[k], stock_j[k], fix_j[k]))
        if tri:
            pre, st, fx = ([t[0] for t in tri], [t[1] for t in tri], [t[2] for t in tri])
            print(f"FIT (grounded judge, n={len(tri)} paired)")
            print(f"  pre-disambiguation {np.mean(pre):.3f}   stock-disambiguated {np.mean(st):.3f}"
                  f"   CONSTRAINED {np.mean(fx):.3f}")
            print(f"  stock vs pre  {np.mean(st)-np.mean(pre):+.3f} (p={wilcoxon(st,pre).pvalue:.3g})"
                  f"   |   constrained vs pre {np.mean(fx)-np.mean(pre):+.3f} "
                  f"(p={wilcoxon(fx,pre).pvalue:.3g})")
            print(f"  RECOVERED: constrained - stock = {np.mean(fx)-np.mean(st):+.3f} "
                  f"(p={wilcoxon(fx,st).pvalue:.3g}); "
                  f"{100*(np.mean(fx)-np.mean(st))/max(np.mean(pre)-np.mean(st),1e-9):.0f}% "
                  f"of the damage")
        keys = sorted(set(a_old) & set(a_stock) & set(a_fix))
        if keys:
            o = [a_old[k] for k in keys]
            s = [a_stock[k] for k in keys]
            f = [a_fix[k] for k in keys]
            print(f"IDENTIFICATION (pairwise, chance 0.50, n={len(keys)})")
            print(f"  old {np.mean(o):.3f}   stock-disambiguated {np.mean(s):.3f}   "
                  f"CONSTRAINED {np.mean(f):.3f}   (constrained vs stock "
                  f"{np.mean(f)-np.mean(s):+.3f}, p={wilcoxon(f,s).pvalue:.3g})")

        emb = make_embedder("all-MiniLM-L6-v2")   # both cells embed with this
        def dist(a, b):
            v = emb.encode([a, b], convert_to_numpy=True, show_progress_bar=False)
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
            return float(1 - v[0] @ v[1])
        d_pre = [dist(r["old_i"], r["old_j"]) for r in rows]
        d_st = [dist(r["new_i"], r["new_j"]) for r in rows]
        d_fx = [dist(r["fix_i"], r["fix_j"]) for r in rows]
        print(f"DE-DUPLICATION (library trigger cap = 0.2 cosine distance; must stay ABOVE)")
        print(f"  pre {np.mean(d_pre):.3f} ({sum(1 for x in d_pre if x<0.2)}/{len(d_pre)} inside cap)"
              f"   stock {np.mean(d_st):.3f} ({sum(1 for x in d_st if x<0.2)}/{len(d_st)})"
              f"   CONSTRAINED {np.mean(d_fx):.3f} ({sum(1 for x in d_fx if x<0.2)}/{len(d_fx)})")
        jf = [len(_toks(r["fix_i"]) & _toks(r["fix_j"])) /
              max(len(_toks(r["fix_i"]) | _toks(r["fix_j"])), 1) for r in rows]
        js = [len(_toks(r["new_i"]) & _toks(r["new_j"])) /
              max(len(_toks(r["new_i"]) | _toks(r["new_j"])), 1) for r in rows]
        jp = [len(_toks(r["old_i"]) & _toks(r["old_j"])) /
              max(len(_toks(r["old_i"]) | _toks(r["old_j"])), 1) for r in rows]
        print(f"  token Jaccard  pre {np.mean(jp):.3f}   stock {np.mean(js):.3f}   "
              f"constrained {np.mean(jf):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["rename", "judge", "arbiter", "report"])
    ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()
    {"rename": stage_rename, "judge": lambda: stage_judge(a.concurrency),
     "arbiter": lambda: stage_arbiter(a.concurrency), "report": stage_report}[a.stage]()


if __name__ == "__main__":
    main()
