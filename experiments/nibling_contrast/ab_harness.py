"""
A/B harness: name a fitted cluster hierarchy twice -- baseline vs. +geometric
contrast -- at temperature 0, and report how often / where names change.

Design (Approach M -- minimal, faithful):
  * Fit the clusterer ONCE (deterministic) so both arms name the identical clusters.
  * Run the real Toponymy.fit twice on that fitted clusterer. The ONLY difference
    between arms is a monkeypatch of toponymy.cluster_layer.topic_name_prompt that
    appends a contrast block (nearby-but-separate finer-layer names) for layers >=1.
  * Force temperature 0 in both arms.
  * Layer 0 never gets contrast, so naming it in BOTH arms is a per-model NOISE
    FLOOR: at temp 0 those names should match; whatever differs is residual model
    nondeterminism that the upper-layer change rate must clear to count as real.

Disambiguation (name_topics -> disambiguate_topics) is DISABLED by default (no-op
patch): it isolates the raw contrast effect, removes its nondeterminism from the
floor, and -- because it no longer masks collisions -- lets us measure whether
contrast REDUCES duplicate names (a free, objective signal). Pass --disambig to
keep it on.

Run one (dataset, model) cell:
  uv run python experiments/nibling_contrast/ab_harness.py --dataset 20ng --model gpt4omini --subsample 2000
  uv run python experiments/nibling_contrast/ab_harness.py --dataset arxiv --model haiku
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import zipfile
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import List, Optional

import numpy as np

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(HERE))
import selector as S  # noqa: E402

import toponymy.cluster_layer as cl_mod  # noqa: E402
from toponymy.clustering import ToponymyClusterer  # noqa: E402
from toponymy.cluster_layer import ClusterLayerText  # noqa: E402
from toponymy.llm_wrappers import LLMWrapper  # noqa: E402
from toponymy.toponymy import Toponymy  # noqa: E402

_ORIG_TNP = cl_mod.topic_name_prompt


# --------------------------------------------------------------------------- #
# Patches
# --------------------------------------------------------------------------- #
@contextmanager
def contrast_patch(cluster_layers, k: int, max_dist: Optional[float], block_version: str = "v1"):
    stats = dict(calls=0, skip=0, nocand=0, filtered=0, injected=0,
                 nearest_dists=[], injected_names={}, by_layer={})

    def bump(layer, key):
        stats["by_layer"].setdefault(layer, dict(injected=0, calls=0))
        stats["by_layer"][layer][key] = stats["by_layer"][layer].get(key, 0) + 1

    def patched(topic_index, layer_id, all_topic_names, **kw):
        prompt = _ORIG_TNP(topic_index, layer_id, all_topic_names, **kw)
        if layer_id < 1:
            return prompt
        if isinstance(prompt, str) and prompt.startswith("[!SKIP!]: "):
            stats["skip"] += 1
            return prompt
        tree = kw.get("cluster_tree")
        if tree is None:
            return prompt
        stats["calls"] += 1
        bump(layer_id, "calls")
        raw = S.contrast_set(cluster_layers, tree, layer_id, topic_index, k=k, max_dist=None)
        if not raw:
            stats["nocand"] += 1
            return prompt
        stats["nearest_dists"].append(raw[0][1])
        kept = [(j, d) for (j, d) in raw if (max_dist is None or d <= max_dist)][:k]
        finer = layer_id - 1
        names, seen = [], set()
        for j, _d in kept:
            if j < len(all_topic_names[finer]):
                nm = all_topic_names[finer][j]
                if nm and nm != "Unlabelled" and nm not in seen:
                    names.append(nm)
                    seen.add(nm)
        if not names:
            stats["filtered"] += 1
            return prompt
        prompt = S.append_contrast(prompt, S.contrast_block(names, version=block_version))
        stats["injected"] += 1
        bump(layer_id, "injected")
        stats["injected_names"][f"{layer_id},{topic_index}"] = names
        return prompt

    cl_mod.topic_name_prompt = patched
    try:
        yield stats
    finally:
        cl_mod.topic_name_prompt = _ORIG_TNP


@contextmanager
def force_temperature(namer, temp=0.0):
    # Opus 4.8 rejects temperature=0 ("temperature is deprecated for this model");
    # it only accepts temperature=1 or unset. So callers pass temp=1.0 for Opus.
    saved = {}
    for attr in ("generate_topic_name", "generate_topic_cluster_names"):
        if hasattr(namer, attr):
            orig = getattr(namer, attr)

            def make(orig):
                def wrapped(*a, **kw):
                    kw["temperature"] = temp
                    return orig(*a, **kw)
                return wrapped

            saved[attr] = orig
            setattr(namer, attr, make(orig))
    try:
        yield
    finally:
        for attr, orig in saved.items():
            setattr(namer, attr, orig)


@contextmanager
def disable_disambiguation():
    orig = ClusterLayerText.disambiguate_topics

    def _noop(self, *a, **kw):
        return None

    ClusterLayerText.disambiguate_topics = _noop
    try:
        yield
    finally:
        ClusterLayerText.disambiguate_topics = orig


# --------------------------------------------------------------------------- #
# Data / namer / embedder
# --------------------------------------------------------------------------- #
def load_dataset(name: str, subsample: Optional[int]):
    if name == "arxiv":
        ex = REPO / "examples"
        emb = np.load(ex / "ai_arxiv_vectors.npy").astype(np.float32)
        coords = np.load(ex / "ai_arxiv_coordinates.npz.npy").astype(np.float32)
        import pandas as pd

        with zipfile.ZipFile(ex / "ai_arxiv_papers.zip") as z:
            with z.open("ai_arxiv_papers") as fh:
                papers = pd.read_csv(fh)
        objects = (papers["title"].astype(str) + ". " + papers["abstract"].astype(str)).tolist()
        meta = dict(emb_model="all-mpnet-base-v2", obj="research papers",
                    corpus="a collection of AI and machine learning research papers")
    elif name == "20ng":
        d = HERE / "data"
        emb = np.load(d / "ng_emb.npy").astype(np.float32)
        coords = np.load(d / "ng_coords.npy").astype(np.float32)
        objects = json.loads((d / "ng_texts.json").read_text())
        meta = dict(emb_model="all-MiniLM-L6-v2", obj="newsgroup posts",
                    corpus="a collection of Usenet newsgroup discussions")
    else:
        raise ValueError(name)
    if subsample:
        objects, emb, coords = objects[:subsample], emb[:subsample], coords[:subsample]
    return objects, emb, coords, meta


def make_namer(model_key: str):
    if model_key == "mock":
        from toponymy.llm_wrappers import HuggingFaceNamer

        return HuggingFaceNamer("Qwen/Qwen2.5-0.5B-Instruct", device="cpu")
    from toponymy.llm_wrappers import AnthropicNamer, OpenAINamer

    return {
        "haiku": lambda: AnthropicNamer(model="claude-haiku-4-5-20251001"),
        "sonnet": lambda: AnthropicNamer(model="claude-sonnet-4-6"),
        "opus": lambda: AnthropicNamer(model="claude-opus-4-8"),
        "gpt4omini": lambda: OpenAINamer(model="openai/gpt-4o-mini"),
    }[model_key]()


def make_embedder(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device="cpu")


# --------------------------------------------------------------------------- #
# Arms / comparison
# --------------------------------------------------------------------------- #
def name_once(objects, emb, coords, clusterer, namer, embedder, meta,
              *, use_contrast: bool, k: int, max_dist: Optional[float], disambig: bool,
              block_version: str = "v1", temperature: float = 0.0):
    model = Toponymy(namer, embedder, clusterer=clusterer,
                     object_description=meta["obj"], corpus_description=meta["corpus"], verbose=False)
    stats = None
    disambig_ctx = nullcontext() if disambig else disable_disambiguation()
    with force_temperature(namer, temperature), disambig_ctx:
        if use_contrast:
            with contrast_patch(clusterer.cluster_layers_, k, max_dist, block_version) as stats:
                model.fit(objects, emb, coords)
        else:
            model.fit(objects, emb, coords)
    return copy.deepcopy(model.topic_names_), stats


def _t(s, n: int = 60) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def compare(baseline: List[List[str]], contrast: List[List[str]]):
    rows = []
    for L, (b, c) in enumerate(zip(baseline, contrast)):
        changed = [(i, bn, cn) for i, (bn, cn) in enumerate(zip(b, c)) if bn != cn]
        rows.append(dict(
            layer=L, n=len(b), changed=len(changed),
            rate=(len(changed) / len(b) if b else 0.0),
            uniq_base=len(set(b)), uniq_contrast=len(set(c)),
            dup_base=len(b) - len(set(b)), dup_contrast=len(c) - len(set(c)),
            renames=changed,
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["arxiv", "20ng"], required=True)
    ap.add_argument("--model", choices=["mock", "haiku", "sonnet", "opus", "gpt4omini"], required=True)
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-dist", type=float, default=None)
    ap.add_argument("--base-min-cluster-size", type=int, default=25)
    ap.add_argument("--min-clusters", type=int, default=4)
    ap.add_argument("--disambig", action="store_true", help="keep disambiguation ON (default: off)")
    args = ap.parse_args()

    t0 = time.time()
    objects, emb, coords, meta = load_dataset(args.dataset, args.subsample)
    print(f"[{args.dataset}/{args.model}] {len(objects)} objects, emb={emb.shape}, "
          f"disambig={'on' if args.disambig else 'OFF'}, max_dist={args.max_dist}")

    embedder = make_embedder(meta["emb_model"])
    clusterer = ToponymyClusterer(min_clusters=args.min_clusters,
                                  base_min_cluster_size=args.base_min_cluster_size, verbose=False)
    clusterer.fit_predict(coords, emb, ClusterLayerText)
    counts = [int(l.centroid_vectors.shape[0]) for l in clusterer.cluster_layers_]
    print(f"clusters per layer (finest->coarsest): {counts}")

    namer = make_namer(args.model)
    if not isinstance(namer, LLMWrapper):
        raise SystemExit(f"Harness v1 assumes sync LLMWrapper; got {type(namer).__name__}")

    temp = 1.0 if args.model == "opus" else 0.0  # Opus 4.8 rejects temperature=0
    if temp:
        print(f"(using temperature={temp} for {args.model})")
    common = dict(k=args.k, max_dist=args.max_dist, disambig=args.disambig, temperature=temp)
    print("naming baseline arm A ...")
    baseline, _ = name_once(objects, emb, coords, clusterer, namer, embedder, meta,
                            use_contrast=False, **common)
    print("naming baseline arm B (noise-floor control) ...")
    baseline2, _ = name_once(objects, emb, coords, clusterer, namer, embedder, meta,
                             use_contrast=False, **common)
    print("naming contrast arm ...")
    contrast, stats = name_once(objects, emb, coords, clusterer, namer, embedder, meta,
                                use_contrast=True, **common)

    rows = compare(baseline, contrast)
    floor_rows = compare(baseline, baseline2)
    nd = np.array(stats["nearest_dists"]) if stats["nearest_dists"] else np.array([np.nan])

    print("\n================ RESULT ================")
    print(f"dataset={args.dataset} model={args.model} k={args.k} max_dist={args.max_dist} "
          f"disambig={'on' if args.disambig else 'OFF'}")
    print(f"contrast injection: injected={stats['injected']} calls={stats['calls']} "
          f"skip(sentinel)={stats['skip']} no-candidates={stats['nocand']} filtered={stats['filtered']}")
    print(f"nearest-contrast cos-dist at naming time: "
          f"min/med/max = {np.nanmin(nd):.3f}/{np.nanmedian(nd):.3f}/{np.nanmax(nd):.3f}")
    print("NOISE FLOOR = baseline-vs-baseline change rate (per layer; the bar to clear)")
    print("\n per-layer  [contrast effect vs noise floor]:")
    for r, fr in zip(rows, floor_rows):
        inj = stats["by_layer"].get(r["layer"], {}).get("injected", 0)
        net = (r["rate"] - fr["rate"]) * 100
        print(f"  L{r['layer']}: contrast {r['changed']:>3}/{r['n']:>3}={r['rate']*100:5.1f}%  "
              f"floor {fr['changed']:>3}/{fr['n']:>3}={fr['rate']*100:5.1f}%  "
              f"NET {net:+5.1f}%  | uniq {r['uniq_base']}->{r['uniq_contrast']}  | injected {inj}")

    print("\n--- sample renames (baseline -> contrast | contrast set) ---")
    for r in rows[1:]:
        shown = 0
        for (i, bn, cn) in r["renames"]:
            cs = stats["injected_names"].get(f"{r['layer']},{i}", [])
            print(f"  (L{r['layer']},{i:>3}) {_t(bn)!r} -> {_t(cn)!r}")
            print(f"           vs: {', '.join(_t(x, 28) for x in cs[:5])}")
            shown += 1
            if shown >= 8:
                break

    out = HERE / "data" / f"result_{args.dataset}_{args.model}.json"
    out.write_text(json.dumps(dict(
        dataset=args.dataset, model=args.model, k=args.k, max_dist=args.max_dist,
        disambig=args.disambig, base_min_cluster_size=args.base_min_cluster_size,
        min_clusters=args.min_clusters, subsample=args.subsample, counts=counts,
        floor_rows=floor_rows, rows=rows,
        injection=dict(injected=stats["injected"], calls=stats["calls"], skip=stats["skip"],
                       nocand=stats["nocand"], filtered=stats["filtered"],
                       nearest_dist=dict(min=float(np.nanmin(nd)), med=float(np.nanmedian(nd)),
                                         max=float(np.nanmax(nd)))),
        injected_names=stats["injected_names"], seconds=round(time.time() - t0, 1),
    ), indent=2))
    print(f"\nsaved -> {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
