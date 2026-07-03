"""
Phase 4 -- the wayfinding lineup (discriminative label eval).

A label's deployment function on a map is identification: a reader must find the right region
among its neighbours given only the name. One lineup per (layer, cluster, candidate-label):
the listener sees the label + k candidate document groups (the true cluster + its k-1 nearest
same-layer clusters by high-D centroid cosine), each shown as N HELD-OUT member docs (sampled
excluding the exemplars the namer saw, so parroting exemplar phrasing can't win), identities
hidden. The listener scores every candidate 0-100 (atlantic-mirror lineup_eval protocol);
prob-mass on the true cluster = score_true / sum. Per cluster the distractor set + doc sample
are frozen (seeded once), so different labels for the same cluster face IDENTICAL lineups and
comparisons are paired. Candidate order reshuffles per self-consistency sample (k=3), so
position bias is marginalized rather than frozen in.

Stages (instrument checks BEFORE use, per PLAN kill criteria):
  check    -- no LLM: replay fit, verify determinism + lineup/doc construction, print examples
  smoke    -- a few units on haiku, print raw listener responses
  floors   -- haiku: shuffled-label floor (expect ~chance), gimme trials (distant distractors,
              expect ~ceiling), repeat floor (gold run twice -> |d pm| band), k-sweep {3,5,7}
  battery  -- sonnet: all 107 gold + 488 variants through frozen k=5 nn lineups (gates a+b)
  report   -- recompute the analysis/report from saved unit files (no LLM)

  uv run python experiments/label_quality/wayfinding.py --stage check
  uv run python experiments/label_quality/wayfinding.py --stage smoke
  uv run python experiments/label_quality/wayfinding.py --stage floors
  uv run python experiments/label_quality/wayfinding.py --stage battery
Writes data/wayfinding_20ng_<stage>.json (resumable; completed units are skipped on re-run).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
NIBLING = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(NIBLING))
sys.path.insert(0, str(HERE))

MODELS = {"haiku": "anthropic/claude-haiku-4-5-20251001", "sonnet": "anthropic/claude-sonnet-4-6"}
K_DEFAULT, N_DOCS, MAXLEN, K_SAMPLES, TEMP = 5, 5, 500, 3, 0.7
LETTERS = "ABCDEFG"

PROMPT = """You will identify which group of documents a LABEL names.

LABEL: "{label}"

Below are {k} candidate document groups ({span}). Each is shown as a few member documents \
(truncated). Exactly one of these groups is the one the label was written to name.

{blocks}

For EACH candidate group, rate from 0 to 100 how well the LABEL fits that group as a name, \
judging only from the documents shown. Be discriminating: reserve high scores for a group the \
label names genuinely and specifically; if the label fits several groups only in a generic way, \
give those groups similar middling scores.

Give your ratings as {k} integers in candidate order ({span})."""

SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {"scores": {"type": "array", "items": {"type": "integer"}}},
          "required": ["scores"]}
RESPONSE_FORMAT = {"type": "json_schema",
                   "json_schema": {"name": "scores", "schema": SCHEMA, "strict": True}}


def md5i(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def unit_rows(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


class Cell:
    """Replayed canonical fit (same recipe as metrics.py) + frozen lineup ingredients."""

    def __init__(self, dataset: str = "20ng"):
        from perturbations import load_fit

        cl, objects, emb, coords, meta = load_fit(dataset, None, 25, 4)
        for layer in cl.cluster_layers_:
            layer.make_exemplar_texts(objects, emb)  # deterministic; what the namer saw
        self.dataset, self.objects, self.tree = dataset, objects, cl.cluster_tree_
        self.emb = emb.astype(np.float64)
        self.layers = cl.cluster_layers_
        self.counts = [int(l.centroid_vectors.shape[0]) for l in cl.cluster_layers_]
        self.cent = [unit_rows(l.centroid_vectors.astype(np.float64)) for l in cl.cluster_layers_]
        self._docs, self._held_fallback = {}, 0

    def held_out(self, L: int, i: int) -> list[str]:
        """Frozen N-doc sample of cluster (L,i) members EXCLUDING the namer's exemplars."""
        if (L, i) in self._docs:
            return self._docs[(L, i)]
        members = np.where(self.layers[L].cluster_labels == i)[0]
        ex = set(map(int, self.layers[L].exemplar_indices[i]))
        pool = np.array([m for m in members if int(m) not in ex])
        if pool.size == 0:  # tiny cluster fully covered by exemplars: fall back, but count it
            pool, self._held_fallback = members, self._held_fallback + 1
        rng = np.random.default_rng(md5i(f"wayfind-docs|{L}|{i}"))
        take = rng.choice(pool, size=min(N_DOCS, pool.size), replace=False)
        docs = [" ".join(str(self.objects[j]).split())[:MAXLEN] for j in take]
        self._docs[(L, i)] = docs
        return docs

    def lineup(self, L: int, i: int, k: int, mode: str) -> list[int]:
        """True cluster + k-1 same-layer distractors; deterministic. mode nn = nearest centroids
        (hard), gimme = farthest (listener sanity check). A layer with < k clusters uses the
        whole layer. Returned order is canonical [true, d1, ...]; display order shuffles later."""
        n = self.counts[L]
        kk = min(k, n)
        sims = self.cent[L] @ self.cent[L][i]
        order = np.argsort(-sims)  # self first (sim 1.0)
        others = [int(j) for j in order if j != i]
        picks = others[: kk - 1] if mode == "nn" else others[::-1][: kk - 1]
        return [i] + picks


def parse_scores(text: str, k: int) -> list[int] | None:
    try:
        obj = json.loads(text)
        sc = obj["scores"] if isinstance(obj, dict) else obj
    except Exception:
        m = re.search(r"\[[\d\s,]+\]", text or "")
        if not m:
            return None
        try:
            sc = json.loads(m.group(0))
        except Exception:
            return None
    try:
        sc = [max(0, min(100, int(x))) for x in sc]
    except Exception:
        return None
    return (sc + [0] * k)[:k]


async def run_unit(u: dict, cell: Cell, model: str, sem: asyncio.Semaphore) -> dict:
    """One (lineup, label) unit: K_SAMPLES listener calls with reshuffled candidate order."""
    import litellm

    cids = u["lineup"]
    k = len(cids)
    groups = {j: cell.held_out(u["L"], j) for j in cids}

    async def one(s: int):
        rng = np.random.default_rng(md5i(f"{u['uid']}|s{s}"))
        disp = [cids[j] for j in rng.permutation(k)]
        blocks = "\n\n".join(
            f"Candidate {LETTERS[a]}:\n" + "\n".join(f"- {d}" for d in groups[j])
            for a, j in enumerate(disp))
        span = f"{LETTERS[0]}-{LETTERS[k-1]}"
        prompt = PROMPT.format(label=u["label"], k=k, span=span, blocks=blocks)
        async with sem:
            try:
                # structured output: without it sonnet deliberates per-candidate in prose and
                # hits max_tokens before any array appears (claude-sonnet-4-6 rejects prefill)
                r = await litellm.acompletion(model=model, temperature=TEMP, max_tokens=300,
                                              num_retries=5, response_format=RESPONSE_FORMAT,
                                              messages=[{"role": "user", "content": prompt}])
                sc = parse_scores(r.choices[0].message.content or "", k)
            except Exception as e:
                print("  listener err:", type(e).__name__, str(e)[:70], flush=True)
                sc = None
        return (disp, sc) if sc else None

    samples = [s for s in await asyncio.gather(*[one(s) for s in range(K_SAMPLES)]) if s]
    pms, top1s, mass = [], [], defaultdict(list)
    for disp, sc in samples:
        tot = sum(sc) or 1
        tpos = disp.index(u["true"])
        pms.append(sc[tpos] / tot)
        top1s.append(1.0 if sum(x > sc[tpos] for x in sc) == 0 else 0.0)
        for a, j in enumerate(disp):
            mass[j].append(sc[a] / tot)
    out = dict(u, k=k, n_valid=len(samples),
               pm=float(np.mean(pms)) if pms else None,
               top1=float(np.mean(top1s)) if top1s else None,
               mass={str(j): float(np.mean(v)) for j, v in mass.items()},
               samples=[{"order": d, "scores": s} for d, s in samples])
    return out


async def run_all(units: list[dict], cell: Cell, model: str, concurrency: int, path: Path):
    done = {}
    if path.exists():
        cached = json.loads(path.read_text()).get("units", {})
        done = {k: v for k, v in cached.items() if v.get("pm") is not None}  # retry failed units
        units = [u for u in units if u["uid"] not in done]
        print(f"  resume: {len(done)} cached, {len(units)} to run")
    if not units:
        return done
    sem = asyncio.Semaphore(concurrency)
    lock, n_done = asyncio.Lock(), 0

    def save():
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"model": model, "config": dict(
            n_docs=N_DOCS, maxlen=MAXLEN, k_samples=K_SAMPLES, temp=TEMP), "units": done}))
        tmp.replace(path)

    async def wrapped(u):
        nonlocal n_done
        r = await run_unit(u, cell, model, sem)
        async with lock:
            done[u["uid"]] = r
            n_done += 1
            if n_done % 40 == 0:
                save()
                print(f"  {n_done}/{len(units)}", flush=True)

    await asyncio.gather(*[wrapped(u) for u in units])
    save()
    return done


def make_unit(cell, L, i, kind, label, k, mode, run=1):
    uid = f"{mode}|k{k}|L{L}.{i}|{kind}|{hashlib.md5(label.encode()).hexdigest()[:10]}|r{run}"
    return dict(uid=uid, L=L, i=i, true=i, kind=kind, label=label, mode=mode, run=run,
                lineup=cell.lineup(L, i, k, mode))


def derangement(n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return [int(x) for x in p]


def load_battery():
    return json.loads((HERE / "data" / "battery_20ng.json").read_text())


def gold_by_cluster(battery):
    return {(it["layer"], it["idx"]): it["gold"] for it in battery}


# ---------------------------------------------------------------- stage unit builders

def units_floors(cell, battery):
    gold = gold_by_cluster(battery)
    units = []
    for (L, i), g in gold.items():                      # gold @ k5 nn = repeat run 1 + baseline
        units.append(make_unit(cell, L, i, "gold", g, K_DEFAULT, "nn", run=1))
        units.append(make_unit(cell, L, i, "gold", g, K_DEFAULT, "nn", run=2))   # repeat floor
        units.append(make_unit(cell, L, i, "gold", g, K_DEFAULT, "gimme", run=1))
        units.append(make_unit(cell, L, i, "gold", g, 3, "nn", run=1))           # k sweep
        units.append(make_unit(cell, L, i, "gold", g, 7, "nn", run=1))
    for L in range(len(cell.counts)):                   # shuffled-label floor (within layer)
        perm = derangement(cell.counts[L], seed=7000 + L)
        for i in range(cell.counts[L]):
            lab = gold.get((L, perm[i]))
            if lab:
                units.append(make_unit(cell, L, i, "shuffled", lab, K_DEFAULT, "nn", run=1))
    return units


def units_battery(cell, battery):
    """595 battery units + same-listener shuffled/gimme catch trials (so gate (a)'s floor
    comparison doesn't cross listeners)."""
    units = []
    gold = gold_by_cluster(battery)
    for it in battery:
        L, i = it["layer"], it["idx"]
        units.append(make_unit(cell, L, i, "gold", it["gold"], K_DEFAULT, "nn", run=1))
        units.append(make_unit(cell, L, i, "gimme", it["gold"], K_DEFAULT, "gimme", run=1))
        for kind, lab in it["variants"].items():
            units.append(make_unit(cell, L, i, kind, lab, K_DEFAULT, "nn", run=1))
    for L in range(len(cell.counts)):
        perm = derangement(cell.counts[L], seed=7000 + L)
        for i in range(cell.counts[L]):
            lab = gold.get((L, perm[i]))
            if lab:
                units.append(make_unit(cell, L, i, "shuffled", lab, K_DEFAULT, "nn", run=1))
    return units


# ---------------------------------------------------------------- reports

def _sel(units, **kw):
    return [u for u in units.values()
            if all(u.get(f) == v for f, v in kw.items()) and u.get("pm") is not None]


def report_floors(units, cell):
    print("\n================ FLOORS (haiku) ================")
    for L in range(len(cell.counts)):
        k = min(K_DEFAULT, cell.counts[L])
        g = _sel(units, kind="gold", mode="nn", run=1, L=L, k=K_DEFAULT)
        s = _sel(units, kind="shuffled", mode="nn", run=1, L=L)
        gi = _sel(units, kind="gold", mode="gimme", run=1, L=L)
        pm = lambda xs: np.mean([u["pm"] for u in xs]) if xs else float("nan")
        t1 = lambda xs: np.mean([u["top1"] for u in xs]) if xs else float("nan")
        print(f"L{L} (n={cell.counts[L]}, chance={1/k:.2f}):  "
              f"gold pm {pm(g):.3f} top1 {t1(g):.0%}  |  shuffled pm {pm(s):.3f} top1 {t1(s):.0%}"
              f"  |  gimme pm {pm(gi):.3f} top1 {t1(gi):.0%}")
    r1 = {u["uid"].rsplit("|r", 1)[0]: u for u in _sel(units, kind="gold", mode="nn", run=1, k=K_DEFAULT)}
    r2 = {u["uid"].rsplit("|r", 1)[0]: u for u in _sel(units, kind="gold", mode="nn", run=2, k=K_DEFAULT)}
    dd = [abs(r1[x]["pm"] - r2[x]["pm"]) for x in r1 if x in r2]
    if dd:
        print(f"repeat floor (gold k5, {len(dd)} labels): mean |d pm| {np.mean(dd):.3f}, "
              f"p90 {np.percentile(dd, 90):.3f}, max {np.max(dd):.3f}")
    for kk in (3, K_DEFAULT, 7):
        g = _sel(units, kind="gold", mode="nn", run=1, k=kk) or \
            [u for u in _sel(units, kind="gold", mode="nn", run=1) if u["k"] == kk]
        if g:
            L0 = [u for u in g if u["L"] == 0]
            print(f"k={kk}: gold pm {np.mean([u['pm'] for u in g]):.3f} "
                  f"top1 {np.mean([u['top1'] for u in g]):.0%} (chance {1/kk:.2f})"
                  f"   [L0 only: pm {np.mean([u['pm'] for u in L0]):.3f} "
                  f"top1 {np.mean([u['top1'] for u in L0]):.0%}]")


def sibling_id(cell, battery_item, gold):
    """Recover which cluster the sibling variant's label belongs to (same layer, prefer same parent)."""
    L, i, lab = battery_item["layer"], battery_item["idx"], battery_item["variants"].get("sibling")
    if lab is None:
        return None
    cands = [j for j in range(cell.counts[L]) if j != i and gold.get((L, j)) == lab]
    if len(cands) > 1:
        parents = {c: p for p, kids in cell.tree.items() for (_, c) in kids if p[0] == L + 1}
        # tree maps (layer+1, parent) -> list of (layer, child); invert for this layer
        kidmap = {}
        for p, kids in cell.tree.items():
            for (lc, c) in kids:
                if lc == L:
                    kidmap[c] = p
        cands = sorted(cands, key=lambda j: 0 if kidmap.get(j) == kidmap.get(i) else 1)
    return cands[0] if cands else None


def report_battery(units, cell, battery):
    print("\n================ BATTERY (gate a) ================")
    gold = gold_by_cluster(battery)
    kinds = ["gold", "ancestor", "sibling", "distant", "generic", "verbose", "shuffled", "gimme"]
    print(f"{'kind':<9} {'n':>4} {'pm':>6} {'top1':>6} {'<chance':>8} {'H(mass)':>8}")
    for kind in kinds:
        xs = _sel(units, kind=kind, mode="gimme" if kind == "gimme" else "nn")
        if not xs:
            continue
        ent = [(-sum(p * np.log(p) for p in u["mass"].values() if p > 0) / np.log(u["k"])) for u in xs]
        below = np.mean([u["pm"] < 1 / u["k"] for u in xs])
        print(f"{kind:<9} {len(xs):>4} {np.mean([u['pm'] for u in xs]):>6.3f} "
              f"{np.mean([u['top1'] for u in xs]):>6.0%} {below:>8.0%} {np.mean(ent):>8.3f}")
    # paired gold-vs-variant on identical lineups
    gmap = {(u["L"], u["i"]): u for u in _sel(units, kind="gold", mode="nn")}
    print("\npaired: gold beats variant on prob-mass (chance 0.50)")
    for kind in kinds[1:]:
        wins = [(gmap[(u['L'], u['i'])]["pm"] > u["pm"])
                for u in _sel(units, kind=kind, mode="nn") if (u["L"], u["i"]) in gmap]
        if wins:
            print(f"  gold > {kind:<9} {np.mean(wins):.0%}  (n={len(wins)})")
    # sibling signature: where did the mass go?
    sib_pm, sib_mass, in_lineup = [], [], 0
    for it in battery:
        sj = sibling_id(cell, it, gold)
        if sj is None:
            continue
        u = next((u for u in _sel(units, kind="sibling", mode="nn")
                  if u["L"] == it["layer"] and u["i"] == it["idx"]), None)
        if u is None:
            continue
        if sj in u["lineup"]:
            in_lineup += 1
            sib_pm.append(u["pm"])
            sib_mass.append(u["mass"].get(str(sj), 0.0))
    if sib_pm:
        print(f"\nsibling signature ({in_lineup} lineups contain the sibling): "
              f"mass on true {np.mean(sib_pm):.3f} vs mass on sibling {np.mean(sib_mass):.3f}")
    # gate (b): correlation with the grounded judge over judged battery candidates
    jr = json.loads((HERE / "data" / "judge_ratings_20ng_sonnet.json").read_text())
    jmap = {(r["layer"], r["idx"], r["type"]): r["overall"] for r in jr if r["overall"] is not None}
    pairs = [(u["pm"], jmap[(u["L"], u["i"], u["kind"])]) for u in _sel(units, mode="nn")
             if (u["L"], u["i"], u["kind"]) in jmap]
    if pairs:
        from scipy.stats import spearmanr
        x, y = zip(*pairs)
        print(f"\ngate (b): Spearman(lineup pm, judge overall) = {spearmanr(x, y)[0]:+.3f} "
              f"(n={len(pairs)}; expect moderate positive, NOT ~1)")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["check", "smoke", "floors", "battery", "report"])
    ap.add_argument("--dataset", default="20ng")
    ap.add_argument("--listener", default=None, help="haiku|sonnet (default: stage-appropriate)")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--limit", type=int, default=6, help="units for smoke")
    args = ap.parse_args()

    cell = Cell(args.dataset)
    battery = load_battery()
    print(f"[{args.dataset}] replayed fit: clusters/layer {cell.counts} "
          f"(battery expects {dict(Counter(it['layer'] for it in battery))})")
    assert cell.counts == [74, 24, 9], "canonical replay mismatch -- do not proceed"

    if args.stage == "check":
        gold = gold_by_cluster(battery)
        for (L, i) in [(0, 0), (1, 3), (2, 5)]:
            lu = cell.lineup(L, i, K_DEFAULT, "nn")
            docs = cell.held_out(L, i)
            ex = set(map(int, cell.layers[L].exemplar_indices[i]))
            print(f"\nL{L}#{i} gold: {gold[(L, i)]!r}")
            print(f"  lineup (nn): {lu}   gimme: {cell.lineup(L, i, K_DEFAULT, 'gimme')}")
            print(f"  exemplars excluded: {len(ex)}; held-out docs: {len(docs)}; "
                  f"first doc: {docs[0][:120]!r}")
        d2 = Cell(args.dataset)
        same = all(d2.lineup(0, i, 5, "nn") == cell.lineup(0, i, 5, "nn") for i in range(10)) \
            and d2.held_out(0, 0) == cell.held_out(0, 0)
        print(f"\ndeterminism across replays: {'OK' if same else 'FAIL'}")
        if cell._held_fallback:
            print(f"held-out fallback used for {cell._held_fallback} clusters")
        return

    model = MODELS[args.listener or ("sonnet" if args.stage == "battery" else "haiku")]
    out = HERE / "data" / f"wayfinding_{args.dataset}_{args.stage}.json"

    if args.stage == "smoke":
        gold = gold_by_cluster(battery)
        us = [make_unit(cell, L, i, "gold", gold[(L, i)], K_DEFAULT, "nn")
              for (L, i) in [(0, 0), (0, 28), (1, 3), (2, 5)]][: args.limit]
        us.append(make_unit(cell, 0, 0, "generic", "Various topics", K_DEFAULT, "nn"))
        units = asyncio.run(run_all(us, cell, model, args.concurrency, out))
        for u in units.values():
            if u["pm"] is None:
                print(f"\n{u['uid']}\n  FAILED (n_valid=0)")
                continue
            print(f"\n{u['uid']}\n  pm={u['pm']:.3f} top1={u['top1']:.0%} n_valid={u['n_valid']} "
                  f"mass={ {k: round(v, 2) for k, v in u['mass'].items()} }")
        return

    if args.stage == "floors":
        us = units_floors(cell, battery)
        print(f"floors: {len(us)} units x {K_SAMPLES} samples on {model}")
        units = asyncio.run(run_all(us, cell, model, args.concurrency, out))
        report_floors(units, cell)
        return

    if args.stage == "battery":
        us = units_battery(cell, battery)
        print(f"battery: {len(us)} units x {K_SAMPLES} samples on {model}")
        units = asyncio.run(run_all(us, cell, model, args.concurrency, out))
        report_battery(units, cell, battery)
        return

    if args.stage == "report":
        for stage, rep in [("floors", report_floors), ("battery", report_battery)]:
            p = HERE / "data" / f"wayfinding_{args.dataset}_{stage}.json"
            if p.exists():
                units = json.loads(p.read_text())["units"]
                rep(units, cell) if stage == "floors" else rep(units, cell, battery)


if __name__ == "__main__":
    main()
