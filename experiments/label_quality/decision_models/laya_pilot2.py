"""
Second pilot: variants of the one format with signal (documents as state, Choice over labels).

Same stratified sample as the first pilot. Bar fixed before running, on BOTH corpora:
gold minus shuffled >= 0.10, and gimme >= 0.50. Local inference only; argv[1] is the scratchpad.
"""
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SP = Path(sys.argv[1])
sys.path.insert(0, str(SP))

import laya  # noqa: E402

from laya_pilot import DEVICE, group_state, log, md5i, spearman  # noqa: E402

REPOS = {
    "typed": "convaiinnovations/laya-typed-decisions",
    "english": "convaiinnovations/laya",
}


def question(u, stock, order_seed=0):
    rng = random.Random(md5i(f"{u['uid']}|o{order_seed}"))
    names = {"c0": u["label"]}
    for n, j in enumerate(u["lineup"][1:], 1):
        s = stock[(u["L"], j)]
        if s != u["label"]:
            names[f"c{n}"] = s
    keys = list(names)
    rng.shuffle(keys)
    return {"q": {"type": "choice", "instructions": "Which label best names this group of documents?",
                  "criteria": {k: names[k] for k in keys}}}


def whole_group(agent, units, groups, stock, **kw):
    out = {}
    for u in units:
        r = agent.predict(group_state(groups[f"{u['L']}|{u['true']}"]), question(u, stock), **kw)
        out[u["uid"]] = float(r["answers"]["q"]["probabilities"]["c0"])
    return out


def per_document(agent, units, groups, stock, **kw):
    out = {}
    for u in units:
        q = question(u, stock)
        q["q"]["instructions"] = "Which label best names the topic of this document?"
        docs = groups[f"{u['L']}|{u['true']}"]
        res = agent.predict_batch(docs, q, batch_size=8, **kw)
        out[u["uid"]] = float(np.mean([r["answers"]["q"]["probabilities"]["c0"] for r in res]))
    return out


def main():
    data = {c: json.loads((SP / f"lineups_{c}.json").read_text()) for c in ("20ng", "arxiv_home")}
    sample = {}
    for c, d in data.items():
        by = defaultdict(list)
        for u in d["units"]["battery"]:
            if u["k"] == 5 and u["run"] == 1 and u["pm"] is not None:
                by[u["kind"]].append(u)
        rng = random.Random(7)
        sample[c] = [u for kind in sorted(by) for u in rng.sample(by[kind], min(12, len(by[kind])))]
    agents = {}
    for name, repo in REPOS.items():
        t0 = time.time()
        agents[name] = laya.load(repo, device=DEVICE)
        log(f"loaded {repo} in {time.time() - t0:.0f}s")
    variants = {
        "typed/per_document": lambda *a: per_document(agents["typed"], *a),
        "english/per_document": lambda *a: per_document(agents["english"], *a),
        "english/whole_group_512": lambda *a: whole_group(agents["english"], *a),
        "english/whole_group_1024": lambda *a: whole_group(agents["english"], *a, max_len=1024),
    }
    results = {}
    for name, fn in variants.items():
        ok = True
        for c, d in data.items():
            stock = {(r["L"], r["i"]): r["stock"] for r in d["rows"]}
            t0 = time.time()
            try:
                got = fn(sample[c], d["groups"], stock)
            except Exception as e:
                log(f"{name} {c} FAILED: {type(e).__name__}: {str(e)[:300]}")
                ok = False
                continue
            units = sample[c]
            ours = np.array([got[u["uid"]] for u in units])
            sonnet = np.array([u["pm"] for u in units])
            by = defaultdict(list)
            for u, p in zip(units, ours):
                by[u["kind"]].append(p)
            line = {k: round(float(np.mean(v)), 3) for k, v in sorted(by.items())}
            gap = line["gold"] - line["shuffled"]
            passed = gap >= 0.10 and line["gimme"] >= 0.50
            ok = ok and passed
            log(f"{name:26s} {c:10s} rho={spearman(ours, sonnet):+.3f} gold-shuf={gap:+.3f} gimme={line['gimme']:.3f} "
                f"{'PASS' if passed else 'fail'} {(time.time() - t0) / len(units):.2f}s/unit {line}")
            results[f"{name}|{c}"] = dict(by_kind=line, gap=gap, passed=passed)
        log(f"{name}: {'PASSES BAR ON BOTH' if ok else 'does not pass on both'}")
        (SP / "laya_pilot2_results.json").write_text(json.dumps(results))


if __name__ == "__main__":
    main()
