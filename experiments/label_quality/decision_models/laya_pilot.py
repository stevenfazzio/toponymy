"""
Pilot: which way of asking a Jev-like decision model reproduces a wayfinding lineup?

Runs a stratified sample of battery units from both corpora through several question formats and
compares against the saved Sonnet listener. Local inference only; argv[1] is the scratchpad.

  group_choice   one Choice over the k document groups, the literal lineup (needs a long context)
  group_noul     one Noul per group; score = P(true) / sum P(all groups)
  group_score    one 5-level Score per group; score = true expected level / sum over groups
  label_choice   true group's documents as state, Choice over the candidate plus neighbours' labels
"""
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import laya

SP = Path(sys.argv[1])
DEVICE = "mps"
REPOS = {
    "typed": "convaiinnovations/laya-typed-decisions",
    "multi": "convaiinnovations/laya-multilingual",
}
LETTERS = "ABCDEFG"
PER_KIND = 12


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def md5i(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def ranks(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i : j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def noul_q(label):
    return {"q": {"type": "noul", "instructions": f'Is "{label}" a good, specific name for this group of documents?'}}


def score_q(label):
    return {"q": {"type": "score", "instructions": f'How well does the label "{label}" name this group of documents?',
                  "criteria": ["Unrelated to these documents", "Only loosely related", "Partly fits, or fits only generically",
                               "Fits well", "Names exactly what these documents are about"]}}


def group_state(docs):
    return {"documents": docs}


def run_group_choice(agent, units, groups, name):
    states, orders = [], []
    for u in units:
        rng = np.random.default_rng(md5i(f"{u['uid']}|s0"))
        disp = [u["lineup"][j] for j in rng.permutation(len(u["lineup"]))]
        orders.append(disp)
        states.append({"label": u["label"], "candidate_groups": {LETTERS[a]: groups[f"{u['L']}|{j}"] for a, j in enumerate(disp)}})
    out = {}
    for k in sorted({len(o) for o in orders}):
        idx = [n for n, o in enumerate(orders) if len(o) == k]
        q = {"q": {"type": "choice", "instructions": "Which candidate group of documents does the label name?",
                   "criteria": {LETTERS[a]: f"Candidate group {LETTERS[a]}" for a in range(k)}}}
        res = agent.predict_batch([states[n] for n in idx], q, batch_size=4, max_len=8192)
        for n, r in zip(idx, res):
            p = r["answers"]["q"]["probabilities"]
            out[units[n]["uid"]] = float(p[LETTERS[orders[n].index(units[n]["true"])]])
    return out


def run_group_each(agent, units, groups, make_q, read):
    out = {}
    for u in units:
        ids = u["lineup"]
        res = agent.predict_batch([group_state(groups[f"{u['L']}|{j}"]) for j in ids], make_q(u["label"]), batch_size=8)
        vals = [read(r["answers"]["q"]) for r in res]
        total = sum(vals)
        out[u["uid"]] = float(vals[0] / total) if total > 0 else 1.0 / len(ids)
    return out


def run_label_choice(agent, units, groups, stock):
    out = {}
    for u in units:
        rng = random.Random(md5i(u["uid"]))
        names = {"c0": u["label"]}
        for n, j in enumerate(u["lineup"][1:], 1):
            s = stock[(u["L"], j)]
            if s != u["label"]:
                names[f"c{n}"] = s
        keys = list(names)
        rng.shuffle(keys)
        q = {"q": {"type": "choice", "instructions": "Which label best names this group of documents?",
                   "criteria": {k: names[k] for k in keys}}}
        r = agent.predict(group_state(groups[f"{u['L']}|{u['true']}"]), q)
        out[u["uid"]] = float(r["answers"]["q"]["probabilities"]["c0"])
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
        sample[c] = [u for kind in sorted(by) for u in rng.sample(by[kind], min(PER_KIND, len(by[kind])))]
        log(f"{c}: {len(sample[c])} units across {sorted(by)}")
    agents = {}
    for name, repo in REPOS.items():
        t0 = time.time()
        agents[name] = laya.load(repo, device=DEVICE)
        log(f"loaded {repo} in {time.time() - t0:.0f}s")
    runs = {
        "typed/group_noul": lambda c, g, s, st: run_group_each(agents["typed"], s, g, noul_q, lambda a: a["noul"]),
        "typed/group_score": lambda c, g, s, st: run_group_each(agents["typed"], s, g, score_q, lambda a: a["score"]),
        "typed/label_choice": lambda c, g, s, st: run_label_choice(agents["typed"], s, g, st),
        "multi/group_noul": lambda c, g, s, st: run_group_each(agents["multi"], s, g, noul_q, lambda a: a["noul"]),
        "multi/group_score": lambda c, g, s, st: run_group_each(agents["multi"], s, g, score_q, lambda a: a["score"]),
        "multi/label_choice": lambda c, g, s, st: run_label_choice(agents["multi"], s, g, st),
        "multi/group_choice": lambda c, g, s, st: run_group_choice(agents["multi"], s, g, "multi"),
    }
    results = {}
    for name, fn in runs.items():
        for c, d in data.items():
            stock = {(r["L"], r["i"]): r["stock"] for r in d["rows"]}
            t0 = time.time()
            try:
                got = fn(c, d["groups"], sample[c], stock)
            except Exception as e:
                log(f"{name} {c} FAILED: {type(e).__name__}: {str(e)[:300]}")
                continue
            secs = time.time() - t0
            units = sample[c]
            ours = np.array([got[u["uid"]] for u in units])
            sonnet = np.array([u["pm"] for u in units])
            by = defaultdict(list)
            for u, p in zip(units, ours):
                by[u["kind"]].append(p)
            rho = spearman(ours, sonnet)
            line = {k: round(float(np.mean(v)), 3) for k, v in sorted(by.items())}
            results[f"{name}|{c}"] = dict(rho=rho, by_kind=line, secs=secs, n=len(units), scores={u["uid"]: got[u["uid"]] for u in units})
            log(f"{name:20s} {c:10s} rho={rho:+.3f} {secs / len(units):.2f}s/unit {line}")
            (SP / "laya_pilot_results.json").write_text(json.dumps(results))
    sonnet_line = {}
    for c in data:
        by = defaultdict(list)
        for u in sample[c]:
            by[u["kind"]].append(u["pm"])
        sonnet_line[c] = {k: round(float(np.mean(v)), 3) for k, v in sorted(by.items())}
    log(f"sonnet reference {sonnet_line}")


if __name__ == "__main__":
    main()
