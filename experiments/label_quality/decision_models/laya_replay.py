"""
Replay saved lineups through Laya's English checkpoint, classifying each held-out document.

  python laya_replay.py <scratchpad> design   length-bias check on a small sample of clusters
  python laya_replay.py <scratchpad> full <stock|rung>   full battery + ladder, resumable

Score for a label = mean P(label) over the true cluster's documents, in a Choice between that
label and its k-1 nearest neighbours' labels. Option order is reseeded per sample, three samples
per unit, matching the Sonnet protocol. Local inference only.
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

from laya_pilot import DEVICE, log, md5i, spearman  # noqa: E402

REPO = "convaiinnovations/laya"
SAMPLES = 3


def load(cell):
    d = json.loads((SP / f"lineups_{cell}.json").read_text())
    ladder = json.loads((SP / f"ladder_{cell}.json").read_text())
    stock = {(r["L"], r["i"]): r["stock"] for r in d["rows"]}
    rung_of = {}
    for r in sorted(ladder, key=int):
        for L, names in enumerate(ladder[r]):
            for i, name in enumerate(names):
                rung_of.setdefault((L, i, name), int(r))
    return d, ladder, stock, rung_of


def distractor(u, j, design, ladder, stock, rung_of):
    if design == "rung":
        r = rung_of.get((u["L"], u["i"], u["label"]))
        if r is not None:
            return ladder[str(r)][u["L"]][j]
    return stock[(u["L"], j)]


def score(agent, u, groups, design, ladder, stock, rung_of, sample):
    names = {"c0": u["label"]}
    for n, j in enumerate(u["lineup"][1:], 1):
        s = distractor(u, j, design, ladder, stock, rung_of)
        if s != u["label"]:
            names[f"c{n}"] = s
    keys = list(names)
    random.Random(md5i(f"{u['uid']}|o{sample}")).shuffle(keys)
    q = {"q": {"type": "choice", "instructions": "Which label best names the topic of this document?",
               "criteria": {k: names[k] for k in keys}}}
    res = agent.predict_batch(groups[f"{u['L']}|{u['true']}"], q, batch_size=8)
    return float(np.mean([r["answers"]["q"]["probabilities"]["c0"] for r in res]))


def demeaned_slope(units, values):
    by = defaultdict(list)
    for u, v in zip(units, values):
        by[(u["L"], u["i"])].append((np.log(len(u["label"].split())), v))
    xs, ys = [], []
    for rows in by.values():
        if len(rows) < 3:
            continue
        a = np.array(rows)
        xs.extend(a[:, 0] - a[:, 0].mean())
        ys.extend(a[:, 1] - a[:, 1].mean())
    xs, ys = np.array(xs), np.array(ys)
    return float((xs * ys).sum() / (xs * xs).sum())


def within_rho(units, a, b):
    by = defaultdict(list)
    for u, x, y in zip(units, a, b):
        by[(u["L"], u["i"])].append((x, y))
    rhos = []
    for rows in by.values():
        r = np.array(rows)
        if len(r) >= 4 and r[:, 0].std() > 0 and r[:, 1].std() > 0:
            rhos.append(spearman(r[:, 0], r[:, 1]))
    return float(np.mean(rhos)), len(rhos)


def design_check():
    agent = laya.load(REPO, device=DEVICE)
    for cell in ("20ng", "arxiv_home"):
        d, ladder, stock, rung_of = load(cell)
        units = [u for u in d["units"]["ladder"] if u["run"] == 1 and u["pm"] is not None]
        clusters = sorted({(u["L"], u["i"]) for u in units})
        pick = set(map(tuple, random.Random(11).sample(clusters, 25)))
        units = [u for u in units if (u["L"], u["i"]) in pick]
        sonnet = [u["pm"] for u in units]
        log(f"{cell}: {len(units)} ladder units, 25 clusters; Sonnet slope per log-word {demeaned_slope(units, sonnet):+.3f}")
        for design in ("stock", "rung"):
            t0 = time.time()
            got = [score(agent, u, d["groups"], design, ladder, stock, rung_of, 0) for u in units]
            rho, n = within_rho(units, got, sonnet)
            log(f"  {design:5s} slope per log-word {demeaned_slope(units, got):+.3f}, mean {np.mean(got):.3f}, "
                f"within-cluster rho vs Sonnet {rho:+.3f} (n={n}), {(time.time() - t0) / len(units):.2f}s/unit")


def full(design):
    agent = laya.load(REPO, device=DEVICE)
    for cell in ("20ng", "arxiv_home"):
        d, ladder, stock, rung_of = load(cell)
        path = SP / f"laya_replay_{cell}_{design}.jsonl"
        done = set()
        if path.exists():
            done = {(r["uid"], r["sample"]) for r in map(json.loads, path.read_text().splitlines())}
        battery = [u for u in d["units"]["battery"] if u["k"] == 5 and u["run"] == 1]
        ladder_units = list(d["units"]["ladder"])
        todo = [(u, s) for u in ladder_units for s in range(SAMPLES)] + [(u, 0) for u in battery]
        todo = sorted((t for t in todo if (t[0]["uid"], t[1]) not in done), key=lambda t: (t[1], t[0]["uid"]))
        log(f"{cell}: {len(done)} done, {len(todo)} to go ({len(ladder_units)} ladder x {SAMPLES}, {len(battery)} battery x 1)")
        t0 = time.time()
        with path.open("a") as fh:
            for n, (u, s) in enumerate(todo, 1):
                try:
                    v = score(agent, u, d["groups"], design, ladder, stock, rung_of, s)
                except Exception as e:
                    log(f"  unit {u['uid']} sample {s} failed: {type(e).__name__}: {str(e)[:160]}")
                    continue
                fh.write(json.dumps(dict(uid=u["uid"], sample=s, v=v)) + "\n")
                if n % 50 == 0:
                    fh.flush()
                if n % 500 == 0:
                    log(f"  {n}/{len(todo)} at {(time.time() - t0) / n:.2f}s each")
        log(f"{cell}: finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    if sys.argv[2] == "design":
        design_check()
    else:
        full(sys.argv[3])
