"""
Score the Laya replay against the saved Sonnet listener. Reads only; argv[1] is the scratchpad.

Selection test: each method picks one name per cluster. Picks are then rated by Sonnet's saved
lineup score, which is independent of Laya's noise, so Laya's picks carry no winner's-curse
inflation. Sonnet's own pick is in-sample and flagged as such.
"""
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

REPO = Path("/Users/stevenfazzio/repos/toponymy")
SP = Path(sys.argv[1])
BRANCH = "experiment/wayfinding-lineup:experiments/label_quality/data/"
ORDER = ["gimme", "gold", "verbose", "ancestor", "generic", "sibling", "shuffled", "distant"]


def saved(name):
    return json.loads(subprocess.run(["git", "show", BRANCH + name], cwd=REPO, capture_output=True, text=True, check=True).stdout)


def words(s):
    return len(s.split())


def within_rho(rows):
    """rows: dict cluster -> list of (a, b). Mean per-cluster Spearman."""
    out = []
    for r in rows.values():
        r = np.array(r, float)
        if len(r) >= 4 and r[:, 0].std() > 0 and r[:, 1].std() > 0:
            out.append(spearmanr(r[:, 0], r[:, 1])[0])
    return float(np.mean(out)), len(out), (float(wilcoxon(out).pvalue) if len(out) >= 8 else float("nan"))


def analyze(cell):
    print(f"\n========== {cell} ==========")
    d = json.loads((SP / f"lineups_{cell}.json").read_text())
    ladder_names = json.loads((SP / f"ladder_{cell}.json").read_text())
    got = defaultdict(dict)
    for line in (SP / f"laya_replay_{cell}_stock.jsonl").read_text().splitlines():
        r = json.loads(line)
        got[r["uid"]][r["sample"]] = r["v"]
    print(f"replayed units {len(got)}")

    # ---- gates on the full battery
    by, pairs = defaultdict(list), []
    for u in d["units"]["battery"]:
        if u["k"] == 5 and u["run"] == 1 and u["uid"] in got and u["pm"] is not None:
            v = got[u["uid"]][0]
            by[u["kind"]].append((v, u["pm"]))
            pairs.append((v, u["pm"]))
    print("kind       n    laya   sonnet")
    for k in ORDER:
        if k in by:
            a = np.array(by[k])
            print(f"{k:9s} {len(a):4d}  {a[:, 0].mean():.3f}  {a[:, 1].mean():.3f}")
    gold, shuf, gim = (np.mean([x[0] for x in by[k]]) for k in ("gold", "shuffled", "gimme"))
    print(f"bar: gold-shuffled {gold - shuf:+.3f} (need >=0.10), gimme {gim:.3f} (need >=0.50) -> "
          f"{'PASS' if gold - shuf >= 0.10 and gim >= 0.50 else 'FAIL'}")
    p = np.array(pairs)
    print(f"pooled rho vs Sonnet across all kinds {spearmanr(p[:, 0], p[:, 1])[0]:+.3f} (n={len(p)})")

    # ---- band, exactly as the Sonnet protocol: p90 |d| over the repeat subset
    units = d["units"]["ladder"]
    mean3 = lambda u: float(np.mean([got[u["uid"]][s] for s in range(3)]))
    r1 = {u["uid"][:-1]: mean3(u) for u in units if u["run"] == 1 and len(got.get(u["uid"], {})) == 3}
    r2 = {u["uid"][:-1]: mean3(u) for u in units if u["run"] == 2 and len(got.get(u["uid"], {})) == 3}
    rep = [abs(r1[k] - r2[k]) for k in r2 if k in r1]
    band = float(np.percentile(rep, 90))
    print(f"laya repeat band p90 {band:.3f} (n={len(rep)}); sonnet band {d['band']:.3f}")

    # ---- does Laya agree with Sonnet among candidates for one cluster? with a noise ceiling
    sonnet_units = saved(f"wayfinding_{cell}_ladder.json")["units"]
    cands, ceiling = defaultdict(list), defaultdict(list)
    for u in units:
        if u["run"] != 1 or u["pm"] is None or len(got.get(u["uid"], {})) != 3:
            continue
        cands[(u["L"], u["i"])].append(dict(label=u["label"], sonnet=u["pm"], laya=mean3(u)))
        s = sonnet_units[u["uid"]]
        fracs = []
        for smp in s["samples"]:
            total = sum(smp["scores"]) or 1
            fracs.append(smp["scores"][smp["order"].index(u["true"])] / total)
        if len(fracs) == 3:
            ceiling[(u["L"], u["i"])].append((fracs[0], np.mean(fracs[1:])))
    agree = {k: [(c["laya"], c["sonnet"]) for c in v] for k, v in cands.items()}
    rho, n, pv = within_rho(agree)
    crho, cn, _ = within_rho(ceiling)
    print(f"within-cluster rho, laya vs sonnet {rho:+.3f} (n={n}, p={pv:.3g})")
    print(f"within-cluster rho, sonnet sample vs its other two samples {crho:+.3f} (n={cn}) <- noise reference")
    for L in sorted({k[0] for k in cands}):
        r = within_rho({k: v for k, v in agree.items() if k[0] == L})
        print(f"  L{L}: {r[0]:+.3f} (n={r[1]})")

    # ---- can Laya tell which candidates are safe to choose?
    tp = fp = fn = tn = 0
    for rows in cands.values():
        sb, lb = max(c["sonnet"] for c in rows), max(c["laya"] for c in rows)
        for c in rows:
            s_ok, l_ok = c["sonnet"] >= sb - d["band"], c["laya"] >= lb - band
            tp += s_ok and l_ok
            fp += (not s_ok) and l_ok
            fn += s_ok and (not l_ok)
            tn += (not s_ok) and (not l_ok)
    prec, base = tp / max(tp + fp, 1), (tp + fn) / (tp + fp + fn + tn)
    print(f"'safe' agreement: precision {prec:.3f} vs base rate {base:.3f} (tp {tp} fp {fp} fn {fn} tn {tn})")

    # ---- selection: rate every method's picks with Sonnet
    idx = {(L, i, name): r for r, layers in ladder_names.items() for L, names in enumerate(layers) for i, name in enumerate(names)}
    rows = {(r["L"], r["i"]): r for r in d["rows"]}
    rng = np.random.default_rng(0)
    picks = defaultdict(dict)
    for key, opts in cands.items():
        L, i = key
        best = max(c["laya"] for c in opts)
        ok = [c for c in opts if c["laya"] >= best - band]
        picks["laya rule"][key] = min(ok, key=lambda c: (words(c["label"]), -c["laya"]))
        picks["laya max"][key] = max(opts, key=lambda c: c["laya"])
        best_s = max(c["sonnet"] for c in opts)
        ok_s = [c for c in opts if c["sonnet"] >= best_s - d["band"]]
        picks["sonnet rule (in-sample)"][key] = min(ok_s, key=lambda c: (words(c["label"]), -c["sonnet"]))
        by_label = {c["label"]: c for c in opts}
        stock = rows[key]["stock"]
        if stock in by_label:
            picks["stock"][key] = by_label[stock]
        picks["always shortest"][key] = min(opts, key=lambda c: (words(c["label"]), c["label"]))
        short = sorted(opts, key=lambda c: (words(c["label"]), c["label"]))
        picks["random candidate"][key] = short[int(rng.integers(len(short)))]
    for r in range(7):
        for key, opts in cands.items():
            name = ladder_names[str(r)][key[0]][key[1]]
            hit = [c for c in opts if c["label"] == name]
            if hit:
                picks[f"fixed rung {r}"][key] = hit[0]
    print("\nmethod                     n   sonnet-rated   words | by layer sonnet/words")
    layers = sorted({k[0] for k in cands})
    report = {}
    for name in ["stock", "sonnet rule (in-sample)", "laya rule", "laya max", "always shortest", "random candidate"] + [f"fixed rung {r}" for r in range(7)]:
        p = picks[name]
        pm, w = np.mean([c["sonnet"] for c in p.values()]), np.mean([words(c["label"]) for c in p.values()])
        per = " ".join(
            f"L{L} {np.mean([c['sonnet'] for k, c in p.items() if k[0] == L]):.3f}/{np.mean([words(c['label']) for k, c in p.items() if k[0] == L]):.1f}"
            for L in layers
        )
        print(f"{name:25s} {len(p):3d}   {pm:.3f}        {w:5.1f} | {per}")
        report[name] = dict(n=len(p), sonnet=float(pm), words=float(w))
    a, b = picks["laya rule"], picks["always shortest"]
    common = sorted(set(a) & set(b))
    delta = np.array([a[k]["sonnet"] - b[k]["sonnet"] for k in common])
    nz = delta[delta != 0]
    print(f"laya rule minus always-shortest on Sonnet score: {delta.mean():+.3f} over {len(delta)} clusters, "
          f"{len(nz)} differ, wilcoxon p={wilcoxon(nz).pvalue:.3g}")
    a = picks["laya rule"]
    sample = sorted(common, key=lambda k: -abs(a[k]["sonnet"] - b[k]["sonnet"]))[:4]
    for k in sample:
        print(f"  {k}: laya '{a[k]['label']}' {a[k]['sonnet']:.2f} vs shortest '{b[k]['label']}' {b[k]['sonnet']:.2f}")
    return dict(band=band, within=rho, ceiling=crho, gold=float(gold), shuffled=float(shuf), gimme=float(gim), methods=report)


if __name__ == "__main__":
    results = {c: analyze(c) for c in ("20ng", "arxiv_home")}
    (SP / "laya_replay_summary.json").write_text(json.dumps(results, indent=2))
