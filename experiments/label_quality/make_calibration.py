"""
Phase 0 -- human-calibration seed for the grounded judge.

Selects ~28 already-judged candidates (stratified across quality tiers, BLINDED), regenerates
the SAME grounding documents the judge saw, and emits a self-contained HTML rating form. You
rate each label 0-4 (overall); we then compute judge-human agreement (weighted kappa + Spearman)
to check the sonnet judge is trustworthy (Krumdick: never trust an un-calibrated judge).

  uv run python experiments/label_quality/make_calibration.py
Then serve the data dir (localhost = secure context, so clipboard works) and open in Chrome:
  python3 -m http.server 8765 --bind 127.0.0.1 --directory experiments/label_quality/data
  -> http://127.0.0.1:8765/calibration.html
Rate every item, click "Copy results", and paste the JSON back to me.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path("/Users/stevenfazzio/repos/toponymy")
HERE = REPO / "experiments" / "label_quality"
NIBLING = REPO / "experiments" / "nibling_contrast"
sys.path.insert(0, str(NIBLING))

from judge_fair import sample_docs  # noqa: E402  (same grounding the judge used)

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Toponymy label calibration</title>
<style>
 :root{--bg:#0f1115;--card:#1a1d24;--ink:#e6e8eb;--muted:#9aa3ad;--acc:#7c83ff;--good:#3fb950;}
 *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
 header{position:sticky;top:0;background:rgba(15,17,21,.97);border-bottom:1px solid #2a2f3a;padding:14px 20px;z-index:5}
 header h1{margin:0 0 4px;font-size:17px}header p{margin:0;color:var(--muted);font-size:13px}
 #bar{height:4px;background:#2a2f3a;border-radius:2px;margin-top:8px;overflow:hidden}#bar>div{height:100%;width:0;background:var(--good);transition:width .2s}
 main{max-width:820px;margin:0 auto;padding:20px}
 .card{background:var(--card);border:1px solid #2a2f3a;border-radius:10px;padding:16px 18px;margin:0 0 18px}.card.done{border-color:#244a2a}
 .lab{font-size:18px;font-weight:600;color:#fff;margin:2px 0 10px}
 .docs{max-height:220px;overflow:auto;border:1px solid #2a2f3a;border-radius:8px;padding:8px 10px;background:#12151b;font-size:13px;color:#c7ccd3}.docs li{margin:0 0 6px}
 .rate{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}
 .rate button{flex:1;min-width:88px;padding:8px 6px;border:1px solid #2a2f3a;background:#12151b;color:var(--ink);border-radius:8px;cursor:pointer;font-size:13px}
 .rate button:hover{border-color:var(--acc)}.rate button.sel{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:600}
 .num{color:var(--muted);font-size:12px}
 #exportbox{position:sticky;bottom:0;background:rgba(15,17,21,.97);border-top:1px solid #2a2f3a;padding:14px 20px}
 #exportbox button{padding:9px 16px;border:0;border-radius:8px;background:var(--acc);color:#fff;font-size:14px;cursor:pointer}#exportbox button:disabled{opacity:.4;cursor:not-allowed}
 textarea{width:100%;height:80px;margin-top:10px;background:#12151b;color:var(--ink);border:1px solid #2a2f3a;border-radius:8px;padding:8px;font-family:ui-monospace,monospace;font-size:12px;display:none}
</style></head><body>
<header><h1>Label-quality calibration — rate each name 0–4</h1>
<p>Read the representative documents, then rate how well the <b>name</b> describes that group, grounded only in the docs.
 <b>0</b>=very poor · <b>1</b>=poor · <b>2</b>=neutral · <b>3</b>=good · <b>4</b>=very good. (Order is shuffled and blinded — intended.)</p>
<div id="bar"><div></div></div></header>
<main id="main"></main>
<div id="exportbox"><button id="copy" disabled>Copy results</button> <span class="num" id="status"></span><textarea id="out" readonly></textarea></div>
<script type="application/json" id="data">__DATA__</script>
<script>
const ITEMS=JSON.parse(document.getElementById('data').textContent),R={};
const main=document.getElementById('main'),bar=document.querySelector('#bar>div'),copyBtn=document.getElementById('copy'),statusEl=document.getElementById('status'),out=document.getElementById('out');
const NAMES=['0 very poor','1 poor','2 neutral','3 good','4 very good'];
function refresh(){const n=Object.keys(R).length,N=ITEMS.length;bar.style.width=(100*n/N)+'%';statusEl.textContent=n+' / '+N+' rated';copyBtn.disabled=n<N;copyBtn.textContent=n<N?('Copy results ('+(N-n)+' left)'):'Copy results';}
ITEMS.forEach((it,k)=>{const card=document.createElement('div');card.className='card';
 const num=document.createElement('div');num.className='num';num.textContent='Item '+(k+1)+' of '+ITEMS.length;
 const lab=document.createElement('div');lab.className='lab';lab.textContent=it.label;
 const docs=document.createElement('ul');docs.className='docs';it.docs.forEach(d=>{const li=document.createElement('li');li.textContent=d;docs.appendChild(li);});
 const rate=document.createElement('div');rate.className='rate';
 for(let s=0;s<=4;s++){const b=document.createElement('button');b.textContent=NAMES[s];b.onclick=()=>{R[it.id]=s;[...rate.children].forEach(x=>x.classList.remove('sel'));b.classList.add('sel');card.classList.add('done');refresh();};rate.appendChild(b);}
 card.append(num,lab,docs,rate);main.appendChild(card);});
copyBtn.onclick=()=>{const j=JSON.stringify(R);out.style.display='block';out.value=j;try{navigator.clipboard.writeText(j);}catch(e){}out.focus();out.select();statusEl.textContent='copied — paste it back to Claude';};
refresh();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="20ng")
    ap.add_argument("--judge", default="sonnet")
    ap.add_argument("--n-gold", type=int, default=8)
    ap.add_argument("--n-each", type=int, default=5, help="per variant type (verbose/ancestor/sibling/generic)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-near", type=int, default=10)
    ap.add_argument("--n-rand", type=int, default=5)
    ap.add_argument("--maxlen", type=int, default=280)
    args = ap.parse_args()

    ratings = json.loads((HERE / "data" / f"judge_ratings_{args.dataset}_{args.judge}.json").read_text())
    by_type = defaultdict(list)
    for r in ratings:
        if r.get("overall") is not None:
            by_type[r["type"]].append(r)

    rng = np.random.default_rng(args.seed)
    plan = {"gold": args.n_gold, "verbose": args.n_each, "ancestor": args.n_each,
            "sibling": args.n_each, "generic": args.n_each}
    chosen = []
    for t, n in plan.items():
        pool = by_type.get(t, [])
        take = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        chosen += [pool[i] for i in take]
    rng.shuffle(chosen)

    items, key = [], {}
    for r in chosen:
        iid = f"{r['layer']}_{r['idx']}_{r['type']}"
        docs = sample_docs(args.dataset, r["layer"], r["idx"],
                           n_near=args.n_near, n_rand=args.n_rand, maxlen=args.maxlen)
        items.append({"id": iid, "label": r["label"], "docs": docs})
        key[iid] = {"type": r["type"], "judge_overall": r["overall"],
                    "layer": r["layer"], "idx": r["idx"]}

    data_json = json.dumps(items).replace("</", "<\\/")  # safe inside <script type=application/json>
    (HERE / "data" / "calibration.html").write_text(HTML_TEMPLATE.replace("__DATA__", data_json))
    (HERE / "data" / "calibration_key.json").write_text(json.dumps(key, indent=2))

    from collections import Counter
    print(f"calibration set: {len(items)} blinded items "
          f"({dict(Counter(k['type'] for k in key.values()))})")
    print("wrote  experiments/label_quality/data/calibration.html  (+ calibration_key.json)")


if __name__ == "__main__":
    main()
