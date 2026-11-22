"""Alpamayo 2 Edge fine-tuning rows: the same (clip, key frame) train set as the 10B track (data/plan_rows/
planning_train.jsonl: decision frames + 3 regular frames per clip, val/test clips and overlapping drives excluded),
each turned into Edge inputs (src/edge_inputs.row_inputs, frames chosen by nearest timestamp) with the 6.4 s human
future (n_valid < 64 when the clip ends early; the tail is masked in training). Rows need key >= 20 (2 s of camera
history) and >= 5 s of future.
  python src/edge_rows.py   -> data/edge_rows_train.json ({"clip@t": row}) + adds "train" to data/edge_members.json
Second frame draw (run B, 2026-11-13): ER_PLAN=data/plan_rows_r2/planning_train.jsonl ER_OUT=data/edge_rows_train_r2.json
ER_MEMBERS=data/edge_members_r2.json -> rows for decision + new regular frames, and a separate members file holding only
the NEW regular rows' images (decision-frame images are already fetched); data/edge_members.json is left alone.
"""
import json, os, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan_rows import Clip, HERE
from edge_inputs import row_inputs
from plan_members import tars


def per_clip(args):
    clip, ts = args
    c = Clip("train", clip); out = {}
    for t in ts:
        r = row_inputs(c, t, by_time=True)
        if r is not None and r["fut64"] is not None: out[f"{clip}@{t}"] = r
    return out


def main():
    by = {}; kind = {}
    for l in open(os.path.join(HERE, os.environ.get("ER_PLAN", "data/plan_rows/planning_train.jsonl"))):
        r = json.loads(l); by.setdefault(r["clip"], []).append(r["t"]); kind[f"{r['clip']}@{r['t']}"] = r["kind"]
    with Pool(8) as p: parts = p.map(per_clip, sorted(by.items()), chunksize=8)
    rows = {k: v for d in parts for k, v in d.items()}
    n_in = sum(len(v) for v in by.values())
    print(f"train rows {len(rows)} of {n_in} (dropped {n_in - len(rows)}: key < 20 or < 5 s future); "
          f"n_valid < 64 in {sum(v['n_valid'] < 64 for v in rows.values())}", flush=True)
    json.dump(rows, open(os.path.join(HERE, os.environ.get("ER_OUT", "data/edge_rows_train.json")), "w"))
    T = tars(); mem = {}; sep = os.environ.get("ER_MEMBERS")
    for k, v in rows.items():
        if sep and kind[k] != "regular": continue
        mem.setdefault(k.split("@")[0], set()).update(i["m"] for i in v["images"])
    p = os.path.join(HERE, sep or "data/edge_members.json"); M = {} if sep else json.load(open(p))
    M["train"] = [{"clip": c, "tar": T[("train", c)][0], "size": T[("train", c)][1], "members": sorted(m), "reasoning": []}
                  for c, m in sorted(mem.items())]
    json.dump(M, open(p, "w"))
    print(f"train fetch: {len(M['train'])} clips, {sum(len(r['members']) for r in M['train'])} images", flush=True)


if __name__ == "__main__":
    main()
