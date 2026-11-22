# edge_cot_rows.py: map Edge train rows (clip@frame) to the challenge's Driving reasoning annotation at that exact frame.
# Output data/edge_cot_train.json {row_key: {"trace", "lon", "lat", "text"}} for rows that have one; text = "Longitudinal: X. Lateral: Y. <trace>".
import json, glob, os, collections
NR = "/path/to/workspace/nureasoning/cache/train"
rows = json.load(open("data/edge_rows_train.json"))
clips = collections.defaultdict(set)
for k in rows: c, f = k.rsplit("@", 1); clips[c].add(int(f))
out, nofile, nodrv = {}, 0, 0
for c, fs in clips.items():
    fl = glob.glob(os.path.join(NR, c, "reasoning", "*.json"))
    if not fl: nofile += len(fs); continue
    byf = {}
    for f in fl:
        d = json.load(open(f))
        if d.get("Driving"): byf[d["frame_index"]] = d["Driving"]
    for f in fs:
        D = byf.get(f)
        if not D or not D.get("Reasoning trace"): nodrv += 1; continue
        dd = D.get("Driving decision", {})
        lon = (dd.get("Longitudinal") or "").replace(" (speed up)", "").strip(); lat = (dd.get("Lateral") or "").strip()
        tr = D["Reasoning trace"].strip()
        # text = what the planner learns to write: the annotated decision (parseable, same option strings as the
        # decision MCQs) then the one-sentence trace (base Edge's CoC style)
        out[f"{c}@{f}"] = {"trace": tr, "lon": lon, "lat": lat,
                           "text": (f"Longitudinal: {lon}. Lateral: {lat}. " if lon and lat else "") + tr}
json.dump(out, open("data/edge_cot_train.json", "w"))
print(f"rows {len(rows)}, with Driving trace {len(out)}, clips without reasoning files {nofile} rows, no trace at frame {nodrv}")
print(collections.Counter(int(k.rsplit('@',1)[1]) for k in out).most_common(8))
L = [len(v["trace"].split()) for v in out.values()]; import numpy as np; print("words median", np.median(L), "p95", np.percentile(L, 95))
