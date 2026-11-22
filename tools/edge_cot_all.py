# edge_cot_all.py ROWS OUT: reasoning text for every DECISION frame of an Edge train-rows file (11-15 big run).
# Trace rows keep data/edge_cot_train.json's "Longitudinal: X. Lateral: Y. <trace>"; decision frames without a trace get
# the decision alone ("Longitudinal: X. Lateral: Y.", the meta-action labels of data/plan_rows/planning_train.jsonl, the
# same option strings). Regular frames get no text (hindsight labels on regular frames are not used).
import sys, json
rows = json.load(open(sys.argv[1])); C = json.load(open("data/edge_cot_train.json")); dec = {}
for l in open("data/plan_rows/planning_train.jsonl"):
    r = json.loads(l)
    if r["kind"] == "decision":
        lon, lat = [x.split(": ", 1)[1] for x in r["conversations"][1]["value"].split("; ")]
        dec[f'{r["clip"]}@{r["t"]}'] = (lon, lat)
out = {}
for k in rows:
    if k in C: out[k] = C[k]
    elif k in dec: out[k] = {"lon": dec[k][0], "lat": dec[k][1], "trace": "", "text": f"Longitudinal: {dec[k][0]}. Lateral: {dec[k][1]}."}
json.dump(out, open(sys.argv[2], "w"))
print(f"{len(rows)} rows: {sum(1 for v in out.values() if v['trace'])} decision+trace, "
      f"{sum(1 for v in out.values() if not v['trace'])} decision only, {len(rows) - len(out)} regular (no text)")
