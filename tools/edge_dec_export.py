# edge_dec_export.py SPLIT TAG MODEL_NAME OUT: decision distributions from a reasoning-on Edge run, for the reasoning thread.
# Reads <TAG>_<SPLIT>_shard*of*.json (edge_infer: per clip K generated texts "Longitudinal: X. Lateral: Y. <trace>").
# One jsonl row per clip: {"clip","model","K","long":{str:count},"lat":{str:count},"joint":{"<Long> + <Lat>":count},
# "n_parse_fail"}; raw counts of the exact emitted strings. Traces go to OUT with .traces.json.
import sys, json, glob, re, collections
split, tag, model, out = sys.argv[1:5]
T = {}
for f in sorted(glob.glob(f"{tag}_{split}_shard*of*.json")):
    if ".partial" in f: continue
    T.update(json.load(open(f)))
assert len(T) == 1000, len(T)
pat = re.compile(r"^\s*Longitudinal:\s*(.+?)\.\s*Lateral:\s*(.+?)\.(?:\s|$)(.*)", re.S)
tr, nf = {}, 0
with open(out, "w") as fo:
    for c in sorted(T):
        L, A, J = collections.Counter(), collections.Counter(), collections.Counter(); bad = 0; tr[c] = []
        for s in T[c]["cot"]:
            m = pat.match(s)
            if not m: bad += 1; tr[c].append({"raw": s}); continue
            lo, la, t = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            L[lo] += 1; A[la] += 1; J[f"{lo} + {la}"] += 1; tr[c].append({"long": lo, "lat": la, "trace": t})
        nf += bad
        fo.write(json.dumps({"clip": c, "model": model, "K": len(T[c]["cot"]), "long": dict(L), "lat": dict(A),
                             "joint": dict(J), "n_parse_fail": bad}) + "\n")
json.dump(tr, open(out.rsplit(".", 1)[0] + ".traces.json", "w"))
print(out, len(T), "clips; parse failures", nf, "of", sum(len(v["cot"]) for v in T.values()))
