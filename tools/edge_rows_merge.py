# edge_rows_merge.py OUT A.json r2:B.json r3:C.json ...: union of Edge train-row files from several frame draws.
# A row present in an earlier file keeps that file's draw (decision frames repeat in every draw and were fetched in A).
# Each row gets "draw" ("" for the first file) so src/edge_train.py checks that draw's done markers.
import sys, json
out, first, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
M = {k: dict(v, draw="") for k, v in json.load(open(first)).items()}
for spec in rest:
    tag, f = spec.split(":", 1); n = 0
    for k, v in json.load(open(f)).items():
        if k not in M: M[k] = dict(v, draw=tag); n += 1
    print(f"{f}: +{n} new rows (draw {tag})")
json.dump(M, open(out, "w")); print(out, len(M), "rows,", len({k.split('@')[0] for k in M}), "clips")
