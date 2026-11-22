"""Dup-fix check for the Edge member on validation: per-clip scores of the unfixed and the dup-fixed runs
(<tag>_val_scores.json and <tag>_dupfix_val_scores.json from src/edge_score.py) on the 31 val clips whose frame 100
copies frame 99, plus the other 969 clips as a control (inputs identical there; differences are sampling noise only,
the per-clip seed is fixed). 31 clips are far too few for a verdict on their own; the point is to catch a broken
fix (a large drop) before the test-default fix is trusted. Also reports the same for the v9.5 + Edge blend at the
chosen weight if present.
  python src/edge_dupcheck.py [runs/edge_ft]
"""
import json, sys
import numpy as np

tag = sys.argv[1] if len(sys.argv) > 1 else "runs/edge_ft"
A = json.load(open(f"{tag}_val_scores.json")); B = json.load(open(f"{tag}_dupfix_val_scores.json"))
dup = {c for c, v in json.load(open("data/edge_inputs_validation_dupfix.json")).items() if v["dupfix"]}
names = sorted(set(A) & set(B))
for k in [k for k in ("edge", "edge_s0") + tuple(x for x in A[names[0]] if x.startswith("w")) if k in A[names[0]]]:
    for lab, cs in (("31 dup clips", [c for c in names if c in dup]), ("969 other clips (control)", [c for c in names if c not in dup])):
        d = np.array([B[c][k] - A[c][k] for c in cs], float)
        rng = np.random.RandomState(0); bs = [np.nanmean(rng.choice(d, len(d))) for _ in range(2000)]
        print(f"{k:8s} {lab:26s} fixed - unfixed {np.nanmean(d):+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}] "
              f"(unfixed {np.nanmean([A[c][k] for c in cs]):.4f})")
