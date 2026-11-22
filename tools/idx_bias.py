# Per-sample-index bias across Edge val runs (fixed seed 43 per clip -> same flow noise per index on every clip?)
import sys, glob, numpy as np
for tag in sys.argv[1:]:
    xy = []
    for f in sorted(glob.glob(f"runs/{tag}_validation_shard?of2.npz")): xy += list(np.load(f)["xy"])
    xy = np.array(xy)  # N,K,64,2 ego frame
    fwd5 = xy[:, :, 49, 0]; lat5 = xy[:, :, 49, 1]; dev = np.linalg.norm(xy[:, :, 49] - xy[:, :, 49].mean(1, keepdims=True), axis=-1)
    rel = fwd5 - fwd5.mean(1, keepdims=True)
    print(f"{tag:22s} K={xy.shape[1]} fwd5-vs-mean by index: " + " ".join(f"{v:+.2f}" for v in rel.mean(0)[:8]) +
          " | |lat5| " + " ".join(f"{v:.2f}" for v in np.abs(lat5).mean(0)[:8]) + " | dev " + " ".join(f"{v:.2f}" for v in dev.mean(0)[:8]))
