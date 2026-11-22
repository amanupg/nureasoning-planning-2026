"""Per-clip comparison of a JEPA run with the nuReasoning v7.1 planner on the SAME validation clips.

v7.1 reference (all read-only from ../nureasoning):
  - 849 route clips: the out-of-fold pick of the v7.1 chooser (valsel_e_ftcv_0v, seed 0 = 0.6294),
    scored by its stored devkit labels;
  - other clips: v7.1's no-route fallback, mean of the two fine-tuned Alpamayo half-models' paths,
    scored live with the devkit.
  python src/compare.py runs/drivejepa_validation.npz [more runs...]
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wajepa_run import NR, score

nps = lambda L: L[..., 0] * L[..., 1] * (.3 * L[..., 2] + .2 * L[..., 3] + .5 * L[..., 4])
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "v71_val_perclip.json")


def v71_reference():
    if os.path.isfile(CACHE):
        return json.load(open(CACHE))
    A = os.path.join(NR, "artifacts")
    S = json.load(open(sorted(glob.glob(f"{A}/versioned/valsel_e_ftcv_0v_*.json"))[0]))
    blocks = []
    for _, dn, _ in S["blocks"]:
        P = [np.load(f) for f in sorted(glob.glob(f"{A}/{dn}/validation_[0-9]*.npz"))]
        blocks.append(dict(zip(np.concatenate([p["names"] for p in P]), np.concatenate([p["labels"] for p in P]).astype(float))))
    ref = {n: float(nps(np.concatenate([b[n] for b in blocks])[k])) for n, k in S["selections"].items()}
    sys.path.insert(0, os.path.join(NR, "src"))
    from a15_calib import mean_global
    for c in sorted(os.listdir(os.path.join(NR, "cache/validation"))):
        if c in ref or not os.path.isfile(os.path.join(NR, "cache/validation", c, "metadata.json")): continue
        G = [mean_global(np.load(os.path.join(NR, f"a15ft_out/{h}/validation/{c}.npz"))) for h in ("h0", "h1")]
        g = np.mean(G, 0); g[:, 2] = np.arctan2(np.mean([np.sin(x[:, 2]) for x in G], 0), np.mean([np.cos(x[:, 2]) for x in G], 0))
        r = score("validation", c, g); ref[c] = float(nps(np.array(r))) if r else float("nan")
    json.dump(ref, open(CACHE, "w"))
    return ref


if __name__ == "__main__":
    ref = v71_reference(); route = set(json.load(open(sorted(glob.glob(os.path.join(NR, "artifacts/versioned/valsel_e_ftcv_0v_*.json")))[0]))["selections"])
    print(f"v7.1 reference: {len(ref)} clips, mean {np.nanmean(list(ref.values())):.4f} "
          f"(route {np.nanmean([v for k, v in ref.items() if k in route]):.4f}, no-route {np.nanmean([v for k, v in ref.items() if k not in route]):.4f})")
    for f in sys.argv[1:]:
        z = np.load(f); names = [str(n) for n in z["names"]]; j = nps(z["labels"])
        v = np.array([ref.get(n, np.nan) for n in names]); m = ~np.isnan(j) & ~np.isnan(v)
        r = np.array([n in route for n in names])
        print(f"{os.path.basename(f):32s} n={m.sum():4d}  JEPA {j[m].mean():.4f}  v7.1 {v[m].mean():.4f}  diff {j[m].mean()-v[m].mean():+.4f}"
              f" | route n={(m&r).sum()} {j[m&r].mean():.4f} vs {v[m&r].mean():.4f} | no-route n={(m&~r).sum()} "
              f"{j[m&~r].mean() if (m&~r).any() else float('nan'):.4f} vs {v[m&~r].mean() if (m&~r).any() else float('nan'):.4f}"
              f" | JEPA wins {(j[m] > v[m] + 1e-9).sum()} / loses {(j[m] < v[m] - 1e-9).sum()}"
              f" | per-clip max {np.maximum(j[m], v[m]).mean():.4f}")
