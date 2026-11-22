"""Alpamayo 2 Edge planning (fine-tuned; the zero-shot test was dropped by Aman 2026-11-11): merge shards, validity line, singles and blend with v9.5 on validation.
  python src/edge_score.py [runs/edge_ft] [--force]   (validity gates the scores)
- merge: runs/<tag>_validation_shard*.npz -> runs/<tag>_validation.npz with names, traj (mean of the K candidates:
  mean xy, circular-mean yaw), traj0 (candidate 0), both 51 global rows (row 0 = key pose, rows 1..50 = 0.1..5.0 s).
- validity: vs the human future (data/plan_rows gt5, ego frame, 1..5 s): lateral correlation at 1 s / 3 s, median
  longitudinal ratio at 1 s and 5 s (moving clips), ADE/FDE over the five 1 Hz points.
- scores (devkit scorer, src/wajepa_run.score): edge mean, edge s0, v9.5 blend, v9.5 + w * edge for w in a grid;
  w is chosen on one RandomState(0) log half and checked on the other (both directions); paired log-level bootstrap
  of the full-set best-w blend vs v9.5 (labelled as selected on the same data).
"""
import collections, glob, json, os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "src")
from wajepa_run import score
from ensemble_eval import wmean, nps
from plan_traj import to_global

TAG = next((x for x in sys.argv[1:] if not x.startswith("--")), "runs/edge_ft")
F = {"b": "runs/v71B_validation.npz", "u4": "runs/wajepa_ft_h10u4_validation.npz", "wa": "runs/wajepa_169_validation.npz",
     "dj": "runs/djft_r2avg_validation.npz", "raw": "runs/a15s1lora_raw_validation.npz"}
V95 = {"b": .4, "u4": .35, "wa": .2, "dj": .25, "raw": .1}
WS = tuple(map(float, os.environ["EDGE_WS"].split(","))) if os.environ.get("EDGE_WS") else (0.05, 0.1, 0.15, 0.2, 0.3, 0.45)


def rows51(xy, yaw):
    return np.vstack([[0.0, 0.0, 0.0], np.column_stack([xy[:50], yaw[:50]])])


def merge():
    names, xy, yaw = [], [], []
    for f in sorted(glob.glob(f"{TAG}_validation_shard*of*.npz")):
        if f.endswith(".partial.npz"): continue
        z = np.load(f); names += list(map(str, z["names"])); xy += list(z["xy"]); yaw += list(z["yaw"])
    o = np.argsort(names); names = [names[i] for i in o]; xy = np.array(xy)[o]; yaw = np.array(yaw)[o]
    inp = json.load(open("data/edge_inputs_validation.json"))
    mxy = xy.mean(1); myaw = np.arctan2(np.sin(yaw).mean(1), np.cos(yaw).mean(1))
    traj = np.array([to_global(rows51(mxy[i], myaw[i]), inp[c]["pose"]) for i, c in enumerate(names)])
    traj0 = np.array([to_global(rows51(xy[i, 0], yaw[i, 0]), inp[c]["pose"]) for i, c in enumerate(names)])
    np.savez(f"{TAG}_validation.npz", names=np.array(names), traj=traj, traj0=traj0, ego_mean=mxy, K=xy.shape[1])
    return names, mxy


def validity(names, mxy):
    gt = {}
    for l in open("data/plan_rows/planning_validation.jsonl"):
        r = json.loads(l); gt[r["clip"]] = np.asarray(r["gt5"], float)
    G = np.array([gt[c] for c in names]); P = mxy[:, [9, 19, 29, 39, 49]]
    mov = G[:, 0, 0] > 1.0
    print(f"validity ({len(names)} clips, {mov.sum()} moving): lateral corr 1 s {np.corrcoef(P[:, 0, 1], G[:, 0, 1])[0, 1]:+.3f}, "
          f"3 s {np.corrcoef(P[:, 2, 1], G[:, 2, 1])[0, 1]:+.3f}; longitudinal ratio pred/gt median 1 s "
          f"{np.median(P[mov, 0, 0] / G[mov, 0, 0]):.3f}, 5 s {np.median(P[mov, 4, 0] / G[mov, 4, 0]):.3f}; "
          f"ADE {np.linalg.norm(P - G, axis=2).mean():.2f} m, FDE5 {np.linalg.norm(P[:, 4] - G[:, 4], axis=1).mean():.2f} m "
          f"(median {np.median(np.linalg.norm(P[:, 4] - G[:, 4], axis=1)):.2f})")
    c1 = np.corrcoef(P[:, 0, 1], G[:, 0, 1])[0, 1]; c3 = np.corrcoef(P[:, 2, 1], G[:, 2, 1])[0, 1]
    r1 = np.median(P[mov, 0, 0] / G[mov, 0, 0]); r5 = np.median(P[mov, 4, 0] / G[mov, 4, 0])
    ok = c1 > 0.5 and c3 > 0.5 and 0.8 < r1 < 1.25 and 0.7 < r5 < 1.4
    print("VALIDITY_PASS (frame, lateral sign and time scale match the human future)" if ok else
          "VALIDITY_FAIL: frame/sign/time-scale mismatch; no verdict on this model until fixed", flush=True)
    return ok


def one(a):
    c, M = a; out = {}
    r = score("validation", c, M["e"]); out["edge"] = nps(r) if r else np.nan
    r = score("validation", c, M["e0"]); out["edge_s0"] = nps(r) if r else np.nan
    base = wmean([M[m] for m in V95], [V95[m] for m in V95])
    r = score("validation", c, base); out["v95"] = nps(r) if r else np.nan
    for w in WS:
        r = score("validation", c, wmean([base, M["e"]], [1 - w, w])); out[f"w{w}"] = nps(r) if r else np.nan
    return c, out


def main():
    names, mxy = merge()
    if not validity(names, mxy) and "--force" not in sys.argv:
        sys.exit("stopping before scores (validity failed); rerun with --force only to debug")
    z = np.load(f"{TAG}_validation.npz"); E = dict(zip(map(str, z["names"]), z["traj"])); E0 = dict(zip(map(str, z["names"]), z["traj0"]))
    Z = {k: dict(zip(map(str, np.load(p)["names"]), np.load(p)["traj"])) for k, p in F.items()}
    names = sorted(set(E).intersection(*[set(v) for v in Z.values()]))
    args = [(c, {**{k: np.asarray(Z[k][c], float) for k in Z}, "e": E[c], "e0": E0[c]}) for c in names]
    with ProcessPoolExecutor(8) as ex: R = dict(ex.map(one, args, chunksize=4))
    json.dump(R, open(f"{TAG}_val_scores{os.environ.get('EDGE_SFX', '')}.json", "w"))
    log = lambda c: "_".join(c.split("_")[:2]); L = sorted({log(c) for c in names}); by = collections.defaultdict(list)
    for c in names: by[log(c)].append(c)
    H = set(np.random.RandomState(0).permutation(L)[:len(L) // 2])
    m = lambda k, cs: np.nanmean([R[c][k] for c in cs])
    A = [c for c in names if log(c) in H]; B = [c for c in names if log(c) not in H]
    print(f"n={len(names)} clips, {len(L)} logs")
    for k in ("edge", "edge_s0", "v95") + tuple(f"w{w}" for w in WS):
        print(f"  {k:8s} {m(k, names):.4f}   (half A {m(k, A):.4f}, half B {m(k, B):.4f})")
    for fit, chk, lab in ((A, B, "A->B"), (B, A, "B->A")):
        w = max(WS, key=lambda w: m(f"w{w}", fit) - m("v95", fit))
        print(f"  weight fit on {lab[0]}: w={w}, gain on {lab[-1]} {m(f'w{w}', chk) - m('v95', chk):+.4f}")
    wb = max(WS, key=lambda w: m(f"w{w}", names)); rng = np.random.RandomState(0)
    d = {l: [R[c][f"w{wb}"] - R[c]["v95"] for c in by[l]] for l in L}
    bs = [np.nanmean(np.concatenate([d[l] for l in rng.choice(L, len(L))])) for _ in range(2000)]
    print(f"  best w on all (selected on the same data) {wb}: {m(f'w{wb}', names) - m('v95', names):+.4f} "
          f"[{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")
    for k in ("edge", "edge_s0"):
        d = {l: [R[c][k] - R[c]["v95"] for c in by[l]] for l in L}; rng = np.random.RandomState(0)
        bs = [np.nanmean(np.concatenate([d[l] for l in rng.choice(L, len(L))])) for _ in range(2000)]
        print(f"  {k} vs v95 {m(k, names) - m('v95', names):+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")


if __name__ == "__main__":
    main()
