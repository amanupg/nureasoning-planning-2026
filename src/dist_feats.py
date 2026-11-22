"""Ego-history features and distance targets for the 5 s distance model (src/dist_net.py).

Rows: train = every (clip, r) of the packed WA-JEPA meta / main-track index (81,523 rows, 7,654 clips);
validation / test = key frame 100 of each clip. Features from the nuReasoning cache (read-only), only frames <= r:
speed at 0, -0.5 ... -10 s (NaN where the clip starts later), acceleration now (x, y), min / max / mean speed over the
available history, time since the last near-stop (< 0.5 m/s), distance covered in the last 1 / 3 / 10 s, available
history length, driving command. Targets (train / validation only, from frames r..r+50): path length at 1..5 s.
  python src/dist_feats.py --split train   -> data/dist_feats_<split>.npz {clip, r, X [N, F], cols, Y [N, 5]}
"""
import argparse, json, os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from wajepa_run import NR, KEY, CMD
COLS = [f"v_{k / 10:.1f}" for k in range(0, 101, 5)] + ["a_x", "a_y", "v_min", "v_max", "v_mean", "since_stop", "d1", "d3", "d10", "hist_s", "cmd"]


def clip_rows(args):
    from nureasoning.common.pickle_io import load_pickle
    split, clip, rs = args
    cd = os.path.join(NR, "cache", split, clip); fr = {f["frame_index"]: f for f in json.load(open(os.path.join(cd, "metadata.json")))["frames"]}
    need = sorted({i for r in rs for i in range(max(0, r - 100), r + 51) if i in fr})
    E = {i: load_pickle(os.path.join(cd, fr[i]["ego_state"])) for i in need}
    out = []
    for r in rs:
        h = [i for i in range(max(0, r - 100), r + 1) if i in E]
        v = np.array([np.hypot(E[i].velocity["vx"], E[i].velocity["vy"]) for i in h]); vi = dict(zip(h, v))
        P = np.array([[E[i].pose["x"], E[i].pose["y"]] for i in h]); step = np.hypot(*np.diff(P, axis=0).T) if len(P) > 1 else np.zeros(0)
        st = [i for i, x in zip(h, v) if x < 0.5]; since = (r - st[-1]) * 0.1 if st else (r - h[0]) * 0.1 + 0.5
        a = E[r].acceleration
        x = [vi.get(r - k, np.nan) for k in range(0, 101, 5)] + [a["ax"], a["ay"], v.min(), v.max(), v.mean(), since,
             step[-10:].sum(), step[-30:].sum(), step.sum(), (r - h[0]) * 0.1, CMD.get(((fr[r].get("mission_goal") or {}).get("command")), 3)]
        y = [np.nan] * 5
        if split != "test" and all(r + k in E for k in range(51)):
            F = np.array([[E[r + k].pose["x"], E[r + k].pose["y"]] for k in range(51)]); s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(F, axis=0).T))])
            y = [s[10 * j] for j in range(1, 6)]
        out.append((clip, r, x, y))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="train"); a = ap.parse_args()
    if a.split == "train":
        rows = [(c, int(r)) for c, r in json.load(open(os.path.join(NR, "artifacts/a15ft_index_train.json")))]
    else:
        root = os.path.join(NR, "cache", a.split); rows = [(c, KEY) for c in sorted(os.listdir(root)) if os.path.isfile(os.path.join(root, c, "metadata.json"))]
    by = {}
    for c, r in rows: by.setdefault(c, []).append(r)
    with ProcessPoolExecutor(8) as ex:
        R = [x for part in ex.map(clip_rows, [(a.split, c, rs) for c, rs in by.items()], chunksize=4) for x in part]
    np.savez(os.path.join(HERE, f"data/dist_feats_{a.split}.npz"), clip=np.array([x[0] for x in R]), r=np.array([x[1] for x in R]),
             X=np.array([x[2] for x in R], np.float32), Y=np.array([x[3] for x in R], np.float32), cols=np.array(COLS))
    Y = np.array([x[3] for x in R], float)
    print(f"{a.split}: {len(R)} rows from {len(by)} clips; 5 s distance mean {np.nanmean(Y[:, 4]):.2f} m, rows with target {np.isfinite(Y[:, 4]).mean():.1%}")


if __name__ == "__main__":
    main()
