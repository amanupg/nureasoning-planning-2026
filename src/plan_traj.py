"""Planning answer text -> 51 submission poses, and the validation GT round trip.

parse_delta: exactly 5 "[a, b]" pairs, summed (per-step displacements -> waypoints at 1..5 s), else None.
spline51: cubic spline through the origin and the 5 waypoints at t = 0..5 s, start slope = current ego-frame
  velocity, natural end; heading from the spline velocity (held below 0.2 m/s), unwrapped; row 0 = 0.
to_global: ego-frame rows -> global poses for the devkit scorer.
  python src/plan_traj.py gt [rows_dir]   # GT -> delta text -> parse -> spline -> score, plus raw GT
"""
import json, os, re, sys
import numpy as np

PAIR = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def parse_delta(text):
    p = PAIR.findall(text)
    return np.cumsum(np.array(p, float), 0) if len(p) == 5 else None


def spline51(points, velocity):
    from scipy.interpolate import CubicSpline
    knots = np.vstack([[0.0, 0.0], points]); t = np.arange(51) * 0.1
    sp = [CubicSpline(np.arange(6.0), knots[:, k], bc_type=((1, velocity[k]), (2, 0.0))) for k in (0, 1)]
    xy = np.column_stack([s(t) for s in sp]); vx, vy = (s.derivative()(t) for s in sp)
    yaw = np.zeros(51)
    for i in range(1, 51): yaw[i] = np.arctan2(vy[i], vx[i]) if np.hypot(vx[i], vy[i]) > 0.2 else yaw[i - 1]
    out = np.column_stack([xy, np.unwrap(yaw)]); out[0] = 0.0
    return out


def to_global(traj, pose):
    x0, y0, h0 = pose; c, s = np.cos(h0), np.sin(h0)
    return np.column_stack([x0 + c * traj[:, 0] - s * traj[:, 1], y0 + s * traj[:, 0] + c * traj[:, 1], traj[:, 2] + h0])


def _score(a):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from wajepa_run import score
    return score("validation", *a) or [np.nan] * 5


def _raw_gt(clip):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from wajepa_run import NR
    from nureasoning.common.pickle_io import load_pickle
    cd = os.path.join(NR, "cache", "validation", clip); fr = json.load(open(os.path.join(cd, "metadata.json")))["frames"]
    e = load_pickle(os.path.join(cd, fr[100]["ego_state"]))
    return np.vstack([[e.pose["x"], e.pose["y"], e.pose["yaw"]], np.asarray(e.trajectory_future[:50], float)[:, :3]])


if __name__ == "__main__" and sys.argv[1] == "gt":
    from concurrent.futures import ProcessPoolExecutor
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plan_rows import delta_text
    d = sys.argv[2] if len(sys.argv) > 2 else "data/plan_rows"
    R = [json.loads(l) for l in open(os.path.join(d, "planning_validation.jsonl"))]
    nps = lambda L: L[:, 0] * L[:, 1] * (.3 * L[:, 2] + .2 * L[:, 3] + .5 * L[:, 4])
    rt = [(r["clip"], to_global(spline51(parse_delta(delta_text(r["gt5"])), r["velocity"]), r["pose"])) for r in R]
    with ProcessPoolExecutor(8) as ex:
        raw = list(ex.map(_raw_gt, [r["clip"] for r in R], chunksize=8))
        L1 = np.array(list(ex.map(_score, rt, chunksize=8)), float)
        L0 = np.array(list(ex.map(_score, [(r["clip"], g) for r, g in zip(R, raw)], chunksize=8)), float)
    for name, L in (("raw GT 51 poses", L0), ("GT -> delta text -> spline", L1)):
        print(f"{name}: NPS {np.nanmean(nps(L)):.4f} on {np.isfinite(L).all(1).sum()} clips | "
              + " ".join(f"{k} {np.nanmean(L[:, i]):.3f}" for i, k in enumerate(["coll", "drive", "prog", "comf", "human"])))
