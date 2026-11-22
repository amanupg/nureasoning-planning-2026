"""Turn command rebuilt from mission_goal.route_path (the raw command is noisy, in test too).
The route is a sparse global polyline (~25-30 m spacing). In the key-frame ego frame: project the ego onto the route,
walk AHEAD m metres along it, and compare the route's direction there with the ego heading:
> +TH deg -> TURN_LEFT (0), < -TH -> TURN_RIGHT (2), else GO_STRAIGHT (1). Empty / too short route -> None (keep raw).
  python src/route_cmd.py   -> agreement of raw vs route command with the human 5 s turn on validation (labels only)
"""
import json, os, sys
import numpy as np

AHEAD, TH = 60.0, 30.0


def route_command(route, x, y, yaw, ahead=AHEAD, th=TH):
    R = np.asarray(route, float)
    if R.ndim != 2 or len(R) < 2: return None
    c, s = np.cos(yaw), np.sin(yaw); d = R - [x, y]
    P = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)          # ego frame, x forward
    seg = np.diff(P, axis=0); L = np.hypot(*seg.T); cum = np.concatenate([[0], np.cumsum(L)])
    best, j0, u0 = np.inf, 0, 0.0                                                      # projection onto the polyline
    for j in range(len(seg)):
        if L[j] < 1e-6: continue
        u = np.clip(-(P[j] @ seg[j]) / L[j] ** 2, 0, 1); q = P[j] + u * seg[j]
        if (dd := q @ q) < best: best, j0, u0 = dd, j, u
    s0 = cum[j0] + u0 * L[j0]; st = s0 + ahead
    if np.sqrt(best) > 15 or cum[-1] - s0 < 15: return None                            # route not near / not ahead
    k = min(int(np.searchsorted(cum, min(st, cum[-1]), side="right")) - 1, len(seg) - 1)
    ang = np.degrees(np.arctan2(seg[k, 1], seg[k, 0]))
    return 0 if ang > th else (2 if ang < -th else 1)


def main():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from wajepa_run import NR, CMD
    from nureasoning.common.pickle_io import load_pickle
    for split in ("validation",):
        rows = []
        for c in sorted(os.listdir(os.path.join(NR, "cache", split))):
            cd = os.path.join(NR, "cache", split, c); fr = {f["frame_index"]: f for f in json.load(open(os.path.join(cd, "metadata.json")))["frames"]}
            E = load_pickle(os.path.join(cd, fr[100]["ego_state"])); mg = fr[100].get("mission_goal") or {}
            raw = CMD.get(mg.get("command"), 3); rc = route_command(mg.get("route_path") or [], E.pose["x"], E.pose["y"], E.pose["yaw"])
            F = np.asarray(E.trajectory_future, float); dyaw = np.degrees(np.angle(np.exp(1j * (F[-1, 2] - E.pose["yaw"])))) if len(F) else 0
            dist = np.hypot(*np.diff(np.vstack([[E.pose["x"], E.pose["y"]], F[:, :2]]), axis=0).T).sum() if len(F) else 0
            gt = 0 if dyaw > 30 else (2 if dyaw < -30 else 1)
            rows.append((raw, rc, gt, dist))
        R = np.array([(a, -1 if b is None else b, g, d) for a, b, g, d in rows], float)
        mv = R[:, 3] > 10; cl = np.where(R[:, 1] >= 0, R[:, 1], R[:, 0])
        print(f"{split}: {len(R)} clips; route command available {np.mean(R[:, 1] >= 0):.3f}; raw != cleaned {np.mean(cl != R[:, 0]):.3f}")
        print(f"  agreement with the human 5 s turn (|dyaw| > 30 deg), clips moving > 10 m ({mv.sum()}): raw {np.mean(R[mv, 0] == R[mv, 2]):.3f} | cleaned {np.mean(cl[mv] == R[mv, 2]):.3f}")
        for g, nm in enumerate(["left", "straight", "right"]):
            m = mv & (R[:, 2] == g); print(f"   human {nm:8s} n {m.sum():4d}: raw {np.mean(R[m, 0] == g):.3f} cleaned {np.mean(cl[m] == g):.3f}")


if __name__ == "__main__":
    main()
