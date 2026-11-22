"""Rebuild v7.1's VALIDATION trajectories (global, 51 x 3) so ensembles with v7.1 can be scored.

Mirrors ../nureasoning/src/make_submission.py (read-only imports) with the v7.1 recipe:
  - route clips: out-of-fold chooser pick (valsel_e_ftcv_0v, the seed-0 run behind v7.1's selections);
    k < 148 -> vocab148.build(route candidate); k >= 148 -> fine-tuned Alpamayo half h0 (148..175) /
    h1 (176..203) path, candidate spec from label_paths.SPECS (re-timed or raw);
  - untrusted / missing route with no Alpamayo pick -> mean of the two half-models' paths (FALLBACK).
  -> runs/v71_validation.npz {names, traj}; check: route-clip picks vs their stored labels.
"""
import glob, json, os, sys
import numpy as np

NR = "/path/to/workspace/nureasoning"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(NR, "src")); sys.path.insert(0, os.path.join(NR, "recon/devkit"))


def main():
    import make_submission as M, planners
    from vocab148 import VOCAB148, build
    from label_paths import SPECS as SP28
    from a15_calib import mean_global, retime
    S = json.load(open(sorted(glob.glob(f"{NR}/artifacts/versioned/valsel_e_ftcv_0v_*.json"))[0]))
    assert [b[1] for b in S["blocks"]] == ["labels148", "labels_a15ft_h0", "labels_a15ft_h1"], S["blocks"]
    sel = S["selections"]; A = lambda h, c: mean_global(np.load(f"{NR}/a15ft_out/{h}/validation/{c}.npz"))
    names, trajs, kinds = [], [], []
    for c in sorted(os.listdir(f"{NR}/cache/validation")):
        cp = f"{NR}/cache/validation/{c}"
        if not os.path.isfile(f"{cp}/metadata.json"): continue
        meta = M._load_metadata(cp); fr = meta["frames"]; ki = M.select_key_frame_idx(fr, M.CFG.key_frame_index)
        ego = M.load_pickle(os.path.join(cp, fr[ki]["ego_state"])); route = M.build_route_line(fr[ki].get("mission_goal"))
        k = sel.get(c); trust = M.route_trustworthy(ego, route)
        if k is not None and k >= 148:
            h, j = ("h0", k - 148) if k < 176 else ("h1", k - 176)
            g = A(h, c); kind, v = SP28[j]
            if kind != "raw":
                v0 = planners._speed(ego); kin = planners.arc_length_kinematic(v0, planners._lon_accel(ego))
                own = float(np.sum(np.hypot(*np.diff(g[:, :2], axis=0).T)))
                L = kin * v if kind == "rel" else (v if kind == "abs" else own)
                g = retime(g, v0, max(0.0, L), ego.pose["yaw"])
            kinds.append("alpamayo")
        elif not trust or k is None:
            G = [A("h0", c), A("h1", c)]; g = np.mean(G, 0)
            g[:, 2] = np.arctan2(np.mean([np.sin(x[:, 2]) for x in G], 0), np.mean([np.cos(x[:, 2]) for x in G], 0))
            kinds.append("fallback")
        else:
            g = build(meta, fr, ki, ego, route, VOCAB148[k]); kinds.append("route")
        names.append(c); trajs.append(np.asarray(g, float))
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    np.savez_compressed(os.path.join(HERE, "runs/v71_validation.npz"), names=np.array(names), traj=np.array(trajs), kind=np.array(kinds))
    print(f"v7.1 validation trajectories: {len(names)} ({ {k: kinds.count(k) for k in set(kinds)} })")


if __name__ == "__main__":
    main()
