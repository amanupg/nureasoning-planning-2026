"""Build a v8.x test submission: JEPA / ensemble planning + v7.1's reasoning answers (unchanged).

Members (global x, y, yaw x 51 on test): runs/wajepa_169_test.npz, runs/drivejepa_test.npz, and
v7.1's own trajectories decoded from its submission file (read-only). Ensemble = per-timestep
weighted mean (yaw circular), the same function scored on validation (src/ensemble_eval.py).
Encoding uses the devkit (global_to_ego, normalize_ego_trajectory, make_clip_record, ...), as
v7.1's builder does. Checks: devkit validate_submission; every trajectory decodes back to its
global plan (max error printed); answers identical to v7.1.

  python src/build_submission.py --name v8.1_... --members wa:1
  python src/build_submission.py --name v8.3_... --members v71:0.6,wa:0.268,drive:0.132
"""
import argparse, hashlib, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NR = "/path/to/workspace/nureasoning"
sys.path.insert(0, os.path.join(HERE, "src")); sys.path.insert(0, os.path.join(NR, "recon/devkit"))
from ensemble_eval import wmean
V71 = os.path.join(NR, "submissions/v7.1_chooser-a15ft-valcv-ens5_a15ft-fallback_reasoning-cot.json")
RUNS = {"wa": "runs/wajepa_169_test.npz", "drive": "runs/drivejepa_test.npz", "djft": "runs/djft_r40_test.npz", "djft3": "runs/djft_r2avg_test.npz",
        "b": "runs/v71B_test.npz", "raw": "runs/a15s1lora_raw_test.npz", "waft": "runs/wajepa_ft_w1_test.npz",
        "u4": "runs/wajepa_ft_h10u4_test.npz",
        # duplicated-key-frame fix (WA_DUPFIX=1: history offsets -1 when frame100 == frame99; LEDGER 2026-11-08)
        "wafix": "runs/wajepa_169_test_dupfix.npz", "waftfix": "runs/wajepa_ft_w1_dupfix_test.npz",
        "u4fix": "runs/wajepa_ft_h10u4_dupfix_test.npz",
        # Alpamayo 2 Edge fine-tuned, epoch 1 final (update-3215, K=6 mean; LEDGER 2026-11-12); neutral key
        "e1": "runs/edge_ft3215_test.npz", "e1k": "runs/edge_ft3215k16_test.npz",
        "e2a": "runs/edge_e2_2400_test.npz", "e2f": "runs/edge_e2_3215_test.npz",
        "e3f": "runs/edge_e3_3215_test.npz",
        "e2ak": "runs/edge_e2a_k16_test.npz",   # e2-2400 with K=16 (LEDGER 2026-11-14)
        "p1nt": "runs/edge_P1_2477nt_k16_test.npz",   # stage-1 final (C-2ep update-2477), no text, K16 (LEDGER 2026-11-17)
        "p1t": "runs/edge_P1_2477_k16_test.npz",      # same, with text
        "b4": "runs/edge_poolB4_test.npz"}            # pool of e2ak + p1nt + p1t + e1k samples (tools/pool_test.py)


def ego_to_global(traj, pose):
    x0, y0, yaw0 = float(pose["x"]), float(pose["y"]), float(pose["yaw"])
    c, s = math.cos(yaw0), math.sin(yaw0); t = np.asarray(traj, float)
    return np.stack([x0 + c * t[:, 0] - s * t[:, 1], y0 + s * t[:, 0] + c * t[:, 1], yaw0 + t[:, 2]], 1)


def main():
    from nureasoning.planning.benchmark import BenchmarkConfig, select_key_frame_idx, _load_metadata
    from nureasoning.common.pickle_io import load_pickle
    from nureasoning.submission.format import (global_to_ego, normalize_ego_trajectory, trajectory_to_list,
                                               make_clip_record, make_submission, write_submission,
                                               load_submission, validate_submission)
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--members", required=True, help="e.g. v71:0.6,wa:0.4")
    ap.add_argument("--answers", default=V71, help="submission whose reasoning answers are copied (default v7.1), or 'fixed-A': every answer 'A' (planning-only test)")
    a = ap.parse_args()
    mem = [(k, float(w)) for k, w in (x.split(":") for x in a.members.split(","))]
    v71 = {r["clip"]: r for r in json.load(open(V71))["clips"]}
    SB = a.answers == "fixed-A"
    ansf = ({c: {"answers": [{"question_id": x["question_id"], "answer": "A"} for x in r["answers"]]} for c, r in v71.items()}
            if SB else {r["clip"]: r for r in json.load(open(a.answers))["clips"]})
    R = {k: dict(zip(map(str, np.load(os.path.join(HERE, p))["names"]), np.load(os.path.join(HERE, p))["traj"]))
         for k, p in RUNS.items() if any(m == k for m, _ in mem)}
    CFG = BenchmarkConfig(); root = os.path.join(NR, "cache/test")
    records, maxerr, n_q, missing = [], 0.0, 0, 0
    for c in sorted(v71):
        cp = os.path.join(root, c); meta = _load_metadata(cp); fr = meta.get("frames", [])
        ki = select_key_frame_idx(fr, CFG.key_frame_index); ego = load_pickle(os.path.join(cp, fr[ki]["ego_state"]))
        G, W = [], []
        for k, w in mem:
            if k == "v71":
                G.append(ego_to_global(v71[c]["trajectory"], ego.pose)); W.append(w)
            elif c in R[k]:
                G.append(np.asarray(R[k][c], float)); W.append(w)
        if len(G) < len(mem): missing += 1
        g = wmean(G, W)
        e = normalize_ego_trajectory(global_to_ego(g, ego.pose))
        maxerr = max(maxerr, float(np.abs(ego_to_global(e, ego.pose)[:, :2] - g[:, :2]).max()))
        ans = ansf[c]["answers"]; n_q += len(ans)
        records.append(make_clip_record(clip=c, clip_token=meta.get("clip_token"), log_name=meta.get("log_name"),
                                        target_frame_index=ki, trajectory=trajectory_to_list(e), answers=ans))
    payload = make_submission(records, extra_meta={"team": "Aman_upg", "method": a.name, "planner": a.members,
                                                   "answer_strategy": "fixed answer A" if SB else f"answers copied from {os.path.basename(a.answers)}"})
    os.makedirs(os.path.join(HERE, "submissions"), exist_ok=True)
    out = os.path.join(HERE, "submissions", f"{a.name}.json")
    if os.path.exists(out): raise SystemExit(f"refusing to overwrite {out}")
    write_submission(payload, out)
    errs = validate_submission(load_submission(out), data_root=root)
    new = {r["clip"]: r for r in json.load(open(out))["clips"]}
    same_ans = all(new[c]["answers"] == ansf[c]["answers"] for c in ansf)
    tr = np.array([new[c]["trajectory"] for c in sorted(new)], float)
    print(f"{a.name}\n  clips={len(records)} questions={n_q} members={a.members} clips missing a member={missing}")
    print(f"  round-trip max |decoded - planned| = {maxerr:.2e} m; answers identical to {a.answers if SB else os.path.basename(a.answers)}: {same_ans}; "
          f"finite: {bool(np.isfinite(tr).all())}; max endpoint {np.hypot(tr[:, -1, 0], tr[:, -1, 1]).max():.1f} m")
    print(f"  sha256={hashlib.sha256(open(out, 'rb').read()).hexdigest()}")
    print(f"  devkit validate_submission: {'PASS (0 errors)' if not errs else 'FAIL ' + str(errs[:3])}")


if __name__ == "__main__":
    main()
