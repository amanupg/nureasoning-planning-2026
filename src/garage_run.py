"""py123d_garage v0.2 (KE:SAI Latent TransFuser, ltf-base-joint64-68M) open-loop on nuReasoning key frames.

Mirrors GarageDriver._plan (py123d_garage evaluation/alpasim/help/driver_service.py) without a scene API: the policy's
own feature builders are bypassed by handing build_features() every tensor it would build, in its exact format:
- camera_feature uint8 [3, 512, 352]: front camera 1 s earlier (row 1) over the current front camera (row 2), each the
  FULL frame cv2-resized to 352x256, RGB. nuReasoning CAM_M_F is served as the nuPlan front camera (dataset "nuplan").
- velocity [1] = ego speed; velocity_vector / acceleration_vector [2] in the ego frame (from the nuReasoning ego state).
- no route: target points NaN (the checkpoint trained with target-point dropout 0.983).
Output: predictions.ego_trajectory_se2 = 64 poses at 0.1 s in the anchor rear-axle frame; the first 50 -> 51 global
poses at 0.1 s (key frame + 5 s), the format of every other member (runs/*_<split>.npz 'traj').
Weights: weights/py123d/v0.2.0/ltf-base-joint64-68M/<stage>/model_<seed>.pth (CC BY-NC-SA 4.0; code Apache-2.0,
ext/py123d_garage = v0.2.2; see BORROWED.md). Run with .venv_garage; score with src/score_npz.py (main .venv).
  .venv_garage/bin/python src/garage_run.py --split validation --stage posttrain --seeds 0
"""
import argparse, ctypes, json, os, sys, time
import numpy as np

if sys.platform == "darwin":                                              # py123d's cache store dlopens Linux libc at import;
    _cdll = ctypes.CDLL                                                   # inference never calls it, so macOS libc stands in
    ctypes.CDLL = lambda name, *a, **k: _cdll(None if name == "libc.so.6" else name, *a, **k)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NR = "/path/to/workspace/nureasoning"                       # read-only cache (metadata, ego states)
KEY, PAST = 100, 10                                                       # key frame; 1 s earlier at 10 Hz
W = os.path.join(HERE, "weights/py123d/v0.2.0/ltf-base-joint64-68M")


def build(stage, seeds, device):
    import torch
    from omegaconf import OmegaConf
    from py123d_garage.config.schema.policy.policy_config import EvaluationPolicyConfig
    from py123d_garage.evaluation.help.policy_building import build_evaluation_policy
    from py123d_garage.common.config_help.config_handling import replay_values, complete
    from py123d_garage.common.config_help.replay import _saved_policy_config
    ck = [os.path.join(W, stage, f"model_{s}.pth") for s in seeds]
    merged = OmegaConf.structured(EvaluationPolicyConfig)                 # as finalize_evaluation: the saved config replayed
    replay_values(merged, _saved_policy_config(ck[0]), skip_unknown=True)  # leaf by leaf onto the typed schema
    replay_values(merged, {"evaluation_checkpoint_files": ck})
    policy = build_evaluation_policy(complete(merged, EvaluationPolicyConfig))
    return policy.to(torch.device(device)).eval()


def ego(cd, fr, i):
    sys.path.insert(0, os.path.join(NR, "recon/devkit"))
    from nureasoning.common.pickle_io import load_pickle
    return load_pickle(os.path.join(cd, fr[i]["ego_state"]))


def inputs(split, clip, img_root):
    import cv2, torch
    cd = os.path.join(NR, "cache", split, clip); fr = {f["frame_index"]: f for f in json.load(open(os.path.join(cd, "metadata.json")))["frames"]}
    rows = []
    for i in (KEY - PAST, KEY):
        p = os.path.join(img_root, clip, fr[i]["sensors"]["cameras"]["front"])
        rgb = cv2.imread(p, cv2.IMREAD_COLOR)[:, :, ::-1]
        rows.append(cv2.resize(np.ascontiguousarray(rgb), (352, 256)))
    cam = torch.as_tensor(np.concatenate(rows, 0)).permute(2, 0, 1).contiguous()
    E = ego(cd, fr, KEY); v, a = E.velocity, E.acceleration
    cached = {"camera_feature": cam, "velocity": torch.tensor([float(np.hypot(v["vx"], v["vy"]))], dtype=torch.float32),
              "velocity_vector": torch.tensor([v["vx"], v["vy"]], dtype=torch.float32),
              "acceleration_vector": torch.tensor([a["ax"], a["ay"]], dtype=torch.float32)}
    return cached, (E.pose["x"], E.pose["y"], E.pose["yaw"])


def to_global(rel, origin):
    P = np.vstack([[0, 0, 0], rel[:50]]); ox, oy, oh = origin; c, s = np.cos(oh), np.sin(oh)
    return np.stack([ox + c * P[:, 0] - s * P[:, 1], oy + s * P[:, 0] + c * P[:, 1], oh + np.unwrap(P[:, 2])], 1)


def main():
    import torch
    from py123d_garage.api.abstract_policy_tensors import NavigationConditioning
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="validation"); ap.add_argument("--stage", default="posttrain")
    ap.add_argument("--seeds", default="0"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu"); a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]; img_root = os.path.join(HERE, "data/img", a.split)
    clips = sorted(c for c in os.listdir(img_root) if os.path.isfile(os.path.join(img_root, c, "_done.json")))[: a.limit or None]
    policy = build(a.stage, seeds, a.device); dev = torch.device(a.device)
    ntp = len(policy.policy_config.required_target_point_distances_m or [])   # posttrain 3 (40 / 60 / 80 m), midtrain 0
    nav = NavigationConditioning(target_points=torch.full((1, ntp, 2), float("nan")), target_point_headings=torch.full((1, ntp), float("nan")))
    names, trajs, rels = [], [], []; t0 = time.time()
    for k, c in enumerate(clips):
        cached, origin = inputs(a.split, c, img_root)
        feats = policy.build_features(None, cached).apply(lambda t: t.unsqueeze(0))
        with torch.no_grad():
            pr = policy.forward(feats.to(dev), nav.to(dev))
        rel = pr.ego_trajectory_se2[0].float().cpu().numpy().astype(np.float64)
        names.append(c); rels.append(rel); trajs.append(to_global(rel, origin))
        if k < 3 or (k + 1) % 100 == 0:
            print(f"  [{k + 1}/{len(clips)}] {(time.time() - t0) / (k + 1):.2f}s/clip v={float(cached['velocity'][0]):.1f} "
                  f"5s=({rel[49, 0]:.1f}, {rel[49, 1]:.1f})", flush=True)
    tag = a.tag or f"garage_{a.stage}_s{''.join(map(str, seeds))}"
    out = os.path.join(HERE, "runs", f"{tag}_{a.split}{'_n' + str(a.limit) if a.limit else ''}.npz")
    np.savez_compressed(out, names=np.array(names), traj=np.array(trajs), rel64=np.array(rels))
    print(f"[garage] {len(names)} clips -> {out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
