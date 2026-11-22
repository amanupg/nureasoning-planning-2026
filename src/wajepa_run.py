"""WA-JEPA (AFARI-Research, NAVSIM-trained, Apache-2.0) zero-shot on nuReasoning clips.

Inputs, matched to the released model (ext/WA-JEPA/close_loop/hugsim_planner.py, NAVSIM dataset):
  history_images [1, 4, 4, 3, 256, 512] in [-1, 1]: frames key-15/-10/-5/0 (0.5 s apart, oldest
      first) x cams [cam_l0, cam_f0, cam_r0, cam_b0] <- [CAM_M_L0, CAM_M_F, CAM_M_R0, CAM_M_B]
  ego_status [1, 8]: NAVSIM command one-hot [left, straight, right, unknown] + vx, vy, ax, ay (ego frame)
  history_trajectory [1, 4, 3]: poses at those 4 frames relative to the key frame (x fwd, y left, yaw)
Never reads ego_state.trajectory_future (absent on test).
Output: 8 waypoints (0.5..4.0 s). Converted to the benchmark's 51 poses (0..5 s at 10 Hz): linear
interpolation from the origin, then 4..5 s extended at the last segment's velocity and heading.

  python src/wajepa_run.py --split validation [--crop 169|full] [--limit N] [--score]
  -> runs/wajepa_<crop>_<split>.npz {names, traj (N, 51, 3) global [x, y, yaw], pred8 (N, 8, 3)}
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NR = os.environ.get("NR_ROOT", "/path/to/workspace/nureasoning")            # read-only
sys.path.insert(0, os.path.join(HERE, "ext/WA-JEPA"))
sys.path.insert(0, os.path.join(NR, "recon/devkit"))
CAMS = ["front_left", "front", "front_right", "back"]
OFFS = [-15, -10, -5, 0]
KEY = 100
CMD = {"TURN_LEFT": 0, "GO_STRAIGHT": 1, "TURN_RIGHT": 2}


def rel_se2(origin, poses):
    ox, oy, oh = origin
    c, s = np.cos(-oh), np.sin(-oh)
    d = poses - np.array([ox, oy, oh])
    x = d[:, 0] * c - d[:, 1] * s; y = d[:, 0] * s + d[:, 1] * c
    return np.stack([x, y, np.arctan2(np.sin(d[:, 2]), np.cos(d[:, 2]))], 1)


def load_model(device):
    import torch
    from omegaconf import OmegaConf
    from models.factory import build_world_model
    from training.checkpoint import load_checkpoint
    cfg = OmegaConf.load(os.path.join(HERE, "ext/WA-JEPA/configs/wa_jepa_navsim_pdms.yaml"))
    OmegaConf.set_struct(cfg, False)
    cfg.model.require_pretrained = False; cfg.model.vjepa2_ckpt = None
    cfg.model.init_from_checkpoint = None; cfg.model.require_init_from_checkpoint = False
    d = OmegaConf.to_container(cfg, resolve=True)
    m = build_world_model(d)
    load_checkpoint(os.path.join(HERE, "weights/wa-jepa/model_state_dict.pt"), m, map_location="cpu",
                    write_key_reports=False)
    return m.to(device).eval(), d


def image(path, crop, hw=(256, 512)):
    import cv2
    im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    if crop == "169":                                     # NAVSIM cameras are 16:9; ours ~3:2
        h, w = im.shape[:2]; nh = int(round(w * 9 / 16)); y0 = (h - nh) // 2
        im = im[y0:y0 + nh]
    im = cv2.resize(im, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
    return (im.astype(np.float32) / 255.0) * 2 - 1


def clip_inputs(split, clip, crop, img_root="data/img", lag=0):
    """lag=1 pairs each pose with the camera frame one earlier (poses lead images by ~0.1 s,
    nureasoning/LEDGER.md 2026-10-03); poses, status and the origin stay at frames 85..100."""
    from nureasoning.common.pickle_io import load_pickle
    cd = os.path.join(NR, "cache", split, clip)
    fr = {f["frame_index"]: f for f in json.load(open(os.path.join(cd, "metadata.json")))["frames"]}
    offs = [int(v) for v in os.environ["WA_OFFS"].split(",")] if os.environ.get("WA_OFFS") else OFFS   # default unchanged
    if os.environ.get("WA_DUPFIX") and fr[KEY]["timestamp_us"] == fr[KEY - 1]["timestamp_us"]:
        offs = [o - 1 if o < 0 else o for o in offs]            # key frame is a copy of KEY-1: true 0.5 s spacing
    egos = [load_pickle(os.path.join(cd, fr[KEY + o]["ego_state"])) for o in offs]
    poses = np.array([[e.pose["x"], e.pose["y"], e.pose["yaw"]] for e in egos], float)
    k = egos[-1]
    mg = fr[KEY].get("mission_goal") or {}; cmd = CMD.get(mg.get("command"), 3)
    if os.environ.get("WA_CMD") == "route":                     # command rebuilt from route_path (src/route_cmd.py)
        from route_cmd import route_command
        rc = route_command(mg.get("route_path") or [], k.pose["x"], k.pose["y"], k.pose["yaw"]); cmd = cmd if rc is None else rc
    st = np.zeros(8, np.float32); st[cmd] = 1
    st[4:8] = [k.velocity["vx"], k.velocity["vy"], k.acceleration["ax"], k.acceleration["ay"]]
    def ip(rel):                                                 # frames outside 85/90/95/100 live in data/img_dup
        f = os.path.join(HERE, img_root, split, clip, rel)
        return f if os.path.isfile(f) else os.path.join(HERE, "data/img_dup", split, clip, rel)
    imgs = np.stack([np.stack([image(ip(fr[KEY + o - lag]["sensors"]["cameras"][c]), crop)
                               for c in CAMS]) for o in offs])            # [T, V, H, W, 3]
    return imgs.transpose(0, 1, 4, 2, 3), st, rel_se2(poses[-1], poses).astype(np.float32), poses[-1], cmd


def to_51(pred8, origin):
    """n poses at 0.5 s steps (n=8: 4 s, extended to 5 s at the last segment's velocity; n=10: 5 s) -> 51 poses."""
    n = len(pred8); tn = np.arange(1, n + 1) * 0.5
    P = np.vstack([[0, 0, 0], pred8]); T = np.concatenate([[0], tn])
    P[:, 2] = np.unwrap(P[:, 2])
    t = np.arange(51) * 0.1
    x = np.interp(t, T, P[:, 0]); y = np.interp(t, T, P[:, 1]); h = np.interp(t, T, P[:, 2])
    v = (P[-1, :2] - P[-2, :2]) / 0.5                     # extend past the last pose at the last segment's velocity
    late = t > tn[-1] + 1e-9
    x[late] = P[-1, 0] + v[0] * (t[late] - tn[-1]); y[late] = P[-1, 1] + v[1] * (t[late] - tn[-1]); h[late] = P[-1, 2]
    ox, oy, oh = origin; c, s = np.cos(oh), np.sin(oh)
    return np.stack([ox + c * x - s * y, oy + s * x + c * y, oh + h], 1)


def score(split, clip, g):
    from nureasoning.planning.benchmark import BenchmarkConfig, evaluate_clip
    r = evaluate_clip(os.path.join(NR, "cache", split, clip), BenchmarkConfig(), g)
    if r.error: return None
    return [r.s_collision, r.s_driveable, r.s_progress, r.s_comfort, r.s_human]


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True); ap.add_argument("--crop", default="169", choices=["169", "full"])
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--score", action="store_true")
    ap.add_argument("--amp", type=int, default=1, help="bf16 autocast, as the model was trained")
    ap.add_argument("--lag", type=int, default=0, help="1: images one frame before the poses (data/img_s1)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    img_root = "data/img_s1" if a.lag else "data/img"; root = os.path.join(HERE, img_root, a.split)
    clips = sorted(c for c in os.listdir(root) if os.path.isfile(os.path.join(root, c, "_done.json")))
    if a.limit: clips = clips[:a.limit]
    m, cfg = load_model(a.device)
    print(f"[wajepa] {a.split} crop={a.crop} clips={len(clips)} device={a.device}", flush=True)
    names, trajs, preds, labels = [], [], [], []
    t0 = time.time()
    for i, c in enumerate(clips):
        imgs, st, hist, origin, cmd = clip_inputs(a.split, c, a.crop, img_root, a.lag)
        feats = {"history_images": torch.from_numpy(imgs).unsqueeze(0).to(a.device),
                 "ego_status": torch.from_numpy(st).unsqueeze(0).to(a.device),
                 "history_trajectory": torch.from_numpy(hist).unsqueeze(0).to(a.device)}
        with torch.no_grad(), torch.autocast(a.device, dtype=torch.bfloat16, enabled=bool(a.amp)):
            p = m.predict_trajectory(feats).squeeze(0).float().cpu().numpy().astype(np.float64)
        g = to_51(p, origin)
        names.append(c); trajs.append(g); preds.append(p)
        if a.score:
            labels.append(score(a.split, c, g) or [np.nan] * 5)
        if i < 3 or (i + 1) % 50 == 0:
            msg = f"  [{i+1}/{len(clips)}] {(time.time()-t0)/(i+1):.2f}s/clip end4s={p[-1,:2].round(1).tolist()} v={st[4]:.1f} cmd={cmd}"
            if labels:
                L = np.array(labels); n = L[:, 0] * L[:, 1] * (.3 * L[:, 2] + .2 * L[:, 3] + .5 * L[:, 4])
                msg += f" NPS so far {np.nanmean(n):.4f}"
            print(msg, flush=True)
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    out = os.path.join(HERE, "runs", f"wajepa_{a.crop}_{a.split}{'_n' + str(a.limit) if a.limit else ''}{'' if a.amp else '_fp32'}{'_lag' + str(a.lag) if a.lag else ''}{os.environ.get('WA_SUFFIX', '')}.npz")
    np.savez_compressed(out, names=np.array(names), traj=np.array(trajs), pred8=np.array(preds),
                        **({"labels": np.array(labels, float)} if labels else {}))
    print(f"[wajepa] {len(names)} clips -> {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
