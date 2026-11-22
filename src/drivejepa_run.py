"""Drive-JEPA perception-free planner (linhanwang/Drive-JEPA, weights CC0) zero-shot on nuReasoning.

Matched to the released checkpoint (drive_jepa_perception_free_agent_vitl.ckpt):
  _keyval_embedding (129, 256) -> front camera only, 256x512 (16x32 patches, 2x2 pooled = 128 + status)
  camera: NAVSIM cam_f0 1920x1080 cropped [28:-28] rows (15:8) then resized to 512x256; one frame,
      repeated as a 2-frame clip (the upstream single-image prehook), ImageNet-normalised in the model
  status_feature (8): NAVSIM command one-hot [left, straight, right, unknown] + vx, vy, ax, ay
  output: 8 poses (0.5..4.0 s), ego x fwd, y left, heading
Our CAM_M_F (2816x1856, stored at 1408x928) is centre-cropped to the same 15:8 aspect first.
The full encoder is inside the planner checkpoint, so the upstream separate pretrained-encoder load
is bypassed (vit_encoder.torch.load patched to return an empty encoder dict).

  python src/drivejepa_run.py --split validation [--limit N] [--score]
  -> runs/drivejepa_<split>.npz {names, traj (N, 51, 3) global, pred8, labels}
"""
import argparse, json, os, sys, time, types
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from wajepa_run import NR, KEY, CMD, to_51, score            # shared adapter pieces
UP = os.path.join(HERE, "ext/Drive-JEPA/navsim_v1")


class _TS:                                                    # nuplan TrajectorySampling stand-in
    def __init__(self, time_horizon=4, interval_length=0.5, num_poses=None):
        self.num_poses = num_poses or int(round(time_horizon / interval_length))


def load_model(device):
    import torch
    mod = types.ModuleType("nuplan.planning.simulation.trajectory.trajectory_sampling"); mod.TrajectorySampling = _TS
    for n in ["nuplan", "nuplan.planning", "nuplan.planning.simulation", "nuplan.planning.simulation.trajectory"]:
        sys.modules.setdefault(n, types.ModuleType(n))
    sys.modules[mod.__name__] = mod
    sys.path.insert(0, UP)
    import vjepa2.evals.image_classification_frozen.modelcustom.vit_encoder as ve
    real = ve.torch.load
    ve.torch.load = lambda *a, **k: {"target_encoder": {}}
    cwd = os.getcwd(); os.chdir(UP)                          # the model reads ./vjepa2/configs/... relative
    try:
        from navsim.agents.drive_jepa_perception_free.drive_jepa_model import DriveJEPAModel
        m = DriveJEPAModel(_TS(4, 0.5), pretrain_pt_path=None, image_architecture="vit_large", tf_d_model=256,
                           tf_d_ffn=1024, tf_num_layers=3, tf_num_head=8, num_keyval=129, front_only=True,
                           freeze_encoder=True, double_image=True)   # double_image only skips the prehook
    finally:
        os.chdir(cwd); ve.torch.load = real
    sd = torch.load(os.path.join(HERE, "weights/drive-jepa/drive_jepa_perception_free_agent_vitl.ckpt"),
                    map_location="cpu", weights_only=False)["state_dict"]
    sd = {k.replace("agent._model.", "", 1): v for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=True), None
    print("[drivejepa] checkpoint loaded strictly", flush=True)
    return m.to(device).eval()


def front(path, hw=(256, 512)):
    import cv2
    im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]; nh = int(round(w * 1024 / 1920)); y0 = (h - nh) // 2
    im = cv2.resize(im[y0:y0 + nh], (hw[1], hw[0]))           # upstream cv2.resize default (bilinear)
    return im.astype(np.float32) / 255.0


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--img-root", default="data/img", help="data/img (HF range fetch) or data/img_cl (cluster copy)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    from nureasoning.common.pickle_io import load_pickle
    root = os.path.join(HERE, a.img_root, a.split)
    clips = sorted(c for c in os.listdir(root) if os.path.isfile(os.path.join(root, c, "_done.json")))
    if a.limit: clips = clips[:a.limit]
    m = load_model(a.device)
    print(f"[drivejepa] {a.split} clips={len(clips)} device={a.device}", flush=True)
    names, trajs, preds, labels = [], [], [], []; t0 = time.time()
    for i, c in enumerate(clips):
        cd = os.path.join(NR, "cache", a.split, c)
        fr = {f["frame_index"]: f for f in json.load(open(os.path.join(cd, "metadata.json")))["frames"]}
        e = load_pickle(os.path.join(cd, fr[KEY]["ego_state"]))
        cmd = CMD.get(((fr[KEY].get("mission_goal") or {}).get("command")), 3)
        st = np.zeros(8, np.float32); st[cmd] = 1
        st[4:8] = [e.velocity["vx"], e.velocity["vy"], e.acceleration["ax"], e.acceleration["ay"]]
        im = front(os.path.join(root, c, fr[KEY]["sensors"]["cameras"]["front"])).transpose(2, 0, 1)
        cam = torch.from_numpy(np.stack([im, im], 1)).unsqueeze(0).to(a.device)   # [1, C, 2, H, W]
        with torch.no_grad():
            p = m(cam, torch.from_numpy(st).unsqueeze(0).to(a.device))["trajectory"][0].float().cpu().numpy().astype(np.float64)
        g = to_51(p, (e.pose["x"], e.pose["y"], e.pose["yaw"]))
        names.append(c); trajs.append(g); preds.append(p)
        if a.score: labels.append(score(a.split, c, g) or [np.nan] * 5)
        if i < 3 or (i + 1) % 50 == 0:
            msg = f"  [{i+1}/{len(clips)}] {(time.time()-t0)/(i+1):.2f}s/clip end4s={p[-1,:2].round(1).tolist()} v={st[4]:.1f} cmd={cmd}"
            if labels:
                L = np.array(labels); n = L[:, 0] * L[:, 1] * (.3 * L[:, 2] + .2 * L[:, 3] + .5 * L[:, 4])
                msg += f" NPS so far {np.nanmean(n):.4f}"
            print(msg, flush=True)
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    out = os.path.join(HERE, "runs", f"drivejepa_{a.split}{'_n' + str(a.limit) if a.limit else ''}.npz")
    np.savez_compressed(out, names=np.array(names), traj=np.array(trajs), pred8=np.array(preds),
                        **({"labels": np.array(labels, float)} if labels else {}))
    print(f"[drivejepa] {len(names)} clips -> {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
