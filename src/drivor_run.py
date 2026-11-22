"""DrivoR (valeoai, CVPR 2026, Apache-2.0; NAVSIM-v1 checkpoint drivor_Nav1_25epochs.pth) zero-shot on nuReasoning clips.

Inputs, as its NAVSIM feature builder (ext/DrivoR/navsim/agents/drivoR/drivor_features.py) builds them, without the NAVSIM
data pipeline: current frame (key frame 100) of 4 cameras in the builder's order [cam_f0, cam_b0, cam_l0, cam_r0] <-
our [front, back, front_left, front_right]; each 16:9 centre-cropped (NAVSIM cameras are 1920x1080, ours 3:2) and resized
to 1148x672, ImageNet-normalised; ego status [pose (0,0,0 = current), velocity vx vy, acceleration ax ay, command one-hot
(left, straight, right, unknown)] in the ego frame. Output: 64 proposals x 8 poses (0.5..4 s) + learned sub-scores; the
model's own choice (argmax of its PDM-style score) is the trajectory, extended to 5 s as for WA-JEPA (wajepa_run.to_51).
pytorch_lightning is only imported by its logger for a decorator; stubbed here (their code is not modified).
  python src/drivor_run.py --split validation [--limit N] [--score]
  -> runs/drivor_<split>.npz {names, traj [N, 51, 3] global, prop [N, 64, 8, 3] ego frame, score [N, 64], origin}
"""
import argparse, json, os, sys, time, types
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from wajepa_run import NR, KEY, CMD, to_51, score
VIEWS = ["front", "back", "front_left", "front_right"]                    # cam_f0, cam_b0, cam_l0, cam_r0


def load_model(device):
    import torch
    pl = types.ModuleType("pytorch_lightning"); ut = types.ModuleType("pytorch_lightning.utilities")
    rz = lambda f: f; rz.rank = 0; ut.rank_zero_only = rz; pl.utilities = ut
    sys.modules.setdefault("pytorch_lightning", pl); sys.modules.setdefault("pytorch_lightning.utilities", ut)
    sys.path.insert(0, os.path.join(HERE, "ext/DrivoR"))
    from omegaconf import OmegaConf
    from navsim.agents.drivoR.drivor_model import DrivoRModel
    cfg = OmegaConf.load(os.path.join(HERE, "ext/DrivoR/navsim/planning/script/config/common/agent/drivoR.yaml")).config
    cfg.image_backbone.model_weights = os.path.join(HERE, "weights/drivor/vit_small_patch14_reg4_dinov2.safetensors")
    m = DrivoRModel(cfg)
    sd = torch.load(os.path.join(HERE, "weights/drivor/drivor_Nav1_25epochs.pth"), map_location="cpu", weights_only=False)["state_dict"]
    mi, un = m.load_state_dict({k.replace("agent._drivor_model.", ""): v for k, v in sd.items()}, strict=False)
    assert not mi and not un, (mi[:3], un[:3])
    return m.to(device).eval()


def inputs(split, clip, img_root):
    import cv2
    from nureasoning.common.pickle_io import load_pickle
    cd = os.path.join(NR, "cache", split, clip); fr = {f["frame_index"]: f for f in json.load(open(os.path.join(cd, "metadata.json")))["frames"]}
    ims = []
    for v in VIEWS:
        im = cv2.cvtColor(cv2.imread(os.path.join(img_root, clip, fr[KEY]["sensors"]["cameras"][v])), cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]; nh = int(round(w * 9 / 16)); y0 = (h - nh) // 2
        im = cv2.resize(im[y0:y0 + nh], (1148, 672), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        ims.append(((im - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)).transpose(2, 0, 1))
    e = load_pickle(os.path.join(cd, fr[KEY]["ego_state"])); cmd = CMD.get(((fr[KEY].get("mission_goal") or {}).get("command")), 3)
    st = np.zeros(11, np.float32); st[3:7] = [e.velocity["vx"], e.velocity["vy"], e.acceleration["ax"], e.acceleration["ay"]]; st[7 + cmd] = 1
    return np.stack(ims).astype(np.float32), st, np.array([e.pose["x"], e.pose["y"], e.pose["yaw"]], float)


def _score(args):
    split, c, g = args; return c, score(split, c, g)


def main():
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="validation"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    a = ap.parse_args(); root = os.path.join(HERE, "data/img", a.split)
    clips = sorted(c for c in os.listdir(root) if os.path.isfile(os.path.join(root, c, "_done.json")))[:a.limit or None]
    out = os.path.join(HERE, f"runs/drivor_{a.split}.npz"); done = {}
    if os.path.isfile(out):
        z = np.load(out); done = {str(c): (t, p, s, o) for c, t, p, s, o in zip(z["names"], z["traj"], z["prop"], z["score"], z["origin"])}
    m = load_model(a.device); t0 = time.time(); todo = [c for c in clips if c not in done]
    print(f"[drivor] {a.split}: {len(done)} done, {len(todo)} to go on {a.device}", flush=True)
    def save():
        ks = list(done); np.savez(out, names=np.array(ks), traj=np.array([done[k][0] for k in ks]), prop=np.array([done[k][1] for k in ks]),
                                  score=np.array([done[k][2] for k in ks]), origin=np.array([done[k][3] for k in ks]))
    for i, c in enumerate(todo):
        im, st, o = inputs(a.split, c, root)
        with torch.no_grad():
            r = m({"image": torch.from_numpy(im)[None].to(a.device), "ego_status": torch.from_numpy(st)[None, None].to(a.device)})
        t8 = r["trajectory"][0].float().cpu().numpy().astype(np.float64)
        done[c] = (to_51(t8, o), r["proposals"][0].float().cpu().numpy(), r["pdm_score"][0].float().cpu().numpy(), o)
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            save(); print(f"  [{len(done)}/{len(clips)}] {(time.time() - t0) / (i + 1):.2f}s/clip", flush=True)
    if a.score and a.split != "test":
        from concurrent.futures import ProcessPoolExecutor
        z = np.load(out)
        with ProcessPoolExecutor(8) as ex: R = dict(ex.map(_score, [(a.split, str(c), t) for c, t in zip(z["names"], z["traj"])], chunksize=8))
        L = np.array([R[c] if R[c] else [np.nan] * 5 for c in map(str, z["names"])], float)
        np.savez(out, **{k: z[k] for k in z.files}, labels=L)
        N = L[:, 0] * L[:, 1] * (.3 * L[:, 2] + .2 * L[:, 3] + .5 * L[:, 4])
        print(f"{a.split} DrivoR zero-shot: NPS {np.nanmean(N):.4f} on {len(N)} clips | collision {np.nanmean(L[:, 0]):.3f} drivable {np.nanmean(L[:, 1]):.3f} "
              f"progress {np.nanmean(L[:, 2]):.3f} comfort {np.nanmean(L[:, 3]):.3f} human {np.nanmean(L[:, 4]):.3f}", flush=True)


if __name__ == "__main__":
    main()
