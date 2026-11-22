"""Half-validation fine-tune test of py123d_garage's planning decoder on nuReasoning (2026-11-06, Mac CPU).

Question: does our data cut the zero-shot model's 5 s distance error (5.5 m MAE) enough to matter (kill rule: held-out
5 s distance MAE must fall well below 4.5 m)? Two folds by drive log (the halves every blend report uses): train the
decoder on one half's clips, predict the other half's key frame, swap. Held-out predictions for all 1000 clips ->
runs/garage_ft2f_validation.npz (scored / blended like any member; trained on validation, so never a submission member).
Rows: anchors 95 and 100 per clip (local full front frames exist at 85 / 90 / 95 / 100; t-1 s = anchor - 10).
Labels from the human log, in the anchor rear-axle frame as py123d builds them: trajectory = 64 poses at 0.1 s (constant
velocity past the clip end), path = 64 points at 1 m along the logged future (straight past its end) with path heading.
Training core adapted from alpasim-jepa/src/closed_loop/train_decoder.py (read-only): backbone and other heads frozen in
eval mode, planning_decoder trained with py123d's own flow loss and weights and _forward_heads' training conditioning.
  .venv_garage/bin/python src/garage_ft.py --epochs 8 --lr 1e-4
"""
import argparse, json, math, os, random, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import garage_run as G                                                    # macOS libc shim, policy builder, inputs
HERE = G.HERE; NR = G.NR
log_of = lambda c: "_".join(c.split("_")[:2])


def rel_se2(origin, P):
    ox, oy, oh = origin; c, s = np.cos(-oh), np.sin(-oh); d = P - np.array([ox, oy, oh])
    return np.stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c, np.arctan2(np.sin(d[:, 2]), np.cos(d[:, 2]))], 1)


def row(clip, anchor, root):
    """Features (as garage_run, anchored at `anchor`) and labels from the logged future."""
    import cv2, torch
    sys.path.insert(0, os.path.join(NR, "recon/devkit"))
    from nureasoning.common.pickle_io import load_pickle
    cd = os.path.join(NR, "cache", "validation", clip); fr = {f["frame_index"]: f for f in json.load(open(os.path.join(cd, "metadata.json")))["frames"]}
    rows = []
    for i in (anchor - 10, anchor):
        rgb = cv2.imread(os.path.join(root, clip, fr[i]["sensors"]["cameras"]["front"]), cv2.IMREAD_COLOR)[:, :, ::-1]
        rows.append(cv2.resize(np.ascontiguousarray(rgb), (352, 256)))
    E0 = load_pickle(os.path.join(cd, fr[anchor]["ego_state"])); v, a = E0.velocity, E0.acceleration
    origin = (E0.pose["x"], E0.pose["y"], E0.pose["yaw"])
    fut = [k for k in sorted(fr) if k > anchor]
    P = rel_se2(origin, np.array([[e.pose["x"], e.pose["y"], e.pose["yaw"]] for e in (load_pickle(os.path.join(cd, fr[k]["ego_state"])) for k in fut)]))
    T = np.zeros((64, 3))
    n = min(64, len(P)); T[:n] = P[:n]
    if n < 64:                                                            # constant velocity past the clip end
        vel = (P[n - 1, :2] - (P[n - 2, :2] if n > 1 else np.zeros(2))) / 0.1
        for j in range(n, 64):
            T[j, :2] = P[n - 1, :2] + vel * 0.1 * (j - n + 1); T[j, 2] = P[n - 1, 2]
    xy = np.vstack([[0, 0], P[:, :2]]); seg = np.linalg.norm(np.diff(xy, axis=0), axis=1); s = np.concatenate([[0], np.cumsum(seg)])
    keep = np.concatenate([[True], seg > 1e-3]); xy, s = xy[keep], s[keep]
    hd = np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0])) if len(xy) > 2 else np.zeros(len(xy))
    q = np.arange(1, 65, dtype=float); path = np.zeros((64, 3))
    for j, d in enumerate(q):
        if len(xy) > 2 and d <= s[-1]:
            path[j, 0], path[j, 1] = np.interp(d, s, xy[:, 0]), np.interp(d, s, xy[:, 1]); path[j, 2] = np.interp(d, s, np.unwrap(hd))
        else:                                                             # straight past the end (or a standing ego)
            h = np.unwrap(hd)[-1] if len(xy) > 2 else 0.0; base = xy[-1] if len(xy) > 2 else np.zeros(2); extra = d - (s[-1] if len(xy) > 2 else 0.0)
            path[j] = [base[0] + extra * np.cos(h), base[1] + extra * np.sin(h), h]
    feats = {"camera_feature": torch.as_tensor(np.concatenate(rows, 0)).permute(2, 0, 1).contiguous(),
             "velocity": torch.tensor([float(np.hypot(v["vx"], v["vy"]))], dtype=torch.float32),
             "velocity_vector": torch.tensor([v["vx"], v["vy"]], dtype=torch.float32),
             "acceleration_vector": torch.tensor([a["ax"], a["ay"]], dtype=torch.float32)}
    return {"clip": clip, "anchor": anchor, "feats": feats, "traj": torch.tensor(T, dtype=torch.float32),
            "path": torch.tensor(path, dtype=torch.float32), "origin": origin, "human5": float(np.linalg.norm(T[49, :2]))}


def main():
    import torch
    from py123d_garage.api.abstract_policy_tensors import NavigationConditioning
    from py123d_garage.policy.transfuser.features import TransfuserFeatures
    from py123d_garage.policy.transfuser.network.decoder.unicycle import fit_actions
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--stage", default="posttrain"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="garage_ft2f"); a = ap.parse_args()
    torch.set_num_threads(8); random.seed(0); np.random.seed(0); torch.manual_seed(0)
    root = os.path.join(HERE, "data/img/validation")
    clips = sorted(c for c in os.listdir(root) if os.path.isfile(os.path.join(root, c, "_done.json")))[: a.limit or None]
    L = sorted({log_of(c) for c in clips}); H = set(np.random.RandomState(0).permutation(L)[:len(L) // 2])   # blend-report halves
    t0 = time.time(); R = {}
    for c in clips:
        for an in (95, 100):
            try: R[(c, an)] = row(c, an, root)
            except Exception as e: print(f"  skip {c} {an}: {e}", flush=True)
    print(f"rows {len(R)} from {len(clips)} clips ({time.time() - t0:.0f}s); halves {sum(log_of(c) in H for c in clips)} / {sum(log_of(c) not in H for c in clips)} clips", flush=True)
    dev = torch.device("cpu")

    def nav_for(policy, n):
        k = len(policy.policy_config.required_target_point_distances_m or [])
        return NavigationConditioning(target_points=torch.full((n, k, 2), float("nan")), target_point_headings=torch.full((n, k), float("nan")))

    def batch_of(rows):
        f = TransfuserFeatures(**{k: torch.stack([r["feats"][k] for r in rows]) for k in rows[0]["feats"]})
        return f, torch.stack([r["traj"] for r in rows]), torch.stack([r["path"] for r in rows])

    def label(policy, features, traj, path):
        cfg = policy.policy_config; lab = traj[:, : cfg.trajectory_num_steps]; ac = cfg.planning_config.unicycle_action_config
        if ac is not None:
            act = torch.stack([fit_actions(t, v, ac) for t, v in zip(traj, features.velocity.reshape(-1))]); lab = torch.cat([lab, act.to(lab.dtype)], -1)
        if cfg.planning_config.path_config is not None:
            lab = torch.cat([lab, torch.nn.functional.pad(path, (0, lab.shape[-1] - path.shape[-1]))], 1)
        return lab

    @torch.no_grad()
    def evaluate(policy, rows):
        policy.eval(); out = []
        for i in range(0, len(rows), 32):
            ch = rows[i:i + 32]; f, T, _ = batch_of(ch)
            out.append(policy.forward(f, nav_for(policy, len(ch))).ego_trajectory_se2.float())
        P = torch.cat(out).numpy(); T = torch.stack([r["traj"] for r in rows]).numpy()
        fde = np.linalg.norm(P[:, 49, :2] - T[:, 49, :2], axis=1); dist = np.linalg.norm(P[:, 49, :2], axis=1) - np.array([r["human5"] for r in rows])
        v = np.array([float(r["feats"]["velocity"][0]) for r in rows]); slow = v < 3
        return P, {"fde5": float(fde.mean()), "dist_mae": float(np.abs(dist).mean()), "dist_bias": float(dist.mean()),
                   "dist_mae_slow": float(np.abs(dist[slow]).mean()), "dist_bias_slow": float(dist[slow].mean())}

    held = {}
    for fold in (0, 1):
        tr = [r for (c, an), r in R.items() if (log_of(c) in H) == (fold == 1)]          # fold 0 trains on the second half
        te = [r for (c, an), r in R.items() if (log_of(c) in H) == (fold == 0) and an == 100]
        policy = G.build(a.stage, [0], "cpu"); weights = policy.loss_weights(0)
        _, m0 = evaluate(policy, te)
        print(f"fold {fold}: train {len(tr)} rows, held-out {len(te)} key frames | before: {json.dumps({k: round(x, 3) for k, x in m0.items()})}", flush=True)
        policy.requires_grad_(False); policy.planning_decoder.requires_grad_(True)
        start = {k: v.detach().clone() for k, v in policy.planning_decoder.state_dict().items() if v.is_floating_point()}
        params = [p for p in policy.parameters() if p.requires_grad]
        steps = a.epochs * math.ceil(len(tr) / a.batch)
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps, 1), pct_start=0.1)
        step = 0
        for ep in range(a.epochs):
            policy.eval(); policy.training = True; policy.planning_decoder.train(); random.shuffle(tr); run = []
            for i in range(0, len(tr), a.batch):
                f, T, Pth = batch_of(tr[i:i + a.batch]); nav = nav_for(policy, len(T))
                with torch.no_grad():
                    lidar = None if policy.policy_config.backbone_config.latent else f.lidar_feature
                    bev, img, reg = policy.backbone(f.camera_feature.float(), lidar)
                pred = policy._forward_heads(f, nav, bev, img, reg)
                losses = policy.planning_decoder.compute_loss(pred.trajectory, label(policy, f, T, Pth))
                tot = sum(weights[k] * v.float() for k, v in losses.items())
                opt.zero_grad(set_to_none=True); tot.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step(); step += 1
                run.append(tot.item())
            moved = math.sqrt(sum(float((v.detach() - start[k]).norm()) ** 2 for k, v in policy.planning_decoder.state_dict().items() if k in start)) / \
                math.sqrt(sum(float(t.norm()) ** 2 for t in start.values()))
            P, m = evaluate(policy, te)
            print(f"  fold {fold} epoch {ep + 1}/{a.epochs} step {step}: train loss {np.mean(run):.4f} | decoder moved {moved:.4f} | held-out "
                  f"{json.dumps({k: round(x, 3) for k, x in m.items()})} ({time.time() - t0:.0f}s)", flush=True)
        for r, p in zip(te, P): held[r["clip"]] = (p, r["origin"])
    names = sorted(held)
    np.savez_compressed(os.path.join(HERE, f"runs/{a.tag}_validation.npz"), names=np.array(names),
                        traj=np.array([G.to_global(held[c][0].astype(np.float64), held[c][1]) for c in names]),
                        rel64=np.array([held[c][0] for c in names]))
    print(f"[garage_ft] held-out predictions for {len(names)} clips -> runs/{a.tag}_validation.npz ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
