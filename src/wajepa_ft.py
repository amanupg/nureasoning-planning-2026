"""WA-JEPA fine-tune on nuReasoning with the V-JEPA 2.1 encoder FROZEN (Aman's decision).

Model = the released WA-JEPA checkpoint, built from the eval config plus the training-only keys of the
upstream NAVSIM training config (EMA, JEPA masks). Trained: flow predictor, trajectory head, history /
ego encoders (and the scene projector unless --freeze-projector). Frozen: encoder + target encoder.
Losses as upstream (configs/causal_future_jepa_mv_navsim.yaml): 0.5 x loss_jepa_pred + 1.0 x loss_traj.

Data: training samples (clip, r) from the main track's a15ft index (read-only); per sample 4 cameras
[front_left, front, front_right, back] x (4 history + 8 future) frames at 2 Hz (r-15 .. r+40), each
centre-cropped to 16:9 and resized to 256x512 exactly as at inference (wajepa_run.image, crop "169").
Ego status / history / future poses come from the nuReasoning cache (read-only), built the same way as
wajepa_run.clip_inputs. Dev set = the same 10% of training drive logs as the Drive-JEPA fine-tune
(dev-seed 0); the checkpoint with the lowest dev FDE@end (flow sampling, 4 steps) is kept.

  python src/wajepa_ft.py train --tag w1 --img-root data/img_wa/train [--limit-clips 5 --steps 20]
  python src/wajepa_ft.py predict --tag w1 --split validation --score
"""
import argparse, functools, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from wajepa_run import NR, CAMS, CMD, rel_se2, image, to_51, score       # shared with zero-shot inference
HIST = [-15, -10, -5, 0]; FUT = list(range(5, 45, 5))
log_of = lambda c: "_".join(c.split("_")[:2])


@functools.lru_cache(maxsize=4096)
def _meta(clip):
    d = os.path.join(NR, "cache/train", clip)
    return {f["frame_index"]: f for f in json.load(open(os.path.join(d, "metadata.json")))["frames"]}


def sample(clip, r, img_root, crop="169"):
    from nureasoning.common.pickle_io import load_pickle
    fr = _meta(clip); d = os.path.join(NR, "cache/train", clip)
    egos = {o: load_pickle(os.path.join(d, fr[r + o]["ego_state"])) for o in HIST + FUT}
    P = np.array([[egos[o].pose["x"], egos[o].pose["y"], egos[o].pose["yaw"]] for o in HIST + FUT], float)
    rel = rel_se2(P[len(HIST) - 1], P).astype(np.float32)
    k = egos[0]; st = np.zeros(8, np.float32)
    st[CMD.get(((fr[r].get("mission_goal") or {}).get("command")), 3)] = 1
    st[4:8] = [k.velocity["vx"], k.velocity["vy"], k.acceleration["ax"], k.acceleration["ay"]]
    im = lambda o: np.stack([image(os.path.join(img_root, clip, fr[r + o]["sensors"]["cameras"][c]), crop)
                             for c in CAMS]).transpose(0, 3, 1, 2)              # [V, 3, H, W]
    return {"history_images": np.stack([im(o) for o in HIST]).astype(np.float32),
            "future_images": np.stack([im(o) for o in FUT]).astype(np.float32),
            "history_trajectory": rel[:len(HIST)], "future_trajectory": rel[len(HIST):], "ego_status": st}


class DS:
    """Training samples. With data/wa_train_meta.npz + wa_train_paths.json.gz (src/wa_meta_train.py) it
    needs only those files and the frames (GPU pod); otherwise it reads the nuReasoning cache (local)."""
    def __init__(self, rows, img_root, packed=None, crop="169", horizon=8, meta_tag=""):
        self.rows, self.root, self.crop, self.h = rows, img_root, crop, horizon   # crop "full" for pre-cropped frames
        self.rl, self.rl_gt, self.rl_mode = None, 0.0, "rwr"                  # set from rl_targets() for the RL rounds
        sfx = f"_{meta_tag}" if meta_tag else ""; mp = os.path.join(HERE, f"data/wa_train_meta{sfx}.npz")
        self.packed = os.path.isfile(mp) if packed is None else packed
        if self.packed:
            import gzip
            z = np.load(mp); key = {(str(c), int(r)): i for i, (c, r) in enumerate(zip(z["clip"], z["r"]))}
            self.ix = [key[(c, r)] for c, r in rows]
            self.st, self.hi = z["status"], z["hist"]; self.fu = z["fut10"] if horizon == 10 else z["fut"]
            with gzip.open(os.path.join(HERE, f"data/wa_train_paths{sfx}.json.gz"), "rt") as f: self.paths = json.load(f)

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        import torch
        if not self.packed:
            return {k: torch.from_numpy(v) for k, v in sample(*self.rows[i], self.root).items()}
        c, r = self.rows[i]; j = self.ix[i]; P = self.paths[c]
        im = lambda o: np.stack([image(os.path.join(self.root, c, p), self.crop) for p in P[str(r + o)]]).transpose(0, 3, 1, 2)
        d = {"history_images": np.stack([im(o) for o in HIST]).astype(np.float32),
             "future_images": np.stack([im(o) for o in FUT]).astype(np.float32),
             "history_trajectory": self.hi[j], "future_trajectory": self.fu[j], "ego_status": self.st[j]}
        if self.rl is not None:                                               # RL: target = own sample (torch RNG: per worker)
            T, Pr, A = self.rl.get((c, r), (None, None, None)); d["rl_w"] = np.float32(1.0)   # mixed-in rows: recorded future
            if Pr is not None and float(torch.rand(())) >= self.rl_gt:
                if self.rl_mode == "rwr":                                     # round 1: draw by reward, pull toward it
                    d["future_trajectory"] = T[int(torch.multinomial(torch.from_numpy(Pr), 1))]
                else:                                                         # round 2: uniform draw, weight = advantage
                    k = int(torch.randint(len(T), ())); d["future_trajectory"] = T[k]; d["rl_w"] = np.float32(A[k])
        return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in d.items()}


def rl_targets(path, tau, min_best, neg=0.25, clip_a=2.0):
    """{(clip, r): (samples [K, 10, 3], P(sample) or None, weight per sample)} from runs/rl_<tag>_graded.npz.
    Round 1 (rwr): P ∝ exp((NPS - best NPS) / tau) over the frame's own samples.
    Round 2 (adv): weight = advantage (NPS - frame mean) / (frame std + 0.05), clipped to +-clip_a; negative
    weights (below-average samples, e.g. collisions) are scaled by neg and push the planner away (see wloss).
    P None (= recorded future) when every sample scores below min_best."""
    q = np.load(path); L = q["labels"]
    S = L[..., 0] * L[..., 1] * (.3 * L[..., 2] + .2 * L[..., 3] + .5 * L[..., 4])
    out = {}
    for c, r, T, s in zip(q["clip"], q["r"], q["traj"], S):
        if np.isnan(s).all() or np.nanmax(s) < min_best: out[(str(c), int(r))] = (T, None, None); continue
        w = np.where(np.isnan(s), 0.0, np.exp((np.nan_to_num(s, nan=-1.0) - np.nanmax(s)) / tau))
        A = np.clip(np.nan_to_num((s - np.nanmean(s)) / (np.nanstd(s) + 0.05), nan=0.0), -clip_a, clip_a)
        A = np.where(A < 0, neg * A, A).astype(np.float32)
        out[(str(c), int(r))] = (T.astype(np.float32), w / w.sum(), A)
    return out


def build(device, freeze_projector, horizon=8, unfreeze=0):
    import torch
    from omegaconf import OmegaConf
    from models.factory import build_world_model
    from training.checkpoint import load_checkpoint
    ev = OmegaConf.load(os.path.join(HERE, "ext/WA-JEPA/configs/wa_jepa_navsim_pdms.yaml")); OmegaConf.set_struct(ev, False)
    tr = OmegaConf.load(os.path.join(HERE, "ext/WA-JEPA/configs/causal_future_jepa_mv_navsim.yaml"))
    for k in ("ema_start", "ema_end", "ema_total_steps", "jepa_mask_configs", "freeze_encoder_steps",
              "target_scene_ema_warmup_steps"):
        ev.model[k] = tr.model[k]
    ev.model.require_pretrained = False; ev.model.vjepa2_ckpt = None
    ev.model.init_from_checkpoint = None; ev.model.require_init_from_checkpoint = False
    m = build_world_model(OmegaConf.to_container(ev, resolve=True))
    load_checkpoint(os.path.join(HERE, "weights/wa-jepa/model_state_dict.pt"), m, map_location="cpu", write_key_reports=False)
    frozen = [m.encoder, m.target_encoder, m.target_scene_projector] + ([m.scene_projector] if freeze_projector else [])
    for mod in frozen: mod.requires_grad_(False)
    if horizon != m.predictor.trajectory_horizon:                         # 4 s -> 5 s head: new poses copy the last
        import torch
        pp = m.predictor.traj_pos.data
        m.predictor.traj_pos = torch.nn.Parameter(torch.cat([pp, pp[:, -1:].repeat(1, horizon - pp.shape[1], 1)], 1))
        m.predictor.trajectory_horizon = horizon
    nb = len(m.encoder.blocks)
    for i in range(nb - unfreeze, nb): m.encoder.blocks[i].requires_grad_(True)  # last N encoder blocks trainable
    if unfreeze: m.encoder.norms_block.requires_grad_(True)
    m._kept = tuple(f"encoder.blocks.{i}." for i in range(nb - unfreeze, nb)) + (("encoder.norms_block.",) if unfreeze else ())
    m.set_loss_weights({"loss_jepa_pred": 0.5, "loss_traj": 1.0})
    return m.to(device)


def own_state(m):
    """Checkpoint = everything except the frozen encoder / target encoder (trainable encoder blocks are kept)."""
    return {k: v for k, v in m.state_dict().items()
            if not k.startswith(("encoder.", "target_encoder.")) or k.startswith(getattr(m, "_kept", ()) or ("\0",))}


def rows_split(limit_clips, dev_frac=0.1, dev_seed=0, meta_tag=""):
    mp = os.path.join(HERE, "data/wa_train_meta.npz")                        # packed (pod/cluster) or main-track index
    rows = ([(str(c), int(r)) for c, r in zip(np.load(mp)["clip"], np.load(mp)["r"])] if os.path.isfile(mp) else
            [(c, int(r)) for c, r in json.load(open(os.path.join(NR, "artifacts/a15ft_index_train.json")))])
    ul = np.unique([log_of(c) for c, _ in rows])                              # same split as djft.py
    dev_logs = set(np.random.RandomState(dev_seed).permutation(ul)[:int(round(dev_frac * len(ul)))])
    if meta_tag:                                                              # e.g. 'all' = base + extra clips: the dev logs stay
        z = np.load(os.path.join(HERE, f"data/wa_train_meta_{meta_tag}.npz"))   # those of the base index (new logs -> train)
        rows = [(str(c), int(r)) for c, r in zip(z["clip"], z["r"])]
    if limit_clips:
        keep = set(sorted({c for c, _ in rows})[:limit_clips]); rows = [x for x in rows if x[0] in keep]
    tr = [x for x in rows if log_of(x[0]) not in dev_logs]; dv = [x for x in rows if log_of(x[0]) in dev_logs]
    if limit_clips and not dv: dv = tr[-4:]                                   # smoke only
    return tr, dv


def train(a):
    import torch
    torch.manual_seed(a.seed); dev = torch.device(a.device)
    m = build(a.device, a.freeze_projector, a.horizon, a.unfreeze_blocks)
    if a.init_from:                                                           # continue from a fine-tuned run (RL round)
        ck0 = torch.load(os.path.join(HERE, f"runs/wajepa_ft_{a.init_from}.pt"), map_location="cpu", weights_only=False)
        _, un = m.load_state_dict(ck0["state_dict"], strict=False); assert not un, un[:5]
        print(f"  initialised from runs/wajepa_ft_{a.init_from}.pt (step {ck0.get('step')})", flush=True)
    m.train(); m.encoder.eval(); m.target_encoder.eval()
    tr, dv = rows_split(a.limit_clips, meta_tag=a.meta_tag)
    rl = rl_targets(os.path.join(HERE, a.rl_graded), a.rl_tau, a.rl_min_best, a.rl_neg) if a.rl_graded else None
    if rl is not None and a.rl_mode == "adv":                                # per-example weighted flow loss
        import types, torch.nn.functional as Fn
        cap = a.rl_cap; m._rl_w = None; m._rl_stats = []

        def wloss(self, preds, zero_ref):
            w = self._rl_w.reshape(-1); out = []                       # loader gives [B, 1]
            for pr in preds:
                if not bool(pr["compute_traj_loss"]): continue
                l = Fn.mse_loss(pr["pred_traj"].float(), pr["target_traj"].float(), reduction="none").flatten(1).mean(1)
                ww = w[pr["idx"]].float()
                out.append(torch.where(ww >= 0, ww * l, ww * torch.clamp(l, max=cap)))   # push away only up to cap
                self._rl_stats.append((l.detach()[ww > 0].sum().item(), (ww > 0).sum().item(),
                                       l.detach()[ww < 0].sum().item(), (ww < 0).sum().item()))
            return torch.cat(out).sum() / w.numel() if out else zero_ref.sum() * 0.0
        m._trajectory_flow_loss = types.MethodType(wloss, m)
    if rl is not None:
        other = [x for x in tr if x not in rl]; tr = [x for x in tr if x in rl]
        if a.rl_mix:                                                          # round 3: ordinary frames against narrowing
            idx = np.random.RandomState(a.seed).permutation(len(other))[:int(round(a.rl_mix * len(tr)))]
            tr = tr + [other[i] for i in idx]
            print(f"  mixed in {len(idx)} ordinary train frames (recorded future), ratio {a.rl_mix}", flush=True)
        print(f"  RL targets: {len(rl)} graded frames, {len(tr)} in the train split; "
              f"{sum(v[1] is None for v in rl.values())} fall back to the recorded future; mode {a.rl_mode}, tau {a.rl_tau}, "
              f"GT share {a.rl_gt}, neg {a.rl_neg}, cap {a.rl_cap}", flush=True)
    if a.dev_max and len(dv) > a.dev_max: dv = [dv[i] for i in np.random.RandomState(0).choice(len(dv), a.dev_max, replace=False)]
    params = [p for p in m.parameters() if p.requires_grad]
    enc = [p for n, p in m.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    rest = [p for n, p in m.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    groups = [{"params": rest, "lr": a.lr}] + ([{"params": enc, "lr": a.enc_lr}] if enc else [])
    print(f"train {len(tr)} / dev {len(dv)} samples; trainable {sum(p.numel() for p in params)/1e6:.1f}M of "
          f"{sum(p.numel() for p in m.parameters())/1e6:.1f}M params (encoder blocks trainable: {a.unfreeze_blocks}, "
          f"horizon {a.horizon} poses)", flush=True)
    def L(rows, sh, rlt=None):
        ds = DS(rows, a.img_root, crop=a.crop, horizon=a.horizon, meta_tag=a.meta_tag); ds.rl, ds.rl_gt, ds.rl_mode = rlt, a.rl_gt, a.rl_mode
        return torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=sh, drop_last=sh,
                                           num_workers=a.workers, persistent_workers=a.workers > 0)
    tl, dl = L(tr, True, rl), L(dv, False)
    steps = a.steps or a.epochs * len(tl)
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=steps, pct_start=0.05)
    amp = lambda: torch.autocast(a.device if a.device != "mps" else "mps", dtype=torch.bfloat16, enabled=bool(a.amp))

    def dev_fde():                                                            # fixed sampling noise: comparable across steps / runs
        m.eval(); err = []
        with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []), torch.no_grad(), amp():
            torch.manual_seed(1234)
            for b in dl:
                f = {k: b[k].to(dev) for k in ("history_images", "ego_status", "history_trajectory")}
                p = m.predict_trajectory(f).float().cpu()
                err.append(torch.linalg.norm(p[:, -1, :2] - b["future_trajectory"][:, -1, :2], dim=-1))
        m.train(); m.encoder.eval(); m.target_encoder.eval()
        return float(torch.cat(err).mean())
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True); ck = os.path.join(HERE, f"runs/wajepa_ft_{a.tag}.pt")
    last = os.path.join(HERE, f"runs/wajepa_ft_{a.tag}_last.pt")
    own = lambda: own_state(m)
    import signal
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))   # Slurm --signal before the wall limit
    if os.path.isfile(last):                                                  # resume (time-limited jobs)
        R = torch.load(last, map_location="cpu", weights_only=False)
        m.load_state_dict(R["model"], strict=False); opt.load_state_dict(R["opt"]); sched.load_state_dict(R["sched"])
        step, best, done_s = R["step"], tuple(R["best"]), R["elapsed"]
        print(f"  resumed from step {step} (best dev FDE@end {best[0]:.3f} m at step {best[1]})", flush=True)
    else:
        f0 = dev_fde(); print(f"  step 0 (released) dev FDE@end {f0:.3f} m", flush=True)
        step, best, done_s = 0, (f0, 0), 0.0
    torch.manual_seed(a.seed + step)                                          # new shuffle order after a resume
    t0 = time.time(); t_ck = t0; start = step

    def save_last():
        if a.no_resume: return                                                # short runs on a full disk: no 2.7 GB resume file
        torch.save({"model": own(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "best": list(best),
                    "elapsed": done_s + time.time() - t0, "args": vars(a)}, last + ".tmp")
        os.replace(last + ".tmp", last)
    eval_every = a.eval_every or len(tl)
    named_tr = [(n, p) for n, p in m.named_parameters() if p.requires_grad]   # validity: how far the weights moved
    grp_of = lambda n: "enc" if n.startswith("encoder.") else ("traj" if "traj" in n else "shared")
    theta0 = {n: p.detach().float().clone() for n, p in named_tr}
    def moved():
        out = {}
        for g in ("enc", "traj", "shared"):
            ns = [(n, p) for n, p in named_tr if grp_of(n) == g]
            if ns:
                d = sum(float((p.detach().float() - theta0[n]).pow(2).sum()) for n, p in ns) ** 0.5
                out[g] = d / max(sum(float(theta0[n].pow(2).sum()) for n, _ in ns) ** 0.5, 1e-12)
        return " ".join(f"{g} {v:.2e}" for g, v in out.items())
    win = []                                                                  # loss means over each log window
    while step < steps and not stop["flag"]:
        for b in tl:
            b = {k: v.to(dev) for k, v in b.items()}
            if "rl_w" in b:
                w_ = b.pop("rl_w")
                if a.rl_mode == "adv": m._rl_w = w_
            with amp():
                out = m(b)
            ld = out["loss"]; loss = a.jepa_w * ld["loss_jepa_pred"] + 1.0 * ld["loss_traj"]
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); step += 1
            m.after_optimizer_step(step); win.append((ld["loss_jepa_pred"].item(), ld["loss_traj"].item()))
            if step % a.log_every == 0 or step - start <= 3:
                mem = f" peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if torch.cuda.is_available() else ""
                print(f"  step {step}/{steps} loss {loss.item():.4f} (jepa {ld['loss_jepa_pred'].item():.4f}, "
                      f"traj {ld['loss_traj'].item():.4f}) {(time.time()-t0)/(step-start):.2f}s/step{mem} | window mean jepa "
                      f"{np.mean([w[0] for w in win]):.4f} traj {np.mean([w[1] for w in win]):.6f} (n {len(win)})", flush=True); win = []
                if getattr(m, "_rl_stats", None):
                    st_ = np.array(m._rl_stats).sum(0); m._rl_stats = []
                    print(f"    rl: pull loss {st_[0] / max(st_[1], 1):.4f} (n {int(st_[1])}), "
                          f"push loss {st_[2] / max(st_[3], 1):.4f} (n {int(st_[3])}, cap {a.rl_cap})", flush=True)
            if step % eval_every == 0 or step == steps:
                fde = dev_fde(); tag = ""
                if fde < best[0] or rl is not None:                          # RL: keep the latest (FDE is not the objective)
                    best = (fde, step); tag = "  <- best"
                    ckp = None if a.no_save else (ck.replace(".pt", f"_s{step}.pt") if a.snapshots else ck)   # snapshots: one file per eval step
                    if ckp:
                        torch.save({"state_dict": own(), "args": vars(a), "step": step, "dev_fde": fde}, ckp + ".tmp")
                        os.replace(ckp + ".tmp", ckp)                         # a full disk never corrupts the kept model
                print(f"  step {step} dev FDE@end {fde:.3f} m{tag} ({done_s + time.time()-t0:.0f}s) | weights moved (rel L2) {moved()}", flush=True)
                save_last(); t_ck = time.time()
            elif time.time() - t_ck > 60 * a.ckpt_minutes:
                save_last(); t_ck = time.time()
            if (a.max_hours and time.time() - t0 > 3600 * a.max_hours) or stop["flag"] or step >= steps:
                break
        if (a.max_hours and time.time() - t0 > 3600 * a.max_hours) or stop["flag"]:
            save_last(); print(f"  stopping at step {step} (time limit / signal); resume with the same command", flush=True)
            return
    save_last()
    print(f"kept step {best[1]} (dev FDE@end {best[0]:.3f} m) -> {ck if best[1] else 'released weights (no checkpoint)'}", flush=True)
    print("TRAINING_DONE", flush=True)


def load_ft(tag, device):
    """Fine-tuned model from runs/wajepa_ft_<tag>.pt, built with that run's horizon / unfrozen blocks."""
    import torch
    ck = torch.load(os.path.join(HERE, f"runs/wajepa_ft_{tag}.pt"), map_location="cpu", weights_only=False)
    ar = ck.get("args", {})
    m = build(device, True, int(ar.get("horizon", 8)), int(ar.get("unfreeze_blocks", 0)))
    missing, unexpected = m.load_state_dict(ck["state_dict"], strict=False)
    assert not unexpected and all(k.startswith(("encoder.", "target_encoder.")) for k in missing), (unexpected, missing[:5])
    assert not any(k.startswith(m._kept) for k in missing) if m._kept else True
    return m, ar


def predict(a):
    """Same inputs and post-processing as the zero-shot run (wajepa_run.py), with fine-tuned weights."""
    import torch
    from wajepa_run import clip_inputs
    m, _ = load_ft(a.tag, a.device)
    m.eval(); root = os.path.join(HERE, "data/img", a.split)
    clips = sorted(c for c in os.listdir(root) if os.path.isfile(os.path.join(root, c, "_done.json")))
    if a.limit: clips = clips[:a.limit]
    out = os.path.join(HERE, f"runs/wajepa_ft_{a.tag}{os.environ.get('WA_SUFFIX', '')}_{a.split}.npz"); part = out.replace(".npz", "_partial.npz")
    done = {}
    if os.path.isfile(part):                                          # resume after a hang / crash
        z = np.load(part); done = {str(c): (t, p) for c, t, p in zip(z["names"], z["traj"], z["pred8"])}
        print(f"  resuming: {len(done)} clips already predicted", flush=True)
    t0 = time.time(); todo = [c for c in clips if c not in done]

    def save_part():
        np.savez(part, names=np.array(list(done)), traj=np.array([v[0] for v in done.values()]),
                 pred8=np.array([v[1] for v in done.values()]))
    for i, c in enumerate(todo):
        imgs, st, hist, origin, cmd = clip_inputs(a.split, c, "169")
        f = {"history_images": torch.from_numpy(imgs).unsqueeze(0).to(a.device),
             "ego_status": torch.from_numpy(st).unsqueeze(0).to(a.device),
             "history_trajectory": torch.from_numpy(hist).unsqueeze(0).to(a.device)}
        with torch.no_grad(), torch.autocast(a.device, dtype=torch.bfloat16, enabled=bool(a.amp)):
            p = m.predict_trajectory(f).squeeze(0).float().cpu().numpy().astype(np.float64)
        done[c] = (to_51(p, origin), p)
        if (i + 1) % 50 == 0:
            save_part(); print(f"  [{len(done)}/{len(clips)}] {(time.time()-t0)/(i+1):.2f}s/clip", flush=True)
    names = clips; trajs = [done[c][0] for c in names]; preds = [done[c][1] for c in names]; extra = {}
    if a.score:                                                       # devkit scoring, 8 processes, after prediction
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(8) as ex:
            labels = [r or [np.nan] * 5 for r in ex.map(_score1, [(a.split, c, g) for c, g in zip(names, trajs)], chunksize=8)]
        Lb = np.array(labels, float); n = Lb[:, 0] * Lb[:, 1] * (.3 * Lb[:, 2] + .2 * Lb[:, 3] + .5 * Lb[:, 4])
        extra["labels"] = Lb; print(f"{a.split} {a.tag}: NPS {np.nanmean(n):.4f} on {np.isfinite(n).sum()} clips", flush=True)
    np.savez_compressed(out, names=np.array(names), traj=np.array(trajs), pred8=np.array(preds), **extra)
    if os.path.isfile(part): os.remove(part)
    print(f"-> {out}", flush=True)


def _score1(args):
    return score(*args)


def check(a):
    """Local check without a backward pass (MPS cannot hold a training step: >42 GB at batch 1).
    Data path, shapes, the training forward + losses (no grad), dev FDE, checkpoint save/load, predict."""
    import torch
    m = build(a.device, a.freeze_projector, a.horizon, a.unfreeze_blocks); m.eval()
    tr, dv = rows_split(a.limit_clips or 5)
    b = DS(tr, a.img_root, horizon=a.horizon)[0]
    print({k: tuple(v.shape) for k, v in b.items()}, flush=True)
    b = {k: v.unsqueeze(0).to(a.device) for k, v in b.items()}
    with torch.no_grad(), torch.autocast(a.device, dtype=torch.bfloat16):
        m.train(); m.encoder.eval(); m.target_encoder.eval()
        ld = m(b)["loss"]; m.eval()
        p = m.predict_trajectory({k: b[k] for k in ("history_images", "ego_status", "history_trajectory")}).float().cpu()
    print("losses:", {k: round(float(v), 4) for k, v in ld.items()}, flush=True)
    gt = b["future_trajectory"][0].cpu()
    print("pred 4 s", p[0, -1].numpy().round(2), "gt 4 s", gt[-1].numpy().round(2),
          f"FDE {float(torch.linalg.norm(p[0, -1, :2] - gt[-1, :2])):.2f} m", flush=True)
    ck = os.path.join(HERE, "runs/wajepa_ft_check.pt")
    torch.save({"state_dict": own_state(m)}, ck)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"checkpoint {os.path.getsize(ck)/1e9:.2f} GB; trainable {n/1e6:.1f}M params", flush=True)


def main():
    import torch
    sys.path.insert(0, os.path.join(HERE, "ext/WA-JEPA"))
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["train", "predict", "check"])
    ap.add_argument("--tag", default="w1"); ap.add_argument("--img-root", default=os.path.join(HERE, "data/img_wa/train"))
    ap.add_argument("--split", default="validation"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--limit-clips", type=int, default=0, help="smoke: first N training clips")
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--bs", type=int, default=4); ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--jepa-w", type=float, default=0.5, help="world-model loss weight (upstream 0.5; 0 = planning loss only)")
    ap.add_argument("--wd", type=float, default=0.04); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--freeze-projector", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=8, choices=[8, 10], help="planner poses at 0.5 s (10 = 5 s head)")
    ap.add_argument("--unfreeze-blocks", type=int, default=0, help="train the last N encoder blocks")
    ap.add_argument("--enc-lr", type=float, default=1e-5, help="learning rate for unfrozen encoder blocks (upstream encoder lr)")
    ap.add_argument("--crop", default="169", help="'full' when frames were pre-cropped to 256x512 at ingest (IMG_WAJEPA=1)")
    ap.add_argument("--workers", type=int, default=0); ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--dev-max", type=int, default=2000); ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--init-from", default="", help="start from runs/wajepa_ft_<tag>.pt (e.g. h10u4)")
    ap.add_argument("--rl-graded", default="", help="runs/rl_<tag>_graded.npz: train toward own high-NPS samples")
    ap.add_argument("--rl-tau", type=float, default=0.05); ap.add_argument("--rl-min-best", type=float, default=0.5)
    ap.add_argument("--rl-gt", type=float, default=0.2, help="share of RL batches that keep the recorded future as target")
    ap.add_argument("--rl-mix", type=float, default=0.0, help="add this many ordinary train frames per RL frame (recorded future)")
    ap.add_argument("--snapshots", type=int, default=0, help="keep every eval step as runs/wajepa_ft_<tag>_s<step>.pt")
    ap.add_argument("--no-resume", type=int, default=0, help="do not write the resume file (_last.pt)")
    ap.add_argument("--no-save", type=int, default=0, help="write no checkpoints at all (memory tests)")
    ap.add_argument("--meta-tag", default="", help="'all': base + extra train clips (data/wa_train_meta_all.npz); dev logs unchanged")
    ap.add_argument("--rl-mode", default="rwr", choices=["rwr", "adv"], help="rwr: round 1; adv: advantage-weighted with push-away")
    ap.add_argument("--rl-neg", type=float, default=0.25, help="scale of negative advantages (adv)")
    ap.add_argument("--rl-cap", type=float, default=0.05, help="push-away stops once a bad sample's loss exceeds this (adv)")
    ap.add_argument("--ckpt-minutes", type=float, default=30); ap.add_argument("--max-hours", type=float, default=0, help="stop cleanly after this many hours (resume later)"); ap.add_argument("--score", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    a = ap.parse_args()
    {"train": train, "predict": predict, "check": check}[a.cmd](a)


if __name__ == "__main__":
    main()
