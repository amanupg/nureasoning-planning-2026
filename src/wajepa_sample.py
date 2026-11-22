"""Step 1 diagnostic: how much better are the BEST of K sampled trajectories than a typical one?

WA-JEPA's planner is a flow-matching generator: different starting noise gives different plausible
trajectories. For each validation clip we encode the scene ONCE (frozen encoder + scene projector),
then run the flow planner K times with different noise (seeds 1000+k), and score every sample with the
devkit. Same inputs and post-processing as src/wajepa_ft.py predict (and src/wajepa_run.py).
  python src/wajepa_sample.py sample --tag w1 --K 16 [--limit N]   -> runs/samples_<tag>_validation.npz (resumable)
  python src/wajepa_sample.py score  --tag w1                      -> runs/samples_<tag>_validation_scores.npz + report
"""
import argparse, collections, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from wajepa_run import to_51, score
nps = lambda L: L[..., 0] * L[..., 1] * (.3 * L[..., 2] + .2 * L[..., 3] + .5 * L[..., 4])


def encode(m, f):
    """Scene context for a batch-1 feature dict (the part of predict_trajectory before the flow loop)."""
    import torch
    f = m.prepare_batch(f)
    history = m._ensure_multiview(m._require_tensor(f, "history_images"))
    bsz, _, views, ch, h, w = history.shape
    future = torch.zeros(bsz, m.num_future_frames, views, ch, h, w, device=history.device, dtype=history.dtype)
    clip = m._make_full_clip(history, future)
    masks = m.mask_sampler.full_future_mask(bsz, device=clip.device)
    idx = masks.indices[0]; masks_x = masks.masks_x[0]
    view_idx = (idx.unsqueeze(1) * m.num_cameras + torch.arange(m.num_cameras, device=clip.device).unsqueeze(0)).reshape(-1)
    cs = m._encode_scene_context(clip.index_select(0, view_idx), masks_x.index_select(0, idx), idx.numel())
    cti = m._history_indices_from_masks(masks_x.index_select(0, idx), idx.numel())
    return f, cs, cti


def flow(m, f, cs, cti, seeds):
    """Run the flow planner once per seed (batched), sharing the encoded scene."""
    import torch
    c = len(seeds); dev = cs.device; dt_ = cs.dtype; A = m.driving_condition_adapter
    fc = {k: (v.expand(c, *v.shape[1:]) if k in ("ego_status", "history_trajectory") else v) for k, v in f.items()}
    cond = A.inference_conditions(fc).select(torch.arange(c, device=dev))
    csc, ctic = cs.expand(c, *cs.shape[1:]), cti.expand(c, *cti.shape[1:])
    fcs = m._future_mask_condition(c, device=dev, dtype=dt_)
    scenes, trajs = [], []
    for s in seeds:                                   # per-seed noise, so results don't depend on the chunking
        g = torch.Generator(device="cpu").manual_seed(int(s))
        scenes.append(torch.randn((1, m.predictor.num_scene_tokens, m.scene_projector.scene_dim), generator=g))
        trajs.append(A.initial_inference_trajectory(m, 1, device=torch.device("cpu"), dtype=torch.float32, generator=g,
                                                   sample_device=torch.device("cpu")))
    scene = torch.cat(scenes).to(dev, dt_) * m.flow_inference_noise_scale; traj = torch.cat(trajs).to(dev, dt_)
    ti = A.prepare_inference_inputs(m, cond, dtype=dt_, device=dev)
    steps = max(m.flow_num_inference_steps, 1); dt = 1.0 / steps
    for step in range(steps):
        tv = min(step / steps, 1.0 - 1e-4); t = torch.full((c,), tv, device=dev, dtype=dt_)
        ti["noisy_trajectory"] = traj
        ps, pt = m.predictor(context_scene=csc, context_token_indices=ctic, noisy_future_scene=scene,
                             future_condition_scene=fcs, t_cont=t, trajectory_inputs=ti)
        den = (1.0 - t).clamp_min(1e-3).view(-1, 1, 1)
        scene = scene + dt * (ps - scene) / den; traj = traj + dt * (pt - traj) / den
    return A.inference_output(m, traj)


def sample(a):
    import torch
    from wajepa_ft import build
    from wajepa_run import clip_inputs
    m = build(a.device, True)
    if a.tag != "zs":
        sd = torch.load(os.path.join(HERE, f"runs/wajepa_ft_{a.tag}.pt"), map_location="cpu")["state_dict"]
        mi, un = m.load_state_dict(sd, strict=False); assert not un
    m.eval(); root = os.path.join(HERE, "data/img", a.split)
    clips = sorted(c for c in os.listdir(root) if os.path.isfile(os.path.join(root, c, "_done.json")))
    if a.limit: clips = clips[:a.limit]
    out = os.path.join(HERE, f"runs/samples_{a.tag}_{a.split}.npz"); done = {}
    if os.path.isfile(out):
        z = np.load(out); done = {str(c): t for c, t in zip(z["names"], z["traj"])}
    seeds = [1000 + k for k in range(a.K)]; t0 = time.time(); todo = [c for c in clips if c not in done]
    print(f"[sample] {a.tag} {a.split}: {len(done)} done, {len(todo)} to go, K={a.K}", flush=True)
    for i, c in enumerate(todo):
        imgs, st, hist, origin, cmd = clip_inputs(a.split, c, "169")
        f = {"history_images": torch.from_numpy(imgs).unsqueeze(0).to(a.device),
             "ego_status": torch.from_numpy(st).unsqueeze(0).to(a.device),
             "history_trajectory": torch.from_numpy(hist).unsqueeze(0).to(a.device)}
        with torch.no_grad(), torch.autocast(a.device, dtype=torch.bfloat16):
            f, cs, cti = encode(m, f)
            P = torch.cat([flow(m, f, cs, cti, seeds[j:j + a.chunk]) for j in range(0, a.K, a.chunk)]).float().cpu().numpy()
        done[c] = np.stack([to_51(p.astype(np.float64), origin) for p in P])
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            np.savez(out, names=np.array(list(done)), traj=np.array(list(done.values())))
            print(f"  [{len(done)}/{len(clips)}] {(time.time()-t0)/(i+1):.2f}s/clip", flush=True)


def _sc(args):
    split, c, G = args
    return c, [score(split, c, g) or [np.nan] * 5 for g in G]


def report(a):
    from concurrent.futures import ProcessPoolExecutor
    z = np.load(os.path.join(HERE, f"runs/samples_{a.tag}_{a.split}.npz"))
    with ProcessPoolExecutor(8) as ex:
        R = dict(ex.map(_sc, [(a.split, str(c), t) for c, t in zip(z["names"], z["traj"])], chunksize=2))
    names = sorted(R); L = np.array([R[c] for c in names], float)                # (N, K, 5)
    np.savez_compressed(os.path.join(HERE, f"runs/samples_{a.tag}_{a.split}_scores.npz"), names=np.array(names), labels=L)
    S = nps(L); K = S.shape[1]; log = lambda c: "_".join(c.split("_")[:2])
    print(f"{len(names)} clips x K={K} samples ({a.tag}, {a.split})")
    print(f"  one random sample (mean over samples)  {np.nanmean(S):.4f}")
    for k in sorted({1, 2, 4, 8, K}):
        print(f"  oracle best of {k:2d}                     {np.nanmean(np.nanmax(S[:, :k], 1)):.4f}")
    print(f"  oracle worst of {K}                     {np.nanmean(np.nanmin(S, 1)):.4f}")
    best = np.nanargmax(S, 1)
    for i, nm in enumerate(["collision", "drivable", "progress", "comfort", "human"]):
        print(f"  {nm:9s} sample mean {np.nanmean(L[..., i]):.4f}  in best-of-{K} {np.nanmean(L[np.arange(len(names)), best, i]):.4f}")
    spread = np.nanmax(S, 1) - np.nanmin(S, 1)
    print(f"  clips where samples differ by >0.1 NPS: {np.mean(spread > 0.1):.1%}; all samples gated to 0: {np.mean(np.nanmax(S, 1) == 0):.1%}")
    L2 = sorted({log(c) for c in names}); rng = np.random.RandomState(0); by = collections.defaultdict(list)
    for c, b, m_ in zip(names, np.nanmax(S, 1), np.nanmean(S, 1)): by[log(c)].append(b - m_)
    bs = [np.mean(np.concatenate([by[l] for l in rng.choice(L2, len(L2))])) for _ in range(1000)]
    print(f"  best-of-{K} minus mean: {np.nanmean(np.nanmax(S, 1) - np.nanmean(S, 1)):+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")


def main():
    import torch
    sys.path.insert(0, os.path.join(HERE, "ext/WA-JEPA"))
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["sample", "score"])
    ap.add_argument("--tag", default="w1"); ap.add_argument("--split", default="validation")
    ap.add_argument("--K", type=int, default=16); ap.add_argument("--chunk", type=int, default=4); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    a = ap.parse_args(); {"sample": sample, "score": report}[a.cmd](a)


if __name__ == "__main__":
    main()
