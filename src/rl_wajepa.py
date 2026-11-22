"""RL round 1 for the fine-tuned WA-JEPA planner (h10u4): sample, grade with the official metric, re-train.

Why: on validation the best of 16 flow samples scores 0.859 vs 0.730 for the model as run, so the model
already produces good trajectories but does not prefer them. Round 1 is reward-weighted fine-tuning
(rejection-sampling / RAFT style): for each training frame draw K samples from the current model, score
every one with the devkit NPS on that frame's own recorded future, then fine-tune the flow planner toward
the high-scoring samples (src/wajepa_ft.py train --rl-graded ...).

  python src/rl_wajepa.py sample --tag h10u4 --n 20000 --shard 0 --nshards 4   (GPU; cluster)
      -> runs/rl_<tag>_samples_s<shard>.npz: clip, r, traj [N, K, 10, 3] (planner frame, as future_trajectory)
  python src/rl_wajepa.py grade --tag h10u4                                   (CPU; needs nureasoning/cache/train)
      -> runs/rl_<tag>_graded.npz: clip, r, traj, labels [N, K, 5] (collision, drivable, progress, comfort, human)
  python src/rl_wajepa.py report --tag h10u4
Rows: a fixed random subset (seed 0) of the TRAIN part of rows_split (dev logs excluded), same for every shard.
"""
import argparse, glob, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from wajepa_ft import rows_split, HIST, NR, _meta
from wajepa_run import image, to_51
nps = lambda L: L[..., 0] * L[..., 1] * (.3 * L[..., 2] + .2 * L[..., 3] + .5 * L[..., 4])


def rl_rows(n, seed=0, pool="train"):
    """pool 'train': frames h10u4 was fine-tuned on; 'dev': the held-out dev logs (unseen by the model);
    'kf100': train-split frames at frame 100, the key frame validation and test are scored at."""
    tr, dv = rows_split(0); src = {"train": tr, "dev": dv, "kf100": [x for x in tr if x[1] == 100], "all": tr + dv}[pool]
    idx = np.sort(np.random.RandomState(seed).permutation(len(src))[:n])
    return [src[i] for i in idx]


def sample(a):
    """History-only inputs (no future frames), encode once, K flow samples per frame (seeds 1000+k)."""
    import gzip, json, torch
    from wajepa_ft import load_ft
    from wajepa_sample import encode, flow
    m, _ = load_ft(a.tag, a.device); m.eval()
    zf = np.load(os.path.join(HERE, f"data/wa_{a.split}_meta.npz"))        # validation/test: src/wa_eval_pack.py
    z = {k: zf[k] for k in ("clip", "r", "status", "hist")}                 # read each npz array ONCE (zf[k] reloads it:
    key = {(str(c), int(r)): i for i, (c, r) in enumerate(zip(z["clip"], z["r"]))}   # per-frame access OOM-killed 3 shards)
    with gzip.open(os.path.join(HERE, f"data/wa_{a.split}_paths.json.gz"), "rt") as f: paths = json.load(f)
    rows = (rl_rows(a.n, pool=a.pool) if a.split == "train" else sorted(key))[a.shard::a.nshards]
    out = os.path.join(HERE, f"runs/rl_{a.tag}{a.sfx}_samples_s{a.shard}.npz"); done = {}
    if os.path.isfile(out):
        q = np.load(out); done = {(str(c), int(r)): t for c, r, t in zip(q["clip"], q["r"], q["traj"])}
    todo = [x for x in rows if x not in done][:a.limit or None]; seeds = [1000 + k for k in range(a.K)]; t0 = time.time()
    print(f"[sample] {a.tag} shard {a.shard}/{a.nshards}: {len(done)} done, {len(todo)} to go, K={a.K}", flush=True)

    sc_out = out.replace("_samples_", "_scene_"); scene = {}                 # pooled scene encoding per frame (scorer input)
    if a.save_scene and os.path.isfile(sc_out):
        q = np.load(sc_out); scene = {(str(c), int(r)): v for c, r, v in zip(q["clip"], q["r"], q["scene"])}
    todo_scene = a.save_scene and any(x not in scene for x in done)
    if todo_scene: done = {k: v for k, v in done.items() if k in scene}      # redo frames sampled without a saved scene

    def save():
        ks = list(done)
        np.savez(out, clip=np.array([c for c, _ in ks]), r=np.array([r for _, r in ks]), traj=np.array([done[k] for k in ks]))
        if a.save_scene:
            np.savez(sc_out, clip=np.array([c for c, _ in ks]), r=np.array([r for _, r in ks]), scene=np.array([scene[k] for k in ks]),
                     status=np.array([z["status"][key[k]] for k in ks]), hist=np.array([z["hist"][key[k]] for k in ks]))
    todo = [x for x in rows if x not in done][:a.limit or None]
    for i, (c, r) in enumerate(todo):
        j = key[(c, r)]; P = paths[c]
        imgs = np.stack([np.stack([image(os.path.join(a.img_root, c, p), a.crop) for p in P[str(r + o)]]).transpose(0, 3, 1, 2)
                         for o in HIST]).astype(np.float32)
        f = {"history_images": torch.from_numpy(imgs).unsqueeze(0).to(a.device),
             "ego_status": torch.from_numpy(z["status"][j]).unsqueeze(0).to(a.device),
             "history_trajectory": torch.from_numpy(z["hist"][j]).unsqueeze(0).to(a.device)}
        with torch.no_grad(), torch.autocast(a.device, dtype=torch.bfloat16):
            f, cs, cti = encode(m, f)
            if a.save_scene: scene[(c, r)] = pool_scene(cs)
            T = (torch.cat([flow(m, f, cs, cti, seeds[s:s + a.chunk]) for s in range(0, a.K, a.chunk)]).float().cpu().numpy()
                 if a.K > 0 else np.zeros((0, 10, 3), np.float32))                  # K=0: encode only (scene features)
        done[(c, r)] = T.astype(np.float32)
        if (i + 1) % (2000 if a.K == 0 else 200) == 0 or i + 1 == len(todo):
            save(); print(f"  [{len(done)}/{len(rows)}] {(time.time() - t0) / (i + 1):.2f}s/frame", flush=True)
    print("SAMPLE_DONE", flush=True)


def pool_scene(cs):
    """[1, 4096, 512] context scene tokens = 4 cameras x 2 time steps x 16 x 32 patches (verified: neighbouring
    patches cos 0.55-0.61, 8 columns apart 0.37, random 0.12) -> mean over time, 4x4 average pool -> [4, 4, 8, 512] fp16."""
    import torch
    x = cs[0].float().reshape(4, 2, 16, 32, -1).mean(1).permute(0, 3, 1, 2)          # [cam, D, 16, 32]
    return torch.nn.functional.avg_pool2d(x, 4).permute(0, 2, 3, 1).cpu().numpy().astype(np.float16)


def _origin(clip, r, split="train"):
    import json
    from nureasoning.common.pickle_io import load_pickle
    fr = _meta(clip) if split == "train" else {f["frame_index"]: f for f in _meta_list(clip, split)}
    e = load_pickle(os.path.join(NR, "cache", split, clip, fr[r]["ego_state"]))
    return np.array([e.pose["x"], e.pose["y"], e.pose["yaw"]], float)


def _grade1(args):
    """All K samples of one frame (plus the recorded future as a check) with the devkit at key frame r."""
    from nureasoning.planning.benchmark import BenchmarkConfig, evaluate_clip
    clip, r, T, split = args
    cp = os.path.join(NR, "cache", split, clip); cfg = BenchmarkConfig(key_frame_index=r)
    fi = [f["frame_index"] for f in _meta_list(clip, split)]
    if fi[r] != r: return clip, r, np.full((len(T), 5), np.nan)             # list position must equal frame index
    o = _origin(clip, r, split); L = []
    for t in T:
        res = evaluate_clip(cp, cfg, to_51(t.astype(np.float64), o))
        L.append([np.nan] * 5 if res.error else [res.s_collision, res.s_driveable, res.s_progress, res.s_comfort, res.s_human])
    return clip, r, np.array(L, float)


def _meta_list(clip, split="train"):
    import json
    return json.load(open(os.path.join(NR, "cache", split, clip, "metadata.json")))["frames"]


def grade(a):
    from concurrent.futures import ProcessPoolExecutor
    fs = sorted(glob.glob(os.path.join(HERE, f"runs/rl_{a.tag}{a.sfx}_samples_s*.npz")))
    C, R, T = [], [], []
    for f in fs:
        q = np.load(f); C += list(map(str, q["clip"])); R += list(map(int, q["r"])); T.append(q["traj"])
    T = np.concatenate(T); out = os.path.join(HERE, f"runs/rl_{a.tag}{a.sfx}_graded.npz")
    if a.limit: C, R, T = C[:a.limit], R[:a.limit], T[:a.limit]
    print(f"[grade] {len(C)} frames x K={T.shape[1]} from {len(fs)} shard files", flush=True); t0 = time.time()
    L = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (c, r, l) in enumerate(ex.map(_grade1, [(c, r, t, a.split) for c, r, t in zip(C, R, T)], chunksize=4)):
            L[(c, r)] = l
            if (i + 1) % 500 == 0: print(f"  [{i + 1}/{len(C)}] {(time.time() - t0) / (i + 1):.2f}s/frame", flush=True)
    np.savez_compressed(out, clip=np.array(C), r=np.array(R), traj=T, labels=np.array([L[(c, r)] for c, r in zip(C, R)]))
    print("GRADE_DONE", flush=True)


def report(a):
    q = np.load(os.path.join(HERE, f"runs/rl_{a.tag}{a.sfx}_graded.npz")); S = nps(q["labels"]); K = S.shape[1]
    ok = ~np.isnan(S).any(1); S = S[ok]
    print(f"{ok.sum()} graded frames (of {len(ok)}), K={K}")
    print(f"  one random sample {S.mean():.4f}; best of {K} {S.max(1).mean():.4f}; worst {S.min(1).mean():.4f}")
    sp = S.max(1) - S.min(1)
    print(f"  frames where samples differ by >0.1: {np.mean(sp > 0.1):.1%}; best sample < 0.5: {np.mean(S.max(1) < 0.5):.1%}")
    for i, nm in enumerate(["collision", "drivable", "progress", "comfort", "human"]):
        print(f"  {nm:9s} mean {np.nanmean(q['labels'][ok][..., i]):.4f}")


def main():
    import torch
    sys.path.insert(0, os.path.join(HERE, "ext/WA-JEPA"))
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["sample", "grade", "report"])
    ap.add_argument("--tag", default="h10u4"); ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--K", type=int, default=16); ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--img-root", default=""); ap.add_argument("--crop", default="full")
    ap.add_argument("--split", default="train", choices=["train", "validation", "test"], help="validation/test: packed key-frame clips")
    ap.add_argument("--save-scene", type=int, default=0, help="also save the pooled scene encoding per frame (scorer input)")
    ap.add_argument("--pool", default="train", choices=["train", "dev", "kf100", "all"])
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    a = ap.parse_args(); a.sfx = ("" if a.pool == "train" else "_" + a.pool) if a.split == "train" else "_" + a.split; a.img_root = a.img_root or os.path.join(HERE, "data/img_wa_full", a.split)
    {"sample": sample, "grade": grade, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
