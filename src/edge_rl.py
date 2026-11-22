"""RL member for Alpamayo 2 Edge: leave-one-out reward on the K-mean, reward-weighted flow loss (Aman 2026-11-13: "yes
go ahead with the rl"). Why this recipe: we submit the MEAN of K samples, so
a sample is good when it pulls that mean toward a higher score. Reward of sample k = NPS(mean of all K) - NPS(mean of
the other K-1). Per-sample NPS rewards favour samples that score well alone, not ones
that improve the mean. Samples with a positive reward become flow-matching targets weighted by that reward
(RAFT / reward-weighted regression; positive-only, stable), mixed with the human-future rows as an anchor.

  select (Mac, CPU): python src/edge_rl.py select --n 6000          -> data/edge_rl_select.json (row keys whose frame
                     index equals the list position, so the devkit can score at that key frame)
  sample (GPU):      python -I src/edge_rl.py sample --model DIR --tag e2a --shard i/n [--K 6]
                     -> runs/edge_rl_<tag>_s<i>of<n>.npz: names, xy [N,K,64,2], yaw [N,K,64] (ego frame of the row's key pose)
                     Noise seed differs per row (hash of the key), unlike edge_infer's fixed seed 43.
  grade (Mac, CPU):  python src/edge_rl.py grade --tag e2a          -> runs/edge_rl_<tag>_graded.npz: per row the K
                     sample scores, the K-mean score, the K leave-one-out mean scores, and the human future's score
  build (Mac):       python src/edge_rl.py build --tag e2a          -> data/edge_rl_rows_<tag>.json for src/edge_train.py:
                     human-future rows (w 1) + positive-reward sample rows ("rl": 1, flow loss only, w = reward / mean
                     positive reward, clipped at 3)
"""
import argparse, glob, hashlib, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
NR = os.environ.get("NR_ROOT", "/path/to/workspace/nureasoning")
sys.path.insert(0, os.path.join(NR, "recon/devkit"))
ROWS = "data/edge_rows_train.json"
nps = lambda L: L[..., 0] * L[..., 1] * (.3 * L[..., 2] + .2 * L[..., 3] + .5 * L[..., 4])


def _frames(clip):
    return [f["frame_index"] for f in json.load(open(os.path.join(NR, "cache", "train", clip, "metadata.json")))["frames"]]


def select(a):
    R = json.load(open(os.path.join(HERE, ROWS))); keys = sorted(R)
    rng = np.random.RandomState(a.seed); cand = [keys[i] for i in rng.permutation(len(keys))]
    out, bad, seen = [], 0, {}
    for k in cand:
        c, t = k.split("@"); t = int(t)
        if c not in seen:
            try: seen[c] = _frames(c)
            except FileNotFoundError: seen[c] = None
        fi = seen[c]
        if fi is None or t >= len(fi) or fi[t] != t or t + 50 >= len(fi) or fi[t + 50] != t + 50: bad += 1; continue
        out.append(k)
        if len(out) >= a.n: break
    json.dump(sorted(out), open(os.path.join(HERE, "data/edge_rl_select.json"), "w"))
    print(f"[select] {len(out)} rows from {len(set(k.split('@')[0] for k in out))} clips ({bad} skipped: frame index != position)")


def sample(a):
    import torch
    from alpamayo2_edge.checkpoint import load_model
    from alpamayo2_edge.helper import prepare_model_inputs, to_device
    from edge_infer import build
    i, n = map(int, a.shard.split("/"))
    R = json.load(open(os.path.join(HERE, ROWS))); keys = json.load(open(os.path.join(HERE, "data/edge_rl_select.json")))[i::n]
    out = os.path.join(HERE, f"runs/edge_rl_{a.tag}_s{i}of{n}")
    names, XY, YAW = [], [], []
    for prev in (out, out + ".partial"):
        if os.path.isfile(prev + ".npz"):
            z = np.load(prev + ".npz"); names = list(map(str, z["names"])); XY = list(z["xy"]); YAW = list(z["yaw"]); break
    done = set(names)
    model, _ = load_model(a.model, device="cuda", dtype="bfloat16", attn_implementation="sdpa")
    print(f"[edge_rl sample] {a.tag} shard {a.shard}: {len(keys)} rows ({len(done)} done) K={a.K} gpu {torch.cuda.get_device_name()}", flush=True)

    def save(p):
        np.savez(p + ".npz", names=np.array(names), xy=np.array(XY, np.float32), yaw=np.array(YAW, np.float32), K=a.K)

    t0 = time.time(); k = 0
    for key in keys:
        if key in done: continue
        prep = to_device(prepare_model_inputs(build(R[key], "train", a.img_root), model.config, model.tokenizer), device="cuda")
        s = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
        torch.manual_seed(s); gen = torch.Generator(device="cuda").manual_seed(s)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            xyz, rot, _, _ = model.sample_trajectories_from_data(
                prep, top_p=0.98, top_k=None, temperature=a.temperature, num_traj_sets=1, num_traj_samples=a.K,
                max_generation_length=256, diffusion_kwargs={"inference_step": 10, "generator": gen}, return_extra=True)
        xyz = xyz[0, 0].float().cpu().numpy(); rot = rot[0, 0].float().cpu().numpy()
        names.append(key); XY.append(xyz[..., :2]); YAW.append(np.arctan2(rot[..., 1, 0], rot[..., 0, 0])); k += 1
        if k % 25 == 0:
            save(out + ".partial")
            print(f"  {len(names)}/{len(keys)} {(time.time() - t0) / k:.2f} s/row peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    save(out)
    print(f"EDGE_RL_SAMPLE_DONE {len(names)} rows {(time.time() - t0) / max(k, 1):.2f} s/row", flush=True)


def _g51(xy, yaw, pose):
    from plan_traj import to_global
    return to_global(np.vstack([[0.0, 0.0, 0.0], np.column_stack([xy[:50], yaw[:50]])]), pose)


def _grade1(args):
    from nureasoning.planning.benchmark import BenchmarkConfig, evaluate_clip
    key, xy, yaw, row = args
    c, t = key.split("@"); cfg = BenchmarkConfig(key_frame_index=int(t)); cp = os.path.join(NR, "cache", "train", c)
    def sc(x, y):
        try: r = evaluate_clip(cp, cfg, _g51(x, y, row["pose"]))
        except FileNotFoundError: return [np.nan] * 5           # a few Mac train caches lack an annotation frame: row skipped
        return [np.nan] * 5 if r.error else [r.s_collision, r.s_driveable, r.s_progress, r.s_comfort, r.s_human]
    cm = lambda Y: np.arctan2(np.sin(Y).mean(0), np.cos(Y).mean(0))
    K = len(xy); f = np.asarray(row["fut64"], float)
    samp = [sc(xy[j], yaw[j]) for j in range(K)]
    mean = sc(xy.mean(0), cm(yaw))
    loo = [sc(np.delete(xy, j, 0).mean(0), cm(np.delete(yaw, j, 0))) for j in range(K)]
    gt = sc(f[:, :2], f[:, 2])
    return key, samp, mean, loo, gt


def grade(a):
    from concurrent.futures import ProcessPoolExecutor
    fs = sorted(f for f in glob.glob(os.path.join(HERE, f"runs/edge_rl_{a.tag}_s*of*.npz")) if not f.endswith(".partial.npz"))
    names, XY, YAW = [], [], []
    for f in fs:
        z = np.load(f); names += list(map(str, z["names"])); XY += list(z["xy"]); YAW += list(z["yaw"])
    if a.limit: names, XY, YAW = names[:a.limit], XY[:a.limit], YAW[:a.limit]
    R = json.load(open(os.path.join(HERE, ROWS)))
    print(f"[grade] {len(names)} rows x K={len(XY[0])} from {len(fs)} shard files", flush=True); t0 = time.time()
    res = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for j, r in enumerate(ex.map(_grade1, [(k, x, y, R[k]) for k, x, y in zip(names, XY, YAW)], chunksize=4)):
            res[r[0]] = r[1:]
            if (j + 1) % 500 == 0: print(f"  [{j + 1}/{len(names)}] {(time.time() - t0) / (j + 1):.2f} s/row", flush=True)
    L = lambda i: np.array([res[k][i] for k in names], float)
    np.savez_compressed(os.path.join(HERE, f"runs/edge_rl_{a.tag}_graded.npz"), names=np.array(names), xy=np.array(XY),
                        yaw=np.array(YAW), samp=L(0), mean=L(1), loo=L(2), gt=L(3))
    report(a)


def report(a):
    z = np.load(os.path.join(HERE, f"runs/edge_rl_{a.tag}_graded.npz"))
    S, M, LO, G = nps(z["samp"]), nps(z["mean"]), nps(z["loo"]), nps(z["gt"])
    ok = ~(np.isnan(S).any(1) | np.isnan(M) | np.isnan(LO).any(1)); rew = (M[:, None] - LO)[ok]
    print(f"{ok.sum()} graded rows (of {len(ok)}): human future {np.nanmean(G[ok]):.4f} | one sample {S[ok].mean():.4f} | "
          f"K-mean {M[ok].mean():.4f} | best sample {S[ok].max(1).mean():.4f}")
    print(f"  loo reward: mean {rew.mean():+.4f}, positive {np.mean(rew > 1e-6):.1%}, zero {np.mean(np.abs(rew) <= 1e-6):.1%}, "
          f"rows with any positive {np.mean((rew > 1e-6).any(1)):.1%}; |reward| p50 {np.median(np.abs(rew)):.4f} p90 {np.percentile(np.abs(rew), 90):.4f}")


def build_rows(a):
    z = np.load(os.path.join(HERE, f"runs/edge_rl_{a.tag}_graded.npz")); R = json.load(open(os.path.join(HERE, ROWS)))
    M, LO = nps(z["mean"]), nps(z["loo"]); names = list(map(str, z["names"]))
    rew = M[:, None] - LO; ok = ~(np.isnan(M) | np.isnan(LO).any(1))
    pos = rew[ok][rew[ok] > 1e-6]; scale = pos.mean()
    out = {}; n_rl = 0
    for i, k in enumerate(names):
        if not ok[i]: continue
        out[k] = dict(R[k])                                                   # human-future anchor, normal loss
        for j in np.where(rew[i] > 1e-6)[0]:
            v = {kk: vv for kk, vv in R[k].items() if kk != "fut64"}
            v["fut64"] = np.column_stack([z["xy"][i, j], z["yaw"][i, j]]).round(4).tolist(); v["n_valid"] = 64
            v["rl"] = 1; v["w"] = float(min(3.0, rew[i, j] / scale)); out[f"{k}#{j}"] = v; n_rl += 1
    p = os.path.join(HERE, f"data/edge_rl_rows_{a.tag}.json"); json.dump(out, open(p, "w"))
    print(f"[build] {p}: {len(out) - n_rl} human rows + {n_rl} sample rows (reward scale {scale:.4f}, "
          f"weights mean {np.mean([v['w'] for v in out.values() if v.get('rl')]):.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["select", "sample", "grade", "report", "build"])
    ap.add_argument("--n", type=int, default=6000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=""); ap.add_argument("--tag", default="e2a"); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--K", type=int, default=6); ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--img-root", default=os.path.join(HERE, "data/img_edge"))
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "sample": sys.path.insert(0, os.path.join(HERE, "external", "alpamayo2-edge", "src"))
    {"select": select, "sample": sample, "grade": grade, "report": report, "build": build_rows}[a.cmd](a)
