"""Fine-tune the Drive-JEPA perception-free planner on nuReasoning (frozen V-JEPA encoder, train the decoder).

Samples: the 81,523 (clip, frame r) training moments of ../nureasoning/artifacts/a15ft_index_train.json
(read-only), front camera at frame r (copied read-only from the cluster's img_full/train, 896x592).
Target: ego poses at r+5, r+10, ..., r+50 (0.5 s steps to 5.0 s; 10 poses, not the released 8) in the
frame-r ego frame (x fwd, y left, heading). Status: frame-r command one-hot + vx, vy, ax, ay.
Validation / test inputs are resized 1408x928 -> 896x592 first so every image takes the same chain.

  python src/djft.py targets          # data/djft_{train,validation,test}_meta.npz
  python src/djft.py cache --split train|validation|test   # pooled encoder features (N, 128, 1024) fp16
  python src/djft.py train --epochs 12 --tag e12
  python src/djft.py predict --tag e12 --split validation --score  -> runs/djft_<tag>_<split>.npz
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from wajepa_run import NR, KEY, CMD, rel_se2, score
NP = 10                                    # poses: 0.5 .. 5.0 s
IMGROOT = {"train": "data/img_train", "validation": "data/img/validation", "test": "data/img_cl/test"}


def _meta(split, c, cache={}):
    if (split, c) not in cache:
        cache.clear()
        cache[(split, c)] = {f["frame_index"]: f for f in json.load(open(os.path.join(NR, "cache", split, c, "metadata.json")))["frames"]}
    return cache[(split, c)]


def targets():
    from nureasoning.common.pickle_io import load_pickle
    for split in ("train", "validation", "test"):
        if split == "train":
            rows = [tuple(x) for x in json.load(open(os.path.join(NR, "artifacts/a15ft_index_train.json")))]
        else:
            rows = [(c, KEY) for c in sorted(os.listdir(os.path.join(NR, "cache", split)))
                    if os.path.isfile(os.path.join(NR, "cache", split, c, "metadata.json"))]
        img, st, tgt, org = [], [], [], []
        t0 = time.time()
        for i, (c, r) in enumerate(rows):
            fr = _meta(split, c); cd = os.path.join(NR, "cache", split, c)
            e = load_pickle(os.path.join(cd, fr[r]["ego_state"]))
            s = np.zeros(8, np.float32); s[CMD.get(((fr[r].get("mission_goal") or {}).get("command")), 3)] = 1
            s[4:8] = [e.velocity["vx"], e.velocity["vy"], e.acceleration["ax"], e.acceleration["ay"]]
            o = np.array([e.pose["x"], e.pose["y"], e.pose["yaw"]]); org.append(o); st.append(s)
            img.append(f"{c}/{fr[r]['sensors']['cameras']['front']}")
            if split == "train":                            # future poses only for training targets
                P = np.array([[q.pose["x"], q.pose["y"], q.pose["yaw"]] for q in
                              (load_pickle(os.path.join(cd, fr[r + 5 * k]["ego_state"])) for k in range(1, NP + 1))])
                tgt.append(rel_se2(o, P).astype(np.float32))
            if (i + 1) % 10000 == 0: print(f"  {split} {i+1}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
        np.savez_compressed(os.path.join(HERE, f"data/djft_{split}_meta.npz"), clip=np.array([c for c, _ in rows]),
                            r=np.array([r for _, r in rows]), img=np.array(img), status=np.array(st), origin=np.array(org),
                            **({"target": np.array(tgt)} if tgt else {}))
        print(f"{split}: {len(rows)} samples", flush=True)


def _front(path, split):
    import cv2
    from PIL import Image
    if split != "train":                                   # match the training chain: -> 896x592 first
        im = np.asarray(Image.open(path).convert("RGB").resize((896, 592), Image.BILINEAR))
    else:
        im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]; nh = int(round(w * 1024 / 1920)); y0 = (h - nh) // 2
    return cv2.resize(im[y0:y0 + nh], (512, 256)).astype(np.float32) / 255.0


def cache(split, device, bs=16):
    import torch
    from drivejepa_run import load_model
    from einops import rearrange
    m = load_model(device); M = np.load(os.path.join(HERE, f"data/djft_{split}_meta.npz"))
    out = os.path.join(HERE, f"data/djft_{split}_feat.npy")
    F = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(len(M["img"]), 128, 1024))
    root = os.path.join(HERE, IMGROOT[split]); t0 = time.time()
    for i in range(0, len(F), bs):
        x = np.stack([_front(os.path.join(root, p), split).transpose(2, 0, 1) for p in M["img"][i:i + bs]])
        x = torch.from_numpy(x).to(device); x = m.transform(x)
        x = torch.stack([x, x], 2)                         # [B, C, 2, H, W]: image as a 2-frame clip
        with torch.no_grad():
            f = m.image_encoder(x)
            f = rearrange(f, "B (H W) D -> B D H W", H=16, W=32); f = m.avg_pool(f).flatten(-2, -1).permute(0, 2, 1)
        F[i:i + len(x)] = f.float().cpu().numpy().astype(np.float16)
        if (i // bs) % 200 == 0: print(f"  {split} {i + len(x)}/{len(F)} {(time.time()-t0)/(i+len(x)):.3f}s/sample", flush=True)
    F.flush(); print(f"{split}: features {F.shape} -> {out} ({time.time()-t0:.0f}s)", flush=True)


class Decoder:
    """The released decoder (image_fc, status encoder, keyval/query embeddings, transformer, head) with the
    horizon extended from 8 to NP poses (new queries start as copies of the last pretrained one)."""

    @staticmethod
    def build(device):
        import torch
        from drivejepa_run import load_model
        m = load_model("cpu")
        q = m._query_embedding.weight.data
        m._query_embedding = torch.nn.Embedding(NP, q.shape[1]); m._query_embedding.weight.data[:8] = q
        m._query_embedding.weight.data[8:] = q[-1:].repeat(NP - 8, 1); m._trajectory_head._num_poses = NP
        del m.image_encoder                                 # features are cached
        return m.to(device)

    @staticmethod
    def forward(m, feat, status):
        import numpy as _np
        img = m.image_fc(feat)
        kv = torch_cat([img, m._status_encoding(status)[:, None]]) + m._keyval_embedding.weight[None]
        q = m._query_embedding.weight[None].repeat(feat.shape[0], 1, 1)
        return m._trajectory_head(m._transformer(src=kv, tgt=q))["trajectory"]


def torch_cat(x):
    import torch
    return torch.cat(x, 1)


def l1_len_norm(pred, gt, alpha=5.0):
    import torch
    per = (pred - gt).abs().mean(dim=[1, 2])
    L = torch.linalg.norm(gt[:, 1:, :2] - gt[:, :-1, :2], dim=-1).sum(1) + torch.linalg.norm(gt[:, 0, :2], dim=-1)
    w = 1.0 / (alpha + L); w = w * (w.numel() / w.sum())
    return (w * per).mean()


def train(a):
    """Upstream recipe (train_drive_jepa_perception_free.sh): AdamW lr 1e-4, batch 32, up to 40 epochs.
    10% of the training DRIVE LOGS are held out as a dev set; the checkpoint with the lowest dev 5 s
    endpoint error (human-likeness is FDE-based) is kept. Validation is never used to choose."""
    import torch
    M = np.load(os.path.join(HERE, "data/djft_train_meta.npz")); F = np.load(os.path.join(HERE, "data/djft_train_feat.npy"), mmap_mode="r")
    T = torch.from_numpy(M["target"]); S = torch.from_numpy(M["status"])
    log = np.array(["_".join(str(c).split("_")[:2]) for c in M["clip"]]); ul = np.unique(log)
    dev_logs = set(np.random.RandomState(a.dev_seed).permutation(ul)[:int(round(a.dev_frac * len(ul)))])
    dev = np.where(np.isin(log, list(dev_logs)))[0]; tr = np.where(~np.isin(log, list(dev_logs)))[0]
    print(f"train {len(tr)} samples / dev {len(dev)} samples ({len(dev_logs)} of {len(ul)} logs)", flush=True)
    torch.manual_seed(a.seed); m = Decoder.build(a.device)
    for mod in m.modules():                                  # released decoder has dropout 0.0
        if isinstance(mod, torch.nn.Dropout): mod.p = a.dropout
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * (len(tr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    rng = np.random.RandomState(a.seed); t0 = time.time(); best = (np.inf, -1)
    Fd = torch.from_numpy(np.asarray(F[np.sort(dev)], np.float32)); Sd = S[np.sort(dev)]; Td = T[np.sort(dev)]

    def dev_eval():
        m.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(Fd), 256):
                out.append(Decoder.forward(m, Fd[i:i + 256].to(a.device), Sd[i:i + 256].to(a.device)).cpu())
        m.train(); P = torch.cat(out)
        return float(torch.linalg.norm(P[:, -1, :2] - Td[:, -1, :2], dim=-1).mean()), float(l1_len_norm(P, Td))
    fde0, l0 = dev_eval(); print(f"  epoch 0 (released decoder, 8->10 poses) dev FDE@5s {fde0:.3f} m, dev loss {l0:.4f}", flush=True)
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True); ck = os.path.join(HERE, f"runs/djft_{a.tag}.pt")
    for ep in range(a.epochs):
        perm = rng.permutation(tr); tot = 0.0; n = 0
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            idx = np.sort(perm[i:i + a.bs])
            f = torch.from_numpy(np.asarray(F[idx], np.float32)).to(a.device)
            loss = l1_len_norm(Decoder.forward(m, f, S[idx].to(a.device)), T[idx].to(a.device))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item(); n += 1
        fde, dl = dev_eval()
        if fde < best[0]:
            best = (fde, ep + 1); torch.save({"state_dict": m.state_dict(), "args": vars(a), "np": NP, "epoch": ep + 1, "dev_fde": fde}, ck)
        print(f"  epoch {ep+1}/{a.epochs} train loss {tot / n:.4f} | dev FDE@5s {fde:.3f} m, dev loss {dl:.4f}"
              f"{'  <- best' if best[1] == ep + 1 else ''} ({time.time()-t0:.0f}s)", flush=True)
    print(f"kept epoch {best[1]} (dev FDE@5s {best[0]:.3f} m) -> {ck}", flush=True)


def to_51_np(pred, origin):
    """NP poses at 0.5 s (to 5.0 s) -> 51 global poses at 0.1 s (linear from the origin; no extrapolation)."""
    T = np.concatenate([[0], np.arange(1, NP + 1) * 0.5]); P = np.vstack([[0, 0, 0], pred]); P[:, 2] = np.unwrap(P[:, 2])
    t = np.arange(51) * 0.1; x, y, h = (np.interp(t, T, P[:, j]) for j in range(3))
    ox, oy, oh = origin; c, s = np.cos(oh), np.sin(oh)
    return np.stack([ox + c * x - s * y, oy + s * x + c * y, oh + h], 1)


def predict(a):
    import torch
    M = np.load(os.path.join(HERE, f"data/djft_{a.split}_meta.npz")); F = np.load(os.path.join(HERE, f"data/djft_{a.split}_feat.npy"))
    m = Decoder.build(a.device); m.load_state_dict(torch.load(os.path.join(HERE, f"runs/djft_{a.tag}.pt"))["state_dict"]); m.eval()
    with torch.no_grad():
        P = Decoder.forward(m, torch.from_numpy(F.astype(np.float32)).to(a.device), torch.from_numpy(M["status"]).to(a.device)).cpu().numpy().astype(np.float64)
    G = np.array([to_51_np(p, o) for p, o in zip(P, M["origin"])]); names = [str(c) for c in M["clip"]]
    extra = {}
    if a.score:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(8) as ex:
            L = [r or [np.nan] * 5 for r in ex.map(_score_one, [(a.split, c, g) for c, g in zip(names, G)], chunksize=8)]
        L = np.array(L, float); n = L[:, 0] * L[:, 1] * (.3 * L[:, 2] + .2 * L[:, 3] + .5 * L[:, 4]); extra["labels"] = L
        print(f"{a.split} {a.tag}: NPS {np.nanmean(n):.4f} on {np.isfinite(n).sum()} clips", flush=True)
    out = os.path.join(HERE, f"runs/djft_{a.tag}_{a.split}.npz")
    np.savez_compressed(out, names=np.array(names), traj=G, pred=P, **extra); print(f"-> {out}", flush=True)


def avg(a):
    """Average several fine-tuned decoders' 10-pose predictions (ego frame; yaw circular), then score."""
    Z = [np.load(os.path.join(HERE, f"runs/djft_{t}_{a.split}.npz")) for t in a.tags.split(",")]
    names = [str(c) for c in Z[0]["names"]]; assert all([str(c) for c in z["names"]] == names for z in Z)
    P = np.mean([z["pred"] for z in Z], 0)
    P[..., 2] = np.arctan2(np.mean([np.sin(z["pred"][..., 2]) for z in Z], 0), np.mean([np.cos(z["pred"][..., 2]) for z in Z], 0))
    M = np.load(os.path.join(HERE, f"data/djft_{a.split}_meta.npz")); assert [str(c) for c in M["clip"]] == names
    G = np.array([to_51_np(p, o) for p, o in zip(P, M["origin"])]); extra = {}
    if a.score:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(8) as ex:
            L = [r or [np.nan] * 5 for r in ex.map(_score_one, [(a.split, c, g) for c, g in zip(names, G)], chunksize=8)]
        L = np.array(L, float); n = L[:, 0] * L[:, 1] * (.3 * L[:, 2] + .2 * L[:, 3] + .5 * L[:, 4]); extra["labels"] = L
        print(f"{a.split} avg({a.tags}): NPS {np.nanmean(n):.4f} on {np.isfinite(n).sum()} clips", flush=True)
    out = os.path.join(HERE, f"runs/djft_{a.tag}_{a.split}.npz")
    np.savez_compressed(out, names=np.array(names), traj=G, pred=P, **extra); print(f"-> {out}", flush=True)


def _score_one(args):
    return score(*args)


def main():
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument("cmd")
    ap.add_argument("--split", default="train"); ap.add_argument("--tag", default="r40")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--dev-frac", type=float, default=0.1); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev-seed", type=int, default=0, help="dev-log split (0 = the r40 split)")
    ap.add_argument("--wd", type=float, default=1e-4); ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--tags", default="", help="avg: comma-separated tags whose predictions are averaged")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    {"targets": lambda: targets(), "cache": lambda: cache(a.split, a.device), "train": lambda: train(a),
     "predict": lambda: predict(a), "avg": lambda: avg(a)}[a.cmd]()


if __name__ == "__main__":
    main()
