"""Scene-aware scorer: pick the best of K sampled trajectories (best-of-16 = 0.86 vs 0.73 as run on validation).

Inputs per frame: the planner's own scene encoding (pooled context tokens [4 cams, 4, 8, 512], saved while
sampling: src/rl_wajepa.py --save-scene), ego status + history, and its K candidate trajectories (planner
frame, 10 poses). The K candidates attend to the scene tokens and to each other (so each is judged against
its alternatives), then heads predict a ranking score, NPS and the collision / drivable components.
Loss: listwise (softmax over the K candidates vs softmax(NPS / 0.05)) + 0.5 MSE(NPS) + 0.5 BCE(collision, drivable).
Train: graded train frames (runs/rl_<src>_kf100_graded.npz + _scene_s*.npz; several samplers' candidates for the
same frames are pooled and K=16 are drawn per step). 20% of drive logs held out to choose the epoch.
Test: validation (runs/rl_<src>_validation_graded.npz + scene), scored once.
  python src/scorer.py train --train rl3_s1987,h10u4 --scene rl3_s1987 --val rl3_s1987
"""
import argparse, collections, glob, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nps = lambda L: L[..., 0] * L[..., 1] * (.3 * L[..., 2] + .2 * L[..., 3] + .5 * L[..., 4])
log_of = lambda c: "_".join(c.split("_")[:2])


def load_scene(tag, sfx):
    S = {}
    for f in sorted(glob.glob(os.path.join(HERE, f"runs/rl_{tag}{sfx}_scene_s*.npz"))):
        q = np.load(f)
        for c, r, s, st, h in zip(q["clip"], q["r"], q["scene"], q["status"], q["hist"]):
            S[(str(c), int(r))] = (s, st.astype(np.float32), h.astype(np.float32).reshape(-1))
    return S


def load_graded(tags, sfx):
    G = collections.defaultdict(lambda: [[], []])
    for t in tags:
        q = np.load(os.path.join(HERE, f"runs/rl_{t}{sfx}_graded.npz"))
        for c, r, T, L in zip(q["clip"], q["r"], q["traj"], q["labels"]):
            ok = ~np.isnan(L).any(1); G[(str(c), int(r))][0].append(T[ok]); G[(str(c), int(r))][1].append(L[ok])
    return {k: (np.concatenate(v[0]).astype(np.float32), np.concatenate(v[1]).astype(np.float32)) for k, v in G.items()}


def traj_feat(T):
    """[.., 10, 3] planner-frame poses -> [.., 10, 6]: x, y scaled, sin/cos yaw, step speed, lateral step."""
    import torch
    d = torch.diff(torch.cat([torch.zeros_like(T[..., :1, :]), T], -2), dim=-2)
    return torch.cat([T[..., :2] / 20.0, torch.sin(T[..., 2:3]), torch.cos(T[..., 2:3]),
                      d[..., :1] / 5.0, d[..., 1:2] / 2.0], -1)


def build_model(d=256, layers=3, heads=8):
    import torch, torch.nn as nn

    class Scorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.scene_in = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, d))
            self.cam = nn.Parameter(torch.zeros(4, 1, 1, d)); self.row = nn.Parameter(torch.zeros(1, 4, 1, d))
            self.col = nn.Parameter(torch.zeros(1, 1, 8, d))
            self.ego = nn.Sequential(nn.Linear(8 + 12, d), nn.GELU(), nn.Linear(d, d))
            self.traj = nn.Sequential(nn.Linear(60, d), nn.GELU(), nn.Linear(d, d))
            for p in (self.cam, self.row, self.col): nn.init.normal_(p, std=0.02)
            dec = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
            self.dec = nn.TransformerDecoder(dec, layers)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4))          # rank score, NPS, collision, drivable

        def forward(self, scene, status, hist, T):
            B = scene.shape[0]
            s = self.scene_in(scene.float()) + self.cam + self.row + self.col     # [B, 4, 4, 8, d]
            mem = torch.cat([s.reshape(B, -1, s.shape[-1]), self.ego(torch.cat([status, hist], -1))[:, None]], 1)
            q = self.traj(traj_feat(T).flatten(-2))                              # [B, K, d]
            return self.head(self.dec(q, mem))
    return Scorer()


def batches(keys, S, G, K, bs, rng, train):
    import torch
    order = rng.permutation(len(keys)) if train else np.arange(len(keys))
    for i in range(0, len(order), bs):
        ks = [keys[j] for j in order[i:i + bs]]; sc, st, hi, T, L = [], [], [], [], []
        for k in ks:
            Tk, Lk = G[k]; sel = rng.choice(len(Tk), K, replace=len(Tk) < K) if train else np.arange(min(K, len(Tk)))
            s, a, h = S[k]; sc.append(s); st.append(a); hi.append(h); T.append(Tk[sel]); L.append(Lk[sel])
        yield ks, [torch.from_numpy(np.stack(x)) for x in (sc, st, hi, T, L)]


def loss_fn(out, L):
    import torch, torch.nn.functional as F
    y = nps(L); tgt = torch.softmax(y / 0.05, -1)
    rank = -(tgt * torch.log_softmax(out[..., 0], -1)).sum(-1).mean()
    reg = F.mse_loss(torch.sigmoid(out[..., 1]), y)
    bce = F.binary_cross_entropy_with_logits(out[..., 2], L[..., 0]) + F.binary_cross_entropy_with_logits(out[..., 3], L[..., 1])
    return rank + 0.5 * reg + 0.5 * bce


def evaluate(m, keys, S, G, K, dev):
    """Per frame: NPS of the picked candidate, random (mean), oracle, and medoid (closest endpoint to the mean)."""
    import torch
    m.eval(); R = []
    with torch.no_grad():
        for ks, (sc, st, hi, T, L) in batches(keys, S, G, K, 64, np.random.RandomState(0), False):
            out = m(sc.to(dev), st.to(dev), hi.to(dev), T.to(dev)).float().cpu(); y = nps(L)
            pick = out[..., 0].argmax(-1); end = T[..., -1, :2]; med = (end - end.mean(1, keepdim=True)).norm(dim=-1).argmin(-1)
            ar = torch.arange(len(ks))
            R += list(zip(ks, y[ar, pick].tolist(), y.mean(-1).tolist(), y.max(-1).values.tolist(), y[ar, med].tolist()))
    m.train(); return R


def train(a):
    import torch
    dev = a.device; torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed)
    S = load_scene(a.scene, "_kf100"); G = load_graded(a.train.split(","), "_kf100")
    keys = sorted(k for k in G if k in S and len(G[k][0]) >= a.K)
    logs = sorted({log_of(c) for c, _ in keys}); hold = set(np.random.RandomState(0).permutation(logs)[:len(logs) // 5])
    tr = [k for k in keys if log_of(k[0]) not in hold]; ho = [k for k in keys if log_of(k[0]) in hold]
    print(f"train {len(tr)} frames / held-out logs {len(ho)} frames; candidates per frame "
          f"{np.mean([len(G[k][0]) for k in keys]):.1f}; K={a.K}", flush=True)
    m = build_model(a.d, a.layers).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * ((len(tr) + a.bs - 1) // a.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    print(f"params {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M, {steps} steps", flush=True)
    best = (-1, 0); t0 = time.time(); ck = os.path.join(HERE, f"runs/scorer_{a.tag}.pt")
    for ep in range(a.epochs):
        tot = 0.0; n = 0
        for _, (sc, st, hi, T, L) in batches(tr, S, G, a.K, a.bs, rng, True):
            out = m(sc.to(dev), st.to(dev), hi.to(dev), T.to(dev)).float(); loss = loss_fn(out, L.to(dev))
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item(); n += 1
        R = np.array([r[1:] for r in evaluate(m, ho, S, G, a.K, dev)])
        tag = ""
        if R[:, 0].mean() > best[0]:
            best = (R[:, 0].mean(), ep); tag = "  <- best"; torch.save({"state_dict": m.state_dict(), "args": vars(a), "epoch": ep}, ck)
        print(f"  epoch {ep} loss {tot / n:.4f} | held-out logs: pick {R[:, 0].mean():.4f} random {R[:, 1].mean():.4f} "
              f"medoid {R[:, 3].mean():.4f} oracle {R[:, 2].mean():.4f}{tag} ({time.time() - t0:.0f}s)", flush=True)
    print(f"kept epoch {best[1]} (held-out pick {best[0]:.4f}) -> {ck}", flush=True)
    if a.val:
        m.load_state_dict(torch.load(ck, map_location="cpu")["state_dict"])
        Sv = load_scene(a.val, "_validation"); Gv = load_graded([a.val], "_validation")
        kv = sorted(k for k in Gv if k in Sv and len(Gv[k][0]) >= a.K); R = evaluate(m, kv, Sv, Gv, a.K, dev)
        np.save(os.path.join(HERE, f"runs/scorer_{a.tag}_validation_rows.npy"), np.array([(k[0], *r[1:]) for k, *r in R], dtype=object))
        X = np.array([r[1:] for r in R]); print(f"VALIDATION ({len(R)} clips, {a.val} candidates): pick {X[:, 0].mean():.4f} "
                                               f"random {X[:, 1].mean():.4f} medoid {X[:, 3].mean():.4f} oracle {X[:, 2].mean():.4f}", flush=True)
    print("SCORER_DONE", flush=True)


def main():
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--tag", default="s1"); ap.add_argument("--train", default="rl3_s1987,h10u4")
    ap.add_argument("--scene", default="rl3_s1987"); ap.add_argument("--val", default="rl3_s1987")
    ap.add_argument("--K", type=int, default=16); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=256); ap.add_argument("--layers", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    a = ap.parse_args(); {"train": train}[a.cmd](a)


if __name__ == "__main__":
    main()
