"""B1 step 1: longitudinal decision head on WA-JEPA scene features, used to fix the planners' 'keep speed' bias.

Ceiling (src/meta_oracle.py): the TRUE 4-class decision (stop / decelerate / keep / accelerate, DriveMA thresholds on the
recorded future) brings the blend path to 0.745 (9-class 0.781) vs 0.701. The blend itself implies the right 4-class
decision on only 67.5% of validation clips (accelerate recall 0.38, decelerate 0.65): planners default to 'keep'.
Classifier: pooled rl4 scene tokens [4 cams, 4, 8, 512] (src/rl_wajepa.py --K 0 --save-scene) + ego history features
(src/dist_feats.py) -> 9 classes (stop + 8 speed-slope bins; the 4 classes are merges). Labels from each row's recorded
future, so all ~73k train rows are labelled. Epoch chosen on dev-log frame-100 rows; validation scored once.
Distance: per-class conditional model (GBDT on history + true class, trained on train), then the expected-score-optimal
5 s length under the predicted class mixture (smoothstep human score, widened by the conditional model's residual
spread per class); optionally averaged with the blend's own length (weight chosen by CV on validation).
  python src/dec_wajepa.py
"""
import collections, json, os, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from meta_oracle import future_speeds, classes
from dist_cal import hscore
from dist_stack import retime_scale, plen
from ensemble_eval import wmean, nps
F = {"b": "v71B", "h4": "wajepa_ft_h10u4", "wa": "wajepa_169", "dj": "djft_r2avg", "raw": "a15s1lora_raw"}
W = {"b": .4, "h4": .35, "wa": .2, "dj": .25, "raw": .1}
C9_TO_4 = np.array([0, 1, 1, 1, 2, 2, 3, 3, 3])                           # stop | slope bins -> decel / keep / accel
log_of = lambda c: "_".join(c.split("_")[:2])


def load_scene(tag, split):
    import glob
    sfx = "_all" if split == "train" else f"_{split}"; S = {}
    for f in sorted(glob.glob(os.path.join(HERE, f"runs/rl_{tag}{sfx}_scene_s*.npz"))):
        q = np.load(f)
        for c, r, s in zip(q["clip"], q["r"], q["scene"]): S[(str(c), int(r))] = s
    return S


def labels(split):
    cache = os.path.join(HERE, f"data/dec9_{split}.npz")
    if os.path.isfile(cache):
        q = np.load(cache); return {(str(c), int(r)): (int(k), float(d)) for c, r, k, d in zip(q["clip"], q["r"], q["k9"], q["dist"])}
    z = np.load(os.path.join(HERE, f"data/dist_feats_{split}.npz")); by = collections.defaultdict(list)
    for c, r in zip(map(str, z["clip"]), map(int, z["r"])): by[c].append(r)
    with ProcessPoolExecutor(8) as ex:
        V = {(c, r): v for part in ex.map(future_speeds, [(split, c, rs) for c, rs in by.items()], chunksize=4) for c, r, v in part if v is not None}
    Y = dict(zip(zip(map(str, z["clip"]), map(int, z["r"])), z["Y"][:, 4]))
    L = {k: (classes(v, float(Y[k]))[1], float(Y[k])) for k, v in V.items() if np.isfinite(Y[k])}
    ks = sorted(L); np.savez(cache, clip=np.array([k[0] for k in ks]), r=np.array([k[1] for k in ks]), k9=np.array([L[k][0] for k in ks]), dist=np.array([L[k][1] for k in ks]))
    return L


def feats(split, mu=None, sd=None):
    z = np.load(os.path.join(HERE, f"data/dist_feats_{split}.npz")); X = z["X"].astype(np.float32)
    if mu is None: mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-3
    Xn = np.concatenate([np.nan_to_num((X - mu) / sd), np.isnan(X)], 1).astype(np.float32)
    return dict(zip(zip(map(str, z["clip"]), map(int, z["r"])), Xn)), dict(zip(zip(map(str, z["clip"]), map(int, z["r"])), z["X"])), mu, sd


def build(nf, d=256):
    import torch, torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, d)); self.pos = nn.Parameter(torch.zeros(1, 128, d)); nn.init.normal_(self.pos, std=0.02)
            self.ego = nn.Sequential(nn.Linear(nf, d), nn.GELU(), nn.Linear(d, d)); self.cls = nn.Parameter(torch.zeros(1, 1, d))
            self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 4 * d, dropout=0.1, batch_first=True, norm_first=True), 2)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 9))

        def forward(self, s, x):
            t = self.inp(s.float().flatten(1, 3)) + self.pos
            h = self.enc(torch.cat([self.cls.expand(len(s), -1, -1) + self.ego(x)[:, None], t], 1))
            return self.head(h[:, 0])
    return Net()


def _score(a):
    from wajepa_run import score
    c, ts = a; return c, [nps(r) if (r := score("validation", c, t)) else np.nan for t in ts]


def main():
    import torch
    from sklearn.ensemble import HistGradientBoostingRegressor
    from wajepa_ft import rows_split
    dev = "mps" if torch.backends.mps.is_available() else "cpu"; torch.manual_seed(0)
    St, Sv = load_scene("rl4_s1987", "train"), load_scene("rl4_s1987", "validation")
    Lt, Lv = labels("train"), labels("validation"); Xt, Rt, mu, sd = feats("train"); Xv, Rv, _, _ = feats("validation", mu, sd)
    _, dv = rows_split(0); dev_logs = {log_of(c) for c, _ in dv}
    tr = [k for k in St if k in Lt and log_of(k[0]) not in dev_logs]; dk = [k for k in St if k in Lt and log_of(k[0]) in dev_logs and k[1] == 100]
    va = sorted(k for k in Sv if k in Lv)
    print(f"train rows {len(tr)}, dev@100 {len(dk)}, validation {len(va)}; 9-class train counts {np.bincount([Lt[k][0] for k in tr], minlength=9).tolist()}", flush=True)
    cnt = np.bincount([Lt[k][0] for k in tr], minlength=9).astype(float); wcls = torch.tensor((cnt.sum() / (9 * np.maximum(cnt, 1))) ** 0.5, dtype=torch.float32, device=dev)
    net = build(Xt[tr[0]].shape[0]).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.05)
    def batch(keys, S, X, L):
        return (torch.from_numpy(np.stack([S[k] for k in keys])).to(dev), torch.from_numpy(np.stack([X[k] for k in keys])).to(dev),
                torch.tensor([L[k][0] for k in keys], device=dev))
    def probs(keys, S, X):
        net.eval(); P = []
        with torch.no_grad():
            for i in range(0, len(keys), 256):
                s, x, _ = batch(keys[i:i + 256], S, X, {k: (0, 0) for k in keys[i:i + 256]}); P.append(torch.softmax(net(s, x), -1).cpu().numpy())
        net.train(); return np.concatenate(P)
    E, bs = 8, 128; steps = E * (len(tr) // bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=0.05)
    best = (-1, None); rng = np.random.RandomState(0); t0 = time.time()
    for ep in range(E):
        o = rng.permutation(len(tr))
        for i in range(0, len(o) - bs + 1, bs):
            s, x, y = batch([tr[j] for j in o[i:i + bs]], St, Xt, Lt)
            loss = torch.nn.functional.cross_entropy(net(s, x), y, weight=wcls); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        Pd = probs(dk, St, Xt); yd = np.array([Lt[k][0] for k in dk]); acc4 = np.mean(C9_TO_4[Pd.argmax(1)] == C9_TO_4[yd])
        nll = -np.mean(np.log(Pd[np.arange(len(yd)), yd] + 1e-9)); tag = ""
        if acc4 > best[0]: best = (acc4, {k: v.detach().clone() for k, v in net.state_dict().items()}); tag = " <- best"
        print(f"  epoch {ep} loss {loss.item():.3f} | dev@100 4-class acc {acc4:.3f}, 9-class NLL {nll:.3f}{tag} ({time.time() - t0:.0f}s)", flush=True)
    net.load_state_dict(best[1]); Pv = probs(va, Sv, Xv); yv = np.array([Lv[k][0] for k in va]); p4 = np.stack([Pv[:, C9_TO_4 == j].sum(1) for j in range(4)], 1)
    M = np.zeros((4, 4), int)
    for t, p in zip(C9_TO_4[yv], p4.argmax(1)): M[t, p] += 1
    print(f"VALIDATION 4-class accuracy {np.mean(p4.argmax(1) == C9_TO_4[yv]):.3f} (blend implied 0.675); recall (stop, decel, keep, accel) {np.round(M.diagonal() / M.sum(1), 2).tolist()}", flush=True)
    # per-class conditional length model on train (true class), residual spread per class
    trk = [k for k in Lt if k in Rt]; Xr = np.array([Rt[k] for k in trk]); K9 = np.array([Lt[k][0] for k in trk]); Yd = np.array([Lt[k][1] for k in trk])
    g = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=0).fit(np.c_[Xr, np.eye(9)[K9]], Yd)
    res = Yd - g.predict(np.c_[Xr, np.eye(9)[K9]]); sig = np.array([np.std(res[K9 == j]) if (K9 == j).sum() > 20 else 3.0 for j in range(9)])
    Xva = np.array([Rv[k] for k in va]); mu9 = np.stack([g.predict(np.c_[Xva, np.tile(np.eye(9)[j], (len(va), 1))]) for j in range(9)], 1)   # [N, 9]
    z = np.linspace(-3, 3, 25); pz = np.exp(-z ** 2 / 2); pz /= pz.sum(); grid = np.linspace(0, 80, 321)
    def opt_len(p, m):
        D = m[:, None] + sig[:, None] * z[None]; ev = (p[:, None, None] * pz[None, None] * hscore(grid[None, :, None] - D[:, None, :])).sum((0, 2)); return grid[np.argmax(ev)]
    dec_len = np.array([opt_len(Pv[i], mu9[i]) for i in range(len(va))])
    Z = {k: dict(zip(map(str, np.load(os.path.join(HERE, f"runs/{f}_validation.npz"))["names"]), np.load(os.path.join(HERE, f"runs/{f}_validation.npz"))["traj"])) for k, f in F.items()}
    cl = [k[0] for k in va]; B = {c: wmean([np.asarray(Z[k][c], float) for k in W], [W[k] for k in W]) for c in cl}; bl = np.array([plen(B[c]) for c in cl])
    yl = np.array([Lv[k][1] for k in va]); G = np.array([log_of(c) for c in cl]); ul = np.unique(G)
    fo = {l: i % 5 for i, l in enumerate(np.random.RandomState(0).permutation(ul))}; f = np.array([fo[x] for x in G]); mix = np.zeros(len(va)); alphas = []
    for k in range(5):
        a_ = max(np.linspace(0, 1, 11), key=lambda a: hscore(a * dec_len[f != k] + (1 - a) * bl[f != k] - yl[f != k]).mean()); alphas.append(a_)
        mix[f == k] = a_ * dec_len[f == k] + (1 - a_) * bl[f == k]
    print(f"5 s length MAE: blend {np.mean(np.abs(bl - yl)):.2f} | decision-optimal {np.mean(np.abs(dec_len - yl)):.2f} | mix (CV alphas {alphas}) {np.mean(np.abs(mix - yl)):.2f}", flush=True)
    with ProcessPoolExecutor(8) as ex:
        R = dict(ex.map(_score, [(c, [B[c], retime_scale(B[c], max(dec_len[i], 0)), retime_scale(B[c], max(mix[i], 0))]) for i, c in enumerate(cl)], chunksize=8))
    S = np.array([R[c] for c in cl]); L = sorted(set(G)); H = set(np.random.RandomState(0).permutation(L)[:len(L) // 2])
    for j, nm in ((1, "decision-optimal length"), (2, "mix with blend length")):
        d = S[:, j] - S[:, 0]; by = collections.defaultdict(list)
        for g_, x in zip(G, d): by[g_].append(x)
        r_ = np.random.RandomState(0); bs_ = [np.nanmean(np.concatenate([by[l] for l in r_.choice(L, len(L))])) for _ in range(2000)]
        print(f"  {nm:24s} NPS {np.nanmean(S[:, j]):.4f} vs v9.5 {np.nanmean(S[:, 0]):.4f}: {np.nanmean(d):+.4f} [{np.percentile(bs_, 2.5):+.4f}, {np.percentile(bs_, 97.5):+.4f}] "
              f"halves {np.nanmean(d[[x in H for x in G]]):+.4f} / {np.nanmean(d[[x not in H for x in G]]):+.4f}", flush=True)
    torch.save({"state": best[1], "mu": mu, "sd": sd}, os.path.join(HERE, "runs/dec_wajepa.pt"))
    np.savez(os.path.join(HERE, "runs/dec_wajepa_validation.npz"), clip=np.array(cl), probs=Pv, mu9=mu9, dec_len=dec_len, mix=mix)


if __name__ == "__main__":
    main()
