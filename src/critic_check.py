"""Can a learned scorer pick the best of K sampled trajectories? (CPU diagnostic, no scene input yet.)

Train: the RL round's graded samples (runs/rl_h10u4_kf100_graded.npz: 3,975 frame-100 train frames x 16 h10u4
samples, devkit NPS). Features per sample: trajectory shape (src/hint_check.tfeat), ego speed, and the sample's
position relative to its own group of 16 (difference from the group median per feature, distance of the endpoint
from the group mean). Target: NPS. Model: gradient-boosted trees. In-domain check on held-out drive logs.
Test: the 300 validation clips with 16 scored w1 samples each (runs/samples_w1_validation*.npz, a different model,
so this also tests transfer). Reports the NPS of the picked sample vs one random sample, the medoid, the model as
run, and the oracle best of 16.
  python src/critic_check.py
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from hint_check import tfeat, gt_traj, TF
from wajepa_run import rel_se2
nps = lambda L: L[..., 0] * L[..., 1] * (.3 * L[..., 2] + .2 * L[..., 3] + .5 * L[..., 4])
log_of = lambda c: "_".join(c.split("_")[:2])


def group_feats(T, v0):
    """T: (K, 10, 3) ego-frame samples -> (K, F) features."""
    B = np.array([[tfeat(t, v0)[k] for k in TF] for t in T], float)
    end = T[:, -1, :2]; dmean = np.linalg.norm(end - end.mean(0), axis=1)
    rank = dmean.argsort().argsort() / (len(T) - 1)
    return np.hstack([B, B - np.median(B, 0), dmean[:, None], rank[:, None], np.full((len(T), 1), v0)])


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    q = np.load(os.path.join(HERE, "runs/rl_h10u4_kf100_graded.npz"))
    X, Y, G, S = [], [], [], []
    for c, r, T, L in zip(q["clip"], q["r"], q["traj"], q["labels"]):
        s = nps(L)
        if np.isnan(s).any(): continue
        _, info = gt_traj("train", str(c), int(r))
        if info is None: continue
        X.append(group_feats(np.asarray(T, float), info[1])); Y.append(s); G.append(log_of(str(c)))
    X, Y, G = np.array(X), np.array(Y), np.array(G); n, K, F = X.shape
    print(f"train: {n} frames x {K} samples, {F} features", flush=True)
    rng = np.random.RandomState(0); ul = np.unique(G); hold = set(rng.permutation(ul)[:len(ul) // 5]); te = np.isin(G, list(hold))
    mk = lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, l2_regularization=1.0, random_state=0)
    m = mk().fit(X[~te].reshape(-1, F), Y[~te].reshape(-1))
    P = m.predict(X[te].reshape(-1, F)).reshape(-1, K); Yt = Y[te]; pick = P.argmax(1)
    med = X[te][..., -3].argmin(1)                                          # sample closest to the group mean
    print(f"held-out train logs ({te.sum()} frames): random {Yt.mean():.4f}, medoid {Yt[np.arange(len(Yt)), med].mean():.4f}, "
          f"critic pick {Yt[np.arange(len(Yt)), pick].mean():.4f}, oracle {Yt.max(1).mean():.4f}", flush=True)
    m = mk().fit(X.reshape(-1, F), Y.reshape(-1))
    z = np.load(os.path.join(HERE, "runs/samples_w1_validation_scores.npz")); names = list(map(str, z["names"])); SV = nps(z["labels"])
    T = dict(zip(map(str, np.load(os.path.join(HERE, "runs/samples_w1_validation.npz"))["names"]),
                 np.load(os.path.join(HERE, "runs/samples_w1_validation.npz"))["traj"]))
    d = np.load(os.path.join(HERE, "runs/wajepa_ft_w1_validation.npz")); D = dict(zip(map(str, d["names"]), nps(d["labels"])))
    rows = []
    for i, c in enumerate(names):
        s = SV[i]
        if np.isnan(s).any(): continue
        _, (o, v0) = gt_traj("validation", c, 100)
        Tc = np.array([rel_se2(o, g[5::5][:10]) for g in T[c]]); Xc = group_feats(Tc, v0)
        p = m.predict(Xc); rows.append((s.mean(), D[c], s[Xc[:, -3].argmin()], s[p.argmax()], s.max()))
    R = np.array(rows)
    print(f"validation ({len(R)} clips, w1 samples): random {R[:, 0].mean():.4f}, as run {R[:, 1].mean():.4f}, "
          f"medoid {R[:, 2].mean():.4f}, critic pick {R[:, 3].mean():.4f}, oracle {R[:, 4].mean():.4f}")
    d = R[:, 3] - R[:, 1]; b = [np.mean(d[np.random.RandomState(k).randint(0, len(d), len(d))]) for k in range(2000)]
    print(f"  critic pick - as run {d.mean():+.4f} [{np.percentile(b, 2.5):+.4f}, {np.percentile(b, 97.5):+.4f}] (clip bootstrap)")


if __name__ == "__main__":
    main()
