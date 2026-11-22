"""Oracle test of the DriveMA idea at inference: do driving decisions (meta-actions) select the right mode?

1. Trajectory -> decision classifier: on TRAINING clips, features of the recorded 5 s future (ego frame;
   src/hint_check.tfeat) -> the annotated longitudinal and lateral decision of that frame (devkit VQA
   generator output in ../nureasoning/vqa_train, read-only). Held-out-log accuracy is reported.
2. On the validation clips that have K WA-JEPA samples (runs/samples_w1_validation*.npz), label each sample
   with its implied decision, then select samples consistent with the ANNOTATED decision of the clip
   (oracle) and compare NPS with: a random sample, the model as run, the current blend.
  python src/decision_oracle.py
"""
import collections, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NR = "/path/to/workspace/nureasoning"
sys.path.insert(0, os.path.join(HERE, "src")); sys.path.insert(0, os.path.join(NR, "recon/devkit"))
from hint_check import tfeat, gt_traj, TF
from wajepa_run import rel_se2
nps = lambda L: L[..., 0] * L[..., 1] * (.3 * L[..., 2] + .2 * L[..., 3] + .5 * L[..., 4])
SUBS = ("driving_decision_longitudinal", "driving_decision_lateral", "driving_decision_joint")
log_of = lambda c: "_".join(c.split("_")[:2])


def decisions(root):
    """{(clip, key_frame): (lon, lat)} from the generated VQA (any decision question carries both fields)."""
    out = {}
    for c in os.listdir(os.path.join(NR, root)):
        d = os.path.join(NR, root, c)
        if not os.path.isdir(d): continue
        for fn in os.listdir(d):
            if not fn.endswith("_vqa.json"): continue
            o = json.load(open(os.path.join(d, fn))); kf = int(o.get("key_frame_index", 100))
            for q in o.get("questions") or []:
                if q.get("subcategory") in SUBS and q.get("longitudinal") and q.get("lateral"):
                    out[(c, kf)] = (q["longitudinal"].strip().lower(), q["lateral"].strip().lower()); break
    return out


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    tr = decisions("vqa_train"); va = decisions("vqa_val")
    print(f"decision labels: train {len(tr)} frames, validation {len(va)} frames", flush=True)
    X, Ylon, Ylat, G = [], [], [], []
    for (c, kf), (lo, la) in tr.items():
        P, info = gt_traj("train", c, kf)
        if P is None: continue
        f = tfeat(P, info[1]); X.append([f[k] for k in TF]); Ylon.append(lo); Ylat.append(la); G.append(log_of(c))
    X = np.array(X, float); Ylon = np.array(Ylon); Ylat = np.array(Ylat); G = np.array(G)
    print("train classes lon:", collections.Counter(Ylon).most_common(), "\n lat:", collections.Counter(Ylat).most_common(), flush=True)
    rng = np.random.RandomState(0); ul = np.unique(G); hold = set(rng.permutation(ul)[:len(ul) // 5])
    te = np.isin(G, list(hold))
    clf = {}
    for nm, Y in (("lon", Ylon), ("lat", Ylat)):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, l2_regularization=1.0, random_state=0).fit(X[~te], Y[~te])
        print(f"  trajectory -> {nm} decision, held-out logs accuracy {np.mean(m.predict(X[te]) == Y[te]):.3f} "
              f"(majority {max(collections.Counter(Y[te]).values()) / te.sum():.3f})", flush=True)
        clf[nm] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, l2_regularization=1.0, random_state=0).fit(X, Y)
    # validation samples
    z = np.load(os.path.join(HERE, "runs/samples_w1_validation_scores.npz")); names = list(map(str, z["names"])); S = nps(z["labels"])
    T = dict(zip(map(str, np.load(os.path.join(HERE, "runs/samples_w1_validation.npz"))["names"]),
                 np.load(os.path.join(HERE, "runs/samples_w1_validation.npz"))["traj"]))
    d = np.load(os.path.join(HERE, "runs/wajepa_ft_w1_validation.npz")); D = dict(zip(map(str, d["names"]), nps(d["labels"])))
    E = json.load(open(os.path.join(HERE, "runs/ensemble3_val_scores.json")))
    rows = []
    for i, c in enumerate(names):
        if (c, 100) not in va: continue
        lo, la = va[(c, 100)]
        _, (o, v0) = gt_traj("validation", c, 100)
        F = np.array([[tfeat(rel_se2(o, g[5::5][:10]), v0)[k] for k in TF] for g in T[c]])
        pl, pa = clf["lon"].predict_proba(F), clf["lat"].predict_proba(F)
        il = list(clf["lon"].classes_).index(lo) if lo in clf["lon"].classes_ else None
        ia = list(clf["lat"].classes_).index(la) if la in clf["lat"].classes_ else None
        p = (pl[:, il] if il is not None else np.ones(len(F))) * (pa[:, ia] if ia is not None else np.ones(len(F)))
        hard = (np.array(clf["lon"].classes_)[pl.argmax(1)] == lo) & (np.array(clf["lat"].classes_)[pa.argmax(1)] == la)
        s = S[i]
        rows.append({"clip": c, "random": s.mean(), "det": D[c], "blend": E[c]["v93"], "best": s.max(),
                     "match_any": hard.any(), "hard": s[hard].mean() if hard.any() else s.mean(),
                     "top_p": s[int(np.argmax(p))], "pweighted": float((s * p).sum() / p.sum()) if p.sum() > 0 else s.mean()})
    R = {k: np.mean([r[k] for r in rows]) for k in rows[0] if k != "clip"}
    print(f"\n{len(rows)} of {len(names)} sampled validation clips have an annotated decision")
    print(f"  samples matching the annotated decision exist on {R['match_any']:.1%} of clips")
    for k, lab in (("random", "one random sample"), ("det", "WA-JEPA ft as run"), ("blend", "current blend (v9.3 mix)"),
                   ("hard", "ORACLE: random sample among decision-matching ones"), ("pweighted", "ORACLE: weighted by P(annotated decision)"),
                   ("top_p", "ORACLE: sample most likely to match the decision"), ("best", "oracle best of 16 (uses the future)")):
        print(f"  {lab:52s} {R[k]:.4f}")
    json.dump(rows, open(os.path.join(HERE, "runs/decision_oracle_rows.json"), "w"), default=float)


if __name__ == "__main__":
    main()
