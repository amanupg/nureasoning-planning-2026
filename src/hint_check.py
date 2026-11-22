"""Maneuver-hint check (validation only): do planner trajectories add information to the
Decision questions (driving_decision_longitudinal / lateral / joint) beyond v7.1's recipe?

Baseline = the main track's v7.1 reasoning recipe, fitted on TRAIN questions only (reason_train.py
recipe b with COTFEAT, the train-only analog of the shipped train+val_cot bundle), so every
validation question is out of sample. Variant = the same models plus trajectory features
(ego-frame x/y at 1..5 s, per-second speeds, speed change, heading, lateral extent), fitted on
train with the RECORDED future and evaluated on validation with each trajectory source:
  gt (recorded future; ceiling), wa (WA-JEPA zero-shot), v71 (v7.1 planner), drive (Drive-JEPA).
Accuracy is compared on the same questions (clips that have every source). Paired log-bootstrap
for the difference vs baseline. Reads ../nureasoning read-only (no .pyc written).

  PYTHONDONTWRITEBYTECODE=1 python src/hint_check.py runs/wajepa_169_validation.npz
"""
import collections, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NR = "/path/to/workspace/nureasoning"
sys.dont_write_bytecode = True
os.environ["COTFEAT"] = "1"
sys.path.insert(0, os.path.join(NR, "src")); sys.path.insert(0, os.path.join(NR, "recon/devkit"))
sys.path.insert(0, os.path.join(HERE, "src"))
SUBS = ("driving_decision_longitudinal", "driving_decision_lateral", "driving_decision_joint")
TF = [f"t_x{k}" for k in range(1, 6)] + [f"t_y{k}" for k in range(1, 6)] + [f"t_v{k}" for k in range(1, 6)] + \
     ["t_yaw5", "t_dv", "t_vmin", "t_latmax", "t_stop"]


def tfeat(P, v0):
    """P: (10, 3) ego-frame poses at 0.5..5.0 s."""
    Q = np.vstack([[0, 0, 0], P]); s = np.hypot(*np.diff(Q[::2, :2], axis=0).T)   # 1 s segments
    f = {f"t_x{k}": P[2 * k - 1, 0] for k in range(1, 6)}
    f.update({f"t_y{k}": P[2 * k - 1, 1] for k in range(1, 6)}); f.update({f"t_v{k}": s[k - 1] for k in range(1, 6)})
    f.update(t_yaw5=P[-1, 2], t_dv=s[-1] - v0, t_vmin=s.min(), t_latmax=np.abs(P[:, 1]).max(), t_stop=float(s[-1] < 0.5))
    return f


def gt_traj(split, clip, kf):
    from nureasoning.common.pickle_io import load_pickle
    from wajepa_run import rel_se2
    cd = os.path.join(NR, "cache", split, clip)
    fr = {f["frame_index"]: f for f in json.load(open(os.path.join(cd, "metadata.json")))["frames"]}
    if kf + 50 not in fr: return None, None
    e = load_pickle(os.path.join(cd, fr[kf]["ego_state"])); o = np.array([e.pose["x"], e.pose["y"], e.pose["yaw"]])
    P = np.array([[q.pose["x"], q.pose["y"], q.pose["yaw"]] for q in
                  (load_pickle(os.path.join(cd, fr[kf + 5 * k]["ego_state"])) for k in range(1, 11))])
    return rel_se2(o, P), (o, float(np.hypot(e.velocity["vx"], e.velocity["vy"])))


def main():
    WA = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "runs/wajepa_169_validation.npz")
    os.chdir(NR)                                             # the main track's loaders use relative artifact paths
    import reason_train as RT, reason_model as RM
    from wajepa_run import rel_se2
    kf_of = lambda q: int(q.get("_kf", 100))
    tr = [q for q in RT.prep("vqa_train", "train") if q["subcategory"] in SUBS]
    va = [q for q in RT.prep("vqa_val", "validation") if q["subcategory"] in SUBS]
    for root, rows in (("vqa_train", tr), ("vqa_val", va)):                  # key frame per question file
        kfm = {}
        for d in os.listdir(root):
            for fn in os.listdir(os.path.join(root, d)):
                if fn.endswith(".json"):
                    o = json.load(open(os.path.join(root, d, fn)))
                    for q in o.get("questions") or []: kfm[q.get("question_id")] = o.get("key_frame_index", 100)
        for q in rows: q["_kf"] = kfm.get(q.get("question_id"), 100)
    print(f"decision questions: train {len(tr)}, validation {len(va)}", flush=True)
    cache = {}
    def gt(split, clip, kf):
        if (split, clip, kf) not in cache: cache[(split, clip, kf)] = gt_traj(split, clip, kf)
        return cache[(split, clip, kf)]
    tr = [q for q in tr if gt("train", q["_clip"], kf_of(q))[0] is not None]
    for q in tr:
        P, (o, v0) = gt("train", q["_clip"], kf_of(q)); q["_t"] = tfeat(P, v0)
    # validation trajectory sources
    SRC = {"wa": WA,
           "v71": os.path.join(HERE, "runs/v71_validation.npz"), "drive": os.path.join(HERE, "runs/drivejepa_validation.npz")}
    G = {k: dict(zip(map(str, np.load(p)["names"]), np.load(p)["traj"])) for k, p in SRC.items()}
    va = [q for q in va if all(q["_clip"] in G[k] for k in G) and gt("validation", q["_clip"], kf_of(q))[0] is not None]
    feats = collections.defaultdict(dict)
    for q in va:
        P, (o, v0) = gt("validation", q["_clip"], kf_of(q)); feats["gt"][q["question_id"]] = tfeat(P, v0)
        P100, (o100, _) = gt("validation", q["_clip"], 100)
        for k in G:                                       # 51 global poses at 0.1 s from the key frame
            feats[k][q["question_id"]] = tfeat(rel_se2(o100, np.asarray(G[k][q["_clip"]])[5::5][:10]), v0)
    logs = lambda qs: sorted({"_".join(q["_clip"].split("_")[:2]) for q in qs})
    print(f"validation questions with all sources: {len(va)} ({len({q['_clip'] for q in va})} clips, {len(logs(va))} logs); "
          f"validation logs also in train: {len(set(logs(va)) & set(logs(tr)))}", flush=True)
    maps = RM.build_maps(tr + va)
    base_num = list(RM.NUM)
    base, _ = RT.fit_all(tr, maps)                               # v7.1 recipe, train only
    RM.NUM[:] = base_num + TF
    for q in tr: q["_f"].update(q["_t"])
    withT, pri = RT.fit_all(tr, maps)
    res = collections.defaultdict(dict)                          # res[variant][qid] = correct
    for q in va:
        RM.NUM[:] = base_num
        res["base"][q["question_id"]] = RM.predict_letter(base[q["subcategory"]], q, maps, pri.get(q["subcategory"], {})) == q["answer"]
        RM.NUM[:] = base_num + TF
        for k in ("gt", "wa", "v71", "drive"):
            f0 = dict(q["_f"]); q["_f"].update(feats[k][q["question_id"]])
            res[k][q["question_id"]] = RM.predict_letter(withT[q["subcategory"]], q, maps, pri.get(q["subcategory"], {})) == q["answer"]
            q["_f"] = f0
    lg = {q["question_id"]: "_".join(q["_clip"].split("_")[:2]) for q in va}
    out = {}
    print(f"\n{'subcategory':32} {'n':>4} " + "".join(f"{k:>8}" for k in ("base", "gt", "wa", "v71", "drive")))
    for sub in SUBS + ("ALL",):
        ids = [q["question_id"] for q in va if sub == "ALL" or q["subcategory"] == sub]
        acc = {k: float(np.mean([res[k][i] for i in ids])) for k in ("base", "gt", "wa", "v71", "drive")}
        out[sub] = {"n": len(ids), **acc}
        print(f"{sub:32} {len(ids):>4} " + "".join(f"{acc[k]:>8.3f}" for k in acc))
    ids = [q["question_id"] for q in va]; L = sorted(set(lg.values())); rng = np.random.RandomState(0)
    by = collections.defaultdict(list)
    for i in ids: by[lg[i]].append(i)
    for k in ("wa", "v71", "drive", "gt"):
        d = {l: [res[k][i] - res["base"][i] for i in by[l]] for l in L}
        bs = [np.mean(np.concatenate([d[l] for l in rng.choice(L, len(L))])) for _ in range(2000)]
        out[f"diff_{k}"] = [float(np.mean([res[k][i] - res["base"][i] for i in ids])), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        print(f"  {k:5s} - base: {out[f'diff_{k}'][0]:+.3f}  95% log-bootstrap [{out[f'diff_{k}'][1]:+.3f}, {out[f'diff_{k}'][2]:+.3f}]")
    json.dump(out, open(os.path.join(HERE, "runs/hint_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
