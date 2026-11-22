import json, os, sys, numpy as np
sys.path.insert(0, "src")
from ensemble_eval import wmean
from wajepa_run import score
from ensemble_eval import nps
NR = "/path/to/workspace/nureasoning"; sys.path.insert(0, os.path.join(NR, "recon/devkit"))
from nureasoning.planning.benchmark import BenchmarkConfig, select_key_frame_idx, _load_metadata
from nureasoning.common.pickle_io import load_pickle
from nureasoning.submission.format import global_to_ego, normalize_ego_trajectory
F = {"b": "runs/v71B_validation.npz", "w1": "runs/wajepa_ft_w1_validation.npz", "wa": "runs/wajepa_169_validation.npz",
     "dj": "runs/djft_r2avg_validation.npz", "raw": "runs/a15s1lora_raw_validation.npz"}
W = {"b": .4, "w1": .35, "wa": .2, "dj": .25, "raw": .1}
Z = {k: dict(zip(map(str, np.load(p)["names"]), np.load(p)["traj"])) for k, p in F.items()}
names = sorted(set.intersection(*[set(v) for v in Z.values()]))
CFG = BenchmarkConfig(); root = os.path.join(NR, "cache/validation"); out = {}; sc = []
for i, c in enumerate(names):
    g = wmean([np.asarray(Z[k][c], float) for k in W], [W[k] for k in W])
    cp = os.path.join(root, c); fr = _load_metadata(cp).get("frames", []); ki = select_key_frame_idx(fr, CFG.key_frame_index)
    ego = load_pickle(os.path.join(cp, fr[ki]["ego_state"])); e = normalize_ego_trajectory(global_to_ego(g, ego.pose))
    out[c] = {"ego_xyyaw": np.round(np.asarray(e, float), 4).tolist(), "global_xyyaw": np.round(g, 4).tolist()}
    if i < 200: r = score("validation", c, g); sc.append(nps(r) if r else np.nan)
json.dump({"_doc": "v9.4-recipe blend (b .4, w1 .35, wa .2, dj .25, raw .1; weighted per-timestep mean, yaw circular) on validation. "
           "51 rows per clip = t 0.0..5.0 s at 0.1 s, key frame = devkit select_key_frame_idx (frame 100). ego_xyyaw: ego frame of the key "
           "frame (x forward, y left, m, rad; row 0 = 0) via devkit global_to_ego + normalize_ego_trajectory; global_xyyaw: as blended. "
           "Model predictions only (no GT). Built by jepa-nureasoning 2026-11-11.", "clips": out}, open("runs/handoff/v94_blend_validation_traj.json", "w"))
print(len(out), "clips; NPS on first 200 clips (sanity)", np.nanmean(sc))
