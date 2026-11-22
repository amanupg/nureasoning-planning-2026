"""What does predicting only 4 s cost? Score the RECORDED future (validation) two ways: exact 51 poses, and
the recorded 8 poses (0.5..4 s) passed through our 4 s -> 5 s extension (wajepa_run.to_51: linear in time,
last second at the last velocity). The gap is the cost of the extrapolation alone, with a perfect model.
"""
import os, sys, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(HERE, "src"))
from wajepa_run import to_51, score, rel_se2, NR
nps = lambda r: r[0] * r[1] * (.3 * r[2] + .2 * r[3] + .5 * r[4])


def one(c):
    from nureasoning.common.pickle_io import load_pickle
    cd = os.path.join(NR, "cache/validation", c); fr = {f["frame_index"]: f for f in json.load(open(os.path.join(cd, "metadata.json")))["frames"]}
    E = [load_pickle(os.path.join(cd, fr[100 + i]["ego_state"])) for i in range(0, 51)]
    G = np.array([[e.pose["x"], e.pose["y"], e.pose["yaw"]] for e in E]); o = G[0]
    p8 = rel_se2(o, G[5:41:5])                                   # recorded 0.5..4.0 s in the ego frame
    ext = to_51(p8, o)
    a, b = score("validation", c, G), score("validation", c, ext)
    return c, (nps(a) if a else np.nan, nps(b) if b else np.nan, a[4] if a else np.nan, b[4] if b else np.nan,
               float(np.hypot(*(ext[-1, :2] - G[-1, :2]))))


if __name__ == "__main__":
    clips = sorted(c for c in os.listdir(os.path.join(NR, "cache/validation")) if os.path.isfile(os.path.join(NR, "cache/validation", c, "metadata.json")))
    with ProcessPoolExecutor(8) as ex: R = dict(ex.map(one, clips, chunksize=8))
    A = np.array(list(R.values()), float)
    print(f"{len(R)} clips: recorded future NPS {np.nanmean(A[:,0]):.4f} | 4 s + extension {np.nanmean(A[:,1]):.4f} | cost {np.nanmean(A[:,1]-A[:,0]):+.4f}")
    print(f"  human-likeness {np.nanmean(A[:,2]):.4f} -> {np.nanmean(A[:,3]):.4f}; 5 s endpoint error from extension: median {np.nanmedian(A[:,4]):.2f} m, mean {np.nanmean(A[:,4]):.2f} m, p90 {np.nanpercentile(A[:,4],90):.2f} m")
