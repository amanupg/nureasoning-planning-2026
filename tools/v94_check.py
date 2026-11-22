import json, sys, numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "src")
from wajepa_run import score
from ensemble_eval import nps
D = json.load(open("runs/handoff/v94_blend_validation_traj.json"))["clips"]
def f(c):
    r = score("validation", c, np.asarray(D[c]["global_xyyaw"])); return nps(r) if r else np.nan
if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex: s = list(ex.map(f, sorted(D), chunksize=8))
    print("full val NPS of handoff file", np.nanmean(s), "n", len(s))
