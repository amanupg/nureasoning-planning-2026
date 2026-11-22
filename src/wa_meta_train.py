"""Pack what the WA-JEPA fine-tune needs from the (read-only) nuReasoning training cache, so a GPU pod
needs only this file plus the frames: per sample (clip, r) the ego status, 4 history and 8 future
ego-frame poses (built exactly as wajepa_ft.sample does), and per clip the 4 camera paths of every
needed frame (r-15 .. r+40, step 5).
  python src/wa_meta_train.py -> data/wa_train_meta.npz, data/wa_train_paths.json.gz
"""
import gzip, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wajepa_ft import _meta, HIST, FUT, NR
FUT10 = list(range(5, 55, 5))                      # 0.5 .. 5.0 s targets for a 5 s planner head
from wajepa_run import CAMS, CMD, rel_se2


def main():
    import argparse
    from nureasoning.common.pickle_io import load_pickle
    ap = argparse.ArgumentParser(); ap.add_argument("--rows", default=os.path.join(NR, "artifacts/a15ft_index_train.json"))
    ap.add_argument("--tag", default="", help="output suffix, e.g. 'extra' -> data/wa_train_meta_extra.npz"); a = ap.parse_args()
    sfx = f"_{a.tag}" if a.tag else ""; mo, po = f"data/wa_train_meta{sfx}.npz", f"data/wa_train_paths{sfx}.json.gz"
    rows = [(c, int(r)) for c, r in json.load(open(a.rows))]
    st, hi, fu, fu10, paths = [], [], [], [], {}; t0 = time.time()
    for i, (c, r) in enumerate(rows):
        fr = _meta(c); d = os.path.join(NR, "cache/train", c)
        egos = {o: load_pickle(os.path.join(d, fr[r + o]["ego_state"])) for o in HIST + FUT}
        P = np.array([[egos[o].pose["x"], egos[o].pose["y"], egos[o].pose["yaw"]] for o in HIST + FUT], float)
        rel = rel_se2(P[len(HIST) - 1], P).astype(np.float32); k = egos[0]; s = np.zeros(8, np.float32)
        s[CMD.get(((fr[r].get("mission_goal") or {}).get("command")), 3)] = 1
        s[4:8] = [k.velocity["vx"], k.velocity["vy"], k.acceleration["ax"], k.acceleration["ay"]]
        st.append(s); hi.append(rel[:len(HIST)]); fu.append(rel[len(HIST):])
        P10 = np.array([[q.pose["x"], q.pose["y"], q.pose["yaw"]] for q in (load_pickle(os.path.join(d, fr[r + o]["ego_state"])) for o in FUT10)], float)
        fu10.append(rel_se2(P[len(HIST) - 1], P10).astype(np.float32))
        pc = paths.setdefault(c, {})
        for o in HIST + FUT: pc[str(r + o)] = [fr[r + o]["sensors"]["cameras"][cam] for cam in CAMS]
        if (i + 1) % 10000 == 0: print(f"  {i+1}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(mo, clip=np.array([c for c, _ in rows]), r=np.array([r for _, r in rows]),
                        status=np.array(st), hist=np.array(hi), fut=np.array(fu), fut10=np.array(fu10))
    with gzip.open(po, "wt") as f: json.dump(paths, f)
    print(f"{len(rows)} samples, {len(paths)} clips -> {mo}, {po} ({os.path.getsize(mo)/1e6:.0f} MB, {os.path.getsize(po)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
