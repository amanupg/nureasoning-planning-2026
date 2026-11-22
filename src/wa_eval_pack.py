"""Pack validation / test clips for cluster-side sampling (scorer evaluation and best-of-K inference).

For each clip: the 16 history frames the planner reads (key frame 100 + offsets -15/-10/-5/0, 4 cameras),
16:9 centre-cropped and resized to 256x512 as JPEG q95 (the same ops as the train ingest, src/img_ingest2.py
IMG_WAJEPA, so they load with crop "full"), plus ego status, history trajectory and the key-frame origin,
built exactly as src/wajepa_run.clip_inputs does from the nuReasoning cache (read-only).
  python src/wa_eval_pack.py --split validation
  -> data/img_wa_full/<split>/<clip>/<camera file>, data/wa_<split>_meta.npz, data/wa_<split>_paths.json.gz
"""
import argparse, gzip, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from wajepa_run import NR, CAMS, OFFS, KEY, CMD, rel_se2


def one(args):
    import cv2
    from nureasoning.common.pickle_io import load_pickle
    split, clip, src_root, dst_root = args
    cd = os.path.join(NR, "cache", split, clip)
    fr = {f["frame_index"]: f for f in json.load(open(os.path.join(cd, "metadata.json")))["frames"]}
    egos = [load_pickle(os.path.join(cd, fr[KEY + o]["ego_state"])) for o in OFFS]
    poses = np.array([[e.pose["x"], e.pose["y"], e.pose["yaw"]] for e in egos], float); k = egos[-1]
    st = np.zeros(8, np.float32); st[CMD.get(((fr[KEY].get("mission_goal") or {}).get("command")), 3)] = 1
    st[4:8] = [k.velocity["vx"], k.velocity["vy"], k.acceleration["ax"], k.acceleration["ay"]]
    paths = {}
    for o in OFFS:
        paths[str(KEY + o)] = []
        for c in CAMS:
            p = fr[KEY + o]["sensors"]["cameras"][c]; dst = os.path.join(dst_root, clip, p)
            if not os.path.isfile(dst):
                im = cv2.imread(os.path.join(src_root, clip, p)); h, w = im.shape[:2]; nh = int(round(w * 9 / 16)); y0 = (h - nh) // 2
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                cv2.imwrite(dst + ".tmp.jpg", cv2.resize(im[y0:y0 + nh], (512, 256), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95])
                os.replace(dst + ".tmp.jpg", dst)
            paths[str(KEY + o)].append(p)
    return clip, st, rel_se2(poses[-1], poses).astype(np.float32), poses[-1], paths


def main():
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="validation"); a = ap.parse_args()
    src_root = os.path.join(HERE, "data/img", a.split); dst_root = os.path.join(HERE, "data/img_wa_full", a.split)
    clips = sorted(c for c in os.listdir(src_root) if os.path.isfile(os.path.join(src_root, c, "_done.json")))
    with ProcessPoolExecutor(8) as ex:
        R = list(ex.map(one, [(a.split, c, src_root, dst_root) for c in clips], chunksize=8))
    np.savez(os.path.join(HERE, f"data/wa_{a.split}_meta.npz"), clip=np.array([r[0] for r in R]), r=np.full(len(R), KEY),
             status=np.stack([r[1] for r in R]), hist=np.stack([r[2] for r in R]), origin=np.stack([r[3] for r in R]))
    with gzip.open(os.path.join(HERE, f"data/wa_{a.split}_paths.json.gz"), "wt") as f: json.dump({r[0]: r[4] for r in R}, f)
    print(f"{a.split}: {len(R)} clips packed -> {dst_root}", flush=True)


if __name__ == "__main__":
    main()
