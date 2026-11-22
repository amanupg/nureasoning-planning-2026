"""Sample check of data/img_plan (src/plan_fetch.py) against an independent earlier ingest (data/img_wa_full: same HF
originals, resized 1408x928 then 16:9 centre crop -> 512x256, JPEG q95).
For N validation clips: the key-frame (t=100) and t=97 front images of img_plan, put through the same crop/resize,
must match the img_wa_full copy of the SAME file (mean abs diff small) and differ more from the copy of a different
frame (negative control: same camera 0.5 s earlier). Also: sizes, camera names, reasoning JSON parse.
  python src/plan_fetch_check.py [N]
"""
import glob, json, os, random, sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def wa_view(p):
    im = cv2.imread(p); im = cv2.resize(im, (1408, 928), interpolation=cv2.INTER_LINEAR)
    h, w = im.shape[:2]; nh = int(round(w * 9 / 16)); y0 = (h - nh) // 2
    return cv2.resize(im[y0:y0 + nh], (512, 256), interpolation=cv2.INTER_AREA).astype(np.float32)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    rows = [json.loads(l) for l in open(os.path.join(HERE, "data/plan_rows/planning_validation.jsonl"))]
    random.Random(0).shuffle(rows); res = []; sizes = set(); cams_ok = 0; checked = 0
    for r in rows:
        if len(res) >= n: break
        if not os.path.isfile(os.path.join(HERE, "data/img_plan/validation", r["clip"], "_done.json")): continue
        for i, m in enumerate(r["images"]):
            p = os.path.join(HERE, "data/img_plan/validation", m); im = cv2.imread(p); sizes.add(im.shape[:2][::-1])
            cam = m.split("/")[2]; cams_ok += cam == ("CAM_M_L0", "CAM_M_F", "CAM_M_R0")[i % 3]; checked += 1
        front = r["images"][10]                                     # t0 (present), front
        wa = glob.glob(os.path.join(HERE, "data/img_wa_full/validation", front))
        cand = sorted(glob.glob(os.path.join(HERE, "data/img_wa_full/validation", r["clip"], "cameras/CAM_M_F/*.jpg")))
        if not wa or len(cand) < 2: continue
        same = np.abs(wa_view(os.path.join(HERE, "data/img_plan/validation", front)) - cv2.imread(wa[0]).astype(np.float32)).mean()
        other = [c for c in cand if os.path.basename(c) != os.path.basename(front)][0]
        neg = np.abs(wa_view(os.path.join(HERE, "data/img_plan/validation", front)) - cv2.imread(other).astype(np.float32)).mean()
        res.append((r["clip"][:30], round(float(same), 2), round(float(neg), 2), os.path.basename(other)))
    for x in res: print("same-file MAD %5.2f   other-frame MAD %5.2f   %s  vs %s" % (x[1], x[2], x[0], x[3]))
    print(f"clips {len(res)}; same-file MAD max {max(x[1] for x in res):.2f}; other-frame MAD min {min(x[2] for x in res):.2f}")
    print(f"image sizes {sizes}; camera order ok {cams_ok}/{checked}")
    rj = glob.glob(os.path.join(HERE, "data/reasoning_extra/train/*/reasoning/*.json")); bad = 0; drv = 0
    for p in rj:
        try: d = json.load(open(p)); drv += bool(d.get("Driving"))
        except Exception: bad += 1
    print(f"reasoning files {len(rj)}, unparsable {bad}, with Driving block {drv}")


if __name__ == "__main__":
    main()
