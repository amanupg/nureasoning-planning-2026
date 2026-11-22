"""Completeness check of data/img_plan against the planning rows: every image named by data/plan_rows/planning_*.jsonl
and dup_images_*.json must exist with non-zero size. Prints per-split counts and up to 10 missing paths.
  python src/plan_img_complete.py
"""
import json, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "data/plan_rows"); I = os.path.join(HERE, "data/img_plan")


def check(split, names):
    miss = [n for n in names if not (os.path.isfile(p := os.path.join(I, split, n)) and os.path.getsize(p) > 0)]
    print(f"{split}: {len(names)} unique images, {len(miss)} missing", flush=True)
    for m in miss[:10]: print("  MISSING", m)
    return len(miss)


def main():
    bad = 0
    for split in ("validation", "test", "train"):
        names = set()
        for l in open(os.path.join(R, f"planning_{split}.jsonl")): names.update(json.loads(l)["images"])
        bad += check(split, sorted(names))
    for split in ("validation", "test"):
        d = json.load(open(os.path.join(R, f"dup_images_{split}.json")))
        vals = d.values() if isinstance(d, dict) else d
        names = set()
        for v in vals:
            if isinstance(v, str): names.add(v)
            elif isinstance(v, dict): names.update(x for x in v.values() if isinstance(x, str))
            else: names.update(x for x in v if isinstance(x, str))
        bad += check(f"{split}", sorted(names)) if names else 0
    print("PLAN_IMG_COMPLETE" if bad == 0 else f"PLAN_IMG_INCOMPLETE {bad}")


if __name__ == "__main__":
    main()
