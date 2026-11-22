"""Fetch list for the planning images (src/plan_fetch.py): per clip, its HF tar, the tar size, every image the
planning rows reference (data/plan_rows/planning_{split}.jsonl) and, for train, every reasoning/*.json that the clip
metadata lists but the main-track cache lacks (those can hold Driving decisions; without them some decision frames are
missed). Tar paths and sizes come from our earlier member files (data/members*.json).
  python src/plan_members.py   -> data/plan_members.json
"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan_rows import CACHE, HERE

D = os.path.join(HERE, "data")


def tars():
    t = {}
    for f in ("members.json", "members_wa_train.json", "members_wa_extra.json"):
        for split, rows in json.load(open(os.path.join(D, f))).items():
            for r in rows: t[(split, r["clip"])] = (r["tar"], r["size"])
    return t


def missing_reasoning(clip):
    d = os.path.join(CACHE, "train", clip)
    fr = json.load(open(os.path.join(d, "metadata.json")))["frames"]
    rels = sorted({f["reasoning"] for f in fr if f.get("reasoning")})
    extra = [os.path.join(D, x, "train", clip) for x in ("reasoning_driving", "reasoning_extra")]   # src/plan_fetch.py
    return [f"{clip}/{r}" for r in rels if not os.path.isfile(os.path.join(d, r))
            and not any(os.path.isfile(os.path.join(e, r)) for e in extra)]


def main():
    T = tars(); out = {}
    for split in ("validation", "test", "train"):
        imgs = {}
        for line in open(os.path.join(D, "plan_rows", f"planning_{split}.jsonl")):
            r = json.loads(line); imgs.setdefault(r["clip"], set()).update(r["images"])
        clips = sorted(imgs)
        reas = {}
        if split == "train":
            with ThreadPoolExecutor(16) as ex:
                for c, m in zip(clips, ex.map(missing_reasoning, clips)): reas[c] = m
        rows = []
        for c in clips:
            tar, size = T[(split, c)]
            rows.append({"clip": c, "tar": tar, "size": size, "members": sorted(imgs[c]), "reasoning": reas.get(c, [])})
        out[split] = rows
        print(split, len(rows), "clips", sum(len(r["members"]) for r in rows), "images",
              sum(len(r["reasoning"]) for r in rows), "reasoning", flush=True)
    json.dump(out, open(os.path.join(D, "plan_members.json"), "w"))


if __name__ == "__main__":
    main()
