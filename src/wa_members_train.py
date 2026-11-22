"""Training frames for the WA-JEPA fine-tune: for every training sample (clip, r) in the main track's
a15ft index (81,523 samples, read-only), WA-JEPA's trainer needs 4 cameras x (4 history + 8 future)
frames at 2 Hz: r-15 .. r+40, step 5. Per clip we fetch the union over its samples.
  python src/wa_members_train.py [--limit N] -> data/members_wa_train.json {"train": [{clip, tar, size, members}]}
"""
import argparse, collections, json, os
NR = "/path/to/workspace/nureasoning"
CAMS = ["front_left", "front", "front_right", "back"]
OFFS = list(range(-15, 45, 5))                      # 4 history (-15..0) + 8 future (+5..+40)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="data/members_wa_train.json"); a = ap.parse_args()
    by = collections.defaultdict(set)
    for c, r in json.load(open(f"{NR}/artifacts/a15ft_index_train.json")): by[c].add(int(r))
    clips = sorted(by)[:a.limit] if a.limit else sorted(by)
    rows, n = [], 0
    for c in clips:
        d = f"{NR}/cache/train/{c}"; pv = json.load(open(f"{d}/_provenance.json"))
        fr = {f["frame_index"]: f for f in json.load(open(f"{d}/metadata.json"))["frames"]}
        idx = sorted({r + o for r in by[c] for o in OFFS})
        mem = [f"{c}/{fr[i]['sensors']['cameras'][cam]}" for i in idx for cam in CAMS]
        rows.append({"clip": c, "tar": pv["source_path"], "size": pv["source_size"], "members": mem}); n += len(mem)
    json.dump({"train": rows}, open(a.out, "w"))
    print(f"{len(rows)} clips, {sum(len(by[c]) for c in clips)} samples, {n} images -> {a.out}")


if __name__ == "__main__":
    main()
