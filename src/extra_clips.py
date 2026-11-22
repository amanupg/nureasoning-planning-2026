"""B5 data: rows and frame list for the 2,236 cached train clips missing from the inherited index (a15ft_index_train.json
covers 7,654 of 9,890). Rows as in that index: r = 30, 40, ... while r - 15 >= first frame and r + 50 <= last frame (so the
5 s future exists), plus r = 100 (the test key frame). Frames: 4 cameras x r-15 .. r+40 step 5 (WA-JEPA's trainer).
  python src/extra_clips.py -> data/extra_rows_train.json [[clip, r]], data/members_wa_extra.json {"train": [{clip, tar, size, members}]}
"""
import json, os
NR = "/path/to/workspace/nureasoning"; HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMS = ["front_left", "front", "front_right", "back"]; OFFS = list(range(-15, 45, 5))
have = {c for c, r in json.load(open(f"{NR}/artifacts/a15ft_index_train.json"))}
clips = sorted(c for c in os.listdir(f"{NR}/cache/train") if c not in have and os.path.isfile(f"{NR}/cache/train/{c}/metadata.json"))
rows, mem_rows, n = [], [], 0
for c in clips:
    d = f"{NR}/cache/train/{c}"; fr = {f["frame_index"]: f for f in json.load(open(f"{d}/metadata.json"))["frames"]}
    lo, hi = min(fr), max(fr); rs = sorted({r for r in range(30, hi - 49, 10) if r - 15 >= lo} | ({100} if 85 >= lo and 150 <= hi else set()))
    rs = [r for r in rs if all(r + o in fr and (fr[r + o].get("sensors") or {}).get("cameras") for o in OFFS) and r + 50 in fr]
    if not rs: continue
    pv = json.load(open(f"{d}/_provenance.json")); idx = sorted({r + o for r in rs for o in OFFS})
    mem = [f"{c}/{fr[i]['sensors']['cameras'][cam]}" for i in idx for cam in CAMS]
    rows += [[c, r] for r in rs]; mem_rows.append({"clip": c, "tar": pv["source_path"], "size": pv["source_size"], "members": mem}); n += len(mem)
json.dump(rows, open(os.path.join(HERE, "data/extra_rows_train.json"), "w")); json.dump({"train": mem_rows}, open(os.path.join(HERE, "data/members_wa_extra.json"), "w"))
print(f"{len(mem_rows)} clips, {len(rows)} rows ({sum(r == 100 for _, r in rows)} at frame 100), {n} images (~{n * 49e3 / 1e9:.1f} GB at 256x512)")
