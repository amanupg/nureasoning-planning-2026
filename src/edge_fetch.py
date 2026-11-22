"""Fetch the Alpamayo 2 Edge planning frames (data/edge_members.json from src/edge_members.py / src/edge_rows.py)
with the planning fetcher (src/plan_fetch.py: HTTP range reads from the HF tars, saved tar indices), stored at a
chosen pixel budget (JPEG q95; note plan_fetch's done record always says "max_pixels 196608", its default):
  val/test: --maxpix 1306624 (1408x928, half of the 2816x1856 originals) for every camera;
  train: front camera --maxpix 735000 (~1056x696, so the 2x centre crop for the tele slot still has ~Edge's front
    budget, 196608 px) and the other cameras --maxpix 98304 (Edge's side/rear budget), as two passes with their own
    member lists (--cams F | other) and done markers.
Output data/img_edge/<split>/<member path>, own byte counter.
  python src/edge_fetch.py --maxpix N [--cams F|other] --shard i --nshards n [plan_fetch args]
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan_fetch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    argv = sys.argv[1:]; maxpix = 1408 * 928; cams = ""
    if "--maxpix" in argv: i = argv.index("--maxpix"); maxpix = int(argv[i + 1]); del argv[i:i + 2]
    if "--cams" in argv: i = argv.index("--cams"); cams = argv[i + 1]; del argv[i:i + 2]
    tag = ""                                   # --tag r2: members from data/edge_members_r2.json, markers _done_<cams>_r2.json
    if "--tag" in argv: i = argv.index("--tag"); tag = argv[i + 1]; del argv[i:i + 2]
    plan_fetch.MAXPIX = maxpix
    members = f"data/edge_members_{tag}.json" if tag else "data/edge_members.json"; done = f"_done_{tag}.json" if tag else "_done.json"
    sfx = f"_{tag}" if tag else ""
    if cams:                                   # camera-filtered member list for the two train passes
        M = json.load(open(os.path.join(HERE, members))); keep = (lambda m: "/CAM_M_F/" in m) if cams == "F" else (lambda m: "/CAM_M_F/" not in m)
        out = {s: [{**r, "members": [m for m in r["members"] if keep(m)]} for r in rows] for s, rows in M.items()}
        members = f"data/edge_members_{cams}{sfx}.json"; done = f"_done_{cams}{sfx}.json"
        tmp = os.path.join(HERE, members + f".{os.getpid()}.tmp"); json.dump(out, open(tmp, "w")); os.replace(tmp, os.path.join(HERE, members))
    sys.argv = [sys.argv[0], "--members", members, "--out", "data/img_edge", "--mode", "images", "--cap-gb", "400",
                "--done-name", done] + argv
    plan_fetch.main()
