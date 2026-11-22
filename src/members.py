"""Which camera frames to fetch for the JEPA planners (copied-in ingest, see img_ingest2.py).

WA-JEPA: cam_l0/f0/r0/b0 x 4 frames 0.5 s apart; Drive-JEPA (perception-free): l0/f0/r0 stitched
at the current frame and 0.5 s earlier. Union: CAM_M_L0, CAM_M_F, CAM_M_R0, CAM_M_B at key-frame
offsets -15, -10, -5, 0 (10 Hz). Tar path/size come from the nuReasoning session's cache
(_provenance.json, read-only). -> data/members.json {split: [{clip, tar, size, members}]}
"""
import json, os, sys
CACHE = "/path/to/workspace/nureasoning/cache"
CAMS = ["front_left", "front", "front_right", "back"]
OFFS = [-15, -10, -5, 0]
KEY = 100


def rows(split):
    out = []
    for c in sorted(os.listdir(f"{CACHE}/{split}")):
        d = f"{CACHE}/{split}/{c}"
        if not os.path.isfile(f"{d}/metadata.json"): continue
        pv = json.load(open(f"{d}/_provenance.json")); fr = json.load(open(f"{d}/metadata.json"))["frames"]
        idx = {f["frame_index"]: f for f in fr}
        mem = [f"{c}/{idx[KEY + o]['sensors']['cameras'][cam]}" for o in OFFS for cam in CAMS]
        out.append({"clip": c, "tar": pv["source_path"], "size": pv["source_size"], "members": mem})
    return out


if __name__ == "__main__":
    M = {s: rows(s) for s in (sys.argv[1:] or ["validation", "test"])}
    os.makedirs("data", exist_ok=True); json.dump(M, open("data/members.json", "w"))
    for s, r in M.items(): print(s, len(r), "clips x", len(r[0]["members"]), "images; e.g.", r[0]["members"][:2])
