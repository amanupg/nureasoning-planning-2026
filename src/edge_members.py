"""Fetch list for Alpamayo 2 Edge zero-shot planning inputs (src/edge_fetch.py).

Edge's driving profile (NVlabs/alpamayo2-edge @ 02f55a7, sample.py / load_physical_aiavdataset.py): six camera slots,
front wide (1) and front tele (6) at -2, -1, -0.5, -0.1, 0 s; cross-left (0), cross-right (2), rear-left (3),
rear-right (5) at -1, -0.1, 0 s. nuReasoning has eight 53-degree cameras (computed from the calibrations:
CAM_M_F 0 deg, L0 +56, R0 -54, L1 +110, R1 -109, L2 +149, R2 -147, B 180). Mapping:
  1 front wide  <- CAM_M_F            (narrower than 120 deg, the only forward camera)
  6 front tele  <- CAM_M_F centre crop (2x, ~28 deg; same as the reasoning session's VQA mapping)
  0 cross-left  <- CAM_M_L0 (+56 deg)  2 cross-right <- CAM_M_R0 (-54 deg)
  3 rear-left   <- CAM_M_L2 (+149 deg) 5 rear-right  <- CAM_M_R2 (-147 deg)
At 10 Hz with key frame K: front frames K-20, K-10, K-5, K-1, K; others K-10, K-1, K.
  python src/edge_members.py [splits]   -> data/edge_members.json
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan_rows import CACHE, HERE
from plan_members import tars

KEY = 100
FRONT_OFFS = (-20, -10, -5, -1, 0)
OTHER_OFFS = (-10, -1, 0)
SLOTS = (("front", 1, FRONT_OFFS), ("front_left", 0, OTHER_OFFS), ("front_right", 2, OTHER_OFFS),
         ("back_left", 3, OTHER_OFFS), ("back_right", 5, OTHER_OFFS))      # tele (6) is cropped from "front"


def clip_members(split, clip, key=KEY):
    fr = json.load(open(os.path.join(CACHE, split, clip, "metadata.json")))["frames"]
    out = []
    for cam, _, offs in SLOTS:
        for o in offs:
            out.append(f"{clip}/{fr[key + o]['sensors']['cameras'][cam]}")
    return sorted(set(out))


def from_inputs(name):
    """{clip: sorted unique image members} of data/edge_inputs_<name>.json (src/edge_inputs.py)."""
    d = json.load(open(os.path.join(HERE, "data", f"edge_inputs_{name}.json")))
    return {c: sorted({i["m"] for i in v["images"]}) for c, v in d.items()}


def main():
    """Members follow the inputs files (single source of truth):
    validation <- edge_inputs_validation.json (no dup fix; 415 clips were fetched from this list before the pause);
    test       <- edge_inputs_test.json (dup fix on by default);
    plus data/edge_members_valdup.json: only the EXTRA images the dup fix needs on the 31 val dup clips (fetched as
    its own pass with its own done marker, since those clips may already carry a done marker)."""
    T = tars(); p = os.path.join(HERE, "data", "edge_members.json")
    old = json.load(open(p)) if os.path.isfile(p) else {}
    row = lambda split, c, m: {"clip": c, "tar": T[(split, c)][0], "size": T[(split, c)][1], "members": m, "reasoning": []}
    val, valfix, test = from_inputs("validation"), from_inputs("validation_dupfix"), from_inputs("test")
    out = {**old, "validation": [row("validation", c, val[c]) for c in sorted(val)],
           "test": [row("test", c, test[c]) for c in sorted(test)]}
    extra = {c: sorted(set(valfix[c]) - set(val[c])) for c in valfix}
    extra = {c: m for c, m in extra.items() if m}
    json.dump(out, open(p, "w"))
    json.dump({"validation": [row("validation", c, extra[c]) for c in sorted(extra)]},
              open(os.path.join(HERE, "data", "edge_members_valdup.json"), "w"))
    for k in ("validation", "test", "train"):
        if k in out: print(k, len(out[k]), "clips", sum(len(r["members"]) for r in out[k]), "images")
    print("valdup extra:", len(extra), "clips", sum(map(len, extra.values())), "images")


if __name__ == "__main__":
    main()
