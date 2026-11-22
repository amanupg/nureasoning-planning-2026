"""Keep only the "Driving" block of each fetched train reasoning file (data/reasoning_extra, ~650 KB each, 10,349 files,
~6.7 GB) in data/reasoning_driving/<split>/<clip>/reasoning/<ts>.json (a few hundred bytes each), so the row builder
can run on the Mac without copying the full files. Writes new files only; skips outputs that already exist.
  python src/reasoning_driving.py [workers]
"""
import glob, json, os, sys
from multiprocessing import Pool
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "data/reasoning_extra"); DST = os.path.join(HERE, "data/reasoning_driving")


def one(p):
    out = os.path.join(DST, os.path.relpath(p, SRC))
    try: return bool(json.load(open(out)).get("Driving"))          # done earlier (a cancelled run may leave a cut file)
    except Exception: pass
    d = json.load(open(p)).get("Driving"); os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"Driving": d}, open(out + ".tmp", "w")); os.replace(out + ".tmp", out)
    return bool(d)


if __name__ == "__main__":
    ps = glob.glob(os.path.join(SRC, "*/*/reasoning/*.json"))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool: r = pool.map(one, ps, chunksize=16)
    print(f"files {len(r)}, with a non-empty Driving block {sum(r)}")
