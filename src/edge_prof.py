"""CPU profile of the Edge training input path (no GPU): per-row time of build() (image decode), prepare_model_inputs
(processor + chat template) and fuse_traj_tokens (trajectory tokens), plus a cProfile of the slowest part.
  python -I src/edge_prof.py --model DIR --rows data/edge_inputs_validation.json --split validation --n 12
"""
import argparse, cProfile, json, os, pstats, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "external", "alpamayo2-edge", "src")); sys.path.insert(0, os.path.join(HERE, "src"))
from edge_infer import build
import edge_train as ET


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--rows", required=True)
    ap.add_argument("--split", default="validation"); ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    import torch, hydra.utils as hyu
    from alpamayo2_edge.checkpoint import load_model
    from alpamayo2_edge.helper import prepare_model_inputs
    print("torch threads", torch.get_num_threads(), "cpus", len(os.sched_getaffinity(0)), flush=True)
    model, _ = load_model(a.model, device="cpu", dtype="bfloat16")
    model.future_traj_tokenizer = hyu.instantiate(model.config.future_traj_tokenizer_cfg, load_weights=False)
    ET.cache_processor()
    rows = json.load(open(a.rows)); keys = sorted(k for k in rows if rows[k].get("fut64"))[:a.n]
    img_root = os.path.join(HERE, "data/img_edge"); T = []
    for k in keys:
        v = rows[k]; t0 = time.time(); d = build(v, a.split, img_root); t1 = time.time()
        p = prepare_model_inputs(d, model.config, model.tokenizer); t2 = time.time()
        ET.encode(model, v, a.split, img_root, "cpu", prep=p); t3 = time.time()
        T.append((t1 - t0, t2 - t1, t3 - t2))
        print(f"  {len(v['images'])} imgs, {p['tokenized_data']['input_ids'].shape[1]} tokens: build {t1 - t0:.2f}s "
              f"prepare {t2 - t1:.2f}s fuse {t3 - t2:.2f}s", flush=True)
    import numpy as np; m = np.mean(T[2:], 0); print(f"MEAN (rows 3+): build {m[0]:.2f} prepare {m[1]:.2f} fuse {m[2]:.2f}")
    worst = ["build", "prepare", "fuse"][int(np.argmax(m))]; v = rows[keys[-1]]
    pr = cProfile.Profile(); pr.enable()
    d = build(v, a.split, img_root)
    if worst != "build": p = prepare_model_inputs(d, model.config, model.tokenizer)
    if worst == "fuse": ET.encode(model, v, a.split, img_root, "cpu", prep=p)
    pr.disable(); print("cProfile through", worst); pstats.Stats(pr).sort_stats("cumulative").print_stats(25)


if __name__ == "__main__":
    main()
