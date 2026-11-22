"""AutoVLA (UCLA; checkpoint AutoVLA_PDMS_89.ckpt, Qwen2.5-VL-3B + action codebook; Academic Software License) on
nuReasoning key frames, using its own prompt builder (ext/AutoVLA/models/autovla.py AutoVLA.get_prompt / predict).

Inputs as its NAVSIM agent (ext/AutoVLA/navsim/navsim/agents/vla_agent.py): front / front-left / front-right cameras at
t-1.5, -1.0, -0.5, 0 s (frames 85, 90, 95, 100; 16:9 centre crops of our 1408x928 frames, data/img_qd169 = NAVSIM aspect),
current velocity / acceleration (it uses their magnitudes), command text 'TURN LEFT' / 'KEEP FORWARD' / 'TURN RIGHT' /
'UNKNOWN' (lower-cased by the prompt). CoT mode as trained (planner reasons, states a lateral + longitudinal action,
then emits 10 action tokens = 10 poses at 0.5 s = 5 s, native horizon). Near-greedy decoding (temperature 0.01, its eval
config). Ego inputs from data/wa_<split>_meta.npz (packed by src/wa_eval_pack.py from the cache).
  python src/avla_run.py --split validation --shard 0 --nshards 4 [--limit N]
  -> runs/avla/<split>_s<shard>.jsonl {clip, traj10 [[x, y, h] x 10] ego frame, text}
"""
import argparse, gzip, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMD_TXT = ["TURN LEFT", "KEEP FORWARD", "TURN RIGHT", "UNKNOWN"]
CAMS = ["front_left", "front", "front_right", "back"]                     # order inside wa_<split>_paths


def main():
    import torch, yaml
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="validation"); ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--img-root", default=""); ap.add_argument("--ckpt", default="weights/autovla/AutoVLA_PDMS_89.ckpt")
    ap.add_argument("--temperature", type=float, default=0.01); ap.add_argument("--tag", default="zs")
    a = ap.parse_args(); img_root = os.path.join(HERE, a.img_root or f"data/img_qd169/{a.split}")
    sys.path.insert(0, os.path.join(HERE, "ext/AutoVLA")); os.chdir(os.path.join(HERE, "ext/AutoVLA"))   # codebook path is relative
    import types                                                          # its NAVSIM PDM reward module (hydra / nuplan-devkit) is
    stub = types.ModuleType("models.utils.score")                         # only needed for its own RL; not imported for inference
    for n in ("PDM_Reward", "TrajectorySampling", "Trajectory"): setattr(stub, n, type(n, (), {}))
    sys.modules["models.utils.score"] = stub
    from models.autovla import AutoVLA
    cfg = yaml.safe_load(open("config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
    cfg["model"]["pretrained_model_path"] = os.path.join(HERE, "weights/autovla/Qwen2.5-VL-3B-Instruct")
    cfg["inference"]["sample"] = {"max_length": 2048, "temperature": a.temperature, "top_k": 0, "top_p": 1.0}
    m = AutoVLA(cfg, inference=True, device="cuda")
    sd = torch.load(os.path.join(HERE, a.ckpt), map_location="cpu", weights_only=False)["state_dict"]
    mi, un = m.load_state_dict({k.replace("autovla.", "").replace("drivevla.", ""): v for k, v in sd.items()}, strict=False)
    print(f"loaded {a.ckpt}: missing {len(mi)} unexpected {len(un)}", flush=True); assert not un, un[:5]
    m.eval(); del sd
    z = np.load(os.path.join(HERE, f"data/wa_{a.split}_meta.npz"))
    with gzip.open(os.path.join(HERE, f"data/wa_{a.split}_paths.json.gz"), "rt") as f: P = json.load(f)
    clips = sorted(map(str, z["clip"]))[a.shard::a.nshards][:a.limit or None]; idx = {str(c): i for i, c in enumerate(z["clip"])}
    out = os.path.join(HERE, f"runs/avla/{a.tag}_{a.split}_s{a.shard}.jsonl"); os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {json.loads(l)["clip"] for l in open(out)} if os.path.isfile(out) else set()
    todo = [c for c in clips if c not in done]; t0 = time.time()
    print(f"[avla] {a.split} shard {a.shard}/{a.nshards}: {len(done)} done, {len(todo)} to go", flush=True)
    with open(out, "a") as fo:
        for n, c in enumerate(todo):
            i = idx[c]; st = z["status"][i]
            imgs = {f"{v}_camera": [os.path.join(c, P[c][str(100 + o)][CAMS.index(v)]) for o in (-15, -10, -5, 0)] for v in ("front", "front_left", "front_right")}
            feat = {"images": imgs, "sensor_data_path": img_root, "vehicle_velocity": [float(st[4]), float(st[5])],
                    "vehicle_acceleration": [float(st[6]), float(st[7])], "driving_command": CMD_TXT[int(np.argmax(st[:4]))]}
            with torch.no_grad():
                traj, text = m.predict(feat)
            fo.write(json.dumps({"clip": c, "traj10": np.asarray(traj, float).round(4).tolist(), "text": text}) + "\n"); fo.flush()
            if n < 2 or (n + 1) % 50 == 0: print(f"  [{n + 1}/{len(todo)}] {(time.time() - t0) / (n + 1):.2f}s/clip | {text[-160:]!r}", flush=True)
    print("AVLA_DONE", flush=True)


if __name__ == "__main__":
    main()
