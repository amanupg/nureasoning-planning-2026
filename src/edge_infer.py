"""Alpamayo 2 Edge zero-shot planning on nuReasoning clips (runs on a GPU node with the Edge env).

Inputs: data/edge_inputs_<split>.json (src/edge_inputs.py) + data/img_edge/<split>/<member> (src/edge_fetch.py,
stored at 1408x928). Per clip: 22 frames (slot 6 = 2x centre crop of the front frame, resized back to 1408x928 so all
frames share one size; the checkpoint processor then resizes each slot to its own pixel budget), 20 ego poses
(z = 0, yaw-only rotations), nav text. The repo's own prepare_model_inputs + sample_trajectories_from_data with its
CLI defaults (T 0.6, top_p 0.98, 256 reasoning tokens, 10 flow steps) and K candidates (each its own reasoning).
Output per shard: <out>_<split>_shard<i>of<n>.npz with names, xy [N,K,64,2] and yaw [N,K,64] in the ego frame of the
key frame (0.1..6.4 s), plus a json of the reasoning texts. Resumable per shard (partial file every 25 clips).
  python -I src/edge_infer.py --model DIR --split validation --out runs/edge_zs --shard i/n --K 6 [--limit N]
"""
import argparse, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "external", "alpamayo2-edge", "src"))


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def build(v, split, img_root):
    import torch
    from PIL import Image
    from alpamayo2_edge.sample import CAMERA_PIXEL_BOUNDS
    frames, cams, ts = [], [], []
    size = None
    for im in v["images"]:
        p = os.path.join(img_root, split, im["m"]); x = Image.open(p).convert("RGB")
        if size is None: size = x.size
        if im.get("crop2"):
            w, h = x.size; x = x.crop((w // 4, h // 4, w // 4 + w // 2, h // 4 + h // 2)).resize((w, h), Image.BICUBIC)
        elif x.size != size:
            x = x.resize(size, Image.BICUBIC)
        frames.append(torch.from_numpy(np.asarray(x).copy()).permute(2, 0, 1).unsqueeze(0))
        cams.append(im["cam"]); ts.append(im["ts"])
    h = np.asarray(v["hist"], float)
    xyz = torch.tensor(np.column_stack([h[:, :2], np.zeros(len(h))]), dtype=torch.float32).unsqueeze(0)
    rot = torch.tensor([rotz(a) for a in h[:, 2]], dtype=torch.float32).unsqueeze(0)
    return {"image_frames": torch.stack(frames).contiguous(), "camera_indices": torch.tensor(cams, dtype=torch.int64),
            "absolute_timestamps": torch.tensor(ts, dtype=torch.int64), "ego_history_xyz": xyz, "ego_history_rot": rot,
            "nav_text": [v["nav"]], "camera_pixel_bounds": dict(CAMERA_PIXEL_BOUNDS), "min_pixels": 163840,
            "max_pixels": 196608, "history_num_steps": 20}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--split", default="validation")
    ap.add_argument("--out", default="runs/edge_zs"); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--K", type=int, default=6); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--img-root", default=os.path.join(HERE, "data/img_edge"))
    ap.add_argument("--nav", default="", help="override nav text for every clip (ablation)")
    ap.add_argument("--inputs", default="", help="inputs file suffix: '' or _dupfix (val), test is dup-fixed by default")
    ap.add_argument("--temperature", type=float, default=0.6); ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--steps", type=int, default=10)   # diffusion inference steps (trajectory head); temperature only affects text
    ap.add_argument("--no-text", action="store_true", help="force <|traj_future_start|> as the first generated token: "
                    "no reasoning/decision text, trajectory straight from the prompt (as e2a, which never emits text)")
    a = ap.parse_args()
    import torch
    from alpamayo2_edge.checkpoint import load_model
    from alpamayo2_edge.helper import prepare_model_inputs, to_device
    i, n = map(int, a.shard.split("/"))
    D = json.load(open(os.path.join(HERE, "data", f"edge_inputs_{a.split}{a.inputs}.json")))
    clips = sorted(D)[i::n]
    if a.limit: clips = clips[:a.limit]
    out = os.path.join(HERE, f"{a.out}{a.inputs}_{a.split}_shard{i}of{n}")
    names, XY, YAW, TXT = [], [], [], {}
    for prev in (out, out + ".partial"):                     # resume from the last full or partial save
        if os.path.isfile(prev + ".npz"):
            z = np.load(prev + ".npz"); names = list(map(str, z["names"])); XY = list(z["xy"]); YAW = list(z["yaw"])
            TXT = json.load(open(prev + ".json")); break
    done = set(names)
    model, _ = load_model(a.model, device="cuda", dtype="bfloat16", attn_implementation="sdpa")
    if a.no_text:   # wrap the shared-prefill generate: a processor that forces the anchor on the first generated step only
        from transformers import LogitsProcessor
        anchor = model.config.traj_ids["future_start"]

        class ForceAnchorFirst(LogitsProcessor):
            def __init__(self): self.calls = 0
            def __call__(self, input_ids, scores):
                self.calls += 1
                if self.calls == 1:
                    scores = torch.full_like(scores, float("-inf")); scores[:, anchor] = 0.0
                return scores
        gen0 = model._generate_with_shared_prefill

        def gen_no_text(*args, logits_processor=None, **kw):
            logits_processor.append(ForceAnchorFirst()); return gen0(*args, logits_processor=logits_processor, **kw)
        model._generate_with_shared_prefill = gen_no_text
    print(f"[edge] {a.split} shard {a.shard}: {len(clips)} clips ({len(done)} done) K={a.K} T={a.temperature} steps={a.steps} seed={a.seed} no_text={a.no_text} "
          f"gpu {torch.cuda.get_device_name()}", flush=True)
    t0 = time.time(); k = 0; skipped = 0

    def save(path):
        np.savez(path + ".npz", names=np.array(names), xy=np.array(XY, np.float32), yaw=np.array(YAW, np.float32),
                 K=a.K, temperature=a.temperature)
        json.dump(TXT, open(path + ".json", "w"))

    for c in clips:
        if c in done: continue
        need = ["_done.json"] + (["_done_dup.json"] if D[c].get("dupfix") and a.split == "validation" else [])
        if not all(os.path.isfile(os.path.join(a.img_root, a.split, c, f)) for f in need):   # not fetched yet: rerun later
            skipped += 1; continue
        v = dict(D[c]); v["nav"] = a.nav or v["nav"]
        data = build(v, a.split, a.img_root)
        prep = to_device(prepare_model_inputs(data, model.config, model.tokenizer), device="cuda")
        torch.manual_seed(a.seed); gen = torch.Generator(device="cuda").manual_seed(a.seed)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            xyz, rot, _, extra = model.sample_trajectories_from_data(
                prep, top_p=0.98, top_k=None, temperature=a.temperature, num_traj_sets=1, num_traj_samples=a.K,
                max_generation_length=256, diffusion_kwargs={"inference_step": a.steps, "generator": gen}, return_extra=True)
        xyz = xyz[0, 0].float().cpu().numpy(); rot = rot[0, 0].float().cpu().numpy()
        names.append(c); XY.append(xyz[..., :2]); YAW.append(np.arctan2(rot[..., 1, 0], rot[..., 0, 0]))
        # The repo splits generated text at the first "Longitudinal:/Lateral:/Lane:" into cot (before) and meta_action
        # (from there on). Our decision text starts with "Longitudinal: ...", so its extra["cot"] is always "" and the
        # text sits in meta_action. "cot" here is the full generated text (cot + meta_action), as edge_dec_export expects.
        cs = [str(s) for s in np.asarray(extra.get("cot", [])).reshape(-1)]
        ms = [str(s) for s in np.asarray(extra.get("meta_action", [""] * len(cs))).reshape(-1)]
        TXT[c] = {"nav": v["nav"], "cot": [f"{x} {y}".strip() for x, y in zip(cs, ms)][:a.K]}
        k += 1
        if k % 25 == 0:
            save(out + ".partial")
            print(f"  {len(names)}/{len(clips)} {(time.time() - t0) / k:.2f} s/clip peak "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    save(out)
    print(f"EDGE_INFER_DONE {len(names)} clips ({skipped} skipped, images not fetched) {(time.time() - t0) / max(k, 1):.2f} s/clip", flush=True)


if __name__ == "__main__":
    main()
