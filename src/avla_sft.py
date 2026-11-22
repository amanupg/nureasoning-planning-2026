"""AutoVLA fine-tuning probe on nuReasoning (Aman 2026-10-23: stop the thread unless validation reaches ~0.62).

Start: AutoVLA_PDMS_89.ckpt (zero-shot 0.383; over-brakes 4.7 m at 5 s). Train rows: every frame-100 row of the train split
(dev logs excluded) + random other train rows, --n in total. Per row, exactly its inference prompt: models.autovla.AutoVLA
.get_prompt builds the messages (captured, not rewritten): front / front-left / front-right at r-15, -10, -5, 0 from
data/img_wa_full/train (16:9, 256x512; the processor resizes to its 109,760-pixel budget as for NAVSIM frames), velocity /
acceleration, command text. Target: its no-CoT answer format ("This is a straightforward scenario ... The final output
action is: <action_i> x 10") with the recorded 5 s future (fut10) tokenised by src/avla_tok.py (round trip 2.5 cm).
Loss: next-token cross-entropy on the assistant turn only (its SFT loss). LoRA on the language model's attention + MLP
(vision frozen), bf16, gradient checkpointing, batch 1 x --accum. Saves merged weights as runs/avla_<tag>.ckpt
{"state_dict"} for src/avla_run.py --ckpt.
  python src/avla_sft.py --tag p1 --n 12000
"""
import argparse, gzip, json, math, os, random, sys, time, types
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
CMD_TXT = ["TURN LEFT", "KEEP FORWARD", "TURN RIGHT", "UNKNOWN"]
CAMS = ["front_left", "front", "front_right", "back"]
log_of = lambda c: "_".join(c.split("_")[:2])


def main():
    import torch, yaml
    from avla_tok import tokenize
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="p1"); ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--accum", type=int, default=8); ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--max-hours", type=float, default=10.5); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    from wajepa_ft import rows_split
    zf = np.load(os.path.join(HERE, "data/wa_train_meta.npz")); Z = {k: zf[k] for k in ("clip", "r", "status", "fut10")}
    key = {(str(c), int(r)): i for i, (c, r) in enumerate(zip(Z["clip"], Z["r"]))}
    with gzip.open(os.path.join(HERE, "data/wa_train_paths.json.gz"), "rt") as f: P = json.load(f)
    tr, _ = rows_split(0); k100 = [x for x in tr if x[1] == 100]; rest = [x for x in tr if x[1] != 100]
    rows = k100 + [rest[i] for i in np.random.RandomState(a.seed).permutation(len(rest))[:max(0, a.n - len(k100))]]
    rows = [rows[i] for i in np.random.RandomState(a.seed).permutation(len(rows))[:a.n]]          # cap at --n (smoke tests)
    random.shuffle(rows); print(f"train rows {len(rows)} ({len(k100)} at frame 100)", flush=True)
    img_root = os.path.join(HERE, "data/img_wa_full/train")
    sys.path.insert(0, os.path.join(HERE, "ext/AutoVLA")); os.chdir(os.path.join(HERE, "ext/AutoVLA"))
    stub = types.ModuleType("models.utils.score")
    for n in ("PDM_Reward", "TrajectorySampling", "Trajectory"): setattr(stub, n, type(n, (), {}))
    sys.modules["models.utils.score"] = stub
    import models.autovla as MA
    cfg = yaml.safe_load(open("config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml"))
    cfg["model"]["pretrained_model_path"] = os.path.join(HERE, "weights/autovla/Qwen2.5-VL-3B-Instruct")
    m = MA.AutoVLA(cfg, inference=True, device="cuda")
    sd = torch.load(os.path.join(HERE, "weights/autovla/AutoVLA_PDMS_89.ckpt"), map_location="cpu", weights_only=False)["state_dict"]
    mi, un = m.load_state_dict({k.replace("autovla.", "").replace("drivevla.", ""): v for k, v in sd.items()}, strict=False); del sd
    assert not un, un[:3]; print(f"start weights loaded (missing {len(mi)})", flush=True)
    from peft import LoraConfig, get_peft_model
    m.vlm.gradient_checkpointing_enable(); m.vlm.enable_input_require_grads(); m.vlm.config.use_cache = False
    m.vlm = get_peft_model(m.vlm, LoraConfig(r=a.lora_r, lora_alpha=a.lora_r, lora_dropout=0.05, bias="none",
                                             target_modules=r".*model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"))
    m.vlm.print_trainable_parameters()
    cap = {}; orig = MA.process_vision_info
    def grab(messages):                                                   # capture the exact messages get_prompt builds
        cap["m"] = messages; return orig(messages)
    MA.process_vision_info = grab
    proc = m.processor; tok = m.action_tokenizer

    def example(c, r):
        j = key[(c, r)]; st = Z["status"][j]
        imgs = {f"{v}_camera": [os.path.join(c, P[c][str(r + o)][CAMS.index(v)]) for o in (-15, -10, -5, 0)] for v in ("front", "front_left", "front_right")}
        m.get_prompt({"images": imgs, "sensor_data_path": img_root, "vehicle_velocity": [float(st[4]), float(st[5])],
                      "vehicle_acceleration": [float(st[6]), float(st[7])], "driving_command": CMD_TXT[int(np.argmax(st[:4]))]})
        ans = ("<think>\nThis is a straightforward scenario, and a direct decision can be made.\n</think>\n<answer>\n"
               "The final output action is: " + tok(tokenize(np.asarray(Z["fut10"][j], float))) + "\n</answer>")
        msgs = cap["m"] + [{"role": "assistant", "content": [{"type": "text", "text": ans}]}]
        im, vid = orig(msgs)
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, add_vision_id=True)
        x = proc(text=[text], images=im, videos=vid, padding=True, return_tensors="pt")
        ids = x["input_ids"][0]; lab = ids.clone(); s = [i for i in range(len(ids) - 1) if ids[i] == 151644 and ids[i + 1] == 77091][-1]
        lab[:s + 3] = -100; x["labels"] = lab[None]
        return {k: v.cuda() for k, v in x.items() if isinstance(v, torch.Tensor)}

    params = [p for p in m.vlm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0); steps = len(rows) // a.accum; t0 = time.time(); step = 0; run = []
    m.vlm.train()
    for i, (c, r) in enumerate(rows):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m.vlm(**example(c, r)).loss
        (loss / a.accum).backward(); run.append(loss.item())
        if (i + 1) % a.accum == 0:
            f = min(1.0, (step + 1) / max(1, int(0.05 * steps))) * 0.5 * (1 + math.cos(math.pi * step / max(1, steps)))
            for g in opt.param_groups: g["lr"] = a.lr * f
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 25 == 0 or step <= 2:
                print(f"  step {step}/{steps} loss {np.mean(run[-a.accum * 25:]):.4f} {(time.time() - t0) / (i + 1):.2f}s/sample peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        if time.time() - t0 > 3600 * a.max_hours: print(f"  time limit at sample {i + 1}", flush=True); break
    merged = m.vlm.merge_and_unload(); m.vlm = merged
    sd = {k: v.to(torch.bfloat16).cpu() for k, v in m.state_dict().items()}
    out = os.path.join(HERE, f"runs/avla_{a.tag}.ckpt"); torch.save({"state_dict": sd, "args": vars(a)}, out + ".tmp"); os.replace(out + ".tmp", out)
    print(f"saved {out} ({len(sd)} tensors)", flush=True); print("SFT_DONE", flush=True)


if __name__ == "__main__":
    main()
