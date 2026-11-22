"""Alpamayo 2 Edge planning fine-tune with its native heads (head (a) of the plan): one loss on the VLM, one on the
expert, as the checkpoint config says it was trained (cotrain_expert_vlm true, cotrain_detach_expert_kv true,
loss_weights future_traj 1.0):
  1. VLM: next-token cross-entropy on the discrete future trajectory (the repo's DiscreteTrajectoryTokenizer: 64 steps
     x (accel, curvature), 128 tokens) after an EMPTY reasoning: prompt (ends "<think></think>") +
     <|traj_future_start|> + 128 future tokens + <|traj_future_end|> + <|im_end|>. The human future is the target.
  2. Expert: flow-matching MSE (the repo's FlowMatching.construct_training_data / compute_loss_from_pred) on the
     action-space target of the same future, the expert attending to the VLM key/value cache of the prefix up to and
     including <|traj_future_start|> (detached, so the expert loss does not reach the VLM, as cotrain_detach_expert_kv).
     The cache comes from a separate no-grad prefill (gradient checkpointing disables HF caching in training).
Inference afterwards: src/edge_infer.py (its reasoning would be empty after this fine-tune; --K samples).
Rows: data/edge_inputs_<split>.json style dicts with images / hist / nav / fut64 (src/edge_inputs.py; train rows
from src/edge_rows.py). Optimizer: AdamW on fp32 master weights, bf16 autocast, grad clip 1.0, cosine schedule.
  python -I src/edge_train.py --model DIR --rows data/edge_inputs_validation.json --split validation --smoke 8
"""
import argparse, hashlib, json, math, os, random, sys, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "external", "alpamayo2-edge", "src"))
sys.path.insert(0, os.path.join(HERE, "src"))
from edge_infer import build, rotz

N_FUT_TOK = 128


def fut_tensors(v):
    import torch
    f = np.asarray(v["fut64"], float)
    xyz = torch.tensor(np.column_stack([f[:, :2], np.zeros(len(f))]), dtype=torch.float32)[None, None]
    rot = torch.tensor([rotz(a) for a in f[:, 2]], dtype=torch.float32)[None, None]
    return xyz, rot


_PROC = {}


def cache_processor():
    """alpamayo2_edge.helper.get_processor runs AutoProcessor.from_pretrained on EVERY call (every row). Same inputs
    give the same processor, so build it once per tokenizer and reuse it (prepare_model_inputs looks the name up in
    the helper module at call time, so patching the module attribute is enough)."""
    import alpamayo2_edge.helper as H
    orig = H.get_processor
    def cached(tokenizer, model_config):
        k = id(tokenizer)
        if k not in _PROC: _PROC[k] = orig(tokenizer, model_config)
        return _PROC[k]
    H.get_processor = cached


def prep_cpu(model, v, split, img_root):
    """CPU part of encode (image decode + processor + text tokens). Thread-safe: no torch.compile'd code inside, so the
    training loop runs it in a thread pool ahead of the GPU."""
    from alpamayo2_edge.helper import prepare_model_inputs
    return prepare_model_inputs(build(v, split, img_root), model.config, model.tokenizer)


def encode(model, v, split, img_root, device, prep=None):
    """Token ids, masks and pixel tensors of one training sequence + its action target."""
    import torch
    from alpamayo2_edge.models.utils import fuse_traj_tokens
    cfg = model.config; T = cfg.traj_ids
    if prep is None: prep = prep_cpu(model, v, split, img_root)
    td = dict(prep["tokenized_data"])
    im_end = model.tokenizer.convert_tokens_to_ids("<|im_end|>")
    # --cot: the challenge's Driving "Reasoning trace" is written after the closed <think></think>, before the trajectory,
    # as base Edge does at inference (README: "The model writes its driving reasoning directly").
    txt = model.tokenizer(v["cot"], add_special_tokens=False)["input_ids"] if v.get("cot") else []
    suffix = torch.tensor([txt + [T["future_start"]] + [T["future_pad"]] * N_FUT_TOK + [T["future_end"], im_end]])
    anchor = td["input_ids"].shape[1] + len(txt)                       # index of <|traj_future_start|>
    ids = torch.cat([td["input_ids"], suffix], 1)
    fx, fr = fut_tensors(v)
    traj = {"ego_history_xyz": prep["ego_history_xyz"], "ego_history_rot": prep["ego_history_rot"],
            "ego_future_xyz": fx, "ego_future_rot": fr}
    ids = fuse_traj_tokens(model.history_traj_tokenizer, model.future_traj_tokenizer, ids, traj, T)
    n_suf = suffix.shape[1]
    td["input_ids"] = ids
    td["attention_mask"] = torch.ones_like(ids)
    if "mm_token_type_ids" in td:
        td["mm_token_type_ids"] = torch.cat([td["mm_token_type_ids"], torch.zeros((1, n_suf), dtype=td["mm_token_type_ids"].dtype)], 1)
    td = {k: (x.to(device) if hasattr(x, "to") else x) for k, x in td.items()}
    traj = {k: x.to(device) for k, x in traj.items()}
    traj["n_valid"] = int(v.get("n_valid") or 64); traj["n_text"] = len(txt)
    return td, traj, anchor, n_suf


FLOW_DRAWS = 1   # set from --flow-draws


def step_losses(model, td, traj, anchor, n_suf):
    import torch
    from alpamayo2_edge.models.expert_utils import build_expert_pos_ids_and_attn_mask
    from alpamayo2_edge.models.utils import compute_next_token_loss
    vis = {k: td[k] for k in ("pixel_values", "image_grid_thw") if k in td}
    # 1) VLM cross-entropy on the future tokens (logits only for the suffix).
    nt = traj.get("n_text", 0); ex1 = 1 if nt else 0               # with text, also learn its first token from the prompt end
    out = model.vlm(input_ids=td["input_ids"], attention_mask=td["attention_mask"],
                    mm_token_type_ids=td.get("mm_token_type_ids"), use_cache=False, logits_to_keep=n_suf + ex1, **vis)
    lab = td["input_ids"][:, -(n_suf + ex1):]
    keep = torch.ones_like(lab, dtype=torch.bool)
    nv = traj.get("n_valid", 64); fs = ex1 + nt                    # index of <|traj_future_start|> in lab
    if nv < 64: keep[:, fs + 1 + 2 * nv:fs + 1 + N_FUT_TOK] = False   # tokens of extrapolated steps (2 per step, step-major)
    ce = compute_next_token_loss(out.logits, lab, keep)
    # 2) Expert flow matching on the detached prefix cache.
    plen = anchor + 1
    was = model.vlm.training; model.vlm.eval()
    with torch.no_grad():
        pre = model.vlm.model(input_ids=td["input_ids"][:, :plen], attention_mask=td["attention_mask"][:, :plen],
                              mm_token_type_ids=td["mm_token_type_ids"][:, :plen] if "mm_token_type_ids" in td else None,
                              use_cache=True, **vis)
    model.vlm.train(was)
    cache = pre.past_key_values; rope = model.vlm.model.rope_deltas
    ex = model.expert; dims = ex.action_space.get_action_space_dims()
    with torch.autocast("cuda", enabled=False):
        x = ex.action_space.traj_to_action(traj["ego_history_xyz"][:, 0].float(), traj["ego_history_rot"][:, 0].float(),
                                           traj["ego_future_xyz"][:, 0].float(), traj["ego_future_rot"][:, 0].float())
    pos, mask = build_expert_pos_ids_and_attn_mask(offset=torch.tensor([plen], device=x.device), rope_deltas=rope,
                                                   kv_cache_seq_len=plen, n_diffusion_tokens=dims[0], b_star=1,
                                                   device=x.device, prefix_mask=td["attention_mask"][:, :plen])
    flows = []
    for _d in range(FLOW_DRAWS):   # --flow-draws: independent noise/timestep draws on the same (detached) prefix cache
      tdata = ex.diffusion.construct_training_data(x)
      emb = ex.action_in_proj(tdata["noisy_x"], tdata["timesteps"]).reshape(1, dims[0], -1)
      eo = ex.expert(inputs_embeds=emb, position_ids=pos, past_key_values=cache, attention_mask=mask, use_cache=True,
                     is_causal=False)
      if FLOW_DRAWS > 1:   # the expert call appends its own keys/values to the cache; drop them before the next draw
          if cache.get_seq_length() != plen: cache.crop(plen)
      pred = ex.action_out_proj(eo.last_hidden_state).reshape(-1, *dims)
      if nv < 64:                                                  # masked flow MSE on the real steps only
          tgt = (tdata["x"] - tdata["noise"]).float()
          flows.append(((pred.float() - tgt) ** 2)[:, :nv].mean())
      else:
          flows.append(ex.diffusion.compute_loss_from_pred(tdata, pred.float()))
    flow = flows[0] if len(flows) == 1 else torch.stack(flows).mean()
    return ce, flow


def save(model, src, out):
    """Weights + config, plus the source checkpoint's tokenizer/processor/template files, so that
    alpamayo2_edge.checkpoint.load_model(out) works (the model builds its tokenizer from its own directory)."""
    import shutil
    model.save_pretrained(out, safe_serialization=True)
    for f in ("tokenizer.json", "tokenizer_config.json", "processor_config.json", "chat_template.jinja",
              "generation_config.json"):
        if os.path.isfile(os.path.join(src, f)) and not os.path.isfile(os.path.join(out, f)):
            shutil.copy(os.path.join(src, f), os.path.join(out, f))


def label_check(model, td, traj, anchor, v):
    """Smoke-time check that the training targets are the human future in OUR frame (x forward, y left, 0.1 s steps):
    (1) the 128 label tokens decoded back through the repo's future tokenizer, (2) the flow target x (action space)
    integrated back through action_to_traj, both compared with fut64 at 1, 3, 5 s and over all valid steps. Also prints
    the sign of the lateral and heading at 5 s for both. Fails (raises) if either round trip is off by > 0.5 m at 5 s
    or flips the lateral sign of a clearly turning row."""
    import torch
    T = model.config.traj_ids; ids = td["input_ids"]
    tok = ids[:, anchor + 1:anchor + 1 + N_FUT_TOK] - T["future_id0"]
    hx, hr = traj["ego_history_xyz"][:, 0].float(), traj["ego_history_rot"][:, 0].float()
    dec = model.future_traj_tokenizer.decode(hx, hr, tok)
    dxyz = dec[0][0].float().cpu().numpy()
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        x = model.expert.action_space.traj_to_action(hx, hr, traj["ego_future_xyz"][:, 0].float(), traj["ego_future_rot"][:, 0].float())
        axyz, arot = model.expert.action_space.action_to_traj(x, hx, hr)
    axyz = axyz[0].float().cpu().numpy(); gt = np.asarray(v["fut64"], float); nv = int(v.get("n_valid") or 64)
    for name, P in (("tokens", dxyz), ("flow target", axyz)):
        e = np.linalg.norm(P[:nv, :2] - gt[:nv, :2], axis=1)
        at = {s: np.round(P[10 * s - 1, :2], 2).tolist() for s in (1, 3, 5)}
        print(f"  label check [{name}]: pred {at} | GT {({s: np.round(gt[10 * s - 1, :2], 2).tolist() for s in (1, 3, 5)})} | "
              f"err mean {e.mean():.3f} m, @5s {e[49]:.3f} m, max {e.max():.3f} m", flush=True)
        if e[49] > 0.5: raise RuntimeError(f"label check failed ({name}): {e[49]:.2f} m off at 5 s")
        if abs(gt[49, 1]) > 2.0 and np.sign(P[49, 1]) != np.sign(gt[49, 1]):
            raise RuntimeError(f"label check failed ({name}): lateral sign flipped")
    print("  LABEL_CHECK_PASS", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--rows", required=True)
    ap.add_argument("--split", default="train"); ap.add_argument("--img-root", default=os.path.join(HERE, "data/img_edge"))
    ap.add_argument("--out", default=os.path.join(HERE, "weights/edge_ft")); ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--accum", type=int, default=16); ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--flow-w", type=float, default=1.0); ap.add_argument("--ce-w", type=float, default=1.0)
    ap.add_argument("--smoke", type=int, default=0, help="N steps on ONE repeated row (lr 1e-5, no accumulation)")
    ap.add_argument("--save-every", type=int, default=500, help="optimizer updates between checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--state-every-min", type=float, default=60.0, help="minutes between resumable state saves")
    ap.add_argument("--stop-after-min", type=float, default=0.0, help="save state and exit 3 after this many minutes")
    ap.add_argument("--stop-after-updates", type=int, default=0, help="test hook: save state and exit 3 at this update")
    ap.add_argument("--max-updates", type=int, default=0, help="test hook: cap the number of optimizer updates")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1, help="CPU threads preparing rows ahead of the GPU (0 = inline)")
    ap.add_argument("--allow-partial", action="store_true", help="train on the rows whose images are fetched so far")
    ap.add_argument("--done-tag", default="", help="r2: clips in data/edge_members_r2.json also need _done_F_r2/_done_other_r2")
    ap.add_argument("--no-proc-cache", action="store_true", help="rebuild the processor per row as the repo does")
    ap.add_argument("--cot", default="", help="json {row_key: {trace: ...}}: train with that reasoning text before the trajectory")
    ap.add_argument("--cot-only", action="store_true", help="with --cot: keep only the rows that have a reasoning text")
    ap.add_argument("--freeze", default="", help="comma-separated parameter-name prefixes to freeze (fits 48 GB cards), e.g. "
                    "vlm.model.visual,vlm.model.language_model.embed_tokens")
    ap.add_argument("--lr-vis-mult", type=float, default=1.0, help="lr multiplier for vlm.model.visual")
    ap.add_argument("--lr-expert-mult", type=float, default=1.0, help="lr multiplier for expert.*")
    ap.add_argument("--no-decay-norms", action="store_true", help="no weight decay on norms, biases (ndim < 2) and embeddings")
    ap.add_argument("--min-lr-ratio", type=float, default=0.0, help="cosine decays to this fraction of the peak lr")
    ap.add_argument("--dup-cot", type=int, default=1, help="rows with a reasoning text appear this many times per epoch")
    ap.add_argument("--flow-draws", type=int, default=1, help="noise/timestep draws per row for the flow loss")
    ap.add_argument("--profile", action="store_true", help="log per-update time split: wait for slowest rank, "
                    "rank imbalance, grad all-reduce, clip+optimizer, saves (adds cuda syncs at update boundaries)")
    ap.add_argument("--io-threads", type=int, default=16); ap.add_argument("--io-ahead", type=int, default=96)
    a = ap.parse_args()
    import torch
    from alpamayo2_edge.checkpoint import load_model
    # Multi-GPU (launched by torch.distributed.run, WORLD_SIZE > 1): data parallel inside each update. Row i of the update
    # sequence goes to rank i % W, every rank backpropagates its rows with the same 1/accum scale, and one bucketed
    # all-reduce (sum) per update gives every rank the gradient of the full accum-row batch, as on one GPU. AdamW state is
    # sharded across ranks (ZeroRedundancyOptimizer, ZeRO-1), so the full model fits 48 GB cards without freezing.
    W = int(os.environ.get("WORLD_SIZE", "1")); RK = int(os.environ.get("RANK", "0"))
    if W > 1:
        import torch.distributed as dist
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"])); import datetime; dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
        if a.smoke: raise RuntimeError("--smoke is single-GPU")
        if a.accum % W: raise RuntimeError(f"--accum {a.accum} must be a multiple of the {W} ranks")
        if RK: sys.stdout = open(os.devnull, "w")   # rank 0 logs; errors still reach stderr
    torch.manual_seed(a.seed); random.seed(a.seed)
    model, _ = load_model(a.model, device="cuda", dtype="float32", attn_implementation="sdpa")   # fp32 master weights
    # The future-trajectory tokenizer is a plain object (not a registered module), so from_pretrained builds its
    # action-space buffers under the meta-device init and never fills them; inference never calls it, training does.
    # Rebuild it from the checkpoint config outside that context, then refuse to train if any tensor is still meta.
    import hydra.utils as hyu
    model.future_traj_tokenizer = hyu.instantiate(model.config.future_traj_tokenizer_cfg, load_weights=False)
    meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    for nm in ("history_traj_tokenizer", "future_traj_tokenizer"):
        sp_ = getattr(getattr(model, nm), "action_space", None)
        if sp_ is not None: meta += [f"{nm}.action_space.{n}" for n, t in sp_.named_buffers() if t.is_meta]
    if meta: raise RuntimeError(f"meta tensors after load: {meta[:10]}")
    fa = model.future_traj_tokenizer.action_space; ea = model.expert.action_space
    print(f"  future tokenizer accel_mean {fa.accel_mean.item():.5f} (expert {ea.accel_mean.item():.5f}), "
          f"curvature_std {fa.curvature_std.item():.5f} (expert {ea.curvature_std.item():.5f})", flush=True)
    if W > 1:
        with torch.no_grad():
            for t_ in list(model.parameters()) + list(model.buffers()):
                if t_.is_cuda: dist.broadcast(t_.data, 0)
            for nm in ("history_traj_tokenizer", "future_traj_tokenizer"):
                sp_ = getattr(getattr(model, nm), "action_space", None)
                if sp_ is not None:
                    for t_ in sp_.buffers(): dist.broadcast(t_.data, 0) if t_.is_cuda else None
        torch.manual_seed(a.seed * 1000 + RK)
    if not a.no_proc_cache: cache_processor()
    model.train(); model.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    rows = json.load(open(a.rows)); keys = sorted(k for k in rows if rows[k].get("fut64"))
    if a.cot:
        C = json.load(open(a.cot))
        for k in keys:
            if k in C: rows[k]["cot"] = C[k].get("text") or C[k]["trace"]
        if a.cot_only: keys = [k for k in keys if rows[k].get("cot")]
        print(f"  --cot: {sum(1 for k in keys if rows[k].get('cot'))} of {len(keys)} rows carry a reasoning text", flush=True)
    n_all = len(keys)
    def fetched(clip, draw=""):  # one-pass fetch (val/test) writes _done.json; the two-pass train fetch _done_F + _done_other
        d = os.path.join(a.img_root, a.split, clip)
        ok = os.path.isfile(os.path.join(d, "_done.json")) or all(
            os.path.isfile(os.path.join(d, f"_done_{c}.json")) for c in ("F", "other"))
        if ok and draw:   # merged rows (tools/edge_rows_merge.py): a row of a later frame draw needs that draw's markers
            ok = all(os.path.isfile(os.path.join(d, f"_done_{c}_{draw}.json")) for c in ("F", "other"))
        elif ok and a.done_tag and clip in tagged:   # a second frame draw (--done-tag r2) needs its own markers too
            ok = all(os.path.isfile(os.path.join(d, f"_done_{c}_{a.done_tag}.json")) for c in ("F", "other"))
        return ok
    tagged = ({r["clip"] for r in json.load(open(os.path.join(HERE, f"data/edge_members_{a.done_tag}.json")))[a.split]}
              if a.done_tag else set())
    cd_ = lambda k: (k.split("@")[0], rows[k].get("draw", ""))
    ok_clip = {x: fetched(*x) for x in {cd_(k) for k in keys}}
    keys = [k for k in keys if ok_clip[cd_(k)]]
    if not a.smoke and not a.max_updates and len(keys) < n_all and not a.allow_partial:
        raise RuntimeError(f"only {len(keys)}/{n_all} rows have fetched images (no done markers); finish the fetch or "
                           f"pass --allow-partial")
    if a.dup_cot > 1:   # oversample decision frames (the test key frames are decision frames)
        extra = [k for k in keys if rows[k].get("cot")] * (a.dup_cot - 1); keys = keys + extra
        print(f"  --dup-cot {a.dup_cot}: +{len(extra)} repeated decision rows -> {len(keys)} rows per epoch", flush=True)
    global FLOW_DRAWS; FLOW_DRAWS = a.flow_draws
    if a.freeze:   # frozen parameters keep no gradient or AdamW state (12 bytes/param less) and no vision-tower activations
        pre = tuple(x for x in a.freeze.split(",") if x); nf = 0
        for n_, p_ in model.named_parameters():
            if n_.startswith(pre): p_.requires_grad_(False); nf += p_.numel()
        if not nf: raise RuntimeError(f"--freeze {a.freeze} matched no parameters")
        print(f"  --freeze {a.freeze}: {nf / 1e9:.3f}B parameters frozen", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[edge_train] {len(keys)} rows, trainable {sum(p.numel() for p in params) / 1e9:.3f}B, "
          f"gpu {torch.cuda.get_device_name()}", flush=True)
    pg = params
    if a.lr_vis_mult != 1.0 or a.lr_expert_mult != 1.0 or a.no_decay_norms:   # parameter groups (default: one group)
        G = {}
        for n_, p_ in model.named_parameters():
            if not p_.requires_grad: continue
            part = "vis" if n_.startswith("vlm.model.visual") else "exp" if n_.startswith("expert.") else "llm"
            nd = bool(a.no_decay_norms and (p_.ndim < 2 or "norm" in n_.lower() or "embed" in n_))
            G.setdefault((part, nd), []).append(p_)
        mult = {"vis": a.lr_vis_mult, "exp": a.lr_expert_mult, "llm": 1.0}
        pg = [{"params": v, "lr": a.lr * mult[k[0]], "weight_decay": 0.0 if k[1] else a.wd} for k, v in sorted(G.items())]
        for k, v in sorted(G.items()):
            print(f"  group {k[0]}{' no-decay' if k[1] else ''}: {sum(p_.numel() for p_ in v) / 1e6:.1f}M params, "
                  f"lr {a.lr * mult[k[0]]:.2e}", flush=True)
    if W > 1:
        from torch.distributed.optim import ZeroRedundancyOptimizer
        opt = ZeroRedundancyOptimizer(pg, optimizer_class=torch.optim.AdamW, lr=a.lr, weight_decay=a.wd,
                                      betas=(0.9, 0.95), fused=True)
    else:
        opt = torch.optim.AdamW(pg, lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95), fused=True)

    def allreduce_grads(bucket=2 ** 26):   # sum over ranks in ~256 MB fp32 buckets; a param no local row touched counts as 0
        if W == 1: return
        gs = []
        for p_ in params:
            if p_.grad is None: p_.grad = torch.zeros_like(p_)
            gs.append(p_.grad)
        i0 = 0
        while i0 < len(gs):
            i1, n_ = i0, 0
            while i1 < len(gs) and (n_ == 0 or n_ + gs[i1].numel() <= bucket): n_ += gs[i1].numel(); i1 += 1
            flat = torch.cat([g.reshape(-1) for g in gs[i0:i1]]); dist.all_reduce(flat)
            o_ = 0
            for g in gs[i0:i1]: g.copy_(flat[o_:o_ + g.numel()].view_as(g)); o_ += g.numel()
            del flat; i0 = i1
    if a.smoke:
        turning = [k for k in keys if abs(rows[k]["fut64"][49][1]) > 2.0]           # exercise the sign check
        k = (turning or keys)[0]; td, traj, anc, ns = encode(model, rows[k], a.split, a.img_root, "cuda")
        ids = td["input_ids"][0]
        print(f"  smoke row {k}: {ids.shape[0]} tokens, anchor {anc}, suffix {ns}, nav '{rows[k]['nav']}'", flush=True)
        print("  decoded tail:", repr(model.tokenizer.decode(ids[anc - 12 - traj["n_text"]:anc + 4])), "...",
              repr(model.tokenizer.decode(ids[-3:])), flush=True)
        print("  future token ids (first 8):", ids[anc + 1:anc + 9].tolist(), flush=True)
        label_check(model, td, traj, anc, rows[k])
        for s in range(a.smoke):
            t = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ce, flow = step_losses(model, td, traj, anc, ns)
            loss = a.ce_w * ce + a.flow_w * flow; loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            print(f"  step {s} ce {ce.item():.4f} flow {flow.item():.4f} gnorm {gn.item():.2f} {time.time() - t:.2f}s "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        print("EDGE_SMOKE_DONE", flush=True); return
    # Full training. Resumable: <out>/state.pt holds weights, optimizer, scheduler, update count and RNG states, written
    # atomically at update boundaries every --state-every-min minutes and before a planned stop (exit 3, which the
    # sbatch turns into a requeue). The row order is a pure function of --seed, so a resume skips the rows already seen.
    n_up = int(math.ceil(len(keys) * a.epochs / a.accum))
    if a.max_updates: n_up = min(n_up, a.max_updates)
    warm = max(1, int(a.warmup * n_up))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / warm) * (a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, u / n_up)))))
    os.makedirs(a.out, exist_ok=True); order = []; u = 0; t0 = time.time(); acc = []
    rng = random.Random(a.seed)
    while len(order) < n_up * a.accum:
        e = keys[:]; rng.shuffle(e); order += e
    order = order[:n_up * a.accum]
    sp = os.path.join(a.out, "state.pt"); okey = hashlib.md5("\n".join(order).encode()).hexdigest()

    spo = lambda r: os.path.join(a.out, f"state_opt{r}of{W}.pt")   # W > 1: each rank's AdamW shard

    def save_state():
        if W == 1:
            tmp = sp + ".tmp"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "u": u,
                        "n_up": n_up, "keys_hash": okey, "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(), "args": vars(a)}, tmp)
            os.replace(tmp, sp)
        else:   # every file is written as .tmp first and renamed only after all ranks finished writing
            torch.save({"opt": opt.optim.state_dict(), "u": u, "cuda_rng": torch.cuda.get_rng_state()}, spo(RK) + ".tmp")
            if RK == 0:
                torch.save({"model": model.state_dict(), "sched": sched.state_dict(), "u": u, "n_up": n_up, "W": W,
                            "keys_hash": okey, "torch_rng": torch.get_rng_state(), "args": vars(a)}, sp + ".tmp")
            dist.barrier(); os.replace(spo(RK) + ".tmp", spo(RK))
            if RK == 0: os.replace(sp + ".tmp", sp)
            dist.barrier()
        print(f"  STATE_SAVED update {u} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    if os.path.isfile(sp):
        st = torch.load(sp, map_location="cpu", weights_only=False)
        if st["n_up"] != n_up or st["keys_hash"] != okey:
            raise RuntimeError(f"state.pt does not match this run (n_up {st['n_up']} vs {n_up} or row order differs)")
        if st.get("W", 1) != W: raise RuntimeError(f"state.pt was written by {st.get('W', 1)} ranks, this run has {W}")
        model.load_state_dict(st["model"]); sched.load_state_dict(st["sched"])
        u = st["u"]; torch.set_rng_state(st["torch_rng"])
        if W == 1:
            opt.load_state_dict(st["opt"]); torch.cuda.set_rng_state(st["cuda_rng"])
        else:
            so = torch.load(spo(RK), map_location="cpu", weights_only=False)
            if so["u"] != u: raise RuntimeError(f"rank {RK} optimizer shard is from update {so['u']}, state.pt from {u}")
            opt.optim.load_state_dict(so["opt"]); torch.cuda.set_rng_state(so["cuda_rng"]); del so
        del st
        print(f"  RESUMED at update {u}/{n_up}", flush=True)
    start = u * a.accum; last_save = time.time(); tim = [0.0, 0.0]
    u0 = u; prof = [0.0] * 5; pc = 0.0   # --profile: wait, all-reduce, opt, save, imbalance (s summed over updates)
    from concurrent.futures import ThreadPoolExecutor
    # Two pools: --io-threads only read the image files ahead (warms the page cache; the first read from the shared
    # fileset costs 1-6 s per row, a cached one 0.36 s, profiled in src/edge_prof.py), --workers decode + run the
    # processor a few rows ahead. File reads release the GIL and use almost no CPU, so they do not slow the GPU loop.
    pool = ThreadPoolExecutor(max_workers=max(1, a.workers)); fut = {}
    iop = ThreadPoolExecutor(max_workers=max(1, a.io_threads)); warmed = set()
    def touch(v):
        for im in v["images"]:
            with open(os.path.join(a.img_root, a.split, im["m"]), "rb") as fh: fh.read()
    def want(j):
        if a.io_threads:
            for jj in range(j, min(j + a.io_ahead, len(order))):
                if jj % W == RK and jj not in warmed: warmed.add(jj); iop.submit(touch, rows[order[jj]])
        if a.workers == 0: return
        for jj in range(j, min(j + 2 * a.workers * W, len(order))):
            if jj % W == RK and jj not in fut: fut[jj] = pool.submit(prep_cpu, model, rows[order[jj]], a.split, a.img_root)
    for i in range(start, len(order)):
      if i % W == RK:
        k = order[i]; te = time.time(); want(i)
        td, traj, anc, ns = encode(model, rows[k], a.split, a.img_root, "cuda", prep=fut.pop(i).result() if i in fut else None)
        tim[0] += time.time() - te; te = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ce, flow = step_losses(model, td, traj, anc, ns)
        ((a.ce_w * ce + a.flow_w * flow) / a.accum).backward(); acc.append((ce.item(), flow.item())); tim[1] += time.time() - te
        del td, traj, ce, flow
      if (i + 1) % a.accum == 0:
            if a.profile:
                torch.cuda.synchronize(); tp = time.time()
                if W > 1:   # this update's compute time per rank; the gather itself waits for the slowest rank
                    ct = torch.tensor([tim[0] + tim[1] - pc], device="cuda", dtype=torch.float64)
                    g_ = [torch.zeros_like(ct) for _ in range(W)]; dist.all_gather(g_, ct)
                    g_ = torch.cat(g_).tolist(); prof[4] += max(g_) - sum(g_) / W
                    torch.cuda.synchronize(); prof[0] += time.time() - tp; tp = time.time()
            allreduce_grads()
            if a.profile: torch.cuda.synchronize(); prof[1] += time.time() - tp; tp = time.time()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); u += 1
            if a.profile: torch.cuda.synchronize(); prof[2] += time.time() - tp
            pc = tim[0] + tim[1]
            if u % a.log_every == 0 or u <= 2:
                m = np.mean(acc[-a.log_every * a.accum // W:], 0); nr = i + 1 - start
                pf = ""
                if a.profile:
                    du = max(1, u - u0)
                    pf = (f" | per update: wait {prof[0] / du:.1f}s (imbalance {prof[4] / du:.1f}s) all-reduce "
                          f"{prof[1] / du:.1f}s opt {prof[2] / du:.1f}s save {prof[3] / du:.1f}s")
                print(f"  update {u}/{n_up} ce {m[0]:.4f} flow {m[1]:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{(time.time() - t0) / nr:.2f} s/row (encode {tim[0] / nr:.2f}, fwd+bwd {tim[1] / nr:.2f}) "
                      f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB{pf}", flush=True)
            if u % a.save_every == 0 or u == n_up:
                ts = time.time()
                if RK == 0: save(model, a.model, os.path.join(a.out, f"update-{u}"))
                if W > 1: dist.barrier()
                prof[3] += time.time() - ts
            stop = bool((a.stop_after_updates and u >= a.stop_after_updates) or
                        (a.stop_after_min and time.time() - t0 > 60 * a.stop_after_min))
            due = time.time() - last_save > 60 * a.state_every_min
            if W > 1:   # rank 0's clock decides, so all ranks save and stop at the same update
                fl = torch.tensor([int(stop), int(due)], device="cuda"); dist.broadcast(fl, 0); stop, due = bool(fl[0]), bool(fl[1])
            if u < n_up and (stop or due):
                ts = time.time(); save_state(); last_save = time.time(); prof[3] += last_save - ts
                if stop:
                    print("EDGE_TRAIN_STOP_FOR_REQUEUE", flush=True)
                    if W > 1:   # torch.distributed.run reports a worker's exit 3 as 1, so the sbatch looks for this file
                        if RK == 0: open(os.path.join(a.out, "STOPPED_FOR_REQUEUE"), "w").close()
                        dist.barrier(); dist.destroy_process_group(); sys.exit(0)
                    sys.exit(3)
    pool.shutdown(wait=False, cancel_futures=True); iop.shutdown(wait=False, cancel_futures=True)
    if W > 1: dist.barrier(); dist.destroy_process_group()
    print("EDGE_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
