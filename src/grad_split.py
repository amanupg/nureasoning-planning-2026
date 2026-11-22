"""How the fine-tuning gradient splits between the world-model (JEPA) loss and the trajectory loss.
Per parameter group (trainable encoder blocks / trajectory-only params / everything else): gradient norm of
0.5 x loss_jepa_pred and of 1.0 x loss_traj, their cosine, and the total norm the trainer clips to 1.0."""
import argparse, os, sys, numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.join(HERE, "ext/WA-JEPA"))
from wajepa_ft import build, DS, rows_split

ap = argparse.ArgumentParser(); ap.add_argument("--init", default="rl4_s1987"); ap.add_argument("--unfreeze", type=int, default=4)
ap.add_argument("--img-root", default=os.path.join(HERE, "data/img_wa_full/train")); ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--batches", type=int, default=8); a = ap.parse_args()
import torch
m = build("cuda", False, 10, a.unfreeze)
ck = torch.load(os.path.join(HERE, f"runs/wajepa_ft_{a.init}.pt"), map_location="cpu", weights_only=False)
m.load_state_dict(ck["state_dict"], strict=False); m.train(); m.encoder.eval(); m.target_encoder.eval()
tr, _ = rows_split(0); rows = [tr[i] for i in np.random.RandomState(0).choice(len(tr), a.bs * a.batches, replace=False)]
dl = torch.utils.data.DataLoader(DS(rows, a.img_root, crop="full", horizon=10), batch_size=a.bs, num_workers=4)
named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
grp = lambda n: "encoder" if n.startswith("encoder.") else ("traj-only" if "traj" in n else "shared/other")
print(f"init {a.init}, unfreeze {a.unfreeze}; groups:", {g: round(sum(p.numel() for n, p in named if grp(n) == g) / 1e6, 1) for g in ("encoder", "traj-only", "shared/other")}, "M params")
def grads(loss):
    m.zero_grad(set_to_none=True); loss.backward(retain_graph=True)
    return {n: (p.grad.detach().float().clone() if p.grad is not None else torch.zeros_like(p, dtype=torch.float32)) for n, p in named}
R = []
for b in dl:
    b = {k: v.cuda() for k, v in b.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16): out = m(b)
    ld = out["loss"]; gj = grads(0.5 * ld["loss_jepa_pred"]); gt = grads(1.0 * ld["loss_traj"])
    row = {"jepa": ld["loss_jepa_pred"].item(), "traj": ld["loss_traj"].item()}
    for g in ("encoder", "traj-only", "shared/other", "all"):
        ns = [n for n, _ in named if g == "all" or grp(n) == g]
        if not ns: continue
        vj = torch.cat([gj[n].flatten() for n in ns]); vt = torch.cat([gt[n].flatten() for n in ns])
        row[g] = (vj.norm().item(), vt.norm().item(), torch.nn.functional.cosine_similarity(vj, vt, dim=0).item())
    R.append(row); print(row, flush=True)
print("\nmean over batches: loss jepa %.4f traj %.5f" % (np.mean([r["jepa"] for r in R]), np.mean([r["traj"] for r in R])))
for g in ("encoder", "traj-only", "shared/other", "all"):
    if g in R[0]:
        j, t, c = (np.mean([r[g][i] for r in R]) for i in range(3))
        print(f"  {g:13s} |grad jepa| {j:.4g}  |grad traj| {t:.4g}  traj share of norm {t / (j + t):.1%}  cosine {c:+.3f}")
print("GRAD_SPLIT_DONE")
