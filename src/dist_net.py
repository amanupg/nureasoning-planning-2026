"""Scene-aware 5 s distance model: how far the ego travels in 1..5 s (the planning bottleneck, LEDGER 2026-10-21).

Inputs: front camera at t and t-0.5 s (one shared DINOv3 ViT-B/16, FULLY fine-tuned; Aman 2026-10-21) + ego-history
features (src/dist_feats.py: 10 s speed history, acceleration, time since stop, ...). Per frame: CLS + mean patch token;
the head sees [now, before, now - before] + an MLP of the history features. Output: path length at 1..5 s (m).
Loss: Huber on the 5 distances (5 s weighted x2). Frames: v0 = the 256x512 16:9 frames already on the cluster
(data/img_wa_full); --root / --hw for full resolution later.
Rows: train split of rows_split (dev logs held out); epoch chosen on DEV-LOG rows at frame 100 (validation untouched);
validation predictions written for every epoch, used once on the Mac (src/dist_apply.py).
  python src/dist_net.py train --tag d0
"""
import argparse, gzip, json, math, os, random, sys, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
FRONT = 1                                                                  # index of "front" in wajepa_run.CAMS
log_of = lambda c: "_".join(c.split("_")[:2])


def dec_labels(split):
    f = os.path.join(HERE, f"data/dec_labels_{split}.npz")
    if not os.path.isfile(f): return {}
    z = np.load(f); return {(str(c), int(r)): int(l) for c, r, l in zip(z["clip"], z["r"], z["lon"])}


def load_split(split, feats, mu=None, sd=None):
    z = np.load(os.path.join(HERE, f"data/dist_feats_{split}.npz")); X = z["X"].astype(np.float32); Y = z["Y"].astype(np.float32)
    C, Rr = list(map(str, z["clip"])), list(map(int, z["r"]))                # read each npz array ONCE (z[...] reloads it)
    with gzip.open(os.path.join(HERE, f"data/wa_{split}_paths.json.gz"), "rt") as f: paths = json.load(f)
    if mu is None:
        mu = np.nanmean(X, 0); sd = np.nanstd(X, 0) + 1e-3
    Xn = np.concatenate([np.nan_to_num((X - mu) / sd), np.isnan(X).astype(np.float32)], 1)
    rows = []; L = dec_labels(split)
    for i, (c, r) in enumerate(zip(C, Rr)):
        P = paths.get(c, {})
        if str(r) in P and str(r - 5) in P: rows.append((c, r, P[str(r)][FRONT], P[str(r - 5)][FRONT], Xn[i].copy(), Y[i].copy(), L.get((c, r), -1)))
    return rows, mu, sd


class DS:
    def __init__(self, rows, root, hw, train):
        self.rows, self.root, self.hw, self.train = rows, root, hw, train

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        import torch, cv2
        c, r, pa, pb, x, y, lab = self.rows[i]; flip = self.train and random.random() < 0.5
        def im(p):
            a = cv2.cvtColor(cv2.imread(os.path.join(self.root, c, p)), cv2.COLOR_BGR2RGB)
            if a.shape[:2] != self.hw: a = cv2.resize(a, (self.hw[1], self.hw[0]), interpolation=cv2.INTER_AREA)
            if flip: a = a[:, ::-1]
            a = (a.astype(np.float32) / 255 - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
            return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))).float()
        return im(pa), im(pb), torch.from_numpy(x), torch.from_numpy(np.nan_to_num(y, nan=-1.0)).float(), torch.tensor(lab)


def build(model_dir, nfeat):
    import torch, torch.nn as nn
    from transformers import AutoModel

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.bb = AutoModel.from_pretrained(model_dir); d = self.bb.config.hidden_size
            self.nreg = int(getattr(self.bb.config, "num_register_tokens", 0))
            self.g = nn.Sequential(nn.Linear(nfeat, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU())
            self.head = nn.Sequential(nn.LayerNorm(6 * d + 256), nn.Linear(6 * d + 256, 512), nn.GELU(), nn.Linear(512, 5 + 8))  # 5 lengths + 8 decision logits

        def emb(self, x):
            h = self.bb(pixel_values=x).last_hidden_state
            return torch.cat([h[:, 0], h[:, 1 + self.nreg:].mean(1)], -1)

        def forward(self, a, b, x):
            ea, eb = self.emb(a), self.emb(b)
            out = self.head(torch.cat([ea, eb, ea - eb, self.g(x)], -1))
            return torch.nn.functional.softplus(out[:, :5]) * 10.0, out[:, 5:]   # metres (positive), decision logits
    return Net()


def predict(net, rows, root, hw, bs=64):
    import torch
    dl = torch.utils.data.DataLoader(DS(rows, root, hw, False), batch_size=bs, num_workers=8)
    net.eval(); P = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for a, b, x, _, _ in dl:
            d, lg = net(a.cuda(), b.cuda(), x.cuda()); P.append(np.c_[d.float().cpu().numpy(), torch.softmax(lg.float(), -1).cpu().numpy()])
    net.train(); P = np.concatenate(P); return P[:, :5], P[:, 5:]


def train(a):
    import torch, torch.nn as nn
    from wajepa_ft import rows_split
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    rows, mu, sd = load_split("train", None); _, dv = rows_split(0); dev_logs = {log_of(c) for c, _ in dv}
    tr = [x for x in rows if log_of(x[0]) not in dev_logs and np.isfinite(x[5][4])]
    dev100 = [x for x in rows if log_of(x[0]) in dev_logs and x[1] == 100 and np.isfinite(x[5][4])]
    val, _, _ = load_split("validation", None, mu, sd)
    if a.limit: tr, dev100, val = tr[:a.limit], dev100[:max(8, a.limit // 8)], val[:max(8, a.limit // 8)]   # smoke
    hw = tuple(a.hw); troot = os.path.join(HERE, a.root, "train"); vroot = os.path.join(HERE, a.root, "validation")
    print(f"train {len(tr)} rows / dev logs at frame 100 {len(dev100)} / validation {len(val)}; input {hw}", flush=True)
    net = build(os.path.join(HERE, a.model), len(tr[0][4])).cuda()
    opt = torch.optim.AdamW([{"params": net.bb.parameters(), "lr": a.lr}, {"params": list(net.g.parameters()) + list(net.head.parameters()), "lr": a.lr * 30}], weight_decay=0.02)
    dl = torch.utils.data.DataLoader(DS(tr, troot, hw, True), batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=True)
    total = a.epochs * len(dl); step = 0; hub = nn.HuberLoss(delta=1.0, reduction="none"); wt = torch.tensor([1, 1, 1, 1, 2.0]).cuda()
    out = os.path.join(HERE, f"runs/dist_{a.tag}"); os.makedirs(out, exist_ok=True); best = (1e9, -1); t0 = time.time()
    for ep in range(a.epochs):
        tot = 0.0
        ce = nn.CrossEntropyLoss(ignore_index=-1)
        for xa, xb, x, y, lab in dl:
            f = min(1.0, step / max(1, int(0.03 * total))) * 0.5 * (1 + math.cos(math.pi * step / total))
            for i, g in enumerate(opt.param_groups): g["lr"] = (a.lr if i == 0 else a.lr * 30) * f
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p, lg = net(xa.cuda(non_blocking=True), xb.cuda(non_blocking=True), x.cuda(non_blocking=True))
            p = p.float(); y = y.cuda(); lab = lab.cuda(); loss = (hub(p, y) * wt).mean()
            if a.dec_w > 0 and (lab >= 0).any(): loss = loss + a.dec_w * ce(lg.float(), lab)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            step += 1; tot += loss.item()
            if step % 200 == 0 or step <= 3:
                print(f"  step {step}/{total} loss {loss.item():.3f} {(time.time() - t0) / step:.2f}s/step peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        (pd, qd), yd = predict(net, dev100, troot, hw), np.array([x[5] for x in dev100])
        (pv, qv), yv = predict(net, val, vroot, hw), np.array([x[5] for x in val])
        lv = np.array([x[6] for x in val]); acc = np.mean(qv[lv >= 0].argmax(1) == lv[lv >= 0]) if (lv >= 0).any() else float("nan")
        mae_d = np.mean(np.abs(pd[:, 4] - yd[:, 4])); mae_v = np.nanmean(np.abs(pv[:, 4] - yv[:, 4]))
        np.savez(os.path.join(out, f"val_ep{ep}.npz"), clip=np.array([x[0] for x in val]), pred=pv, dec=qv)
        tag = ""
        if mae_d < best[0]:
            best = (mae_d, ep); tag = "  <- best (dev)"; torch.save({"state": net.state_dict(), "mu": mu, "sd": sd, "args": vars(a), "epoch": ep}, os.path.join(out, "model.pt"))
        print(f"  epoch {ep} loss {tot / len(dl):.3f} | 5 s distance MAE: dev logs @100 {mae_d:.2f} m, validation {mae_v:.2f} m; validation decision accuracy {acc:.3f} (n {int((lv >= 0).sum())}){tag} ({time.time() - t0:.0f}s)", flush=True)
    print(f"kept epoch {best[1]} (dev MAE {best[0]:.2f} m) -> {out}/model.pt; validation predictions: {out}/val_ep{best[1]}.npz", flush=True)
    if os.path.isfile(os.path.join(HERE, "data/dist_feats_test.npz")):                     # test predictions from the kept model
        net.load_state_dict(torch.load(os.path.join(out, "model.pt"), map_location="cpu", weights_only=False)["state"])
        te, _, _ = load_split("test", None, mu, sd); pt, qt = predict(net, te, os.path.join(HERE, a.root, "test"), hw)
        np.savez(os.path.join(out, "test.npz"), clip=np.array([x[0] for x in te]), pred=pt, dec=qt); print(f"test predictions: {len(te)} clips", flush=True)
    print("DIST_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["train"]); ap.add_argument("--tag", default="d0")
    ap.add_argument("--root", default="data/img_wa_full"); ap.add_argument("--hw", type=int, nargs=2, default=[256, 512])
    ap.add_argument("--model", default="weights/dinov3-vitb16"); ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N train rows")
    ap.add_argument("--dec-w", type=float, default=0.5, help="weight of the longitudinal-decision loss (labelled frame-100 rows)")
    a = ap.parse_args(); {"train": train}[a.cmd](a)


if __name__ == "__main__":
    main()
