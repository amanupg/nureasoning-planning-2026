# edge_state_export.py RUN_DIR BASE_MODEL OUT: write an inference checkpoint (same format as src/edge_train.save) from
# the model weights in RUN_DIR/state.pt (for multi-GPU runs, rank 0's file holds the full model). Used when a run stops
# at its time limit between --save-every checkpoints (D 11-15: stop near update 935/1060, tail skipped).
# BASE_MODEL: the directory the run was initialised from (architecture/config + tokenizer files).
# Run on a compute node: CPU only, ~25 GB RAM (fp32 model + state dict). Refuses to overwrite OUT.
import os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
import torch
from edge_train import save
from alpamayo2_edge.checkpoint import load_model

run, base, out = sys.argv[1:4]
if os.path.exists(out): raise SystemExit(f"{out} exists")
st = torch.load(os.path.join(run, "state.pt"), map_location="cpu", weights_only=False)
print(f"state.pt: update {st['u']} of {st['n_up']}, W {st.get('W', 1)}", flush=True)
model, _ = load_model(base, device="cpu", dtype="float32", attn_implementation="sdpa")
model.load_state_dict(st["model"], strict=True)
save(model, base, out)
print("EXPORT_DONE", out, flush=True)
