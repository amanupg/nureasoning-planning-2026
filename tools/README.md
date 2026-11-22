# tools/

Small analysis scripts that produced numbers quoted in LEDGER.md (moved here from the session scratchpad on 2026-11-14).
Run from the repo root with `.venv/bin/python tools/<script>`.

| script | what it does | ledger use |
|---|---|---|
| paired.py A.json B.json [key] | paired log-level bootstrap (506 val logs) of two `*_val_scores.json` files | RL member vs e2a, Edge checkpoints |
| edge_pool.py TAG1 TAG2 ... | pool the K samples of several Edge val runs, score the union mean vs TAG1 | run A + run B pooling |
| pool2.py TAG1 TAG2 ... | same as edge_pool.py with a worker initializer; writes runs/pool_*.json | RL + e2a pool (-0.0042) |
| edge_oracle.py | best-of-K headroom of Edge candidates on val | 11-13 Edge headroom |
| idx_bias.py, subset_mean.py | sample-index bias and subset-mean checks of the K candidates | 11-13 K-mean checks |
| edge_ckpt_mix.py | checkpoint weight averaging of Edge fine-tunes | 11-12 |
| cmp_v97_v129.py, v94_check.py, v94_val_handoff.py | submission comparisons and handoff checks | 11-11 / 11-12 |
