"""Ground-truth trajectory -> AutoVLA action tokens, reimplemented from its NAVSIM agent
(ext/AutoVLA/navsim/navsim/agents/autovla_agent.py TokenProcessor._match_agent_token, num_k = 1) without navsim imports:
for each 0.5 s step, the codebook token whose rolled-out vehicle contour (width 2.0, length 4.8) is closest (sum of corner
distances) to the true contour, rolled out from the previous matched token's end state.
  tokenize(traj10 [10, 3] ego frame x, y, heading) -> [10] token ids;  decode with models.action_tokenizer.ActionTokenizer.
"""
import os, pickle
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOK = None


def _token_traj():
    global _TOK
    if _TOK is None:
        d = pickle.load(open(os.path.join(HERE, "ext/AutoVLA/codebook_cache/agent_vocab.pkl"), "rb"))
        _TOK = torch.tensor(d["token_all"]["veh"], dtype=torch.float32)[:, -1]          # [n_token, 4, 2] end contour
    return _TOK


def contour(pos, head, w=2.0, l=4.8):
    """[2], scalar -> [4, 2] corners (left-front, right-front, right-rear, left-rear) as cal_polygon_contour."""
    c, s = np.cos(head) * 0.5, np.sin(head) * 0.5
    x, y = pos
    return np.array([[x + l * c - w * s, y + l * s + w * c], [x + l * c + w * s, y + l * s - w * c],
                     [x - l * c + w * s, y - l * s - w * c], [x - l * c - w * s, y - l * s + w * c]])


def tokenize(traj):
    T = _token_traj().numpy(); prev_p, prev_h = np.zeros(2), 0.0; ids = []
    for p in traj:
        R = np.array([[np.cos(prev_h), np.sin(prev_h)], [-np.sin(prev_h), np.cos(prev_h)]])   # as transform_to_global
        world = T.reshape(-1, 2) @ R + prev_p; world = world.reshape(T.shape)
        gt = contour(p[:2], p[2]); k = int(np.argmin(np.linalg.norm(world - gt[None], axis=-1).sum(-1))); ids.append(k)
        tc = world[k]; d = tc[0] - tc[3]; prev_h = float(np.arctan2(d[1], d[0])); prev_p = tc.mean(0)
    return np.array(ids)
