"""nuReasoning clips -> Qwen-Drive-1.0 planning scene files (their benchmark JSONL format, docs/data.md).

Per clip (key frame 100): 3 views [front, front-left, front-right] x frames 85/90/95/100 (t-1.5..0 s at 2 Hz),
full frames as stored (1408x928; the model's own pixel budgets then apply: ~0.17 MP history, ~0.92 MP current);
ego history at 10 Hz for frames 85..100: poses relative to frame 100 (x forward, y left, heading), and velocity /
acceleration in each frame's own body frame (verified on their demo scenes: during a 0.45 rad turn the stored
velocity has ~0 lateral component); current velocity / acceleration; driving_command one-hot [left, straight,
right, unknown] and nav_command (0 straight, 1 left, 2 right; unknown -> straight) from the clip's mission goal.
The instruction text is left to the model (it synthesizes it from history + nav command, as for scratch scenes).
Never reads ego_state.trajectory_future.
  python src/qd_scenes.py --split validation  -> data/qd/<split>_scenes.jsonl, data/qd/<split>_files.txt
  python src/qd_scenes.py collect --split validation --pred <predictions.jsonl> --tag rl  -> runs/qwendrive_<tag>_<split>.npz
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from wajepa_run import NR, KEY, CMD, rel_se2
VIEWS = ["front", "front_left", "front_right"]                              # order of CAMERA_VIEWS in qwen_drive/scene.py
VIEW_TAGS = ["<FRONT VIEW>", "<FRONT LEFT VIEW>", "<FRONT RIGHT VIEW>"]
NAV = {0: 1, 1: 0, 2: 2, 3: 0}                                              # our CMD index (left, straight, right, unknown) -> nav_command


def scene(split, clip):
    from nureasoning.common.pickle_io import load_pickle
    cd = os.path.join(NR, "cache", split, clip)
    fr = {f["frame_index"]: f for f in json.load(open(os.path.join(cd, "metadata.json")))["frames"]}
    E = [load_pickle(os.path.join(cd, fr[KEY - 15 + i]["ego_state"])) for i in range(16)]
    P = np.array([[e.pose["x"], e.pose["y"], e.pose["yaw"]] for e in E], float)
    hist = rel_se2(P[-1], P); hist[-1] = 0.0
    vel = [[float(e.velocity["vx"]), float(e.velocity["vy"])] for e in E]
    acc = [[float(e.acceleration["ax"]), float(e.acceleration["ay"])] for e in E]
    cmd = CMD.get(((fr[KEY].get("mission_goal") or {}).get("command")), 3); dc = [0, 0, 0, 0]; dc[cmd] = 1
    content, files = [], []
    for tag, v in zip(VIEW_TAGS, VIEWS):
        content.append({"text": tag})
        for i, o in enumerate((-15, -10, -5, 0)):
            p = os.path.join(clip, fr[KEY + o]["sensors"]["cameras"][v]); files.append(p)
            content += [{"text": f"frame: {i}"}, {"image": p, "timestamp_idx": i, "timestamp_offset_sec": o / 10.0}]
    content.append({"text": "__SYNTHESIZE__"})                              # placeholder, replaced below
    rec = {"type": "chatml", "messages": [{"role": "user", "content": content}, {"role": "assistant", "content": []}],
           "meta_info": {"dataset": "nuReasoning", "token": clip, "scene_token": clip, "cam_order": VIEWS,
                         "history_time_offsets": [-1.5, -1.0, -0.5, 0.0], "image_hz": 2.0},
           "trajectory": {"hist_traj_10hz": hist.round(6).tolist(), "hist_vel_10hz": vel, "hist_acc_10hz": acc,
                          "ego_status": {"ego_velocity": vel[-1], "ego_acceleration": acc[-1], "driving_command": dc},
                          "nav_command": NAV[cmd]}}
    return rec, files, P[-1]


def instruction_fn():
    """DrivingScene.instruction() from ext/Qwen-Drive-1.0/src/qwen_drive/scene.py, with its constants read from that
    file (ast), so the prompt matches the model's own synthesized text exactly (that module needs transformers)."""
    import ast
    src = open(os.path.join(HERE, "ext/Qwen-Drive-1.0/src/qwen_drive/scene.py")).read(); C = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in (
                "HISTORY_FRAME_LABELS", "NAV_COMMANDS", "_INSTRUCTION_HEADER"):
            C[node.targets[0].id] = ast.literal_eval(node.value)

    def f(history, nav_command, num_frames=4):
        labels = C["HISTORY_FRAME_LABELS"][-num_frames:]
        stride = max(1, (len(history) - 1) // max(1, num_frames - 1))
        lines = "".join(" -{}: ({:.4f}, {:.4f}, {:.4f});\n".format(label, *history[index * stride]) for index, label in enumerate(labels))
        header = C["_INSTRUCTION_HEADER"].format(num_frames=num_frames, interval=1.5 / max(1, num_frames - 1),
                                                 first_label=labels[0], last_frame=num_frames - 1)
        return f"{header}{lines}2. Active navigation command: [{C['NAV_COMMANDS'][int(nav_command)]}]"
    return f


def build(a):
    from concurrent.futures import ProcessPoolExecutor
    instruction = instruction_fn()
    root = os.path.join(HERE, "data/img", a.split)
    clips = sorted(c for c in os.listdir(root) if os.path.isfile(os.path.join(root, c, "_done.json")))
    with ProcessPoolExecutor(8) as ex: R = list(ex.map(scene, [a.split] * len(clips), clips, chunksize=8))
    os.makedirs(os.path.join(HERE, "data/qd"), exist_ok=True)
    with open(os.path.join(HERE, f"data/qd/{a.split}_scenes.jsonl"), "w") as f, open(os.path.join(HERE, f"data/qd/{a.split}_files.txt"), "w") as g:
        for rec, files, _ in R:
            t = rec["trajectory"]
            txt = instruction(np.asarray(t["hist_traj_10hz"]), t["nav_command"])
            rec["messages"][0]["content"][-1] = {"text": txt}
            f.write(json.dumps(rec) + "\n"); g.write("\n".join(files) + "\n")
    np.savez(os.path.join(HERE, f"data/qd/{a.split}_origin.npz"), clip=np.array(clips), origin=np.stack([r[2] for r in R]))
    print(f"{a.split}: {len(R)} scenes; example instruction:\n{txt}", flush=True)


def collect(a):
    """predictions.jsonl (token, trajectories [S, 50, 3] ego frame at 0.1..5.0 s) -> global 51-pose trajectories."""
    z = np.load(os.path.join(HERE, f"data/qd/{a.split}_origin.npz")); O = dict(zip(map(str, z["clip"]), z["origin"]))
    names, S = [], []
    for line in open(a.pred):
        r = json.loads(line); names.append(r["token"]); S.append(np.asarray(r["trajectories"], float))
    def glob(t, o):
        t = np.vstack([[0, 0, 0], t]); c, s = np.cos(o[2]), np.sin(o[2])
        return np.stack([o[0] + c * t[:, 0] - s * t[:, 1], o[1] + s * t[:, 0] + c * t[:, 1], o[2] + t[:, 2]], 1)
    G = np.array([[glob(t, O[n]) for t in s] for n, s in zip(names, S)])     # [N, S, 51, 3]
    mean = G.mean(1); mean[..., 2] = np.arctan2(np.sin(G[..., 2]).mean(1), np.cos(G[..., 2]).mean(1))
    np.savez(os.path.join(HERE, f"runs/qwendrive_{a.tag}_{a.split}.npz"), names=np.array(names), traj=mean, samples=G, first=G[:, 0])
    print(f"{len(names)} clips, {G.shape[1]} samples each -> runs/qwendrive_{a.tag}_{a.split}.npz (traj = per-timestep mean)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", nargs="?", default="build", choices=["build", "collect"])
    ap.add_argument("--split", default="validation"); ap.add_argument("--pred", default=""); ap.add_argument("--tag", default="rl")
    a = ap.parse_args(); {"build": build, "collect": collect}[a.cmd](a)
