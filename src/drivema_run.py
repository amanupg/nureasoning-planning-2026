"""DriveMA-2B (Tsinghua MARS Lab, Qwen3.5-2B VLA, Apache-2.0; weights/drivema-2b = HF zwc2003/DriveMA-2B @ d5f4905)
zero-shot on nuReasoning key frames, with its own two-turn prompt (ext/DriveMA/tools/infer_scripts/
run_vllm_infer_mutil_turn.py, copied verbatim below) and near-greedy decoding (temperature 0 here; theirs 0.01).

Turn 1: three front views at the key frame + intent + current / 4 s past (16 steps at 4 Hz) velocity, acceleration and
trajectory in the current ego frame -> "longitudinal action: ..; lateral action: ..".
Turn 2: -> 5 waypoints at 1 Hz [x, y] (x forward, y left).
Camera match to its Waymo training views (f 1110 px on 972 x 1079, HFOV 47 deg, ~f 556 px after its 262,144-pixel cap):
our 1408 x 928 frames (f ~1403) are centre-cropped to HFOV 47 deg (1220 px wide, full height) and resized to f ~556 px
(480 x 368), so objects keep the angular size the model saw in training. front_left / front_right = CAM_M_L0 / R0
(yaw +56 / -54 deg; Waymo +-45 deg). Intent from the dataset driving command (TURN LEFT -> GO_LEFT, KEEP FORWARD ->
GO_STRAIGHT, TURN RIGHT -> GO_RIGHT, UNKNOWN -> null, the token it was trained with for masked fields).
Output -> 51 poses at 0.1 s: cubic spline through (0, 0) and the 5 waypoints, start slope = current velocity; heading
from the path tangent (held when the car barely moves).
  .venv_drivema/bin/python src/drivema_run.py --split validation [--limit N] [--shard i --nshards n]
  -> runs/drivema/<tag>_<split>_s<shard>.jsonl ; collect: --collect -> runs/drivema_<tag>_<split>.npz (names, traj)
"""
import argparse, glob, json, os, re, sys, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NR = "/path/to/workspace/nureasoning"
KEY = 100
W = os.path.join(HERE, "weights/drivema-2b")
VIEWS = (("front_left", "front_left"), ("front", "front"), ("front_right", "front_right"))   # DriveMA key, cache key
INTENT = {"TURN LEFT": "GO_LEFT", "KEEP FORWARD": "GO_STRAIGHT", "TURN RIGHT": "GO_RIGHT", "UNKNOWN": "null"}
CMD_TXT = ["TURN LEFT", "KEEP FORWARD", "TURN RIGHT", "UNKNOWN"]
F_SRC, F_DST, HFOV = 1403.0, 556.0, np.radians(47.0)


# ---- verbatim prompt builders (DriveMA tools/infer_scripts/run_vllm_infer_mutil_turn.py) ----
def _fmt_num(x):
    value = float(x)
    if round(value, 2) == 0:
        value = 0.0
    return f"{value:.2f}"


def _fmt_xy(pt):
    return f"[{_fmt_num(pt[0])}, {_fmt_num(pt[1])}]"


def _fmt_traj(points):
    return ", ".join(_fmt_xy(p) for p in points)


def turn1(intent, past_traj, past_accel, past_vel, cur_vel, cur_accel):
    return ("You are an expert driver.\n"
            "Input:\n"
            "- 1 frame of multi-view images collected from the ego-vehicle at the present timestep: "
            "front_left_view: <image>; front_view:<image>; front_right_view:<image>\n"
            f"- Current high-level intent:{intent}\n"
            f"- Current acceleration is {_fmt_xy(cur_accel)}\n"
            f"- Current velocity is {_fmt_xy(cur_vel)}\n"
            f"- 4-second past trajectory (16 steps at 4 Hz):{_fmt_traj(past_traj)}\n"
            f"- 4-second past acceleration (16 steps at 4 Hz):{_fmt_traj(past_accel)}\n"
            f"- 4-second past velocity (16 steps at 4 Hz):{_fmt_traj(past_vel)}\n"
            "Coordinate System Definition: X-axis: positive forward, negative backward; Y-axis: positive left, negative right.\n"
            "Task: Inspect the input and make the decision.\n"
            "Output format:\n"
            "longitudinal action: xx, lateral action: xx")


TURN2 = ("Task: Given the above information, predict the optimal 5-second future trajectory (5 steps at 1 Hz) of the ego vehicle.\n"
         "Output format:\n"
         "[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4], [x_5, y_5]")


def content(text, imgs):
    parts = text.split("<image>"); out = []
    for i, p in enumerate(parts):
        if p: out.append({"type": "text", "text": p})
        if i < len(parts) - 1: out.append({"type": "image", "image": imgs[i]})
    return out


# ---- our inputs ----
def ego_inputs(split, clip):
    """Current-frame ego quantities and the 16-step 4 Hz past (t = -3.75 .. 0 s), all in the key-frame ego frame."""
    sys.path.insert(0, os.path.join(NR, "recon/devkit"))
    from nureasoning.common.pickle_io import load_pickle
    cd = os.path.join(NR, "cache", split, clip); fr = {f["frame_index"]: f for f in json.load(open(os.path.join(cd, "metadata.json")))["frames"]}
    idx = list(range(KEY - 40, KEY + 1)); E = {i: load_pickle(os.path.join(cd, fr[i]["ego_state"])) for i in idx if i in fr}
    e0 = E[KEY]; ox, oy, oh = e0.pose["x"], e0.pose["y"], e0.pose["yaw"]; c, s = np.cos(oh), np.sin(oh)
    ks = sorted(E); t = (np.array(ks) - KEY) * 0.1
    gx = np.array([E[k].pose["x"] for k in ks]) - ox; gy = np.array([E[k].pose["y"] for k in ks]) - oy
    px, py = c * gx + s * gy, -s * gx + c * gy
    yaw = np.unwrap(np.array([E[k].pose["yaw"] for k in ks]))
    def body_to_key(k, vx, vy):                                           # ego-state vectors are in that frame's body frame
        d = E[k].pose["yaw"] - oh; cd_, sd = np.cos(d), np.sin(d); return cd_ * vx - sd * vy, sd * vx + cd_ * vy
    vel = np.array([body_to_key(k, E[k].velocity["vx"], E[k].velocity["vy"]) for k in ks])
    acc = np.array([body_to_key(k, E[k].acceleration["ax"], E[k].acceleration["ay"]) for k in ks])
    tq = -3.75 + 0.25 * np.arange(16); tq = np.clip(tq, t[0], 0)            # clips start >= 4 s before frame 100
    I = lambda a: np.interp(tq, t, a)
    past = np.c_[I(px), I(py)]; pv = np.c_[I(vel[:, 0]), I(vel[:, 1])]; pa = np.c_[I(acc[:, 0]), I(acc[:, 1])]
    return past, pv, pa, vel[-1], acc[-1], (ox, oy, oh)


def views(split, clip, img_root):
    import cv2
    from PIL import Image
    cd = os.path.join(NR, "cache", split, clip); fr = {f["frame_index"]: f for f in json.load(open(os.path.join(cd, "metadata.json")))["frames"]}
    out = []
    for _, key in VIEWS:
        im = cv2.imread(os.path.join(img_root, clip, fr[KEY]["sensors"]["cameras"][key]), cv2.IMREAD_COLOR)[:, :, ::-1]
        h, w = im.shape[:2]; cw = int(round(2 * F_SRC * np.tan(HFOV / 2))); x0 = (w - cw) // 2
        im = im[:, x0:x0 + cw]; sc = F_DST / F_SRC
        out.append(Image.fromarray(cv2.resize(np.ascontiguousarray(im), (int(round(cw * sc)), int(round(h * sc))), interpolation=cv2.INTER_AREA)))
    return out


NUM = r"(-?\d+(?:\.\d+)?)"
def parse_wp(text):
    pts = re.findall(r"\[\s*" + NUM + r"\s*,\s*" + NUM + r"\s*\]", text)
    return np.array([[float(a), float(b)] for a, b in pts[:5]]) if len(pts) >= 5 else None


def to_51(wp, v0, origin):
    from scipy.interpolate import CubicSpline
    T = np.arange(6.0); P = np.vstack([[0, 0], wp])
    sx = CubicSpline(T, P[:, 0], bc_type=((1, float(v0[0])), (2, 0.0))); sy = CubicSpline(T, P[:, 1], bc_type=((1, float(v0[1])), (2, 0.0)))
    t = np.arange(51) * 0.1; x, y = sx(t), sy(t); dx, dy = sx(t, 1), sy(t, 1)
    h = np.zeros(51); sp = np.hypot(dx, dy)
    for i in range(51):
        h[i] = np.arctan2(dy[i], dx[i]) if sp[i] > 0.5 else (h[i - 1] if i else 0.0)
    h = np.unwrap(h); h[0] = 0.0
    ox, oy, oh = origin; c, s = np.cos(oh), np.sin(oh)
    return np.stack([ox + c * x - s * y, oy + s * x + c * y, oh + h], 1)


def collect(a):
    sys.path.insert(0, os.path.join(HERE, "src"))
    R = {}
    for f in sorted(glob.glob(os.path.join(HERE, f"runs/drivema/{a.tag}_{a.split}_s*.jsonl"))):
        for l in open(f): r = json.loads(l); R[r["clip"]] = r
    names = sorted(c for c in R if R[c]["wp"] is not None)
    T = np.array([to_51(np.asarray(R[c]["wp"]), R[c]["v0"], R[c]["origin"]) for c in names])
    print(f"[drivema] {len(names)} of {len(R)} clips parsed")
    np.savez(os.path.join(HERE, f"runs/drivema_{a.tag}_{a.split}.npz"), names=np.array(names), traj=T,
             meta=np.array([R[c]["meta"] for c in names]), wp=np.array([R[c]["wp"] for c in names]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="validation"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--tag", default="zs")
    ap.add_argument("--device", default="mps"); ap.add_argument("--collect", action="store_true"); a = ap.parse_args()
    if a.collect: return collect(a)
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    proc = AutoProcessor.from_pretrained(W, max_pixels=262144, min_pixels=3136)
    model = AutoModelForMultimodalLM.from_pretrained(W, dtype=torch.bfloat16).to(a.device).eval()
    z = np.load(os.path.join(HERE, f"data/wa_{a.split}_meta.npz")); cmd = {str(c): CMD_TXT[int(np.argmax(st[:4]))] for c, st in zip(z["clip"], z["status"])}
    img_root = os.path.join(HERE, "data/img", a.split)
    clips = sorted(cmd)[a.shard::a.nshards][: a.limit or None]
    out = os.path.join(HERE, f"runs/drivema/{a.tag}_{a.split}_s{a.shard}.jsonl"); os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {json.loads(l)["clip"] for l in open(out)} if os.path.isfile(out) else set()
    todo = [c for c in clips if c not in done]; t0 = time.time()
    print(f"[drivema] {a.split} shard {a.shard}/{a.nshards}: {len(done)} done, {len(todo)} to go", flush=True)

    def gen(conv):
        x = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
                                     enable_thinking=False).to(a.device)
        with torch.no_grad():
            o = model.generate(**x, max_new_tokens=128, do_sample=False)
        return proc.decode(o[0, x["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    with open(out, "a") as fo:
        for n, c in enumerate(todo):
            past, pv, pa, v0, a0, origin = ego_inputs(a.split, c); imgs = views(a.split, c, img_root)
            u1 = content(turn1(INTENT[cmd[c]], past, pa, pv, v0, a0), imgs)
            meta = gen([{"role": "user", "content": u1}])
            wp_txt = gen([{"role": "user", "content": u1}, {"role": "assistant", "content": [{"type": "text", "text": meta}]},
                          {"role": "user", "content": [{"type": "text", "text": TURN2}]}])
            wp = parse_wp(wp_txt)
            fo.write(json.dumps({"clip": c, "meta": meta, "wp_text": wp_txt, "wp": None if wp is None else wp.round(3).tolist(),
                                 "v0": [float(v0[0]), float(v0[1])], "origin": [float(o) for o in origin], "intent": INTENT[cmd[c]]}) + "\n"); fo.flush()
            if n < 3 or (n + 1) % 50 == 0:
                print(f"  [{n + 1}/{len(todo)}] {(time.time() - t0) / (n + 1):.2f}s/clip v={np.hypot(*v0):.1f} {INTENT[cmd[c]]} | {meta!r} | {wp_txt[:90]!r}", flush=True)
    print("DRIVEMA_DONE", flush=True)


if __name__ == "__main__":
    main()
