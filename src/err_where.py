"""Where does the v9.5 blend lose its 5 s distance? (2026-11-05, validation 1000 clips, analysis only.)
Signed along-track error e = blend 5 s path length - human 5 s path length, split by situation tags:
- kinematics: ego speed now; start from stop (v0 < 0.5, human > 3 m); stopping (human 5 s distance < 40% of v0 x 5 s).
- lead vehicle from the frame-100 object boxes (all clips): nearest vehicle ahead in the ego lane (0-60 m, |y| < 1.8 m),
  its gap and its speed along the ego heading.
- the frame-100 driving annotation where present (~52% of clips): traffic light state / visibility, lead behaviour,
  vulnerable road users, annotated decision.
Validation labels are used only to describe errors, never as a planning input.
  python src/err_where.py   -> runs/err_where_validation.json
"""
import collections, json, os, re, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from ensemble_eval import wmean
from dist_stack import plen
from wajepa_run import NR, KEY
F = {"b": "v71B", "h4": "wajepa_ft_h10u4", "wa": "wajepa_169", "dj": "djft_r2avg", "raw": "a15s1lora_raw"}
W = {"b": .4, "h4": .35, "wa": .2, "dj": .25, "raw": .1}


def tags(clip):
    from nureasoning.common.pickle_io import load_pickle
    cd = os.path.join(NR, "cache", "validation", clip); M = json.load(open(os.path.join(cd, "metadata.json")))
    fr = {f["frame_index"]: f for f in M["frames"]}; f0 = fr[KEY]
    E = load_pickle(os.path.join(cd, f0["ego_state"])); A = load_pickle(os.path.join(cd, f0["annotations"]))
    ex, ey, yaw = E.pose["x"], E.pose["y"], E.pose["yaw"]; c, s = np.cos(yaw), np.sin(yaw)
    lead = (np.inf, np.nan)
    for o in A.objects:
        if not o.category.startswith("vehicle"): continue
        dx, dy = o.pose["x"] - ex, o.pose["y"] - ey; x, y = c * dx + s * dy, -s * dx + c * dy
        if 0 < x < 60 and abs(y) < 1.8 and x < lead[0]:
            lead = (x, c * o.velocity["vx"] + s * o.velocity["vy"])
    t = {"scenario": M.get("scenario_type") or "", "lead_gap": lead[0], "lead_v": lead[1],
         "tl_states": sorted({x.state for x in A.traffic_light_states})}
    rp = f0.get("reasoning")
    if rp and os.path.isfile(os.path.join(cd, rp)):
        D = json.load(open(os.path.join(cd, rp))).get("Driving") or {}
        if isinstance(D, dict) and D:
            comp = D.get("Critical components") or {}
            tl = [v for k, v in comp.items() if k.lower().startswith("traffic light") and isinstance(v, dict)]
            veh = [v for k, v in comp.items() if k.lower().startswith("vehicle") and isinstance(v, dict)]
            t.update({"ann": True, "tl_type": (tl[0].get("Type") or "").lower() if tl else "none",
                      "tl_vis": (tl[0].get("Visibility") or "").lower() if tl else "",
                      "lead_same_lane": any("same lane" in (v.get("Relative location") or "").lower() and "front" in (v.get("Relative location") or "").lower() for v in veh),
                      "vru": any(re.match(r"(pedestrian|cyclist|bicycl|vru)", k.lower()) for k in comp),
                      "decision": ((D.get("Driving decision") or {}).get("Longitudinal") or "").lower()})
    return clip, t


def main():
    fz = np.load(os.path.join(HERE, "data/dist_feats_validation.npz")); C = list(map(str, fz["clip"]))
    v0 = fz["X"][:, 0].astype(float); hum = fz["Y"][:, 4].astype(float)
    Z = {k: dict(zip(map(str, np.load(os.path.join(HERE, f"runs/{f}_validation.npz"))["names"]), np.load(os.path.join(HERE, f"runs/{f}_validation.npz"))["traj"])) for k, f in F.items()}
    bl = np.array([plen(wmean([np.asarray(Z[k][c], float) for k in W], [W[k] for k in W])) for c in C])
    with ProcessPoolExecutor(8) as ex: T = dict(ex.map(tags, C, chunksize=8))
    e = bl - hum; ok = np.isfinite(e); tot = np.nansum(e[ok] ** 2)
    def row(name, m):
        m = m & ok; n = int(m.sum())
        if n == 0: return
        print(f"  {name:44s} n {n:4d} ({n / ok.sum():5.1%}) | share of sq. error {np.sum(e[m] ** 2) / tot:5.1%} | mean |e| {np.mean(np.abs(e[m])):5.2f} m"
              f" | bias {np.mean(e[m]):+5.2f} m | human 5 s {np.mean(hum[m]):5.1f} m")
    g = lambda k, d=None: np.array([T[c].get(k, d) for c in C], dtype=object)
    print(f"v9.5 blend, {ok.sum()} clips: mean |e| {np.mean(np.abs(e[ok])):.2f} m, bias {np.mean(e[ok]):+.2f} m, RMS {np.sqrt(tot / ok.sum()):.2f} m")
    print("by ego speed now:")
    for lo, hi in [(0, .5), (.5, 3), (3, 8), (8, 12), (12, 40)]: row(f"v0 {lo}-{hi} m/s", (v0 >= lo) & (v0 < hi))
    print("by manoeuvre (from the human future):")
    start = (v0 < .5) & (hum > 3); stop = (v0 > 3) & (hum < .4 * v0 * 5); stay = (v0 < .5) & (hum <= 3)
    row("start from stop", start); row("stopping", stop); row("remain stopped", stay); row("other (moving, not stopping)", ~start & ~stop & ~stay)
    print("by lead vehicle (object boxes, all clips):")
    lg = g("lead_gap").astype(float); lv = g("lead_v").astype(float)
    row("no lead within 60 m", ~np.isfinite(lg)); row("lead 0-15 m", lg < 15); row("lead 15-30 m", (lg >= 15) & (lg < 30)); row("lead 30-60 m", (lg >= 30) & (lg < 60))
    row("lead stopped (< 0.5 m/s)", np.isfinite(lg) & (np.abs(lv) < .5)); row("lead moving", np.isfinite(lg) & (np.abs(lv) >= .5))
    print("by traffic lights in the frame annotation (any state listed):")
    tls = g("tl_states")
    row("no light states", np.array([len(x) == 0 for x in tls])); row("red among states", np.array(["red" in x for x in tls]))
    row("green only", np.array([set(x) == {"green"} for x in tls])); row("yellow among states", np.array(["yellow" in x for x in tls]))
    ann = np.array([bool(T[c].get("ann")) for c in C]); print(f"by the driving annotation ({ann.sum()} clips):")
    tt = g("tl_type", ""); tv = g("tl_vis", "")
    for k in ("none", "green", "red", "yellow"): row(f"annotated light: {k}", ann & (tt == k))
    row("annotated light, not fully visible", ann & (tt != "none") & np.array([("visible" != x) for x in tv]))
    row("annotated lead in same lane, ahead", ann & (g("lead_same_lane", False) == True)); row("annotated VRU present", ann & (g("vru", False) == True))
    dec = g("decision", "")
    for k in sorted({d for d in dec[ann] if d}): row(f"decision: {k}", ann & (dec == k))
    sc = g("scenario"); print("by scenario tag (contains):")
    for k in ["work_zone", "limited_visibility", "vulnerable_road_user", "stationary_vehicle", "unusual_traffic", "traffic_control"]:
        row(k, np.array([k in x for x in sc]))
    json.dump({c: {**{k: (v if not isinstance(v, float) or np.isfinite(v) else None) for k, v in T[c].items()}, "e": float(e[i]), "v0": float(v0[i]), "human": float(hum[i])}
               for i, c in enumerate(C)}, open(os.path.join(HERE, "runs/err_where_validation.json"), "w"))


if __name__ == "__main__":
    main()
