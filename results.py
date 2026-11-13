"""Aggregate timings (out/pred_torch, out/pred_trt) and Dice (eval.json) into out/results.json."""
import json, os, statistics as st

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
cases = pd.read_csv(os.path.join(HERE, "data/test_cases.csv"))
res = {"cases": []}
dice_trt = json.load(open(os.path.join(HERE, "out/pred_trt/eval.json")))
dice_torch = json.load(open(os.path.join(HERE, "out/pred_torch/eval.json")))
for _, r in cases.iterrows():
    c = r.image_id
    t = {k: json.load(open(os.path.join(HERE, f"out/pred_{k}/{c}.json"))) for k in ("torch", "trt")}
    med = lambda k, f: st.median(x[f] for x in t[k]["runs"])
    d = np.array(list(dice_trt[c].values()))
    res["cases"].append({"case": c, "age": float(r.age), "study": r.study_type, "grid": t["trt"]["runs"][0]["model_grid"],
                         "torch_s": round(med("torch", "total_s"), 2), "trt_s": round(med("trt", "total_s"), 2),
                         "vram_gb": round(max(x["peak_vram_gb"] for x in t["trt"]["runs"]), 2),
                         "found": t["trt"]["labels_found"], "scored": len(d), "median_dice": round(float(np.median(d)), 4),
                         "torch_trt_max_abs_dice_diff": round(max(abs(dice_trt[c][k] - dice_torch[c][k]) for k in dice_trt[c]), 4)})
C = pd.DataFrame(res["cases"])
allv = np.array([v for c in dice_trt.values() for v in c.values()])
per_struct = {}
for c in dice_trt.values():
    for k, v in c.items():
        per_struct.setdefault(k, []).append(v)
ps = sorted(((k, float(np.median(v)), len(v)) for k, v in per_struct.items()), key=lambda x: x[1])
kids = C[C.age < 18]
res["summary"] = {
    "n_cases": len(C), "structure_checks": int(len(allv)), "median_dice": round(float(np.median(allv)), 4),
    "mean_dice": round(float(allv.mean()), 4), "pct_above_090": round(float((allv >= 0.9).mean() * 100), 1),
    "pct_above_080": round(float((allv >= 0.8).mean() * 100), 1),
    "trt_s": {"min": float(C.trt_s.min()), "median": float(C.trt_s.median()), "max": float(C.trt_s.max())},
    "torch_s": {"min": float(C.torch_s.min()), "median": float(C.torch_s.median()), "max": float(C.torch_s.max())},
    "trt_speedup_median": round(float((C.torch_s / C.trt_s).median()), 3),
    "vram_max_gb": float(C.vram_gb.max()),
    "adults": {"n": int((C.age >= 18).sum()), "median_case_dice": float(C[C.age >= 18].median_dice.median())},
    "children": {"n": len(kids), "median_case_dice": float(kids.median_dice.median()) if len(kids) else None},
    "worst_structures": [{"name": k, "median_dice": round(v, 3), "n": n} for k, v, n in ps[:10]],
    "best_structures": [{"name": k, "median_dice": round(v, 3), "n": n} for k, v, n in ps[-10:]],
    "n_structure_types_scored": len(ps),
    "max_abs_dice_diff_torch_vs_trt": float(C.torch_trt_max_abs_dice_diff.max()),
}
json.dump(res, open(os.path.join(HERE, "out/results.json"), "w"), indent=1)
print(C.to_string())
print(json.dumps(res["summary"], indent=1))
