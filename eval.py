"""Dice of NV-Segment-CT predictions against the TotalSegmentator reference masks.

  python eval.py out/pred s2032 [s1806 ...]      -> out/pred/eval.json + summary

Only structures present in the reference (>= 1000 voxels, i.e. >= 3.4 ml at 1.5 mm)
are scored, so a CT that does not cover the head is not punished for "missing" the brain.
"""
import json, os, sys

import nibabel as nib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CD = json.load(open(os.path.join(HERE, "models/nv-segment-ct/metadata.json")))
CD = CD["network_data_format"]["outputs"]["pred"]["channel_def"]
VID = {v: int(k) for k, v in CD.items()}
MIN_VOX = 1000


def ts_to_vista(name):
    """kidney_right -> right kidney, rib_left_1 -> left rib 1, vertebrae_L1 -> vertebrae L1"""
    if name == "urinary_bladder":
        return "bladder"
    t = name.split("_")
    for side in ("left", "right"):
        if side in t:
            t.remove(side)
            return side + " " + " ".join(t)
    return " ".join(t)


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * np.logical_and(a, b).sum() / s if s else np.nan


def eval_case(pred_dir, case, ver="v3"):
    pred = np.asarray(nib.load(os.path.join(pred_dir, f"{case}.nii.gz")).dataobj)
    seg_dir = os.path.join(HERE, "data", ver, case, "segmentations")
    res = {}
    for f in sorted(os.listdir(seg_dir)):
        name = f[:-7]
        vname = ts_to_vista(name)
        if vname not in VID:
            continue
        gt = np.asarray(nib.load(os.path.join(seg_dir, f)).dataobj) > 0
        if gt.sum() < MIN_VOX:
            continue
        res[name] = round(float(dice(pred == VID[vname], gt)), 4)
    return res


if __name__ == "__main__":
    pred_dir, cases = sys.argv[1], sys.argv[2:]
    out = {}
    for c in cases:
        r = eval_case(pred_dir, c)
        out[c] = r
        v = np.array(list(r.values()))
        worst = sorted(r.items(), key=lambda x: x[1])[:5]
        print(f"{c}: {len(r)} structures, median Dice {np.median(v):.3f}, mean {v.mean():.3f}, "
              f">=0.9: {(v >= 0.9).mean():.0%}; worst {worst}", flush=True)
    json.dump(out, open(os.path.join(pred_dir, "eval.json"), "w"), indent=1)
    allv = np.array([d for r in out.values() for d in r.values()])
    print(f"ALL: {len(allv)} structure-scans, median {np.median(allv):.3f}, mean {allv.mean():.3f}")
