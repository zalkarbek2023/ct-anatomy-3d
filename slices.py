"""Axial CT slices + coloured label overlays for the hook scene, placed in the same viewer
frame as mesh.py (x = -R, y = S, z = A, cm, minus the mesh centre).

  python slices.py data/v3/s1922/ct.nii.gz out/pred_trt/s1922.nii.gz out/hero_s1922 [--n 48]
"""
import argparse, json, os

import nibabel as nib
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ct"); ap.add_argument("pred"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=48)
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.out, "anatomy.json")))
    centre = np.array(meta["centre_cm"])
    col = {s["id"]: np.array(s["color"]) for s in meta["structures"]}
    ct = nib.as_closest_canonical(nib.load(args.ct))
    pr = nib.as_closest_canonical(nib.load(args.pred))
    img = np.asarray(ct.dataobj).astype(np.float32)
    lab = np.asarray(pr.dataobj).astype(np.uint8)
    A = ct.affine
    lut = np.zeros((256, 4), np.uint8)
    for k, c in col.items():
        lut[k, :3] = (np.asarray(c) * 255).astype(np.uint8); lut[k, 3] = 255
    os.makedirs(os.path.join(args.out, "slices"), exist_ok=True)
    X, Y, Z = img.shape
    ks = np.linspace(Z - 1, 0, args.n).round().astype(int)          # head -> feet
    lo, hi = -160.0, 240.0                                            # abdominal soft-tissue window (W400 L40)
    out = []
    for i, k in enumerate(ks):
        s = img[:, :, k]
        g = (np.clip((s - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        g = g[::-1, ::-1].T                                          # rows anterior->posterior, cols patient R->L
        m = lut[lab[:, :, k][::-1, ::-1].T]
        Image.fromarray(g).save(os.path.join(args.out, "slices", f"ct_{i:02d}.jpg"), quality=88)
        Image.fromarray(m, "RGBA").save(os.path.join(args.out, "slices", f"mask_{i:02d}.png"))
        p0 = A @ [0, 0, k, 1]; p1 = A @ [X - 1, Y - 1, k, 1]
        xs = sorted([-p0[0] / 10 - centre[0], -p1[0] / 10 - centre[0]])
        zs = sorted([p0[1] / 10 - centre[2], p1[1] / 10 - centre[2]])
        out.append({"y": round(p0[2] / 10 - centre[1], 3), "x": [round(a, 3) for a in xs],
                    "z": [round(a, 3) for a in zs], "labels": int(len(np.unique(lab[:, :, k])) - 1)})
    json.dump({"slices": out, "n_slices_ct": int(Z), "spacing_mm": [float(a) for a in ct.header.get_zooms()[:3]]},
              open(os.path.join(args.out, "slices.json"), "w"), indent=1)
    print(len(out), "slices ->", args.out)


if __name__ == "__main__":
    main()
