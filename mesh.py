"""Label map -> one GLB with a named mesh per structure + a JSON sidecar for the video.

  python mesh.py out/pred_trt/s1806.nii.gz out/hero  [--target-tris 600000]

vtkSurfaceNets3D extracts every label in one pass (shared boundaries stay watertight),
then each structure is decimated and written with its VISTA3D name. Coordinates are
NIfTI world (RAS, mm) mapped to a Y-up viewer frame: x = patient left->right as seen
from the front, y = superior, z = anterior (towards the camera), centimetres, centred.
"""
import argparse, json, os

import nibabel as nib
import numpy as np
import trimesh
import vtk
from vtk.util import numpy_support as ns

HERE = os.path.dirname(os.path.abspath(__file__))
CD = json.load(open(os.path.join(HERE, "models/nv-segment-ct/metadata.json")))
CD = CD["network_data_format"]["outputs"]["pred"]["channel_def"]

BONE = ("vertebrae", "rib", "skull", "sternum", "costal", "hip", "femur", "humerus", "scapula",
        "clavicula", "sacrum")
ARTERY = ("aorta", "artery", "brachiocephalic trunk")
VEIN = ("vena", "vein")
MUSCLE = ("gluteus", "iliopsoas", "autochthon")
LUNG = ("lung",)
HEART = ("heart", "atrial appendage")


def system(name):
    for group, keys in (("bone", BONE), ("artery", ARTERY), ("vein", VEIN), ("muscle", MUSCLE),
                        ("lung", LUNG), ("heart", HEART)):
        if any(k in name for k in keys):
            return group
    return "organ"


BASE = {"bone": (0.91, 0.88, 0.80), "organ": (0.85, 0.45, 0.32), "artery": (1.0, 0.27, 0.24),
        "vein": (0.30, 0.48, 1.0), "muscle": (0.62, 0.22, 0.24), "lung": (0.55, 0.75, 0.92),
        "heart": (0.93, 0.25, 0.30)}
ORGAN = {"liver": (0.62, 0.27, 0.20), "spleen": (0.55, 0.22, 0.38), "kidney": (0.80, 0.42, 0.28),
         "pancreas": (0.93, 0.70, 0.30), "stomach": (0.90, 0.58, 0.52), "bowel": (0.86, 0.64, 0.46),
         "colon": (0.82, 0.56, 0.40), "duodenum": (0.88, 0.62, 0.44), "gallbladder": (0.30, 0.70, 0.32),
         "bladder": (0.92, 0.82, 0.40), "esophagus": (0.95, 0.65, 0.62), "trachea": (0.60, 0.82, 0.92),
         "adrenal": (0.95, 0.76, 0.35), "thyroid": (0.78, 0.49, 0.86), "prostate": (0.72, 0.54, 0.84),
         "brain": (0.95, 0.76, 0.78), "spinal cord": (1.0, 0.88, 0.54), "cyst": (1.0, 0.92, 0.3)}


def colour(name, sys_, v):
    if sys_ == "organ":
        for k, c in ORGAN.items():
            if k in name:
                return c
    c = np.array(BASE[sys_])
    if sys_ == "bone":  # alternate shades so neighbouring vertebrae/ribs read as separate pieces
        c = c * (0.86 + 0.14 * (v % 2))
    return tuple(float(a) for a in np.clip(c, 0, 1))


def surface_nets(lab):
    img = vtk.vtkImageData()
    img.SetDimensions(*lab.shape)
    arr = ns.numpy_to_vtk(lab.ravel(order="F"), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    img.GetPointData().SetScalars(arr)
    sn = vtk.vtkSurfaceNets3D()
    sn.SetInputData(img)
    sn.SetBackgroundLabel(0)
    sn.GenerateLabels(1, 1, 255)  # placeholder, replaced below
    labels = [int(v) for v in np.unique(lab) if v]
    sn.SetNumberOfLabels(len(labels))
    for i, v in enumerate(labels):
        sn.SetLabel(i, v)
    sn.SetOutputMeshTypeToTriangles()
    sn.SmoothingOn()
    sn.SetNumberOfIterations(40)
    sn.Update()
    out = sn.GetOutput()
    pts = ns.vtk_to_numpy(out.GetPoints().GetData()).astype(np.float64)
    tris = ns.vtk_to_numpy(out.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    bl = ns.vtk_to_numpy(out.GetCellData().GetArray("BoundaryLabels")).astype(np.int64)
    return pts, tris, bl, labels


def to_viewer(pts_ijk, affine):
    ras = np.c_[pts_ijk, np.ones(len(pts_ijk))] @ affine.T
    x, y, z = ras[:, 0], ras[:, 1], ras[:, 2]
    # RAS+: +x patient right. Seen from the front the patient's right is on the viewer's left.
    return np.c_[-x, z, y] / 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("out")
    ap.add_argument("--target-tris", type=int, default=600000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    nii = nib.load(args.pred)
    lab = np.asarray(nii.dataobj).astype(np.uint8)
    pts, tris, bl, labels = surface_nets(lab)
    xyz = to_viewer(pts, nii.affine)
    centre = (xyz.min(0) + xyz.max(0)) / 2
    xyz -= centre
    print(f"surface nets: {len(tris)} triangles for {len(labels)} labels", flush=True)
    keep = args.target_tris / max(len(tris), 1)
    scene, meta = trimesh.Scene(), []
    vox_ml = float(np.prod(nii.header.get_zooms()[:3])) / 1000.0
    for v in labels:
        name = CD[str(v)]
        side0, side1 = bl[:, 0] == v, bl[:, 1] == v
        f = np.r_[tris[side0], tris[side1][:, ::-1]]  # consistent outward winding
        if len(f) < 20:
            continue
        m = trimesh.Trimesh(xyz, f, process=True)
        m.remove_unreferenced_vertices()
        tgt = max(200, int(len(m.faces) * keep))
        if tgt < len(m.faces):
            try:
                m = m.simplify_quadric_decimation(face_count=tgt)
            except Exception as e:  # keep full resolution if decimation is unavailable
                print("decimation skipped:", e)
        trimesh.repair.fix_normals(m)
        scene.add_geometry(m, node_name=name, geom_name=name)
        c = m.vertices.mean(0)
        sys_ = system(name)
        meta.append({"name": name, "id": v, "system": sys_, "tris": int(len(m.faces)),
                     "color": [round(a, 3) for a in colour(name, sys_, v)],
                     "centroid": [round(float(a), 2) for a in c],
                     "volume_ml": round(float((lab == v).sum()) * vox_ml, 1)})
    glb = os.path.join(args.out, "anatomy.glb")
    scene.export(glb, include_normals=True)
    ext = (xyz.max(0) - xyz.min(0)).round(1).tolist()
    json.dump({"structures": meta, "extent_cm": ext, "centre_cm": [round(float(a), 3) for a in centre]}, open(os.path.join(args.out, "anatomy.json"), "w"), indent=1)
    tot = sum(s["tris"] for s in meta)
    print(f"{len(meta)} meshes, {tot} triangles, extent {ext} cm, {os.path.getsize(glb) / 1e6:.1f} MB -> {glb}")


if __name__ == "__main__":
    main()
