"""NVIDIA NV-Segment-CT (VISTA3D) "segment everything": 117 structures from one CT.

  python segment.py data/v3/s1806/ct.nii.gz [more.nii.gz ...] --out out/pred [--repeat 3] [--trt]

Writes <out>/<case>.nii.gz (uint8 label map in the original CT grid, VISTA3D label ids)
and <out>/<case>.json (stage timings, peak VRAM). Preprocessing mirrors the official
pipeline (hugging_face nvidia/NV-Segment-CT, vista3d_pipeline.py).
"""
import argparse, json, os, sys, time

import nibabel as nib
import numpy as np
import torch
from monai.apps.vista3d.transforms import VistaPostTransformd
from monai.data.utils import decollate_batch
from monai.inferers import SlidingWindowInfererAdapt
from monai.networks.nets import vista3d132
from monai.transforms import (CastToTyped, Compose, CropForegroundd, EnsureChannelFirstd, EnsureTyped,
                              Invertd, LoadImaged, Orientationd, ScaleIntensityRanged, Spacingd)
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "models/nv-segment-ct/vista3d_pretrained_model/model.safetensors")
META = os.path.join(HERE, "models/nv-segment-ct/metadata.json")
# bundle configs/inference.json: everything_labels = 1..132 minus merged/unsupported ids
EVERYTHING = sorted(set(range(1, 133)) - {2, 16, 18, 20, 21, 23, 24, 25, 26, 27, 128, 129, 130, 131, 132})
DEV = torch.device("cuda")


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def build_model():
    net = vista3d132(encoder_embed_dim=48, in_channels=1)
    sd = load_file(WEIGHTS)
    sd = {k.removeprefix("network."): v for k, v in sd.items()}
    net.load_state_dict(sd)
    return net.to(DEV).eval()


def patch_monai_trt():
    """torch.onnx exports fp16 InstanceNorm as (float input, half scale/bias); TensorRT 11's
    parser rejects mixed types. Cast those scale/bias initializers to float before building."""
    import onnx
    from onnx import numpy_helper
    from monai.networks import trt_compiler as tc

    orig = tc.TrtCompiler._onnx_to_trt

    def fixed(self, onnx_path):
        m = onnx.load(onnx_path)
        inits = {i.name: i for i in m.graph.initializer}
        n = 0
        for node in m.graph.node:
            if node.op_type == "InstanceNormalization":
                for name in node.input[1:]:
                    t = inits.get(name)
                    if t is not None and t.data_type == onnx.TensorProto.FLOAT16:
                        t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), name))
                        n += 1
        onnx.save(m, onnx_path)
        print(f"[trt] cast {n} InstanceNorm scale/bias tensors to float", flush=True)
        return orig(self, onnx_path)

    tc.TrtCompiler._onnx_to_trt = fixed


class SplitInstanceNorm(torch.nn.Module):
    """InstanceNorm3d with a two-stage reduction. TensorRT 11 turns a plain InstanceNorm over
    a 128^3 map into a reduction with one block per channel (48 blocks on a 36-SM GPU), which
    made the encoder 10x slower than PyTorch. Splitting the spatial axis first gives it
    `split` times more parallel work. Math is identical (two-pass mean/variance in fp32)."""

    def __init__(self, norm, split=64):
        super().__init__()
        # keep plain tensors, not the InstanceNorm module: MONAI's exporter wraps every
        # InstanceNorm it finds in CastToFloat, which would break this module
        self.split, self.eps, self.affine = split, norm.eps, norm.affine
        if norm.affine:
            self.weight = torch.nn.Parameter(norm.weight.detach().clone())
            self.bias = torch.nn.Parameter(norm.bias.detach().clone())

    def forward(self, x):
        B, C = x.shape[:2]
        xf = x.float()
        # reduce W first (B*C*D*H parallel outputs, no reshape -> no layout shuffle), then D,H
        m = xf.mean(-1, keepdim=True).mean((-3, -2), keepdim=True)
        v = (xf - m).pow(2).mean(-1, keepdim=True).mean((-3, -2), keepdim=True)
        y = (xf - m) * torch.rsqrt(v + self.eps)
        if self.affine:
            y = y * self.weight.float().view(1, C, 1, 1, 1) + self.bias.float().view(1, C, 1, 1, 1)
        return y.to(x.dtype)


def split_instance_norms(module):
    n = 0
    for name, child in module.named_children():
        if isinstance(child, torch.nn.InstanceNorm3d):
            setattr(module, name, SplitInstanceNorm(child)); n += 1
        else:
            n += split_instance_norms(child)
    return n


class HalfEncoder(torch.nn.Module):
    """fp16 copy of the image encoder; the input cast is part of the exported graph."""

    def __init__(self, enc):
        super().__init__()
        self.enc = enc.half()

    def forward(self, x):
        return self.enc(x.half())


class LabelImageEncoder(torch.nn.Module):
    """Whole image encoder (encoder + label-branch decoder) in fp16, label prompts only."""

    def __init__(self, ie):
        super().__init__()
        self.ie = ie.half()

    def forward(self, x):
        return self.ie(x.half(), with_point=False, with_label=True)[1]


class ImageEncoderAdapter(torch.nn.Module):
    """Keeps VISTA3D's call signature image_encoder(x, with_point, with_label) -> (point, label)."""

    def __init__(self, label_enc):
        super().__init__()
        self.label_enc = label_enc

    def forward(self, x, with_point=True, with_label=True):
        assert not with_point, "TensorRT build covers the automatic (label prompt) branch only"
        return None, self.label_enc(x)


PRE = Compose([
    LoadImaged(keys="image", image_only=True),
    EnsureChannelFirstd(keys="image"),
    EnsureTyped(keys="image", device=DEV, track_meta=True),
    Spacingd(keys="image", pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
    CropForegroundd(keys="image", allow_smaller=True, margin=10, source_key="image"),
    ScaleIntensityRanged(keys="image", a_min=-963.8247715525971, a_max=1053.678477684517,
                         b_min=0, b_max=1, clip=True),
    Orientationd(keys="image", axcodes="RAS"),
    CastToTyped(keys="image", dtype=torch.float32),
])
POST = Compose([
    VistaPostTransformd(keys="pred"),
    Invertd(keys="pred", transform=PRE, orig_keys="image", nearest_interp=True, to_tensor=True),
])
# same as the official Vista3dInferer when no point prompts are given
INFER = SlidingWindowInfererAdapt(roi_size=(128, 128, 128), sw_batch_size=1, with_coord=True,
                                  padding_mode="replicate")


INVERT = Invertd(keys="pred", transform=PRE, orig_keys="image", nearest_interp=True, to_tensor=True)


def _starts(n, roi, step):
    """window start positions, same rule as MONAI dense_patch_slices"""
    if n <= roi:
        return [0]
    k = int(np.ceil((n - roi) / step)) + 1
    return [min(i * step, n - roi) for i in range(k)]


@torch.no_grad()
def gpu_window_infer(x, net, cls, roi=128, overlap=0.25, budget_gb=None):
    """Sliding window with the 117-class logit sum kept on the GPU in fp16.

    Same windows and constant blending as MONAI's SlidingWindowInfererAdapt, but the final
    label is argmax(sum of logits) with background where the max is <= 0. Dividing by the
    window count changes neither the sign nor the argmax, so it is skipped. MONAI keeps an
    fp32 buffer of 117 x volume; for a whole-body CT that does not fit in 16 GB and it falls
    back to host memory (5x slower). If even fp16 does not fit, the volume is done in z-slabs
    that each use every window touching them, so the result is identical.
    """
    D0, H0, W0 = x.shape[2:]
    pad = [max(0, roi - s) for s in (D0, H0, W0)]
    if any(pad):
        x = torch.nn.functional.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]), mode="replicate")
    D, H, W = x.shape[2:]
    step = int(roi * (1 - overlap))
    zs, ys, xs = (_starts(n, roi, step) for n in (D, H, W))
    K, ev = cls.shape[0], cls[:, 0]
    lab = torch.zeros((D, H, W), dtype=torch.uint8, device=x.device)
    if budget_gb is None:  # what is free now, minus headroom for one window's activations
        torch.cuda.empty_cache()
        budget_gb = max(0.5, torch.cuda.mem_get_info()[0] / 1e9 - 3.0)
    slab = max(4, min(D, int(budget_gb * 1e9 // (K * H * W * 2))))  # a slab may be thinner than a window
    for a in range(0, D, slab):
        b = min(D, a + slab)
        buf = torch.zeros((K, b - a, H, W), dtype=torch.float16, device=x.device)
        for z in zs:
            if z + roi <= a or z >= b:
                continue
            z0, z1 = max(z, a), min(z + roi, b)
            for y in ys:
                for w in xs:
                    out = net(x[..., z:z + roi, y:y + roi, w:w + roi], class_vector=cls,
                              prompt_class=cls, transpose=True)[0]
                    buf[:, z0 - a:z1 - a, y:y + roi, w:w + roi] += out[:, z0 - z:z1 - z].half()
        for c in range(0, b - a, 16):
            mx, am = buf[:, c:c + 16].max(0)
            lab[a + c:a + c + mx.shape[0]] = torch.where(mx > 0, ev[am], 0).to(torch.uint8)
        del buf
    return lab[:D0, :H0, :W0]


@torch.no_grad()
def segment(net, path, stitch="gpu"):
    t = {}
    torch.cuda.reset_peak_memory_stats()
    t0 = sync()
    data = PRE({"image": path})
    t1 = sync(); t["load_preprocess_s"] = t1 - t0
    img = data["image"]
    cls = torch.tensor(EVERYTHING, device=DEV).unsqueeze(-1)
    if stitch == "gpu":
        with torch.autocast("cuda"):
            lab = gpu_window_infer(img.as_tensor().unsqueeze(0), net, cls)
        t2 = sync(); t["inference_s"] = t2 - t1
        pred = img.clone()
        pred.set_array(lab.unsqueeze(0).float())
        lab = INVERT({"image": img, "pred": pred})["pred"][0].to(torch.uint8).cpu().numpy()
    else:  # reference path: MONAI sliding window + official post-transform
        with torch.autocast("cuda"):
            pred = INFER(img.unsqueeze(0), net, transpose=True, point_coords=None, point_labels=None,
                         class_vector=cls, prompt_class=cls, prev_mask=None, labels=None, label_set=None)
        t2 = sync(); t["inference_s"] = t2 - t1
        out = POST({"image": img, "pred": pred[0], "label_prompt": cls})
        lab = out["pred"][0].to(torch.uint8).cpu().numpy()
    t3 = sync(); t["postprocess_s"] = t3 - t2
    t["total_s"] = t3 - t0
    t["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
    t["model_grid"] = list(img.shape[1:])
    return lab, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cts", nargs="+")
    ap.add_argument("--out", default=os.path.join(HERE, "out/pred"))
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--trt", action="store_true", help="TensorRT engine for the image encoder (MONAI trt_compile)")
    ap.add_argument("--stitch", default="gpu", choices=["gpu", "monai"])
    ap.add_argument("--trt-scope", default="image_encoder", choices=["encoder", "image_encoder"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.perf_counter()
    net = build_model()
    if args.trt:
        # TensorRT 11 is strongly typed: there is no fp16 builder flag any more, the precision
        # must live in the ONNX graph. So the encoder itself is made fp16 before export and the
        # engine is built with precision="fp32" (= no flags) and no precision constraints.
        from monai.networks import trt_compile
        patch_monai_trt()
        os.makedirs(os.path.join(HERE, "models/trt"), exist_ok=True)
        trt_args = {"dynamic_batchsize": [1, 1, 1], "precision": "fp32",
                    "build_args": {"builder_optimization_level": 5}}
        if args.trt_scope == "encoder":  # what the official bundle compiles
            print("[trt] split InstanceNorm layers:", split_instance_norms(net.image_encoder.encoder), flush=True)
            net.image_encoder.encoder = HalfEncoder(net.image_encoder.encoder)
            trt_compile(net, os.path.join(HERE, "models/trt/vista3d_fp16"), args=trt_args,
                        submodule=["image_encoder.encoder"])
        else:  # encoder + label-branch decoder in one engine
            print("[trt] split InstanceNorm layers:", split_instance_norms(net.image_encoder), flush=True)
            net.image_encoder = ImageEncoderAdapter(LabelImageEncoder(net.image_encoder))
            trt_compile(net, os.path.join(HERE, "models/trt/vista3d_full_fp16"), args=trt_args,
                        submodule=["image_encoder.label_enc"])
    print(f"model loaded in {time.perf_counter() - t0:.1f} s", flush=True)
    for path in args.cts:
        case = os.path.basename(os.path.dirname(path))
        runs = []
        try:
            for r in range(args.repeat):
                lab, t = segment(net, path, args.stitch)
                runs.append(t)
                print(case, r, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in t.items()}, flush=True)
        except torch.OutOfMemoryError as e:
            print(case, "OOM", str(e)[:120], flush=True)
            torch.cuda.empty_cache()
            continue
        ref = nib.load(path)
        nib.save(nib.Nifti1Image(lab, ref.affine), os.path.join(args.out, f"{case}.nii.gz"))
        json.dump({"case": case, "runs": runs, "labels_found": int(len(np.unique(lab)) - 1)},
                  open(os.path.join(args.out, f"{case}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
