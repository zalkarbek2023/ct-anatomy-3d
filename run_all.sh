#!/usr/bin/env bash
# Segment all held-out CTs with PyTorch AMP and with TensorRT, 3 timed runs each, then score.
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CTS=$(tail -n +2 data/test_cases.csv | cut -d, -f1 | sed 's#.*#data/v3/&/ct.nii.gz#')
IDS=$(tail -n +2 data/test_cases.csv | cut -d, -f1)
.venv/bin/python -u segment.py $CTS --repeat 3 --out out/pred_torch
.venv/bin/python -u segment.py $CTS --repeat 3 --trt --out out/pred_trt
.venv/bin/python -u eval.py out/pred_torch $IDS
.venv/bin/python -u eval.py out/pred_trt $IDS
echo "RUN ALL DONE"
