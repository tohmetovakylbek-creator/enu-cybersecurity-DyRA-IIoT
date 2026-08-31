#!/usr/bin/env python3
"""Export the real Vanilla-Transformer (baselines_v2.VanillaTransformer) to ONNX FP32."""
import torch
from baselines_v2 import VanillaTransformer

CKPT = "artifacts/checkpoints/transformer_seed42_best.pt"
OUT  = "artifacts/onnx/transformer_seed42_best_fp32.onnx"

# 1) build the model with the exact training config
model = VanillaTransformer(seq_len=50, num_features=36, d_model=128, num_heads=4, dropout=0.1)

# 2) load weights STRICTLY (so any mismatch raises instead of being swallowed)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model") or ckpt
model.load_state_dict(state, strict=True)
model.eval()

n = sum(p.numel() for p in model.parameters())
print(f"loaded VanillaTransformer OK — parameters: {n:,}")

# 3) sanity: output must be a per-window probability in [0,1]
with torch.no_grad():
    y = model(torch.zeros(4, 50, 36))
print(f"output shape: {tuple(y.shape)}   range: [{y.min():.3f}, {y.max():.3f}]")

# 4) export
torch.onnx.export(
    model, torch.zeros(1, 50, 36), OUT,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=18,
)
print(f"exported: {OUT}")