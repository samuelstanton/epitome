"""
MPS smoke test: verifies that EpitomeNet forward/backward pass works on Apple Silicon GPU.
"""

import itertools
import numpy as np
import torch

# ── dataset ──────────────────────────────────────────────────────────────────
from epitome.dataset import EpitomeDataset
from epitome.constants import Dataset

dataset = EpitomeDataset(
    targets=["CTCF"],
    cells=["K562", "HepG2", "H1"],
    data_dir="epitome/test/data",
    assembly="test",
)
print(f"Dataset: {len(dataset.cellmap)} cell types, {len(dataset.targetmap)} targets")

# ── model (device auto-selected) ──────────────────────────────────────────────
from epitome.models import EpitomeModel

model = EpitomeModel(dataset, test_celltypes=["K562"], batch_size=32)
print(f"Params: {sum(p.numel() for p in model.model.parameters())}")

# ── forward pass ──────────────────────────────────────────────────────────────
results = model.test(32)
preds = results["preds"]
print(f"Forward pass OK  — preds {preds.shape}, mean={preds.mean():.4f}")

# ── training steps ────────────────────────────────────────────────────────────
_, steps, _ = model.train(10)
print(f"Backward pass OK — trained {steps} steps")

results2 = model.test(32)
print(f"Post-train preds — mean={results2['preds'].mean():.4f}")

print(f"✓ MPS smoke test passed (device={model.device})")
