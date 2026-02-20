"""
MPS smoke test: verifies that EpitomeNet forward/backward pass works on Apple Silicon GPU.
"""

import itertools
import numpy as np
import torch

device = torch.device("mps")
print(f"Device: {device}")

# ── dataset ──────────────────────────────────────────────────────────────────
from epitome.dataset import EpitomeDataset
from epitome.constants import Dataset

data_dir = "epitome/test/data"
assembly = "test"

dataset = EpitomeDataset(
    targets=["CTCF"],
    cells=["K562", "HepG2", "H1"],
    data_dir=data_dir,
    assembly=assembly,
)
print(f"Dataset: {len(dataset.cellmap)} cell types, {len(dataset.targetmap)} targets")

# ── model ─────────────────────────────────────────────────────────────────────
from epitome.models import EpitomeModel

model = EpitomeModel(dataset, test_celltypes=["K562"], batch_size=32)
model.model.to(device)
print(f"EpitomeNet on {device}: {sum(p.numel() for p in model.model.parameters())} params")

# ── forward pass ──────────────────────────────────────────────────────────────
batch = next(iter(model.train_iter))
features = batch[0].to(device)
labels   = batch[1].to(device)
weights  = batch[2].to(device)

model.model.eval()
with torch.no_grad():
    logits = model.model(features)
    preds  = torch.sigmoid(logits)

print(f"Forward pass OK  — logits {logits.shape}, device {logits.device}")

# ── backward pass ─────────────────────────────────────────────────────────────
import torch.nn.functional as F

model.model.train()
optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-3)

losses = []
for batch in itertools.islice(model.train_iter, 10):
    features = batch[0].to(device)
    labels   = batch[1].to(device)
    weights  = batch[2].to(device)

    optimizer.zero_grad()
    logits = model.model(features)
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    total = (loss * weights).sum()
    total.backward()
    optimizer.step()
    losses.append(total.item())

print(f"Backward pass OK — 10 steps, loss {losses[0]:.4f} → {losses[-1]:.4f}")
print("✓ MPS smoke test passed")
