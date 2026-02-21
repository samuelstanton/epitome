# Epitome

Predicts ChIP-seq peaks (transcription factor binding, histone modifications) in novel cell types from chromatin accessibility (DNase-seq or ATAC-seq).

This is a fork of [YosefLab/epitome](https://github.com/YosefLab/epitome) ported to PyTorch and modernised for Python 3.9+. The core method is unchanged — see the original repo and [Morrow et al., NAR 2021](https://doi.org/10.1093/nar/gkab676) for details.

## Installation

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/samuelstanton/epitome
cd epitome
uv sync --dev
```

## Quick start

```python
from epitome.dataset import EpitomeDataset
from epitome.models import EpitomeModel

# Define targets and training cell types
dataset = EpitomeDataset(
    targets=['CTCF', 'RAD21', 'SMC3'],
    cells=['K562', 'A549', 'GM12878'],
)

# Train (device auto-selected: MPS > CUDA > CPU)
model = EpitomeModel(dataset, test_celltypes=['K562'])
model.train(5000, val_every=500)   # logs val loss to ~/.epitome/experiments/

# Evaluate on held-out test chromosomes
results = model.test(1000, calculate_metrics=True)
print(results['auROC'], results['auPRC'])
```

## Scoring new samples

```python
# Score specific genomic regions
results = model.score_peak_file(
    ['/path/to/atac.bed'],      # chromatin accessibility peaks
    '/path/to/regions.bed',     # regions to score
)

# Whole-genome scan
model.score_whole_genome(['/path/to/atac.bed'], 'output_prefix')
```

## Hyperparameter tuning

Local sweep (uses early stopping to find optimal step count):

```python
from epitome.tuning import tune

results = tune(dataset, lr_values=[1e-4, 1e-3, 1e-2], max_train_batches=5000)
best = results[0]  # sorted by val loss
print(f"Best lr={best['lr']:.0e}, retrain with model.train({best['stopped_at']})")
```

Parallel sweep on Modal (survives laptop disconnect):

```bash
# install modal first: uv add modal --dev && modal setup
modal run --detach modal_sweep.py \
    --targets CTCF,RAD21,SMC3 \
    --lr-values "1e-3,1e-2,4e-2" \
    --max-train-batches 10000 \
    --group my_sweep
```

Results land at `~/.epitome/sweeps/<group>.json` (local) or on the `epitome-cache` Modal volume.

## Key options

| Parameter | Default | Description |
|---|---|---|
| `lr` | `1e-3` | Peak learning rate |
| `warmup_steps` | `0` | Linear LR warmup before cosine decay |
| `min_lr` | `0.` | Cosine decay floor |
| `device` | auto | `"cpu"`, `"cuda"`, or `"mps"` |
| `similarity_kernel` | `"dot_agree"` | `"dot_agree"` or `"jaccard"` |
| `group` | `None` | Tag for grouping experiment logs |

## Experiment logging

Every run writes a JSONL file to `~/.epitome/experiments/<timestamp>_<hash>.jsonl` with `config`, `train_step`, `val_loss`, `train_complete`, and `eval` records. Use the `group` parameter to tag related runs for easy querying.

## Tests

```bash
uv run pytest epitome/test/
```
