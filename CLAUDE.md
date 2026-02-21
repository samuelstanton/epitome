# Epitome — coding agent orientation

## What this repo does

Predicts ChIP-seq peaks (TF binding, histone marks) in novel cell types by transferring signal from ENCODE reference cell types, using chromatin accessibility (DNase/ATAC-seq) as the bridge. The key idea: if your new cell type has similar open-chromatin patterns to K562 near a locus, it probably has similar CTCF binding there too.

## Stack

- **Python 3.11+**, **PyTorch**, packaged with **uv** (`pyproject.toml`)
- No TensorFlow anywhere — this is a fork of YosefLab/epitome ported to PyTorch
- Tests: `uv run pytest epitome/test/` — must stay green
- Experiments run on **Modal** (`modal_sweep.py`) with T4 GPUs

## Key files

| File | Purpose |
|---|---|
| `epitome/dataset.py` | `EpitomeDataset` — loads `~/.epitome/data/<assembly>/data.h5`, builds `cellmap`/`targetmap`/`matrix` |
| `epitome/generators.py` | `load_data()` generator + `build_dataloader()` → PyTorch `DataLoader` |
| `epitome/models.py` | `EpitomeNet` (nn.Module), `PeakModel`, `EpitomeModel` |
| `epitome/experiment.py` | JSONL experiment logging to `~/.epitome/experiments/` |
| `epitome/tuning.py` | `tune()` — local LR sweep using early stopping |
| `epitome/metrics.py` | `get_performance()`, `gini_normalized()` — pure numpy |
| `epitome/conversion.py` | `RegionConversion` — maps user bed files to dataset genomic regions |
| `modal_sweep.py` | Modal parallel LR sweep (`run_trial`, `run_sweep`, `sweep` entrypoint) |

## Data model

```
data.h5
├── /data          int8 (n_experiments × n_genomic_bins)  — 1=peak, 0=no peak
├── /rows/celltypes, /rows/targets  — row labels
├── /columns/chr, start, binSize   — 200bp bins across hg19/hg38
└── /columns/index/TRAIN,VALID,TEST — chromosome-based splits
```

`EpitomeDataset.matrix[cellmap[cell], targetmap[target]]` = row index in `/data`, or `-1` if that experiment doesn't exist. Missing entries are masked out of the loss.

Train/val/test split is **chromosome-based** (default: val=chr7, test=chr8+chr9). Never random.

## Model architecture

Input: flat feature vector per sample (one genomic locus × one label cell type).

For each eval cell type, features = raw binary peaks at locus `i` (one per target) + similarity statistics at 4 radii `[1,3,10,30]` bins (±200–6000bp). Similarity is `dot_agree` (mean(A·B) + mean(A==B)) or `jaccard` (|A∩B|/|A∪B|) per shell. Missing (cell, target) entries are zeroed out by a mask.

`EpitomeNet`: multi-branch MLP — one Dense branch per input group, 2 layers with halving width + Tanh, branches concatenated, single linear output. Model size auto-adjusts to feature vector length.

Loss: `F.binary_cross_entropy_with_logits` × mask (ignores missing data), summed over batch.

## PeakModel constructor params worth knowing

```python
EpitomeModel(
    dataset,
    test_celltypes=[],       # held-out cell types (labels only, not features)
    lr=1e-3,
    warmup_steps=0,          # linear warmup before cosine decay
    min_lr=0.,               # cosine decay floor
    device=None,             # auto: MPS > CUDA > CPU
    num_workers=0,           # DataLoader workers; >0 requires Linux (fork)
    similarity_kernel='dot_agree',  # or 'jaccard'
    group=None,              # experiment log tag
    max_valid_batches=None,  # enables early stopping on held-out train chr
)
```

## train() signature

```python
best_batch, stopped_at, losses = model.train(
    max_train_batches,
    patience=3,        # early stopping patience (requires max_valid_batches)
    min_delta=0.01,
    val_every=500,     # periodic VALID split evaluation
    val_batches=50,
)
```

## Experiment logging

Every `EpitomeModel` auto-creates an `Experiment` that writes NDJSON to `~/.epitome/experiments/<run_id>.jsonl`. Events: `config`, `train_step` (every 1000 batches), `val_loss` (every `val_every`), `valid_loss` (early stopping checks), `train_complete`, `eval`. All records carry `run_id` and UTC `ts`. Use `group=` to tag related runs.

## Modal sweep

`modal_sweep.py` runs parallel LR sweeps on T4 GPUs. The orchestration (`run_sweep`) runs on Modal so it survives laptop disconnection:

```bash
modal run --detach modal_sweep.py \
    --lr-values "1e-3,1e-2,4e-2" \
    --max-train-batches 10000 \
    --group my_sweep \
    --similarity-kernel dot_agree
```

Results saved to Modal volume `epitome-cache` at `sweeps/<group>.json`. Checkpoints at `checkpoints/<run_id>/`. Data cached at `.epitome/data/`.

## Important constraints

- **`num_workers > 0` only works on Linux** (Modal). macOS uses `spawn` multiprocessing which can't pickle the generator closure. Use `num_workers=0` locally.
- **Always call `model.shutdown()`** before a Modal function returns to terminate DataLoader workers; otherwise the container hangs.
- The `train_valid_iter` uses `mode=Dataset.VALID` (not TRAIN) to skip MLSMOTE oversampling, which caused feature size inconsistencies with non-default similarity kernels.
- `similarity_kernel` must be set before DataLoaders are built (it's stored as `self.similarity_kernel` early in `PeakModel.__init__`).

## Conventions

- Operational logging: `logging.getLogger('epitome')` with `NullHandler` (library pattern — callers configure output)
- Structured experiment data: `Experiment.log_*()` → JSONL
- Warnings about data quality: `warnings.warn()`
- No `print()` in library code
