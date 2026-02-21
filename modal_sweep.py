"""
Parallel LR sweep on Modal.

Each trial runs in its own CPU container. All trials launch simultaneously
via starmap, so a 5-point sweep takes as long as one trial.

Usage
-----
    # dry run (print config, don't execute)
    modal run modal_sweep.py --dry-run

    # hg19 sweep with defaults
    modal run modal_sweep.py

    # custom targets / lr grid
    modal run modal_sweep.py \\
        --assembly hg19 \\
        --targets CTCF,RAD21,SMC3 \\
        --test-celltypes K562 \\
        --lr-values "1e-4,3e-4,1e-3,3e-3,1e-2" \\
        --max-train-batches 5000 \\
        --group hg19_lr_sweep_v1
"""

import json
from pathlib import Path
from typing import Optional

import modal

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = modal.App("epitome-tune")

# Persistent volume: caches downloaded ENCODE data and experiment JSONL logs
# between runs so we don't re-download on every sweep.
volume = modal.Volume.from_name("epitome-cache", create_if_missing=True)
VOLUME_PATH = "/epitome-cache"

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

# CUDA torch for T4 GPU.
_torch = "pip install torch --index-url https://download.pytorch.org/whl/cu124"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .run_commands(_torch)
    .pip_install(
        "numpy",
        "pyyaml",
        "seaborn>=0.11.2",
        "matplotlib>=3.4.3",
        "scikit-learn",
        "tqdm",
        "pyranges>=0.0.104",
        "h5py",
        "pandas",
        "scipy",
        "requests",
    )
    # Install epitome from source so the container picks up local changes.
    .add_local_dir(".", remote_path="/app/epitome", copy=True, ignore=["**/.venv", "**/__pycache__", "**/.git"])
    .run_commands("pip install -e /app/epitome --no-deps --quiet")
)

# ---------------------------------------------------------------------------
# Trial function (runs in container)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    gpu="T4",
    timeout=7200,
)
def run_trial(
    lr: float,
    assembly: str,
    targets: list,
    cells: Optional[list],
    test_celltypes: list,
    max_train_batches: int,
    max_valid_batches: int,
    val_every: int,
    val_batches: int,
    test_batches: int,
    patience: int,
    min_delta: float,
    warmup_steps: int,
    min_lr: float,
    batch_size: int,
    group: str,
    similarity_kernel: str,
) -> dict:
    import os
    from pathlib import Path

    # Redirect ~/.epitome to the shared volume so data is cached across runs.
    os.environ["HOME"] = VOLUME_PATH

    from epitome.dataset import EpitomeDataset
    from epitome.constants import Dataset
    from epitome.models import EpitomeModel
    from epitome.tuning import _best_val_loss

    dataset = EpitomeDataset(targets=targets, cells=cells or None, assembly=assembly)

    model = EpitomeModel(
        dataset,
        test_celltypes=test_celltypes,
        lr=lr,
        max_valid_batches=max_valid_batches,
        warmup_steps=warmup_steps,
        min_lr=min_lr,
        batch_size=batch_size,
        group=group,
        num_workers=4,
        similarity_kernel=similarity_kernel,
    )

    best_batch, stopped_at, _ = model.train(
        max_train_batches,
        patience=patience,
        min_delta=min_delta,
        val_every=val_every,
        val_batches=val_batches,
    )

    # Evaluate on held-out test chromosomes
    test_results = model.test(
        test_batches * batch_size,
        mode=Dataset.TEST,
        calculate_metrics=True,
    )
    test_auROC = test_results["auROC"]
    test_auPRC = test_results["auPRC"]

    # Save checkpoint to volume
    checkpoint_path = str(Path(VOLUME_PATH) / "checkpoints" / model.experiment.run_id)
    model.save(checkpoint_path)

    best_val = _best_val_loss(model.experiment.log_path)
    log = open(model.experiment.log_path).read()
    run_id = model.experiment.run_id
    model.experiment.close()
    model.shutdown()  # terminate DataLoader workers before returning
    volume.commit()  # persist checkpoint + JSONL logs

    return {
        "lr": lr,
        "best_batch": best_batch,
        "stopped_at": stopped_at,
        "best_val_loss": best_val,
        "test_auROC": test_auROC,
        "test_auPRC": test_auPRC,
        "run_id": run_id,
        "checkpoint_path": checkpoint_path,
        "log": log,
    }


# ---------------------------------------------------------------------------
# Sweep orchestrator — runs on Modal so it survives laptop disconnection
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=86400,  # 24 hours
)
def run_sweep(
    parsed_lr: list,
    assembly: str,
    parsed_targets: list,
    parsed_cells: list,
    parsed_test: list,
    max_train_batches: int,
    max_valid_batches: int,
    val_every: int,
    val_batches: int,
    test_batches: int,
    patience: int,
    min_delta: float,
    warmup_steps: int,
    min_lr: float,
    batch_size: int,
    sweep_group: str,
    similarity_kernel: str,
) -> dict:
    from pathlib import Path

    args = [
        (lr, assembly, parsed_targets, parsed_cells, parsed_test,
         max_train_batches, max_valid_batches, val_every, val_batches,
         test_batches, patience, min_delta, warmup_steps, min_lr, batch_size,
         sweep_group, similarity_kernel)
        for lr in parsed_lr
    ]

    print(f"Launching {len(args)} trials in parallel…")
    raw = list(run_trial.starmap(args, return_exceptions=True))

    results, failures = [], []
    for arg, outcome in zip(args, raw):
        if isinstance(outcome, Exception):
            failures.append({"lr": arg[0], "error": str(outcome)})
            print(f"  FAILED lr={arg[0]:.0e}: {outcome}")
        else:
            results.append(outcome)

    if not results:
        raise RuntimeError("All trials failed: " + str(failures))

    results.sort(key=lambda r: r["best_val_loss"])

    # Write JSONL logs to volume
    log_dir = Path(VOLUME_PATH) / "experiments"
    log_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        (log_dir / f"{r['run_id']}.jsonl").write_text(r["log"])

    # Save summary JSON to volume
    summary = {
        "group": sweep_group,
        "best": {k: v for k, v in results[0].items() if k != "log"},
        "results": [{k: v for k, v in r.items() if k != "log"} for r in results],
    }
    summary_path = Path(VOLUME_PATH) / "sweeps" / f"{sweep_group}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    volume.commit()

    return summary


# ---------------------------------------------------------------------------
# Local entrypoint — thin launcher, safe to disconnect after submission
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def sweep(
    assembly: str = "hg19",
    targets: str = "CTCF,RAD21,SMC3",
    cells: str = "",
    test_celltypes: str = "K562",
    lr_values: str = "1e-4,3e-4,1e-3,3e-3,1e-2",
    max_train_batches: int = 5000,
    max_valid_batches: int = 100,
    val_every: int = 500,
    val_batches: int = 50,
    test_batches: int = 200,
    patience: int = 5,
    min_delta: float = 0.001,
    warmup_steps: int = 200,
    min_lr: float = 0.0,
    batch_size: int = 1024,
    group: str = "",
    similarity_kernel: str = "dot_agree",
    dry_run: bool = False,
):
    from epitome.experiment import _make_run_id

    parsed_lr      = [float(x.strip()) for x in lr_values.split(",")]
    parsed_targets = [x.strip() for x in targets.split(",")]
    parsed_cells   = [x.strip() for x in cells.split(",")] if cells else []
    parsed_test    = [x.strip() for x in test_celltypes.split(",")]
    sweep_group    = group or f"modal_tune_{_make_run_id()}"

    print(f"group         : {sweep_group}")
    print(f"assembly      : {assembly}")
    print(f"targets       : {parsed_targets}")
    print(f"test_celltypes: {parsed_test}")
    print(f"lr_values     : {parsed_lr}")
    print(f"max_train_batches: {max_train_batches}  warmup_steps: {warmup_steps}")
    print(f"similarity_kernel: {similarity_kernel}")

    if dry_run:
        print("\n[dry run] exiting without submitting jobs")
        return

    # run_sweep executes on Modal — safe to close laptop after this returns
    summary = run_sweep.remote(
        parsed_lr, assembly, parsed_targets, parsed_cells, parsed_test,
        max_train_batches, max_valid_batches, val_every, val_batches,
        test_batches, patience, min_delta, warmup_steps, min_lr, batch_size,
        sweep_group, similarity_kernel,
    )

    # Print summary table
    results = summary["results"]
    print(f"\n── Sweep results ({sweep_group}) ──────────────────────────────────────────")
    print(f"{'rank':<6}{'lr':<10}{'val_loss':<14}{'test_auROC':<14}{'test_auPRC':<14}{'stopped_at':<13}best_batch")
    for i, r in enumerate(results):
        auroc = f"{r['test_auROC']:.4f}" if r['test_auROC'] is not None else "n/a"
        auprc = f"{r['test_auPRC']:.4f}" if r['test_auPRC'] is not None else "n/a"
        print(f"{i:<6}{r['lr']:<10.0e}{r['best_val_loss']:<14.4f}{auroc:<14}{auprc:<14}{r['stopped_at']:<13}{r['best_batch']}")

    best = summary["best"]
    print(f"\nBest: lr={best['lr']:.0e}  run_id={best['run_id']}")
    print(f"Checkpoint: {best['checkpoint_path']}")
    print(f"Retrain:    model.train({best['stopped_at']})")
    print(f"Summary on volume: {VOLUME_PATH}/sweeps/{sweep_group}.json")
