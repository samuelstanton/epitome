"""
Hyperparameter tuning for Epitome models.

Sweeps learning rate values, using early stopping against the TRAIN_VALID
split to find the optimal stopping step for each, then ranks runs by their
minimum val_loss on the held-out VALID split.
"""

import json
import logging
from typing import List, Optional, Sequence

logger = logging.getLogger('epitome')


def _best_val_loss(log_path: str) -> float:
    """Return the minimum val_loss recorded in an experiment JSONL file."""
    losses = []
    with open(log_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec['event'] == 'val_loss':
                losses.append(rec['loss'])
    return min(losses) if losses else float('inf')


def tune(
    dataset,
    lr_values: Sequence[float] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
    max_train_batches: int = 5000,
    max_valid_batches: int = 100,
    val_every: int = 200,
    val_batches: int = 50,
    patience: int = 5,
    min_delta: float = 0.001,
    group: Optional[str] = None,
    **model_kwargs,
) -> List[dict]:
    """
    Sweep over ``lr_values``, using early stopping to find the optimal number
    of training steps for each, and rank runs by minimum val_loss on the
    held-out VALID split.

    Parameters
    ----------
    dataset : EpitomeDataset
    lr_values : sequence of float
        Learning rates to evaluate. Default: (1e-4, 3e-4, 1e-3, 3e-3, 1e-2).
    max_train_batches : int
        Upper bound on training steps per run (default 5000).
    max_valid_batches : int
        Batches used per early-stopping check on TRAIN_VALID (default 100).
    val_every : int
        How often to evaluate on the VALID split (default 200).
    val_batches : int
        Batches used for each VALID evaluation (default 50).
    patience : int
        Early-stopping patience in units of early-stopping checks (default 5).
    min_delta : float
        Minimum improvement required to reset early-stopping counter (default 0.001).
    group : str, optional
        Experiment group tag shared across all runs in this sweep.
        Auto-generated as ``tune_<run_id>`` if None.
    **model_kwargs
        Passed through to ``EpitomeModel`` (e.g. ``test_celltypes``, ``radii``,
        ``warmup_steps``, ``batch_size``, ...).

    Returns
    -------
    list[dict]
        One entry per lr value, sorted by ``best_val_loss`` ascending. Each dict:

        ============  =================================================
        lr            learning rate used
        best_batch    batch at which early stopping found the best model
        stopped_at    batch at which training actually stopped
        best_val_loss minimum val_loss recorded on the VALID split
        run_id        experiment run ID
        log_path      path to the JSONL experiment log
        ============  =================================================

    The best configuration is ``results[0]``. To retrain with it::

        best = results[0]
        model = EpitomeModel(dataset, lr=best['lr'], **model_kwargs)
        model.train(best['stopped_at'])
    """
    from .models import EpitomeModel
    from .experiment import _make_run_id

    if group is None:
        group = f"tune_{_make_run_id()}"

    logger.info(
        "Starting LR sweep group=%s lr_values=%s max_train_batches=%d",
        group, list(lr_values), max_train_batches,
    )

    results = []

    for lr in lr_values:
        logger.info("sweep lr=%.2e group=%s", lr, group)

        model = EpitomeModel(
            dataset,
            lr=lr,
            max_valid_batches=max_valid_batches,
            group=group,
            **model_kwargs,
        )
        best_batch, stopped_at, _ = model.train(
            max_train_batches,
            patience=patience,
            min_delta=min_delta,
            val_every=val_every,
            val_batches=val_batches,
        )
        model.experiment.close()

        best_val = _best_val_loss(model.experiment.log_path)

        results.append({
            'lr': lr,
            'best_batch': best_batch,
            'stopped_at': stopped_at,
            'best_val_loss': best_val,
            'run_id': model.experiment.run_id,
            'log_path': model.experiment.log_path,
        })
        logger.info(
            "lr=%.2e best_val_loss=%.4f stopped_at=%d run_id=%s",
            lr, best_val, stopped_at, model.experiment.run_id,
        )

    results.sort(key=lambda x: x['best_val_loss'])

    best = results[0]
    logger.info(
        "Best: lr=%.2e best_val_loss=%.4f stopped_at=%d run_id=%s",
        best['lr'], best['best_val_loss'], best['stopped_at'], best['run_id'],
    )

    return results
