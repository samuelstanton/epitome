"""
Experiment tracking for Epitome training runs.

Each run gets a unique ID (UTC timestamp + 8-char hex hash) and a JSONL log
file.  Every record is a self-contained JSON object on its own line with at
minimum the fields ``run_id``, ``ts`` (ISO-8601 UTC), and ``event``.

Defined event types
-------------------
config          – emitted once at model initialisation
train_step      – emitted every ``log_every`` training batches
valid_loss      – emitted at each early-stopping validation check
train_complete  – emitted when ``model.train()`` returns
eval            – emitted by ``model.test()`` / ``run_predictions()``
"""

import json
import os
import uuid
from datetime import datetime, timezone

import numpy as np


DEFAULT_LOG_DIR = os.path.join(os.path.expanduser('~'), '.epitome', 'experiments')


def _make_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    hash8 = uuid.uuid4().hex[:8]
    return f"{ts}_{hash8}"


class _JsonEncoder(json.JSONEncoder):
    """Handles numpy scalars and arrays so they serialise cleanly."""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class Experiment:
    """
    Tracks a single Epitome training run by writing structured JSONL records.

    Parameters
    ----------
    log_dir : str, optional
        Directory to write the ``.jsonl`` file.  Defaults to
        ``~/.epitome/experiments``.

    Examples
    --------
    Auto-created by the model (default usage)::

        model = EpitomeModel(dataset)
        print(model.experiment.run_id)    # e.g. '20240601_143022_a3f9c1b8'
        print(model.experiment.log_path)  # path to the JSONL file

    Explicit construction (custom log dir)::

        exp = Experiment(log_dir='/data/logs')
        model = EpitomeModel(dataset, experiment=exp)
    """

    def __init__(self, log_dir: str = None, group: str = None):
        self.run_id = _make_run_id()
        self.group = group
        self.log_dir = log_dir or DEFAULT_LOG_DIR
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, f"{self.run_id}.jsonl")
        self._file = open(self.log_path, 'w')

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write(self, record: dict):
        record['run_id'] = self.run_id
        if self.group is not None:
            record['group'] = self.group
        record['ts'] = datetime.now(timezone.utc).isoformat()
        self._file.write(json.dumps(record, cls=_JsonEncoder) + '\n')
        self._file.flush()

    # ------------------------------------------------------------------
    # Public logging methods
    # ------------------------------------------------------------------

    def log_config(self, **kwargs):
        """Record model and dataset configuration at initialisation."""
        self._write({'event': 'config', **kwargs})

    def log_train_step(self, batch: int, loss: float):
        """Record training loss at a given batch."""
        self._write({'event': 'train_step', 'batch': batch, 'loss': round(float(loss), 6)})

    def log_valid_loss(self, batch: int, loss: float):
        """Record early-stopping validation loss."""
        self._write({'event': 'valid_loss', 'batch': batch, 'loss': round(float(loss), 6)})

    def log_train_complete(self, best_batch: int, total_batches: int, duration_s: float):
        """Record summary at the end of a training run."""
        self._write({
            'event': 'train_complete',
            'best_batch': best_batch,
            'total_batches': total_batches,
            'duration_s': round(duration_s, 3),
        })

    def log_eval(self, mode: str, n_samples: int, auROC, auPRC, per_target):
        """
        Record evaluation metrics.

        Parameters
        ----------
        mode       : 'train', 'valid', or 'test'
        n_samples  : number of samples evaluated
        auROC      : macro-average auROC across targets (float or None)
        auPRC      : macro-average auPRC across targets (float or None)
        per_target : dict mapping target name -> {'AUC', 'auPRC', 'GINI'}
        """
        self._write({
            'event': 'eval',
            'mode': mode,
            'n_samples': n_samples,
            'auROC': auROC,
            'auPRC': auPRC,
            'per_target': per_target,
        })

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        if not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return f"Experiment(run_id={self.run_id!r}, log_path={self.log_path!r})"
