#####################################################################################
## Metric functions for evaluating model.
#####################################################################################

import sklearn.metrics
import numpy as np


def gini(actual, pred, sample_weight):
    sort_idx = np.argsort(-pred)
    df = np.stack([actual, pred], axis=0)[:, sort_idx]
    n = df.shape[1]
    linsp = (np.arange(1, n + 1) / n).astype(np.float32)
    totalPos = np.sum(actual)
    cumPosFound = np.cumsum(df[0])
    Lorentz = cumPosFound / totalPos
    Gini = Lorentz - linsp
    return float(np.sum(Gini[sample_weight.astype(bool)]))


def gini_normalized(actual, pred, sample_weight=None):
    normalized_gini = gini(actual, pred, sample_weight) / gini(actual, actual, sample_weight)
    return normalized_gini


def get_performance(targetmap, preds, truth, sample_weight, predicted_targets):

    assert(preds.shape == truth.shape)
    assert(preds.shape == sample_weight.shape)

    evaluated_targets = {}

    for j in range(preds.shape[1]):  # for all targets
        try:
            roc_score = sklearn.metrics.roc_auc_score(truth[:, j],
                                                      preds[:, j],
                                                      sample_weight=sample_weight[:, j],
                                                      average='macro')
        except ValueError:
            roc_score = np.NaN

        try:
            pr_score = sklearn.metrics.average_precision_score(truth[:, j],
                                                               preds[:, j],
                                                               sample_weight=sample_weight[:, j])
        except ValueError:
            pr_score = np.NaN

        try:
            gini_score = gini_normalized(truth[:, j],
                                         preds[:, j],
                                         sample_weight=sample_weight[:, j])
        except ValueError:
            gini_score = np.NaN

        evaluated_targets[predicted_targets[j]] = {"AUC": roc_score, "auPRC": pr_score, "GINI": gini_score}

    return evaluated_targets
