"""Label-specific validation thresholds (实施方案 section 28).

For each label, pick the threshold maximizing the validation F1 over a fine
grid.  These thresholds are fixed after validation and reused on the internal
test set.
"""
import numpy as np
from sklearn.metrics import f1_score


def select_thresholds(y_true, y_prob, grid=np.arange(0.05, 0.95, 0.01)):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_prob.ndim == 1:
        y_prob = y_prob[:, None]

    n, c = y_true.shape
    thresholds = []
    for j in range(c):
        yt = y_true[:, j]
        yp = y_prob[:, j]
        if len(np.unique(yt)) < 2:
            thresholds.append(0.5)
            continue
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            f1 = f1_score(yt, (yp >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds.append(best_t)
    return np.array(thresholds, dtype=float)
