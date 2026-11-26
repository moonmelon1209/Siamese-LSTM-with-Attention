"""Select one threshold by the equally weighted mean of language-pair Macro-F1."""
import numpy as np

from original_fixed.util import f1_scores


def calibrate(truth_by_group, scores_by_group):
    percents = sorted(range(101), key=lambda value: abs(value - 50))
    thresholds = np.asarray(percents, dtype=np.float32) / 100
    group_curves = []
    for group, truth in truth_by_group.items():
        scores = scores_by_group[group]
        if not np.isfinite(scores).all():
            raise ValueError(f"{group}: non-finite validation score")
        predicted = scores[:, None] >= thresholds
        positive = truth[:, None] == 1
        tp = (predicted & positive).sum(axis=0)
        tn = (~predicted & ~positive).sum(axis=0)
        # For each class, F1 = 2 * correct / (predicted count + true count).
        f1s = []
        for correct, denominator in ((tp, predicted.sum(axis=0) + positive.sum()),
                                     (tn, (~predicted).sum(axis=0) + (~positive).sum())):
            f1s.append(np.divide(2 * correct, denominator, out=np.zeros(len(thresholds)), where=denominator != 0))
        group_curves.append(np.mean(f1s, axis=0))
    means = np.mean(group_curves, axis=0)
    best = int(np.argmax(means))
    threshold = percents[best] / 100
    details = {g: f1_scores(y, scores_by_group[g] >= threshold) for g, y in truth_by_group.items()}
    return {"mean_macro_f1": float(means[best]), "threshold": threshold, "by_group": details}


def group_scores(all_scores, groups):
    by_group = {}
    offset = 0
    for name, rows in groups.items():
        by_group[name] = all_scores[offset:offset + len(rows)]
        offset += len(rows)
    if offset != len(all_scores):
        raise ValueError("Prediction count differs from the number of input pairs.")
    return by_group
