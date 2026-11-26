"""Recompute test F1 and paired bootstrap intervals from the published predictions."""
import csv
from pathlib import Path

import numpy as np


RESULTS = Path(__file__).resolve().parent
MODELS = {"A": "english_baseline", "B": "multilingual", "C": "multilingual_attention"}
LANGUAGES = {"english": ("en", "en"), "korean": ("ko", "ko"), "cross_en_ko": ("en", "ko")}


def read_predictions(model, group):
    path = RESULTS / MODELS[model] / f"{group}_test_predictions.tsv"
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    ids = [int(row["row_id"]) for row in rows]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError(f"Missing or repeated row IDs in {path}")
    return sorted(rows, key=lambda row: int(row["row_id"]))


def question_groups(rows, languages):
    """Keep pairs that share a question in the same bootstrap sampling unit."""
    parents = list(range(len(rows)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, row in enumerate(rows):
        for language, column in zip(languages, ("question1", "question2")):
            key = (language, " ".join(row[column].casefold().split()))
            if key in seen:
                parents[root(index)] = root(seen[key])
            else:
                seen[key] = index

    clusters = {}
    for index in range(len(rows)):
        clusters.setdefault(root(index), []).append(index)
    return list(clusters.values())


def macro_f1(matrices):
    tn, fp, fn, tp = np.moveaxis(matrices, -1, 0)
    scores = []
    for correct in (tn, tp):
        denominator = 2 * correct + fp + fn
        scores.append(np.divide(2 * correct, denominator, out=np.zeros_like(correct, dtype=float),
                                where=denominator != 0))
    return (scores[0] + scores[1]) / 2


def interval(draws):
    low, high = np.quantile(draws, [0.025, 0.975])
    return f"[{low:+.4f}, {high:+.4f}]"


def main():
    rng = np.random.default_rng(42)
    differences, resamples = [], []
    for group, languages in LANGUAGES.items():
        names = ["B", "C", "A"] if group == "english" else ["B", "C"]
        rows = read_predictions("B", group)
        keys = ("row_id", "question1", "question2", "is_duplicate")
        predictions = []
        for name in names:
            other = rows if name == "B" else read_predictions(name, group)
            if len(rows) != len(other) or any(r[k] != s[k] for r, s in zip(rows, other) for k in keys):
                raise ValueError(f"{group}: {name} test rows differ from B")
            predictions.append(np.array([int(row["prediction"]) for row in other]))
        truth = np.array([int(row["is_duplicate"]) for row in rows])
        clusters = question_groups(rows, languages)
        counts = np.array([[np.bincount(2 * truth[ids] + pred[ids], minlength=4)
                            for pred in predictions] for ids in clusters])
        observed = macro_f1(counts.sum(axis=0))
        draws = []
        for _ in range(100):
            sample = rng.integers(len(clusters), size=(100, len(clusters)))
            draws.append(macro_f1(counts[sample].sum(axis=1)))
        draws = np.concatenate(draws)
        delta = observed[1] - observed[0]
        paired = draws[:, 1] - draws[:, 0]
        resamples.append(paired)
        differences.append(delta)
        scores = "  ".join(f"{n}={v:.4f}" for n, v in zip(names, observed))
        print(f"{group} (n={len(rows)}): {scores}  C-B={delta:+.4f}  95% CI {interval(paired)}")
        if group == "english":
            print(f"  C-A={observed[1] - observed[2]:+.4f}  95% CI {interval(draws[:, 1] - draws[:, 2])}")
    print(f"Mean C-B={np.mean(differences):+.4f}  95% CI {interval(np.mean(resamples, axis=0))}")


if __name__ == "__main__":
    main()
