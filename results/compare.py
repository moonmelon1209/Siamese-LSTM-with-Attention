"""Recompute test F1 and paired bootstrap intervals from the published predictions."""
import csv
from pathlib import Path

import numpy as np


RESULTS = Path(__file__).resolve().parent
MODELS = {"A": "english_baseline", "B": "multilingual", "C": "multilingual_attention",
          "B100": "multilingual_full_100", "Half100": "multilingual_half_100"}
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
    multilingual = ["B", "C", "B100", "Half100"]
    contrasts = [("C", "B"), ("B100", "B"), ("B", "Half100")]
    group_scores, group_draws = [], []
    for group, languages in LANGUAGES.items():
        names = multilingual + (["A"] if group == "english" else [])
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
        scores = "  ".join(f"{n}={v:.4f}" for n, v in zip(names, observed))
        print(f"{group} (n={len(rows)}): {scores}")
        for newer, older in contrasts + ([("C", "A")] if group == "english" else []):
            i, j = names.index(newer), names.index(older)
            print(f"  {newer}-{older}={observed[i] - observed[j]:+.4f}  "
                  f"95% CI {interval(draws[:, i] - draws[:, j])}")
        group_scores.append(observed[:len(multilingual)])
        group_draws.append(draws[:, :len(multilingual)])
    means = np.mean(group_scores, axis=0)
    mean_draws = np.mean(group_draws, axis=0)
    print("Mean Macro-F1: " + "  ".join(f"{model}={score:.4f}" for model, score in zip(multilingual, means)))
    for newer, older in contrasts:
        i, j = multilingual.index(newer), multilingual.index(older)
        print(f"Mean {newer}-{older}={means[i] - means[j]:+.4f}  "
              f"95% CI {interval(mean_draws[:, i] - mean_draws[:, j])}")


if __name__ == "__main__":
    main()
