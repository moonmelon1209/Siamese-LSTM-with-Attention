"""Evaluate the saved English baseline on the fixed, labeled English test set."""
import argparse
import csv
from pathlib import Path
from tempfile import NamedTemporaryFile

from predict import predict_pairs
from util import DATA_DIR, DEFAULT_RUN, f1_scores, labels, read_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    args = parser.parse_args()
    rows = read_pairs(DATA_DIR / "test" / "english_test.txt")
    scores, threshold = predict_pairs(rows, args.run_dir, args.device)
    predicted = (scores >= threshold).astype(int)
    metrics = f1_scores(labels(rows), predicted)

    destination = args.run_dir / "english_test_predictions.tsv"
    with NamedTemporaryFile("w", encoding="utf-8", newline="", dir=args.run_dir, delete=False) as file:
        temp_path = Path(file.name)
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["row_id", "question1", "question2", "is_duplicate", "score", "prediction"])
        for i, (row, score, decision) in enumerate(zip(rows, scores, predicted)):
            writer.writerow([i, row["question1"], row["question2"], row["is_duplicate"], repr(float(score)), decision])
    temp_path.replace(destination)
    print(f"English test pairs: {len(rows)}   validation threshold: {threshold:.2f}")
    print(f"Test Macro-F1:     {metrics['macro_f1']:.4f}")
    print(f"Test duplicate F1: {metrics['duplicate_f1']:.4f}")
    print(f"Predictions: {destination.resolve()}")


if __name__ == "__main__":
    main()
