"""Evaluate the saved multilingual checkpoint on each held-out language pair."""
import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

# Support both "python improved/script.py" and "python -m improved.script".
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from improved.data import DEFAULT_RUN, MANIFEST, encode_split, fingerprint, load_split
from improved.metrics import group_scores
from original_fixed.util import f1_scores, labels


def parse_args(description, default_run):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--run-dir", type=Path, default=default_run)
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--split-manifest", type=Path, default=MANIFEST, help="Split manifest used for this experiment")
    return parser.parse_args()


def evaluate_model(args, load_run):
    from original_fixed.model import configure_device

    configure_device(args.device)
    model, state = load_run(args.run_dir)
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if fingerprint(manifest) != state["split_fingerprint"]:
        raise ValueError("The current splits do not match the splits recorded for this checkpoint.")
    groups = load_split("test", manifest)
    inputs = encode_split(groups, state["vocabulary"], state["architecture"]["sequence_length"])
    raw = model.predict(inputs, batch_size=256, verbose=0).ravel()
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite test scores")
    scores = group_scores(raw, groups)
    threshold = state["validation"]["threshold"]
    report = {"threshold": threshold, "by_group": {}}
    print(f"Threshold selected on validation data: {threshold:.2f}")
    with TemporaryDirectory(dir=args.run_dir) as temporary:
        stage = Path(temporary)
        for name, rows in groups.items():
            predicted = (scores[name] >= threshold).astype(int)
            metrics = f1_scores(labels(rows), predicted)
            report["by_group"][name] = {"n": len(rows), **metrics}
            print(f"{name} (n={len(rows)}): Macro-F1={metrics['macro_f1']:.4f}, duplicate F1={metrics['duplicate_f1']:.4f}")
            with (stage / f"{name}_test_predictions.tsv").open("w", encoding="utf-8", newline="") as output:
                writer = csv.writer(output, delimiter="\t")
                writer.writerow(["row_id", "question1", "question2", "is_duplicate", "score", "prediction"])
                for row, score, decision in zip(rows, scores[name], predicted):
                    writer.writerow([row["row_id"], row["question1"], row["question2"], row["is_duplicate"],
                                     repr(float(score)), decision])
        report["mean_macro_f1"] = float(np.mean([v["macro_f1"] for v in report["by_group"].values()]))
        for prediction_file in stage.glob("*.tsv"):
            prediction_file.replace(args.run_dir / prediction_file.name)
        summary = stage / "test_metrics.json"
        summary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        summary.replace(args.run_dir / summary.name)
    print(f"Mean of the three Macro-F1 scores: {report['mean_macro_f1']:.4f}")
    print(f"Results: {args.run_dir.resolve()}")


def main():
    args = parse_args(__doc__, DEFAULT_RUN)
    from improved.model import load_run

    evaluate_model(args, load_run)


if __name__ == "__main__":
    main()
