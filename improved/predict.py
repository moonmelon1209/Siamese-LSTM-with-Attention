"""Score one pair using explicit question languages and a saved multilingual checkpoint."""
import argparse
from pathlib import Path

import numpy as np

# Support both "python improved/script.py" and "python -m improved.script".
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from improved.data import DEFAULT_RUN, encode_pairs


def parse_args(description, default_run):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--questions", nargs=2, required=True, metavar=("QUESTION1", "QUESTION2"))
    parser.add_argument("--languages", nargs=2, choices=["en", "ko"], required=True)
    parser.add_argument("--run-dir", type=Path, default=default_run)
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    return parser.parse_args()


def predict_pair(args, load_run):
    from original_fixed.model import configure_device

    configure_device(args.device)
    model, state = load_run(args.run_dir)
    row = dict(zip(("question1", "question2"), args.questions))
    inputs = encode_pairs([row], args.languages, state["vocabulary"], state["architecture"]["sequence_length"])
    score = float(model.predict(inputs, verbose=0).item())
    if not np.isfinite(score):
        raise ValueError("Non-finite similarity")
    threshold = state["validation"]["threshold"]
    print(f"Similarity: {score:.6f}, threshold: {threshold:.2f}, predicted_duplicate: {int(np.float32(score) >= np.float32(threshold))}")


def main():
    args = parse_args(__doc__, DEFAULT_RUN)
    from improved.model import load_run

    predict_pair(args, load_run)


if __name__ == "__main__":
    main()
