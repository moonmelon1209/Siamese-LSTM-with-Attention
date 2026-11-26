"""Score a pair of questions or a file with question1 and question2 columns."""
import argparse
from pathlib import Path

import numpy as np

from util import DEFAULT_RUN, HERE, encode_pairs, read_pairs


def predict_pairs(rows, run_dir, device):
    from model import configure_device, load_run

    configure_device(device)
    model, state = load_run(run_dir)
    inputs = encode_pairs(rows, state["vocabulary"], set(state["stop_words"]),
                          state["architecture"]["sequence_length"])
    scores = model.predict(inputs, batch_size=256, verbose=0).ravel()
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite prediction scores")
    return scores, state["validation"]["threshold"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--questions", nargs=2, metavar=("FIRST", "SECOND"))
    source.add_argument("--input", type=Path, default=HERE.parent / "original" / "data" / "test-20.csv")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    args = parser.parse_args()
    if args.questions:
        rows = [dict(zip(("question1", "question2"), args.questions))]
    else:
        rows = read_pairs(args.input, labeled=False)
    scores, threshold = predict_pairs(rows, args.run_dir, args.device)
    print("similarity\tpredicted_duplicate")
    decisions = scores >= threshold
    for score, decision in zip(scores, decisions):
        print(f"{score:.8f}\t{int(decision)}")


if __name__ == "__main__":
    main()
