"""Evaluate the saved multilingual attention checkpoint on each held-out language pair."""
from pathlib import Path

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from improved.evaluate import evaluate_model, parse_args
from improved_attention import DEFAULT_RUN


def main():
    args = parse_args(__doc__, DEFAULT_RUN)
    from improved_attention.model import load_run

    evaluate_model(args, load_run)


if __name__ == "__main__":
    main()
