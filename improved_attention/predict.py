"""Score one pair using explicit question languages and a saved multilingual attention checkpoint."""
from pathlib import Path

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from improved.predict import predict_pair, parse_args
from improved_attention import DEFAULT_RUN


def main():
    args = parse_args(__doc__, DEFAULT_RUN)
    from improved_attention.model import load_run

    predict_pair(args, load_run)


if __name__ == "__main__":
    main()
