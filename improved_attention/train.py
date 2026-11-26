"""Train the multilingual attention MaLSTM on English, Korean, and English–Korean question pairs."""
from pathlib import Path

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from improved.train import train_model, parse_args
from improved_attention import DEFAULT_RUN, MODEL_TYPE


def main():
    args = parse_args(__doc__, DEFAULT_RUN)
    from improved_attention.model import build_model

    train_model(args, build_model, MODEL_TYPE)


if __name__ == "__main__":
    main()
