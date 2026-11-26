"""Train the multilingual last-state MaLSTM on English, Korean, and English–Korean question pairs."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

# Support both "python improved/script.py" and "python -m improved.script".
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from improved.data import (DEFAULT_RUN, EMBEDDING_DIM, HIDDEN_SIZE, MANIFEST, SEQUENCE_LENGTH,
                           count_training_tokens, encode_split, fingerprint, load_split)
from improved.embeddings import build_embeddings, vector_path
from improved.metrics import calibrate, group_scores
from original_fixed.util import labels


def parse_args(description, default_run):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--en-vectors", type=Path, help="Existing aligned English fastText .vec or .vec.gz")
    parser.add_argument("--ko-vectors", type=Path, help="Existing aligned Korean fastText .vec or .vec.gz")
    parser.add_argument("--device", choices=["gpu", "cpu", "auto"], default="gpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path, default=default_run)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.run_dir.exists():
        parser.error(f"Run already exists: {args.run_dir}. Choose a fresh --run-dir.")

    return args


def train_model(args, build_model, model_type):
    import tensorflow as tf
    from original_fixed.model import configure_device

    configure_device(args.device)
    tf.keras.utils.set_random_seed(args.seed)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    train = load_split("train", manifest)
    validation = load_split("validation", manifest)
    try:
        paths = {"en": vector_path("en", args.en_vectors), "ko": vector_path("ko", args.ko_vectors)}
    except OSError as error:
        raise SystemExit(str(error)) from error
    counts = count_training_tokens(train, SEQUENCE_LENGTH)
    vocab, embeddings, coverage = build_embeddings(counts, paths)
    train_x = encode_split(train, vocab, SEQUENCE_LENGTH)
    train_y = np.concatenate([labels(rows) for rows in train.values()]).astype(np.float32).reshape(-1, 1)
    val_x = encode_split(validation, vocab, SEQUENCE_LENGTH)
    val_y = {name: labels(rows) for name, rows in validation.items()}

    architecture = dict(embedding_dim=EMBEDDING_DIM, sequence_length=SEQUENCE_LENGTH, hidden_size=HIDDEN_SIZE)
    model = build_model(len(embeddings), **architecture)
    model.get_layer("shared_encoder").get_layer("aligned_vectors").set_weights([embeddings])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mean_squared_error")
    print(f"Training pairs: {len(train_y)}, validation pairs: {len(val_x[0])}", flush=True)

    args.run_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.run_dir.parent) as temporary:
        stage = Path(temporary) / "run"
        stage.mkdir()

        class BestValidation(tf.keras.callbacks.Callback):
            best = None

            def on_epoch_end(self, epoch, logs=None):
                scores = self.model.predict(val_x, batch_size=args.batch_size, verbose=0).ravel()
                current = calibrate(val_y, group_scores(scores, validation))
                print(f"Validation mean Macro-F1={current['mean_macro_f1']:.4f}, threshold={current['threshold']:.2f}")
                if self.best is None or current["mean_macro_f1"] > self.best["mean_macro_f1"]:
                    self.best = {"epoch": epoch + 1, **current}
                    self.model.save_weights(str(stage / "model.weights.h5"))

        selection = BestValidation()
        model.fit(train_x, train_y, batch_size=args.batch_size, epochs=args.epochs,
                  shuffle=True, callbacks=[selection], verbose=2)
        state = {
            "model": model_type, "architecture": architecture,
            "vocabulary": vocab, "validation": selection.best, "seed": args.seed,
            "batch_size": args.batch_size, "epochs": args.epochs,
            "split_fingerprint": fingerprint(manifest),
            "train_sizes": {name: len(rows) for name, rows in train.items()},
            "validation_sizes": {name: len(rows) for name, rows in validation.items()},
            "vector_files": {lang: path.name for lang, path in paths.items()},
            "coverage": coverage, "tokenizer": "nfkc_lower_word_v1", "padding": "right",
        }
        (stage / "run.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        stage.rename(args.run_dir)

    best = selection.best
    print(f"Best epoch {best['epoch']}, validation mean Macro-F1={best['mean_macro_f1']:.4f}")
    for group, result in best["by_group"].items():
        print(f"  {group}: validation Macro-F1={result['macro_f1']:.4f}")
    print(f"Saved: {args.run_dir.resolve()}")


def main():
    args = parse_args(__doc__, DEFAULT_RUN)
    from improved.model import build_model

    train_model(args, build_model, "multilingual_last_state")


if __name__ == "__main__":
    main()
