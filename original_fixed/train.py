"""Train the English MaLSTM on the prepared training and validation splits."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from util import (DATA_DIR, DEFAULT_RUN, EMBEDDING_DIM, EMBEDDING_DIR, HIDDEN_SIZE, SEQUENCE_LENGTH,
                  best_threshold, build_vocabulary, encode_pairs, labels, read_pairs)
from word2vec import load_embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, default=EMBEDDING_DIR / "GoogleNews-vectors-negative300.bin.gz")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["gpu", "cpu", "auto"], default="gpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.run_dir.exists():
        parser.error(f"Run directory already exists: {args.run_dir}. Choose a new --run-dir for another run.")
    if not args.vectors.is_file():
        parser.error(f"Missing Google News vectors: {args.vectors}. See README.md or pass --vectors PATH.")

    import tensorflow as tf
    from nltk.corpus import stopwords
    from model import build_model, configure_device

    configure_device(args.device)
    tf.keras.utils.set_random_seed(args.seed)
    stop_words = set(stopwords.words("english"))
    train = read_pairs(DATA_DIR / "train" / "english_train.txt")
    validation = read_pairs(DATA_DIR / "validation" / "english_validation.txt")
    vocabulary = build_vocabulary(train, stop_words)
    print(f"Train: {len(train)} pairs, validation: {len(validation)} pairs. Loading Google News vectors...", flush=True)
    embeddings, matched = load_embeddings(args.vectors, vocabulary, EMBEDDING_DIM, args.seed)
    print(f"Pretrained vectors found for {matched}/{len(vocabulary)} training words.", flush=True)
    train_inputs = encode_pairs(train, vocabulary, stop_words, SEQUENCE_LENGTH)
    val_inputs = encode_pairs(validation, vocabulary, stop_words, SEQUENCE_LENGTH)
    val_labels = labels(validation)

    architecture = dict(embedding_dim=EMBEDDING_DIM, sequence_length=SEQUENCE_LENGTH, hidden_size=HIDDEN_SIZE)
    model = build_model(len(vocabulary) + 2, **architecture)
    model.get_layer("shared_encoder").get_layer("word_vectors").set_weights([embeddings])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mean_squared_error")

    args.run_dir.parent.mkdir(parents=True, exist_ok=True)
    # Publish the weights and their matching vocabulary/threshold together.
    with TemporaryDirectory(dir=args.run_dir.parent) as temporary:
        stage = Path(temporary) / "run"
        stage.mkdir()

        class BestValidation(tf.keras.callbacks.Callback):
            best = None

            def on_epoch_end(self, epoch, logs=None):
                scores = self.model.predict(val_inputs, batch_size=args.batch_size, verbose=0).ravel()
                if not np.isfinite(scores).all():
                    raise ValueError("Non-finite validation scores")
                current = best_threshold(val_labels, scores)
                print(f"Validation Macro-F1={current['macro_f1']:.4f}  threshold={current['threshold']:.2f}")
                if self.best is None or current["macro_f1"] > self.best["macro_f1"]:
                    self.best = {"epoch": epoch + 1, **current}
                    self.model.save_weights(str(stage / "model.weights.h5"))

        selection = BestValidation()
        model.fit(train_inputs, labels(train).astype(np.float32).reshape(-1, 1),
                  batch_size=args.batch_size, epochs=args.epochs, callbacks=[selection], verbose=2)
        state = {
            "architecture": architecture, "vocabulary": vocabulary, "stop_words": sorted(stop_words),
            "validation": selection.best, "seed": args.seed, "batch_size": args.batch_size,
            "epochs": args.epochs, "train_size": len(train), "validation_size": len(validation),
            "vectors_file": args.vectors.name, "pretrained_words": matched,
        }
        (stage / "run.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        stage.rename(args.run_dir)

    best = selection.best
    print(f"Best epoch {best['epoch']}: validation Macro-F1={best['macro_f1']:.4f}, "
          f"duplicate F1={best['duplicate_f1']:.4f}, threshold={best['threshold']:.2f}")
    print(f"Saved: {args.run_dir.resolve()}")


if __name__ == "__main__":
    main()
