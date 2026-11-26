"""Multilingual MaLSTM with a single shared, padding-aware LSTM."""
import json

import tensorflow as tf

from original_fixed.model import ManDist


def build_model(vocab_size, embedding_dim, sequence_length, hidden_size):
    layers = tf.keras.layers
    encoder = tf.keras.Sequential([
        layers.Input(shape=(sequence_length,), dtype="int32"),
        layers.Embedding(vocab_size, embedding_dim, mask_zero=True, trainable=False,
                         embeddings_initializer="zeros", name="aligned_vectors"),
        layers.LSTM(hidden_size, name="lstm"),
    ], name="shared_encoder")
    left = layers.Input(shape=(sequence_length,), dtype="int32", name="question1")
    right = layers.Input(shape=(sequence_length,), dtype="int32", name="question2")
    similarity = ManDist(name="similarity")([encoder(left), encoder(right)])
    return tf.keras.Model([left, right], similarity, name="multilingual_MaLSTM")


def load_run(directory):
    state = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if state.get("model") != "multilingual_last_state":
        raise ValueError("Expected a multilingual last-state MaLSTM checkpoint.")
    size = 2 + sum(len(words) for words in state["vocabulary"].values())
    model = build_model(size, **state["architecture"])
    model.load_weights(str(directory / "model.weights.h5"))
    return model, state
