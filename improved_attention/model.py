"""A shared MaLSTM encoder with additive attention over each question's LSTM states."""
import json

import tensorflow as tf

from improved_attention import MODEL_TYPE
from original_fixed.model import ManDist


@tf.keras.utils.register_keras_serializable(package="MaLSTM")
class AttentionPooling(tf.keras.layers.Layer):
    """Turn (batch, time, hidden) into one weighted sentence vector per row."""

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.supports_masking = True
        self.projection = tf.keras.layers.Dense(units, activation="tanh", name="projection")
        self.score = tf.keras.layers.Dense(1, use_bias=False, name="score")

    def call(self, states, mask=None):
        logits = tf.squeeze(self.score(self.projection(states)), axis=-1)
        # Use float32 for the masked softmax even if the caller uses mixed precision.
        logits = tf.cast(logits, tf.float32)
        if mask is not None:
            logits = tf.where(mask, logits, tf.constant(-1e30, tf.float32))
        weights = tf.nn.softmax(logits, axis=1)
        if mask is not None:
            # A completely padded row has no contributing timesteps.
            weights = tf.where(mask, weights, tf.zeros_like(weights))
        weights = tf.cast(weights[..., None], states.dtype)
        return tf.reduce_sum(states * weights, axis=1)

    def compute_mask(self, inputs, mask=None):
        # Pooling removes the time axis.
        return None

    def get_config(self):
        return {**super().get_config(), "units": self.units}


def build_model(vocab_size, embedding_dim, sequence_length, hidden_size):
    layers = tf.keras.layers
    encoder = tf.keras.Sequential([
        layers.Input(shape=(sequence_length,), dtype="int32"),
        layers.Embedding(vocab_size, embedding_dim, mask_zero=True, trainable=False,
                         embeddings_initializer="zeros", name="aligned_vectors"),
        layers.LSTM(hidden_size, return_sequences=True, name="lstm"),
        AttentionPooling(hidden_size, name="attention_pooling"),
    ], name="shared_encoder")
    left = layers.Input(shape=(sequence_length,), dtype="int32", name="question1")
    right = layers.Input(shape=(sequence_length,), dtype="int32", name="question2")
    similarity = ManDist(name="similarity")([encoder(left), encoder(right)])
    return tf.keras.Model([left, right], similarity, name="multilingual_attention_MaLSTM")


def load_run(directory):
    state = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if state.get("model") != MODEL_TYPE:
        raise ValueError("Expected a multilingual attention-pooling MaLSTM checkpoint.")
    size = 2 + sum(len(words) for words in state["vocabulary"].values())
    model = build_model(size, **state["architecture"])
    model.load_weights(str(directory / "model.weights.h5"))
    return model, state
