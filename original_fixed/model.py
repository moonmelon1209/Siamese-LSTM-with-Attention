"""The original shared LSTM encoder and Manhattan similarity, using public Keras APIs."""
import json

import tensorflow as tf


def configure_device(mode):
    devices = tf.config.list_physical_devices("GPU")
    if mode == "cpu" or not devices:
        if mode == "gpu":
            raise RuntimeError(
                "TensorFlow could not find a GPU. Activate the Conda environment in README.md "
                "and check the NVIDIA driver, CUDA 11.2 and cuDNN 8.1."
            )
        tf.config.set_visible_devices([], "GPU")
        print("Device: CPU", flush=True)
        return
    gpu = devices[0]
    tf.config.set_visible_devices(gpu, "GPU")
    tf.config.experimental.set_memory_growth(gpu, True)
    name = tf.config.experimental.get_device_details(gpu).get("device_name", gpu.name)
    print(f"Device: GPU 0 ({name})", flush=True)


class ManDist(tf.keras.layers.Layer):
    def call(self, inputs):
        left, right = inputs
        return tf.exp(-tf.reduce_sum(tf.abs(left - right), axis=-1, keepdims=True))


def build_model(vocab_size, embedding_dim, sequence_length, hidden_size):
    layers = tf.keras.layers
    encoder = tf.keras.Sequential([
        layers.Input(shape=(sequence_length,), dtype="int32"),
        layers.Embedding(vocab_size, embedding_dim, embeddings_initializer="zeros",
                         trainable=False, name="word_vectors"),
        layers.LSTM(hidden_size),
    ], name="shared_encoder")
    left = layers.Input(shape=(sequence_length,), dtype="int32", name="question1")
    right = layers.Input(shape=(sequence_length,), dtype="int32", name="question2")
    score = ManDist(name="similarity")([encoder(left), encoder(right)])
    return tf.keras.Model([left, right], score, name="MaLSTM")


def load_run(run_dir):
    with (run_dir / "run.json").open(encoding="utf-8") as file:
        state = json.load(file)
    model = build_model(len(state["vocabulary"]) + 2, **state["architecture"])
    model.load_weights(str(run_dir / "model.weights.h5"))
    return model, state
