"""Load only the training vocabulary from the Google News word2vec binary."""
import gzip
from io import BufferedReader

import numpy as np


def load_embeddings(path, vocabulary, dimensions, seed):
    # Keep the original random initialization for words absent from Google News.
    matrix = np.random.default_rng(seed).standard_normal(
        (len(vocabulary) + 2, dimensions), dtype=np.float32
    )
    matrix[0] = 0
    wanted = {word.encode("utf-8"): index for word, index in vocabulary.items()}
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rb") as raw, BufferedReader(raw) as stream:
        count, width = map(int, stream.readline().split())
        if width != dimensions:
            raise ValueError(f"Expected {dimensions}-dimensional word2vec vectors, got {width}")
        vector_bytes = 4 * width
        for _ in range(count):
            if not wanted:
                break
            # Some word2vec writers put a newline between binary records.
            word = bytearray()
            while True:
                char = stream.read(1)
                if char == b" ":
                    break
                if not char:
                    raise EOFError("Truncated word2vec token. Check the downloaded file.")
                if char not in (b"\r", b"\n"):
                    word.extend(char)
            vector = stream.read(vector_bytes)
            if len(vector) != vector_bytes:
                raise EOFError("Truncated word2vec vector. Check the downloaded file.")
            index = wanted.pop(bytes(word), None)
            if index is not None:
                matrix[index] = np.frombuffer(vector, dtype="<f4")
    matched = len(vocabulary) - len(wanted)
    if not matched:
        raise ValueError("No English training words matched the supplied embedding file.")
    return matrix, matched
