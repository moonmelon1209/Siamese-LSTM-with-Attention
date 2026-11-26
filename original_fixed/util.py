"""English QQP preprocessing and metrics shared by training and inference."""
import csv
from pathlib import Path
import re

import numpy as np
from sklearn.metrics import f1_score


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "dataset"
EMBEDDING_DIR = HERE.parent / "embedding"
DEFAULT_RUN = HERE / "artifacts" / "english_baseline"
SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 300
HIDDEN_SIZE = 50
PAD_ID = 0
UNK_ID = 1


def text_to_word_list(text):
    # Pre process and convert texts to a list of words
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text


def read_pairs(path, *, labeled=True):
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        required = {"question1", "question2"}
        if labeled:
            required.add("is_duplicate")
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: expected columns {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no question pairs")
    for i, row in enumerate(rows, start=2):
        if not all((row[q] or "").strip() for q in ("question1", "question2")):
            raise ValueError(f"{path}, row {i}: empty question")
        if labeled and row["is_duplicate"] not in ("0", "1"):
            raise ValueError(f"{path}, row {i}: expected a 0/1 label")
    return rows


def tokens(question, stop_words):
    return [word for word in text_to_word_list(question) if word not in stop_words]


def build_vocabulary(rows, stop_words):
    vocabulary = {}
    for row in rows:
        for column in ("question1", "question2"):
            for word in tokens(row[column], stop_words):
                if word not in vocabulary:
                    vocabulary[word] = len(vocabulary) + 2
    return vocabulary


def encode_pairs(rows, vocabulary, stop_words, max_length):
    inputs = []
    for column in ("question1", "question2"):
        encoded = np.full((len(rows), max_length), PAD_ID, dtype=np.int32)
        for i, row in enumerate(rows):
            words = tokens(row[column], stop_words)[:max_length]
            if words:
                encoded[i, -len(words):] = [vocabulary.get(w, UNK_ID) for w in words]
        inputs.append(encoded)
    return inputs


def labels(rows):
    return np.array([int(row["is_duplicate"]) for row in rows], dtype=np.int32)


def f1_scores(truth, predicted):
    negative, positive = f1_score(truth, predicted, labels=[0, 1], average=None, zero_division=0)
    return {"macro_f1": float((negative + positive) / 2), "duplicate_f1": float(positive)}


def best_threshold(truth, scores):
    # Search the same grid at each epoch. When tied, prefer the threshold nearest 0.50.
    best = None
    for percent in sorted(range(101), key=lambda n: abs(n - 50)):
        threshold = percent / 100
        metrics = f1_scores(truth, scores >= threshold)
        if best is None or metrics["macro_f1"] > best["macro_f1"]:
            best = {"threshold": threshold, **metrics}
    return best
