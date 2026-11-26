"""Language-aware QQP inputs. Language is supplied by the split, including for mixed-script text."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np

from original_fixed.util import DATA_DIR, EMBEDDING_DIM, HIDDEN_SIZE, SEQUENCE_LENGTH, read_pairs


HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "artifacts" / "multilingual"
MANIFEST = HERE / "split_manifest.json"
LANGUAGES = {"english": ("en", "en"), "korean": ("ko", "ko"), "cross_en_ko": ("en", "ko")}
PAD_ID, UNK_ID = 0, 1
TOKEN_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)*")


def source_path(group, split):
    return DATA_DIR / split / f"{group}_{split}.txt"


def file_digest(path):
    # A Windows checkout can use CRLF for the same source rows.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_split(split, manifest):
    groups = {}
    for group in LANGUAGES:
        path = source_path(group, split)
        rule = manifest[split][group]
        if file_digest(path) != rule["sha256"]:
            raise ValueError(f"{path} changed. Regenerate split_manifest.json before training.")
        excluded = set(rule["excluded_row_ids"])
        groups[group] = [dict(row, row_id=i) for i, row in enumerate(read_pairs(path)) if i not in excluded]
    return groups


def tokenize(text):
    text = unicodedata.normalize("NFKC", text).replace("’", "'").lower()
    return TOKEN_PATTERN.findall(text)


def count_training_tokens(groups, max_length):
    counts = {"en": Counter(), "ko": Counter()}
    for group, rows in groups.items():
        for column, lang in zip(("question1", "question2"), LANGUAGES[group]):
            for row in rows:
                counts[lang].update(tokenize(row[column])[:max_length])
    return counts


def encode_pairs(rows, languages, vocabulary, max_length):
    result = []
    for column, lang in zip(("question1", "question2"), languages):
        lookup = vocabulary[lang]
        ids = np.full((len(rows), max_length), PAD_ID, dtype=np.int32)
        for i, row in enumerate(rows):
            words = tokenize(row[column])[:max_length]
            # At least one real timestep, even for a punctuation-only question.
            tokens = [lookup.get(word, UNK_ID) for word in words] or [UNK_ID]
            ids[i, :len(tokens)] = tokens
        result.append(ids)
    return result


def encode_split(groups, vocabulary, max_length):
    batches = [encode_pairs(rows, LANGUAGES[group], vocabulary, max_length) for group, rows in groups.items()]
    return [np.concatenate(side) for side in zip(*batches)]


def fingerprint(manifest):
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
