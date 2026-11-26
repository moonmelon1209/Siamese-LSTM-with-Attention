"""Read fixed Korean and English vectors aligned by fastText's RCSLS mapping."""
import gzip
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np

from original_fixed.util import EMBEDDING_DIM, EMBEDDING_DIR


VECTOR_BASE = "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/"


def vector_path(lang, supplied):
    if supplied is not None:
        if not supplied.is_file():
            raise FileNotFoundError(f"Vector file not found: {supplied}")
        return supplied
    path = EMBEDDING_DIR / f"wiki.{lang}.align.vec"
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = VECTOR_BASE + path.name
    print(f"Downloading {url} to {path} ...", flush=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as file:
        temporary = Path(file.name)
    try:
        with urlopen(url, timeout=90) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
            size = response.headers.get("Content-Length")
            if size and output.tell() != int(size):
                raise OSError(f"Incomplete download: {url}")
        with temporary.open("rb") as check:
            header = check.readline(128).split()
            if len(header) != 2 or not header[0].isdigit() or header[1] != str(EMBEDDING_DIM).encode():
                raise ValueError(f"Unexpected fastText file header: {url}")
        temporary.replace(path)
    except (URLError, TimeoutError) as error:
        raise OSError(f"Download failed. Save {url} to {path}, or supply --{lang}-vectors PATH.") from error
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_selected(path, counts):
    wanted = {word.encode(): word for word in counts}
    selected = {}
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rb") as stream:
        count, dimensions = map(int, stream.readline().split())
        if count <= 0 or dimensions != EMBEDDING_DIM:
            raise ValueError(f"{path}: expected aligned {EMBEDDING_DIM}-dimensional vectors")
        rows = 0
        for line in stream:
            if not line.strip():
                continue
            rows += 1
            token, _, values = line.partition(b" ")
            word = wanted.get(token)
            if word is not None:
                vector = np.fromstring(values, sep=" ", dtype=np.float32)
                if vector.size != dimensions or not np.isfinite(vector).all():
                    raise ValueError(f"{path}: invalid vector at row {rows}")
                selected[word] = vector
        if rows != count:
            raise ValueError(f"{path}: expected {count} vectors, found {rows}. Download the file again.")
    if not selected:
        raise ValueError(f"{path}: no training words matched; check the vector language.")
    return selected


def build_embeddings(counts, paths):
    # IDs are global, but lookup dictionaries are separate for each language.
    vocabulary, coverage = {}, {}
    vectors = [np.zeros(EMBEDDING_DIM, dtype=np.float32) for _ in range(2)]
    for lang in ("en", "ko"):
        print(f"Reading {paths[lang].name} ...", flush=True)
        selected = read_selected(paths[lang], counts[lang])
        vocabulary[lang] = {}
        for word in sorted(selected):
            vocabulary[lang][word] = len(vectors)
            vectors.append(selected[word])
        covered = sum(counts[lang][word] for word in selected)
        total = sum(counts[lang].values())
        coverage[lang] = {"matched_types": len(selected), "total_types": len(counts[lang]),
                          "matched_tokens": covered, "total_tokens": total}
        print(f"{lang} training token coverage: {covered}/{total} ({covered / total:.1%})", flush=True)
    matrix = np.stack(vectors)
    # UNK is a real, unmasked timestep. Its frozen vector is the training vocabulary mean.
    matrix[1] = matrix[2:].mean(axis=0)
    return vocabulary, matrix, coverage
