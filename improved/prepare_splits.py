"""Record duplicate exclusions once so training never needs to open the test set."""
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

# Support both "python improved/script.py" and "python -m improved.script".
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from improved.data import LANGUAGES, MANIFEST, file_digest, source_path
from original_fixed.util import read_pairs


def main():
    manifest = {}
    seen = {}
    for split in ("test", "validation", "train"):
        manifest[split] = {}
        for group, langs in LANGUAGES.items():
            path = source_path(group, split)
            rows = read_pairs(path)
            excluded = []
            for i, row in enumerate(rows):
                pair = tuple(sorted(zip(langs, (row["question1"].strip(), row["question2"].strip()))))
                if pair in seen:
                    if row["is_duplicate"] != seen[pair]:
                        raise ValueError(f"Conflicting labels in {path.name}, source row {i}")
                    excluded.append(i)
                else:
                    seen[pair] = row["is_duplicate"]
            manifest[split][group] = {"sha256": file_digest(path), "excluded_row_ids": excluded,
                                      "kept": len(rows) - len(excluded)}
            print(f"{group}/{split}: {len(rows) - len(excluded)}/{len(rows)}")
    with NamedTemporaryFile("w", encoding="utf-8", dir=MANIFEST.parent, delete=False) as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")
    Path(output.name).replace(MANIFEST)


if __name__ == "__main__":
    main()
