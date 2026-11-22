"""Rebuild the paper's WikiText-103 context sampling from a local copy."""

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np


HEADING = re.compile(r"( \n \n = [^=]*[^=] = \n \n )")


def read_training_split(source):
    if source.is_file() and source.suffix == ".zip":
        with zipfile.ZipFile(source) as archive:
            return archive.read("wikitext-103/wiki.train.tokens").decode("utf-8")
    if source.is_file() and source.suffix == ".tokens":
        return source.read_text(encoding="utf-8")

    import pyarrow.parquet as pq

    shards = sorted(source.glob("train-*-of-*.parquet"))
    if len(shards) != 2:
        raise ValueError("Expected both WikiText-103-v1 training Parquet shards")
    rows = []
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(columns=["text"]):
            for line in batch.column(0).to_pylist():
                # Parquet export encodes the token file's space-only lines
                # as empty strings. The upstream heading regexp uses them.
                rows.append(line if line else " \n")
    return "".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261119)
    args = parser.parse_args()

    text = read_training_split(args.source)
    pattern_matches = list(HEADING.finditer(text))
    if len(pattern_matches) < 20_000:
        raise ValueError(f"Invalid WikiText-103 formatting ({len(pattern_matches)} article headers)")
    eligible = []
    for i, match in enumerate(pattern_matches):
        start = match.end()
        end = pattern_matches[i + 1].start() if i + 1 < len(pattern_matches) else len(text)
        if end - start >= 1024:
            eligible.append((i, match.group().strip()[1:-1].strip(), start))
    if not 1 <= args.count <= len(eligible):
        raise ValueError("Requested context count is outside the eligible article population")

    random = np.random.default_rng(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as out:
        for case, sample in enumerate(random.permutation(len(eligible))[: args.count]):
            article, title, start = eligible[sample]
            context = text[start : start + 1024]
            record = {"id": case, "article_index": article, "title": title, "context": context}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"{len(pattern_matches)} articles, {len(eligible)} eligible, {args.count} contexts")
    print(f"{args.output} SHA-256 {digest}")


if __name__ == "__main__":
    main()
