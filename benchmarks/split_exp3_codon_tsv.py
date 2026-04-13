from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split an Exp3 TSV into fixed-size shards.")
    parser.add_argument("--input-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows-per-shard", type=int, default=100)
    parser.add_argument("--prefix", default="shard")
    args = parser.parse_args()

    rows = read_rows(args.input_tsv)
    if not rows:
        raise ValueError(f"input TSV is empty: {args.input_tsv}")
    if args.rows_per_shard <= 0:
        raise ValueError("--rows-per-shard must be positive")

    shard_count = 0
    for start in range(0, len(rows), args.rows_per_shard):
        shard_count += 1
        shard_rows = rows[start:start + args.rows_per_shard]
        shard_path = args.output_dir / f"{args.prefix}_{shard_count:03d}.tsv"
        write_rows(shard_path, shard_rows)
        print(f"[shard] {shard_path} rows={len(shard_rows)}")


if __name__ == "__main__":
    main()
