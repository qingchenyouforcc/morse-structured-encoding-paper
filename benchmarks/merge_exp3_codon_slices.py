from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from benchmarks.run_exp3_codon_experiments import (  # noqa: E402
    build_appendix_outputs,
    build_natural_distribution_appendix,
    build_outputs,
)


INT_FIELDS = {
    "num_codons",
    "raw_codon_length",
    "normalized_raw_codon_length",
    "encoded_length",
    "character_reduction",
    "normalized_reduction",
    "regularity_bin_n",
}
FLOAT_FIELDS = {
    "compression_ratio",
    "normalized_compression_ratio",
    "encode_ns",
    "decode_ns",
    "gc_content",
    "dominant_reference_coverage",
    "avg_window_heterogeneity",
    "sparse_difference_ratio",
}
BOOL_FIELDS = {
    "sequence_round_trip_ok",
    "token_round_trip_ok",
}


def convert_row(row: dict[str, str]) -> dict[str, object]:
    converted: dict[str, object] = {}
    for key, value in row.items():
        if value == "":
            converted[key] = value
        elif key in INT_FIELDS:
            converted[key] = int(float(value))
        elif key in FLOAT_FIELDS:
            converted[key] = float(value)
        elif key in BOOL_FIELDS:
            converted[key] = value.lower() == "true"
        else:
            converted[key] = value
    return converted


def read_rows(csv_path: Path) -> list[dict[str, object]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return [convert_row(row) for row in csv.DictReader(handle)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Exp3 codon slice CSVs into standard outputs.")
    parser.add_argument("--dataset", choices=("balanced", "natural", "short_mid"), required=True)
    parser.add_argument("--slice-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    input_name = "natural_sample_method_details.csv" if args.dataset == "natural" else "sample_method_details.csv"
    csv_paths = sorted(path for path in args.slice_dir.rglob(input_name) if path.is_file())
    if not csv_paths:
        raise ValueError(f"no slice CSVs found under {args.slice_dir} for {input_name}")

    sample_rows: list[dict[str, object]] = []
    for csv_path in csv_paths:
        sample_rows.extend(read_rows(csv_path))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset == "balanced":
        build_outputs(sample_rows, args.output_dir)
    elif args.dataset == "natural":
        build_natural_distribution_appendix(sample_rows, args.output_dir)
    else:
        build_appendix_outputs(sample_rows, args.output_dir)
    print(f"Merged {len(csv_paths)} slice files -> {args.output_dir}")


if __name__ == "__main__":
    main()
