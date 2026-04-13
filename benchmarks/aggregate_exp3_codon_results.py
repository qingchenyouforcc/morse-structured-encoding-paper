from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from benchmarks.run_exp3_codon_experiments import (  # noqa: E402
    aggregate_rows,
    build_base_vs_dp_rows,
    build_outputs as build_exp3_outputs,
    build_markdown_summary,
    pivot_method_rows,
    write_csv,
)


def read_sample_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def coerce_row_types(row: dict[str, str]) -> dict[str, Any]:
    int_fields = {
        "num_codons",
        "raw_codon_length",
        "normalized_raw_codon_length",
        "encoded_length",
        "character_reduction",
        "normalized_reduction",
        "regularity_bin_n",
    }
    float_fields = {
        "compression_ratio",
        "normalized_compression_ratio",
        "encode_ns",
        "decode_ns",
        "gc_content",
        "dominant_reference_coverage",
        "avg_window_heterogeneity",
        "sparse_difference_ratio",
    }
    bool_fields = {"sequence_round_trip_ok", "token_round_trip_ok"}

    typed: dict[str, Any] = dict(row)
    for field in int_fields:
        if field in row and row[field] != "":
            typed[field] = int(row[field])
    for field in float_fields:
        value = row.get(field, "")
        typed[field] = None if value == "" else float(value)
    for field in bool_fields:
        typed[field] = str(row[field]).lower() == "true"
    return typed


def build_outputs(sample_rows: list[dict[str, Any]], output_dir: Path) -> None:
    results_overall = aggregate_rows(sample_rows, ("species", "method", "category"))
    results_groups = aggregate_rows(sample_rows, ("species", "length_group", "method", "category"))
    worthwhile_analysis = aggregate_rows(
        sample_rows,
        ("species", "length_group", "regularity_group", "method", "category"),
    )

    overall_wide = pivot_method_rows(results_overall, row_fields=("species",))
    groups_wide = pivot_method_rows(results_groups, row_fields=("species", "length_group"))
    worthwhile_wide = pivot_method_rows(
        [
            row
            for row in worthwhile_analysis
            if row["regularity_group"] in {"high_regularity", "low_regularity"}
        ],
        row_fields=("species", "length_group", "regularity_group"),
    )
    base_vs_dp_rows = build_base_vs_dp_rows(
        results_overall,
        group_fields=("species",),
        group_type="species",
    ) + build_base_vs_dp_rows(
        results_groups,
        group_fields=("species", "length_group"),
        group_type="species_length_group",
    )

    write_csv(output_dir / "sample_method_details.csv", sample_rows)
    write_csv(output_dir / "results_overall.csv", overall_wide)
    write_csv(output_dir / "results_groups.csv", groups_wide)
    write_csv(output_dir / "results_base_vs_dp.csv", base_vs_dp_rows)
    write_csv(output_dir / "worthwhile_analysis.csv", worthwhile_wide)
    (output_dir / "exp3_results_summary.md").write_text(
        build_markdown_summary(overall_wide, groups_wide, worthwhile_wide, base_vs_dp_rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate batched Exp3 sample-detail CSV files.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sample_paths = sorted(args.input_dir.rglob("sample_method_details.csv"))
    if not sample_paths:
        raise FileNotFoundError(f"no sample_method_details.csv found under {args.input_dir}")

    sample_rows: list[dict[str, Any]] = []
    for path in sample_paths:
        for row in read_sample_rows(path):
            sample_rows.append(coerce_row_types(row))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_exp3_outputs(sample_rows, args.output_dir)
    print(f"Wrote aggregated results: {args.output_dir}")


if __name__ == "__main__":
    main()
