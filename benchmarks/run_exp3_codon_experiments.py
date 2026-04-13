from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Callable


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from baseline_codecs import (  # noqa: E402
    decode_gzip,
    decode_raw,
    decode_repair_grammar,
    decode_sequitur_grammar,
    decode_zstd,
    encode_gzip,
    encode_raw,
    encode_repair_grammar,
    encode_sequitur_grammar,
    encode_zstd,
    zstd_available,
)
from codon_utils import (  # noqa: E402
    DEFAULT_WINDOW_SIZE,
    DEFAULT_WINDOW_STRIDE,
    codon_words,
    extract_codon_features,
    normalize_codon_sequence,
    read_tsv_rows,
)
from structured_codon_codecs import (  # noqa: E402
    StructuredCodonEncoderConfig,
    decode_structured_codon,
    encode_structured_codon,
)


DEFAULT_INPUT_TSV = ROOT_DIR / "exp3_real_codon" / "sampled_balanced" / "overall.tsv"
DEFAULT_NATURAL_INPUT_TSV = ROOT_DIR / "exp3_real_codon" / "sampled_natural" / "overall.tsv"
DEFAULT_SHORT_MID_INPUT_TSV = ROOT_DIR / "exp3_real_codon" / "sampled_short_mid_balanced" / "overall.tsv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "exp3_real_codon" / "outputs"
MIN_REGULARITY_BIN_N = 40
ADAPTIVE_BASE_CONFIG = StructuredCodonEncoderConfig(
    name="structured_base_codon_adaptive",
    group_mode="adaptive_window",
    search_mode="base",
)
ADAPTIVE_DP_CONFIG = StructuredCodonEncoderConfig(
    name="structured_dp_codon_adaptive",
    group_mode="adaptive_window",
    search_mode="dp",
)
FIXED_BASE_CONFIG = StructuredCodonEncoderConfig(
    name="structured_base_codon_fixed",
    group_mode="fixed_window",
    search_mode="base",
)
FIXED_DP_CONFIG = StructuredCodonEncoderConfig(
    name="structured_dp_codon_fixed",
    group_mode="fixed_window",
    search_mode="dp",
)
METHOD_RATIO_COLUMNS: dict[str, str] = {
    "structured_base_codon_adaptive": "adaptive_base_norm_ratio",
    "structured_dp_codon_adaptive": "adaptive_dp_norm_ratio",
    "structured_base_codon_fixed": "fixed_base_norm_ratio",
    "structured_dp_codon_fixed": "fixed_dp_norm_ratio",
    "raw_codon": "raw_codon_norm_ratio",
    "gzip": "gzip_norm_ratio",
    "zstd": "zstd_norm_ratio",
    "repair": "repair_norm_ratio",
    "sequitur_style": "sequitur_norm_ratio",
}


@dataclass(frozen=True)
class ExperimentMethod:
    name: str
    category: str
    encode: Callable[[str], Any]
    decode: Callable[[Any], str]
    length: Callable[[Any], int]
    preview: Callable[[Any], str]
    available: bool = True
    note: str = ""


def text_byte_length(payload: str) -> int:
    return len(payload.encode("utf-8"))


def bytes_length(payload: bytes) -> int:
    return len(payload)


def byte_preview(payload: bytes, limit: int = 24) -> str:
    return payload[:limit].hex()


def structured_method(name: str, config: StructuredCodonEncoderConfig, category: str) -> ExperimentMethod:
    return ExperimentMethod(
        name=name,
        category=category,
        encode=lambda sequence: encode_structured_codon(sequence, config),
        decode=decode_structured_codon,
        length=text_byte_length,
        preview=lambda payload: str(payload),
    )


def experiment_methods() -> list[ExperimentMethod]:
    methods = [
        structured_method("structured_base_codon_adaptive", ADAPTIVE_BASE_CONFIG, "structured_main"),
        structured_method("structured_dp_codon_adaptive", ADAPTIVE_DP_CONFIG, "structured_main"),
        structured_method("structured_base_codon_fixed", FIXED_BASE_CONFIG, "structured_ablation"),
        structured_method("structured_dp_codon_fixed", FIXED_DP_CONFIG, "structured_ablation"),
        ExperimentMethod(
            name="raw_codon",
            category="baseline",
            encode=encode_raw,
            decode=decode_raw,
            length=text_byte_length,
            preview=lambda payload: str(payload),
        ),
        ExperimentMethod(
            name="gzip",
            category="baseline",
            encode=encode_gzip,
            decode=decode_gzip,
            length=bytes_length,
            preview=byte_preview,
        ),
        ExperimentMethod(
            name="sequitur_style",
            category="grammar_baseline",
            encode=encode_sequitur_grammar,
            decode=decode_sequitur_grammar,
            length=bytes_length,
            preview=byte_preview,
        ),
        ExperimentMethod(
            name="repair",
            category="grammar_baseline",
            encode=encode_repair_grammar,
            decode=decode_repair_grammar,
            length=bytes_length,
            preview=byte_preview,
        ),
        ExperimentMethod(
            name="zstd",
            category="baseline",
            encode=encode_zstd,
            decode=decode_zstd,
            length=bytes_length,
            preview=byte_preview,
            available=zstd_available(),
            note="" if zstd_available() else "zstandard package not installed",
        ),
    ]
    return methods


def timed_call(func: Callable[[], Any], repeat: int) -> tuple[Any, float]:
    total = 0
    result: Any = None
    for _ in range(repeat):
        start = perf_counter_ns()
        result = func()
        total += perf_counter_ns() - start
    return result, total / repeat


def token_round_trip_ok(reference_sequence: str, decoded_sequence: str) -> bool:
    return codon_words(reference_sequence) == codon_words(decoded_sequence)


def load_dataset_rows(input_tsv: Path) -> list[dict[str, str]]:
    rows = read_tsv_rows(input_tsv)
    if not rows:
        raise ValueError(f"input TSV is empty: {input_tsv}")
    required_fields = {
        "species",
        "seq_id",
        "codon_tokens",
        "num_codons",
        "length_group",
        "regularity_group",
        "regularity_quartile",
    }
    missing = required_fields - set(rows[0].keys())
    if missing:
        raise ValueError(f"missing required fields in {input_tsv}: {sorted(missing)}")
    return rows


def filter_dataset_rows(
    rows: list[dict[str, str]],
    *,
    species_filter: set[str] | None,
    length_group_filter: set[str] | None,
) -> list[dict[str, str]]:
    filtered = rows
    if species_filter:
        filtered = [row for row in filtered if row["species"] in species_filter]
    if length_group_filter:
        filtered = [row for row in filtered if row["length_group"] in length_group_filter]
    return filtered


def evaluate_method(
    method: ExperimentMethod,
    row: dict[str, str],
    *,
    time_repeat: int,
    window_size: int,
    stride: int,
) -> dict[str, Any]:
    sequence = normalize_codon_sequence(row["codon_tokens"])
    payload, encode_ns = timed_call(lambda: method.encode(sequence), time_repeat)
    encoded_length = method.length(payload)
    decoded_sequence, decode_ns = timed_call(lambda: method.decode(payload), time_repeat)
    decoded_sequence = normalize_codon_sequence(decoded_sequence)
    features = extract_codon_features(sequence, window_size=window_size, stride=stride)
    raw_length = len(sequence.encode("utf-8"))
    # Under the current codon benchmark protocol, every method operates on the
    # same compact printable canonical input string, so the normalized
    # denominator is identical to the ordinary raw byte length.
    normalized_raw_length = raw_length

    return {
        "species": row["species"],
        "seq_id": row["seq_id"],
        "length_group": row["length_group"],
        "regularity_group": row["regularity_group"],
        "regularity_quartile": row["regularity_quartile"],
        "regularity_bin_n": int(row.get("regularity_bin_n", str(MIN_REGULARITY_BIN_N)) or MIN_REGULARITY_BIN_N),
        "regularity_is_reportable": row.get("regularity_is_reportable", "true"),
        "sampling_view": row.get("sampling_view", ""),
        "terminal_stop_policy": row.get("terminal_stop_policy", ""),
        "sequence": sequence,
        "method": method.name,
        "category": method.category,
        "num_codons": int(row["num_codons"]),
        "raw_codon_length": raw_length,
        "normalized_raw_codon_length": normalized_raw_length,
        "encoded_length": encoded_length,
        "character_reduction": raw_length - encoded_length,
        "normalized_reduction": normalized_raw_length - encoded_length,
        "compression_ratio": encoded_length / raw_length if raw_length else 0.0,
        "normalized_compression_ratio": encoded_length / normalized_raw_length if normalized_raw_length else 0.0,
        "encode_ns": encode_ns,
        "decode_ns": decode_ns,
        "sequence_round_trip_ok": decoded_sequence == sequence,
        "token_round_trip_ok": token_round_trip_ok(sequence, decoded_sequence),
        "gc_content": 0.0 if row.get("gc_content", "") == "" else float(row["gc_content"]),
        "dominant_reference_coverage": features.dominant_reference_coverage,
        "avg_window_heterogeneity": features.avg_window_heterogeneity,
        "sparse_difference_ratio": features.sparse_difference_ratio,
        "encoded_preview": method.preview(payload),
        "method_note": method.note,
    }


def aggregate_rows(rows: list[dict[str, Any]], group_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        buckets.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for key, bucket in buckets.items():
        total_raw = sum(item["raw_codon_length"] for item in bucket)
        total_normalized_raw = sum(item["normalized_raw_codon_length"] for item in bucket)
        total_encoded = sum(item["encoded_length"] for item in bucket)
        summary = {field: value for field, value in zip(group_fields, key)}
        summary.update(
            {
                "samples": len(bucket),
                "total_raw_codon_length": total_raw,
                "total_normalized_raw_codon_length": total_normalized_raw,
                "total_encoded_length": total_encoded,
                "total_character_reduction": total_raw - total_encoded,
                "total_normalized_reduction": total_normalized_raw - total_encoded,
                "compression_ratio": total_encoded / total_raw if total_raw else 0.0,
                "normalized_compression_ratio": total_encoded / total_normalized_raw if total_normalized_raw else 0.0,
                "average_encode_ns": statistics.fmean(item["encode_ns"] for item in bucket),
                "average_decode_ns": statistics.fmean(item["decode_ns"] for item in bucket),
                "sequence_round_trip_success_rate": (
                    sum(1 for item in bucket if item["sequence_round_trip_ok"]) / len(bucket)
                ),
                "token_round_trip_success_rate": (
                    sum(1 for item in bucket if item["token_round_trip_ok"]) / len(bucket)
                ),
                "average_num_codons": statistics.fmean(item["num_codons"] for item in bucket),
            }
        )
        if "regularity_bin_n" in bucket[0]:
            summary["regularity_bin_n"] = bucket[0]["regularity_bin_n"]
        if "regularity_is_reportable" in bucket[0]:
            summary["regularity_is_reportable"] = bucket[0]["regularity_is_reportable"]
        summaries.append(summary)
    return sorted(summaries, key=lambda row: tuple(row[field] for field in group_fields))


def pivot_method_rows(
    summary_rows: list[dict[str, Any]],
    *,
    row_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in summary_rows:
        key = tuple(row[field] for field in row_fields)
        buckets.setdefault(key, {})[row["method"]] = row

    wide_rows: list[dict[str, Any]] = []
    for key in sorted(buckets):
        methods = buckets[key]
        wide_row = {field: value for field, value in zip(row_fields, key)}
        first_row = next(iter(methods.values()))
        wide_row["samples"] = first_row["samples"]
        for method_name, column_name in METHOD_RATIO_COLUMNS.items():
            method_row = methods.get(method_name)
            wide_row[column_name] = "" if method_row is None else method_row["normalized_compression_ratio"]
        wide_row["base_norm_ratio"] = wide_row.get("adaptive_base_norm_ratio", "")
        wide_row["dp_norm_ratio"] = wide_row.get("adaptive_dp_norm_ratio", "")

        sequence_success_rates = [row["sequence_round_trip_success_rate"] for row in methods.values()]
        token_success_rates = [row["token_round_trip_success_rate"] for row in methods.values()]
        wide_row["sequence_round_trip_success_rate"] = min(sequence_success_rates) if sequence_success_rates else ""
        wide_row["token_round_trip_success_rate"] = min(token_success_rates) if token_success_rates else ""
        wide_rows.append(wide_row)
    return wide_rows


def build_base_vs_dp_rows(
    summary_rows: list[dict[str, Any]],
    *,
    group_fields: tuple[str, ...],
    group_type: str,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in summary_rows:
        key = tuple(row[field] for field in group_fields)
        buckets.setdefault(key, {})[row["method"]] = row

    rows: list[dict[str, Any]] = []
    for key, methods in sorted(buckets.items()):
        adaptive_base = methods.get("structured_base_codon_adaptive")
        adaptive_dp = methods.get("structured_dp_codon_adaptive")
        fixed_base = methods.get("structured_base_codon_fixed")
        fixed_dp = methods.get("structured_dp_codon_fixed")
        if not adaptive_base or not adaptive_dp:
            continue

        group_values = {field: value for field, value in zip(group_fields, key)}
        row = {
            "group_type": group_type,
            "species": group_values.get("species", ""),
            "length_group": group_values.get("length_group", ""),
        }
        row.update(
            {
                "samples": adaptive_base["samples"],
                "adaptive_base_norm_ratio": adaptive_base["normalized_compression_ratio"],
                "adaptive_dp_norm_ratio": adaptive_dp["normalized_compression_ratio"],
                "fixed_base_norm_ratio": "" if fixed_base is None else fixed_base["normalized_compression_ratio"],
                "fixed_dp_norm_ratio": "" if fixed_dp is None else fixed_dp["normalized_compression_ratio"],
                "adaptive_dp_improvement_pp": (
                    (adaptive_base["normalized_compression_ratio"] - adaptive_dp["normalized_compression_ratio"]) * 100.0
                ),
                "fixed_to_adaptive_base_improvement_pp": (
                    ""
                    if fixed_base is None
                    else (fixed_base["normalized_compression_ratio"] - adaptive_base["normalized_compression_ratio"]) * 100.0
                ),
                "sequence_round_trip_success_rate": min(
                    method_row["sequence_round_trip_success_rate"] for method_row in methods.values()
                ),
                "token_round_trip_success_rate": min(
                    method_row["token_round_trip_success_rate"] for method_row in methods.values()
                ),
            }
        )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def add_all_species_rows(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_rows = list(sample_rows)
    for row in sample_rows:
        cloned = dict(row)
        cloned["species"] = "ALL"
        all_rows.append(cloned)
    return all_rows


def is_reportable_regularity_row(row: dict[str, Any]) -> bool:
    if row["regularity_group"] not in {"high_regularity", "low_regularity"}:
        return False
    if "regularity_bin_n" in row and int(row.get("regularity_bin_n", 0) or 0) < MIN_REGULARITY_BIN_N:
        return False
    if "regularity_is_reportable" in row:
        return str(row.get("regularity_is_reportable", "true")).lower() == "true"
    return True


def build_markdown_summary(
    overall_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    worthwhile_rows: list[dict[str, Any]],
    base_vs_dp_rows: list[dict[str, Any]],
) -> str:
    def fmt(value: Any) -> str:
        if value == "" or value is None:
            return ""
        return f"{float(value):.4f}"

    lines = [
        "# Exp3 Real CDS Codon Results Summary",
        "",
        "Primary view: balanced length-stratified real CDS codon-token benchmark.",
        "Main structured method: adaptive-window codon recoding. Fixed-window codon recoding is an ablation.",
        "",
        "## Balanced Overall Results",
        "",
        "| Species | Samples | Adaptive Base | Adaptive DP | Fixed Base | Fixed DP | Raw | gzip | zstd | repair | sequitur | Token RT |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in overall_rows:
        lines.append(
            f"| {row['species']} | {row['samples']} | {fmt(row['adaptive_base_norm_ratio'])} | {fmt(row['adaptive_dp_norm_ratio'])} | {fmt(row['fixed_base_norm_ratio'])} | {fmt(row['fixed_dp_norm_ratio'])} | {fmt(row['raw_codon_norm_ratio'])} | {fmt(row['gzip_norm_ratio'])} | {fmt(row['zstd_norm_ratio'])} | {fmt(row['repair_norm_ratio'])} | {fmt(row['sequitur_norm_ratio'])} | {row['token_round_trip_success_rate']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Length-Controlled Results",
            "",
            "| Species | Length Group | Samples | Adaptive Base | Adaptive DP | Fixed Base | Fixed DP | Raw | gzip | repair | sequitur |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in group_rows:
        lines.append(
            f"| {row['species']} | {row['length_group']} | {row['samples']} | {fmt(row['adaptive_base_norm_ratio'])} | {fmt(row['adaptive_dp_norm_ratio'])} | {fmt(row['fixed_base_norm_ratio'])} | {fmt(row['fixed_dp_norm_ratio'])} | {fmt(row['raw_codon_norm_ratio'])} | {fmt(row['gzip_norm_ratio'])} | {fmt(row['repair_norm_ratio'])} | {fmt(row['sequitur_norm_ratio'])} |"
        )

    lines.extend(
        [
            "",
            "## Worthwhile Analysis",
            "",
            "| Species | Length Group | Regularity Group | Samples | Adaptive Base | Adaptive DP | Fixed Base | Fixed DP |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in worthwhile_rows:
        lines.append(
            f"| {row['species']} | {row['length_group']} | {row['regularity_group']} | {row['samples']} | {fmt(row['adaptive_base_norm_ratio'])} | {fmt(row['adaptive_dp_norm_ratio'])} | {fmt(row['fixed_base_norm_ratio'])} | {fmt(row['fixed_dp_norm_ratio'])} |"
        )

    lines.extend(
        [
            "",
            "## Base vs DP",
            "",
            "| Group Type | Species | Length Group | Samples | Adaptive Base | Adaptive DP | Fixed Base | Fixed DP | Adaptive DP Improvement (pp) | Fixed-to-Adaptive Base Improvement (pp) |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in base_vs_dp_rows:
        lines.append(
            f"| {row['group_type']} | {row['species']} | {row['length_group']} | {row['samples']} | {fmt(row['adaptive_base_norm_ratio'])} | {fmt(row['adaptive_dp_norm_ratio'])} | {fmt(row['fixed_base_norm_ratio'])} | {fmt(row['fixed_dp_norm_ratio'])} | {fmt(row['adaptive_dp_improvement_pp'])} | {fmt(row['fixed_to_adaptive_base_improvement_pp'])} |"
        )
    return "\n".join(lines)


def build_outputs(sample_rows: list[dict[str, Any]], output_dir: Path) -> None:
    overall_rows = add_all_species_rows(sample_rows)
    results_overall = aggregate_rows(overall_rows, ("species", "method", "category"))
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
            if is_reportable_regularity_row(row)
        ],
        row_fields=("species", "length_group", "regularity_group"),
    )
    base_vs_dp_rows = build_base_vs_dp_rows(
        results_overall,
        group_fields=("species",),
        group_type="balanced_overall",
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


def build_natural_distribution_appendix(sample_rows: list[dict[str, Any]], output_dir: Path) -> None:
    natural_overall = aggregate_rows(add_all_species_rows(sample_rows), ("species", "method", "category"))
    natural_wide = pivot_method_rows(natural_overall, row_fields=("species",))
    write_csv(output_dir / "natural_distribution_appendix.csv", natural_wide)
    write_csv(output_dir / "natural_sample_method_details.csv", sample_rows)


def build_appendix_outputs(sample_rows: list[dict[str, Any]], output_dir: Path) -> None:
    build_outputs(sample_rows, output_dir)


def evaluate_dataset(
    dataset_rows: list[dict[str, str]],
    *,
    methods: list[ExperimentMethod],
    time_repeat: int,
    window_size: int,
    stride: int,
    label: str,
) -> list[dict[str, Any]]:
    sample_rows: list[dict[str, Any]] = []
    total_rows = len(dataset_rows)
    for index, row in enumerate(dataset_rows, start=1):
        if index == 1 or index % 100 == 0 or index == total_rows:
            print(
                f"[{label}] {index}/{total_rows} "
                f"{row['species']} {row['length_group']} {row['seq_id']} codons={row['num_codons']}"
            )
        for method in methods:
            sample_rows.append(
                evaluate_method(
                    method,
                    row,
                    time_repeat=time_repeat,
                    window_size=window_size,
                    stride=stride,
                )
            )
    return sample_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp3 main and appendix codon benchmarks.")
    parser.add_argument("--input-tsv", type=Path, default=DEFAULT_INPUT_TSV)
    parser.add_argument("--natural-input-tsv", type=Path, default=DEFAULT_NATURAL_INPUT_TSV)
    parser.add_argument("--short-mid-input-tsv", type=Path, default=DEFAULT_SHORT_MID_INPUT_TSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--time-repeat", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_WINDOW_STRIDE)
    parser.add_argument("--species", action="append", default=[])
    parser.add_argument("--length-group", action="append", default=[])
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--skip-natural-appendix", action="store_true")
    parser.add_argument("--skip-short-mid-appendix", action="store_true")
    args = parser.parse_args()

    dataset_rows = filter_dataset_rows(
        load_dataset_rows(args.input_tsv),
        species_filter=set(args.species) if args.species else None,
        length_group_filter=set(args.length_group) if args.length_group else None,
    )
    if not dataset_rows:
        raise ValueError("no rows matched the requested species/length-group filters")

    all_methods = experiment_methods()
    requested_methods = set(args.method) if args.method else None
    if requested_methods:
        all_methods = [method for method in all_methods if method.name in requested_methods]
        if not all_methods:
            raise ValueError(f"no methods matched {sorted(requested_methods)}")
    available_methods = [method for method in all_methods if method.available]
    unavailable_methods = [method for method in all_methods if not method.available]

    sample_rows = evaluate_dataset(
        dataset_rows,
        methods=available_methods,
        time_repeat=max(args.time_repeat, 1),
        window_size=args.window_size,
        stride=args.stride,
        label="balanced",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    main_output_dir = args.output_dir / "main"
    natural_output_dir = args.output_dir / "appendix_natural"
    short_mid_output_dir = args.output_dir / "appendix_short_mid"
    build_outputs(sample_rows, main_output_dir)
    if not args.skip_natural_appendix and args.natural_input_tsv.exists():
        natural_rows = filter_dataset_rows(
            load_dataset_rows(args.natural_input_tsv),
            species_filter=set(args.species) if args.species else None,
            length_group_filter=set(args.length_group) if args.length_group else None,
        )
        if natural_rows:
            natural_sample_rows = evaluate_dataset(
                natural_rows,
                methods=available_methods,
                time_repeat=max(args.time_repeat, 1),
                window_size=args.window_size,
                stride=args.stride,
                label="natural",
            )
            build_natural_distribution_appendix(natural_sample_rows, natural_output_dir)
    if not args.skip_short_mid_appendix and args.short_mid_input_tsv.exists():
        short_mid_rows = filter_dataset_rows(
            load_dataset_rows(args.short_mid_input_tsv),
            species_filter=set(args.species) if args.species else None,
            length_group_filter=set(args.length_group) if args.length_group else None,
        )
        if short_mid_rows:
            short_mid_sample_rows = evaluate_dataset(
                short_mid_rows,
                methods=available_methods,
                time_repeat=max(args.time_repeat, 1),
                window_size=args.window_size,
                stride=args.stride,
                label="short_mid",
            )
            build_appendix_outputs(short_mid_sample_rows, short_mid_output_dir)
    unavailable_rows = [{"method": method.name, "note": method.note} for method in unavailable_methods]
    if unavailable_rows:
        write_csv(main_output_dir / "unavailable_methods.csv", unavailable_rows)
    print(f"Wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
