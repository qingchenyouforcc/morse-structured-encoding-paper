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
from binary_digit_utils import BCD_SYMBOLS, bcd_symbols_to_digits, digits_to_bcd_symbols  # noqa: E402
from sequence_utils import extract_sequence_features, normalized_sequence_length  # noqa: E402
from structured_codecs import (  # noqa: E402
    BASE_CONFIG,
    DP_CONFIG,
    StructuredEncoderConfig,
    decode_structured,
    encode_structured,
)


DATASETS_DIR = ROOT_DIR / "datasets" / "binary_digits"
OUTPUT_DIR = ROOT_DIR / "output" / "exp2_binary_digits"

BINARY_BASE_CONFIG = StructuredEncoderConfig(
    name=BASE_CONFIG.name,
    group_mode=BASE_CONFIG.group_mode,
    search_mode=BASE_CONFIG.search_mode,
    use_alternating=BASE_CONFIG.use_alternating,
    use_fallback=BASE_CONFIG.use_fallback,
    symbols=BCD_SYMBOLS,
)
BINARY_DP_CONFIG = StructuredEncoderConfig(
    name=DP_CONFIG.name,
    group_mode=DP_CONFIG.group_mode,
    search_mode=DP_CONFIG.search_mode,
    use_alternating=DP_CONFIG.use_alternating,
    use_fallback=DP_CONFIG.use_fallback,
    symbols=BCD_SYMBOLS,
)


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


def byte_preview(payload: bytes, limit: int = 24) -> str:
    return payload[:limit].hex()


def structured_method(name: str, config: StructuredEncoderConfig) -> ExperimentMethod:
    return ExperimentMethod(
        name=name,
        category="structured",
        encode=lambda sequence: encode_structured(sequence, config),
        decode=lambda payload: decode_structured(payload, BCD_SYMBOLS),
        length=len,
        preview=lambda payload: str(payload),
    )


def baseline_methods() -> list[ExperimentMethod]:
    methods = [
        ExperimentMethod(
            name="raw_binary",
            category="baseline",
            encode=encode_raw,
            decode=decode_raw,
            length=len,
            preview=lambda payload: str(payload),
        ),
        structured_method("morsefold_base", BINARY_BASE_CONFIG),
        structured_method("morsefold_dp", BINARY_DP_CONFIG),
        ExperimentMethod(
            name="gzip",
            category="baseline",
            encode=encode_gzip,
            decode=decode_gzip,
            length=len,
            preview=byte_preview,
        ),
        ExperimentMethod(
            name="sequitur_style",
            category="grammar_baseline",
            encode=encode_sequitur_grammar,
            decode=decode_sequitur_grammar,
            length=len,
            preview=byte_preview,
        ),
        ExperimentMethod(
            name="repair",
            category="grammar_baseline",
            encode=encode_repair_grammar,
            decode=decode_repair_grammar,
            length=len,
            preview=byte_preview,
        ),
    ]
    methods.append(
        ExperimentMethod(
            name="zstd",
            category="baseline",
            encode=encode_zstd,
            decode=decode_zstd,
            length=len,
            preview=byte_preview,
            available=zstd_available(),
            note="" if zstd_available() else "zstandard package not installed",
        )
    )
    return methods


def load_dataset(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_groups() -> list[tuple[str, list[str]]]:
    return [
        ("S1_single_digit", load_dataset(DATASETS_DIR / "s1_single_digit.txt")),
        ("S2_repeated_digit", load_dataset(DATASETS_DIR / "s2_repeated_digit.txt")),
        ("S3_mixed_digit", load_dataset(DATASETS_DIR / "s3_mixed_digit.txt")),
        ("S4_long_digit", load_dataset(DATASETS_DIR / "s4_long_digit.txt")),
    ]


def timed_call(func: Callable[[], Any], repeat: int) -> tuple[Any, float]:
    total = 0
    result: Any = None
    for _ in range(repeat):
        start = perf_counter_ns()
        result = func()
        total += perf_counter_ns() - start
    return result, total / repeat


def evaluate_method(
    method: ExperimentMethod,
    group_name: str,
    digits: str,
    sequence: str,
    raw_length: int,
    normalized_raw_length: int,
    time_repeat: int,
) -> dict[str, Any]:
    payload, encode_ns = timed_call(lambda: method.encode(sequence), time_repeat)
    encoded_length = method.length(payload)
    decoded_sequence, decode_ns = timed_call(lambda: method.decode(payload), time_repeat)
    decoded_digits = bcd_symbols_to_digits(decoded_sequence)
    features = extract_sequence_features(digits, sequence, BCD_SYMBOLS)

    return {
        "group": group_name,
        "digits": digits,
        "method": method.name,
        "category": method.category,
        "digit_count": len(digits.replace(" ", "")),
        "word_count": features.word_count,
        "code_count": features.code_count,
        "raw_binary_length": raw_length,
        "normalized_raw_binary_length": normalized_raw_length,
        "encoded_length": encoded_length,
        "character_reduction": raw_length - encoded_length,
        "normalized_reduction": normalized_raw_length - encoded_length,
        "compression_ratio": encoded_length / raw_length if raw_length else 0.0,
        "normalized_compression_ratio": encoded_length / normalized_raw_length if normalized_raw_length else 0.0,
        "encode_ns": encode_ns,
        "decode_ns": decode_ns,
        "sequence_round_trip_ok": decoded_sequence == sequence,
        "digit_round_trip_ok": decoded_digits == digits,
        "same_length_run_count": features.same_length_run_count,
        "same_length_run_coverage": features.same_length_run_coverage,
        "mean_same_length_run_size": features.mean_same_length_run_size,
        "max_same_length_run_size": features.max_same_length_run_size,
        "alternating_share": features.alternating_share,
        "mean_code_length": features.mean_code_length,
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
        total_raw = sum(item["raw_binary_length"] for item in bucket)
        total_normalized_raw = sum(item["normalized_raw_binary_length"] for item in bucket)
        total_encoded = sum(item["encoded_length"] for item in bucket)
        summary = {field: value for field, value in zip(group_fields, key)}
        summary.update(
            {
                "samples": len(bucket),
                "total_raw_binary_length": total_raw,
                "total_normalized_raw_binary_length": total_normalized_raw,
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
                "digit_round_trip_success_rate": (
                    sum(1 for item in bucket if item["digit_round_trip_ok"]) / len(bucket)
                ),
                "available_note": bucket[0]["method_note"],
            }
        )
        summaries.append(summary)
    return sorted(summaries, key=lambda row: tuple(row[field] for field in group_fields))


def compare_base_vs_dp(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[str, dict[str, dict[str, Any]]] = {}
    for row in summary_rows:
        if row["method"] not in {"morsefold_base", "morsefold_dp"}:
            continue
        by_group.setdefault(row["group"], {})[row["method"]] = row

    comparisons: list[dict[str, Any]] = []
    for group, methods in sorted(by_group.items()):
        base_row = methods.get("morsefold_base")
        dp_row = methods.get("morsefold_dp")
        if not base_row or not dp_row:
            continue
        comparisons.append(
            {
                "group": group,
                "samples": base_row["samples"],
                "base_total_encoded_length": base_row["total_encoded_length"],
                "dp_total_encoded_length": dp_row["total_encoded_length"],
                "extra_reduction": base_row["total_encoded_length"] - dp_row["total_encoded_length"],
                "base_compression_ratio": base_row["compression_ratio"],
                "dp_compression_ratio": dp_row["compression_ratio"],
                "base_normalized_compression_ratio": base_row["normalized_compression_ratio"],
                "dp_normalized_compression_ratio": dp_row["normalized_compression_ratio"],
                "normalized_ratio_improvement_pp": (
                    (base_row["normalized_compression_ratio"] - dp_row["normalized_compression_ratio"]) * 100.0
                ),
                "base_sequence_round_trip_success_rate": base_row["sequence_round_trip_success_rate"],
                "dp_sequence_round_trip_success_rate": dp_row["sequence_round_trip_success_rate"],
                "base_digit_round_trip_success_rate": base_row["digit_round_trip_success_rate"],
                "dp_digit_round_trip_success_rate": dp_row["digit_round_trip_success_rate"],
            }
        )
    return comparisons


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_markdown_summary(
    group_summary: list[dict[str, Any]],
    overall_summary: list[dict[str, Any]],
    base_vs_dp: list[dict[str, Any]],
) -> str:
    overall_rows = [row for row in overall_summary if row["group"] == "ALL"]
    overall_rows.sort(key=lambda row: row["normalized_compression_ratio"])
    grouped_rows = [row for row in group_summary if row["group"] != "ALL"]
    grouped_by_name: dict[str, list[dict[str, Any]]] = {}
    for row in grouped_rows:
        grouped_by_name.setdefault(row["group"], []).append(row)
    for rows in grouped_by_name.values():
        rows.sort(key=lambda row: row["normalized_compression_ratio"])

    lines = [
        "# Exp2 Results Summary",
        "",
        "This experiment evaluates fixed-length BCD digit sequences using the same structured encoder used in the Morse experiment,",
        "but with the binary symbol alphabet switched from `.`/`-` to `0`/`1`.",
        "",
        "## Overall Results",
        "",
        "| Method | Samples | Compression Ratio | Normalized Compression Ratio | Avg Encode ns | Sequence Round-trip | Digit Round-trip |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in overall_rows:
        lines.append(
            "| {method} | {samples} | {compression_ratio:.4f} | {normalized_compression_ratio:.4f} | "
            "{average_encode_ns:.0f} | {sequence_round_trip_success_rate:.2%} | {digit_round_trip_success_rate:.2%} |".format(
                **row
            )
        )

    lines.extend(["", "## Per-Group Results", ""])
    for group_name in sorted(grouped_by_name):
        lines.extend(
            [
                f"### {group_name}",
                "",
                "| Method | Samples | Compression Ratio | Normalized Compression Ratio | Total Reduction | Digit Round-trip |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in grouped_by_name[group_name]:
            lines.append(
                "| {method} | {samples} | {compression_ratio:.4f} | {normalized_compression_ratio:.4f} | "
                "{total_character_reduction} | {digit_round_trip_success_rate:.2%} |".format(**row)
            )
        lines.append("")

    lines.extend(
        [
            "## Base vs DP",
            "",
            "| Group | Extra Reduction | Base Norm Ratio | DP Norm Ratio | Improvement (pp) | Base Digit RT | DP Digit RT |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in base_vs_dp:
        lines.append(
            "| {group} | {extra_reduction} | {base_normalized_compression_ratio:.4f} | "
            "{dp_normalized_compression_ratio:.4f} | {normalized_ratio_improvement_pp:.2f} | "
            "{base_digit_round_trip_success_rate:.2%} | {dp_digit_round_trip_success_rate:.2%} |".format(**row)
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp2 fixed-length binary digit experiments.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--time-repeat", type=int, default=1)
    args = parser.parse_args()

    groups = build_groups()
    all_methods = baseline_methods()
    main_methods = [method for method in all_methods if method.available]
    unavailable_methods = [method for method in all_methods if not method.available]

    sample_rows: list[dict[str, Any]] = []
    for group_name, samples in groups:
        print(f"[group] {group_name}: {len(samples)} samples")
        for digits in samples:
            sequence = digits_to_bcd_symbols(digits, word_sep=" / ")
            raw_length = len(sequence)
            normalized_raw_length = normalized_sequence_length(digits, sequence)
            for method in main_methods:
                sample_rows.append(
                    evaluate_method(
                        method,
                        group_name,
                        digits,
                        sequence,
                        raw_length,
                        normalized_raw_length,
                        max(args.time_repeat, 1),
                    )
                )

    overall_rows = list(sample_rows)
    for row in sample_rows:
        cloned = dict(row)
        cloned["group"] = "ALL"
        overall_rows.append(cloned)

    group_summary = aggregate_rows(sample_rows, ("group", "method", "category"))
    overall_summary = aggregate_rows(overall_rows, ("group", "method", "category"))
    base_vs_dp = compare_base_vs_dp(overall_summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "sample_method_details.csv", sample_rows)
    write_csv(args.output_dir / "group_method_summary.csv", group_summary)
    write_csv(args.output_dir / "overall_method_summary.csv", overall_summary)
    write_csv(args.output_dir / "base_vs_dp_summary.csv", base_vs_dp)

    unavailable_rows = [{"method": method.name, "note": method.note} for method in unavailable_methods]
    if unavailable_rows:
        write_csv(args.output_dir / "unavailable_methods.csv", unavailable_rows)

    summary_text = build_markdown_summary(group_summary, overall_summary, base_vs_dp)
    (args.output_dir / "exp2_results_summary.md").write_text(summary_text, encoding="utf-8")
    print(f"Wrote: {args.output_dir}")
    if unavailable_methods:
        for method in unavailable_methods:
            print(f"Skipped {method.name}: {method.note}")


if __name__ == "__main__":
    main()
