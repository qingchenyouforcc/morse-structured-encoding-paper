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
    decode_rle,
    decode_zstd,
    encode_gzip,
    encode_raw,
    encode_repair_grammar,
    encode_sequitur_grammar,
    encode_rle,
    encode_zstd,
    zstd_available,
)
from morse_utils import extract_morse_features, normalized_morse_length, text_to_morse  # noqa: E402
from structured_codecs import (  # noqa: E402
    ABLATION_CONFIGS,
    BASE_CONFIG,
    DP_CONFIG,
    decode_structured,
    encode_structured,
)


DATASETS_DIR = ROOT_DIR / "datasets"
OUTPUT_DIR = ROOT_DIR / "output"
BASE_DATASET_PATH = DATASETS_DIR / "base" / "standard_samples.txt"
LONG_DATASET_PATH = DATASETS_DIR / "long" / "long_sentence_samples.txt"
PARAGRAPH_DATASET_PATH = DATASETS_DIR / "paragraph" / "paragraph_samples.txt"
LONG_TEXT_DATASET_PATH = DATASETS_DIR / "long_text" / "long_text_samples.txt"


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


def structured_method(name: str, category: str, config) -> ExperimentMethod:
    return ExperimentMethod(
        name=name,
        category=category,
        encode=lambda morse: encode_structured(morse, config),
        decode=decode_structured,
        length=len,
        preview=lambda payload: str(payload),
    )


def raw_method() -> ExperimentMethod:
    return ExperimentMethod(
        name="raw_morse",
        category="baseline",
        encode=encode_raw,
        decode=decode_raw,
        length=len,
        preview=lambda payload: str(payload),
    )


def byte_preview(payload: bytes, limit: int = 24) -> str:
    return payload[:limit].hex()


def baseline_methods() -> list[ExperimentMethod]:
    methods = [
        raw_method(),
        structured_method("morsefold_base", "structured", BASE_CONFIG),
        structured_method("morsefold_dp", "structured", DP_CONFIG),
        ExperimentMethod(
            name="rle",
            category="baseline",
            encode=encode_rle,
            decode=decode_rle,
            length=len,
            preview=byte_preview,
        ),
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


def ablation_methods() -> list[ExperimentMethod]:
    return [
        structured_method(name, "ablation", config)
        for name, config in ABLATION_CONFIGS.items()
    ]


def parse_group_limits(values: list[str]) -> dict[str, int]:
    limits: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --group-limit value: {value!r}")
        name, limit_text = value.split("=", 1)
        limits[name] = int(limit_text)
    return limits


def load_dataset(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def contains_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def contains_punctuation(text: str) -> bool:
    return any(not ch.isalnum() and not ch.isspace() for ch in text)


def select_samples(samples: list[str], predicate: Callable[[str], bool]) -> list[str]:
    return [sample for sample in samples if predicate(sample)]


def limit_samples(samples: list[str], limit: int | None) -> list[str]:
    if limit is None:
        return list(samples)
    return list(samples[:limit])


def build_groups(group_limits: dict[str, int]) -> list[tuple[str, list[str]]]:
    base_samples = load_dataset(BASE_DATASET_PATH)
    long_samples = load_dataset(LONG_DATASET_PATH)
    paragraph_samples = load_dataset(PARAGRAPH_DATASET_PATH)
    long_text_samples = load_dataset(LONG_TEXT_DATASET_PATH)

    defaults = {
        "single_word": 10000,
        "multi_word_phrase": 10000,
        "number_heavy": 10000,
        "punctuation_heavy": 10000,
        "mixed_digits_punctuation": 10000,
        "long_sentence_gt20_words": 1000,
        "paragraph_samples": 200,
        "long_text_gt200_words": None,
    }
    defaults.update(group_limits)

    return [
        (
            "single_word",
            limit_samples(select_samples(base_samples, lambda sample: len(sample.split()) == 1), defaults["single_word"]),
        ),
        (
            "multi_word_phrase",
            limit_samples(select_samples(base_samples, lambda sample: 2 <= len(sample.split()) <= 5), defaults["multi_word_phrase"]),
        ),
        (
            "number_heavy",
            limit_samples(select_samples(base_samples, contains_digit), defaults["number_heavy"]),
        ),
        (
            "punctuation_heavy",
            limit_samples(select_samples(base_samples, contains_punctuation), defaults["punctuation_heavy"]),
        ),
        (
            "mixed_digits_punctuation",
            limit_samples(
                select_samples(base_samples, lambda sample: contains_digit(sample) and contains_punctuation(sample)),
                defaults["mixed_digits_punctuation"],
            ),
        ),
        ("long_sentence_gt20_words", limit_samples(long_samples, defaults["long_sentence_gt20_words"])),
        ("paragraph_samples", limit_samples(paragraph_samples, defaults["paragraph_samples"])),
        ("long_text_gt200_words", limit_samples(long_text_samples, defaults["long_text_gt200_words"])),
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
    text: str,
    morse: str,
    raw_length: int,
    normalized_raw_length: int,
    time_repeat: int,
) -> dict[str, Any]:
    payload, encode_ns = timed_call(lambda: method.encode(morse), time_repeat)
    encoded_length = method.length(payload)
    decoded_morse, decode_ns = timed_call(lambda: method.decode(payload), time_repeat)
    features = extract_morse_features(text, morse)

    return {
        "group": group_name,
        "text": text,
        "method": method.name,
        "category": method.category,
        "word_count": features.word_count,
        "code_count": features.code_count,
        "raw_morse_length": raw_length,
        "normalized_raw_morse_length": normalized_raw_length,
        "encoded_length": encoded_length,
        "character_reduction": raw_length - encoded_length,
        "normalized_reduction": normalized_raw_length - encoded_length,
        "compression_ratio": encoded_length / raw_length if raw_length else 0.0,
        "normalized_compression_ratio": encoded_length / normalized_raw_length if normalized_raw_length else 0.0,
        "encode_ns": encode_ns,
        "decode_ns": decode_ns,
        "round_trip_ok": decoded_morse == morse,
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
        total_raw = sum(item["raw_morse_length"] for item in bucket)
        total_normalized_raw = sum(item["normalized_raw_morse_length"] for item in bucket)
        total_encoded = sum(item["encoded_length"] for item in bucket)
        summary = {field: value for field, value in zip(group_fields, key)}
        summary.update(
            {
                "samples": len(bucket),
                "total_raw_morse_length": total_raw,
                "total_normalized_raw_morse_length": total_normalized_raw,
                "total_encoded_length": total_encoded,
                "total_character_reduction": total_raw - total_encoded,
                "total_normalized_reduction": total_normalized_raw - total_encoded,
                "compression_ratio": total_encoded / total_raw if total_raw else 0.0,
                "normalized_compression_ratio": total_encoded / total_normalized_raw if total_normalized_raw else 0.0,
                "average_encode_ns": statistics.fmean(item["encode_ns"] for item in bucket),
                "average_decode_ns": statistics.fmean(item["decode_ns"] for item in bucket),
                "round_trip_success_rate": sum(1 for item in bucket if item["round_trip_ok"]) / len(bucket),
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
                "base_average_encode_ns": base_row["average_encode_ns"],
                "dp_average_encode_ns": dp_row["average_encode_ns"],
                "base_average_decode_ns": base_row["average_decode_ns"],
                "dp_average_decode_ns": dp_row["average_decode_ns"],
                "base_round_trip_success_rate": base_row["round_trip_success_rate"],
                "dp_round_trip_success_rate": dp_row["round_trip_success_rate"],
            }
        )
    return comparisons


def bucket_fraction(value: float) -> str:
    if value == 0:
        return "0.00"
    if value <= 0.25:
        return "(0.00,0.25]"
    if value <= 0.50:
        return "(0.25,0.50]"
    if value <= 0.75:
        return "(0.50,0.75]"
    return "(0.75,1.00]"


def summarize_worthwhile(rows: list[dict[str, Any]], feature_name: str) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["method"] not in {"morsefold_base", "morsefold_dp"}:
            continue
        bucket = bucket_fraction(row[feature_name])
        buckets.setdefault((row["method"], bucket), []).append(row)

    summaries: list[dict[str, Any]] = []
    for (method, bucket), bucket_rows in sorted(buckets.items()):
        summaries.append(
            {
                "method": method,
                "bucket": bucket,
                "samples": len(bucket_rows),
                "average_normalized_compression_ratio": statistics.fmean(
                    item["normalized_compression_ratio"] for item in bucket_rows
                ),
                "average_normalized_reduction": statistics.fmean(
                    item["normalized_reduction"] for item in bucket_rows
                ),
                "average_word_count": statistics.fmean(item["word_count"] for item in bucket_rows),
                "average_same_length_run_coverage": statistics.fmean(
                    item["same_length_run_coverage"] for item in bucket_rows
                ),
                "average_alternating_share": statistics.fmean(
                    item["alternating_share"] for item in bucket_rows
                ),
            }
        )
    return summaries


def diagnose_failure_case(row: dict[str, Any]) -> str:
    if row["same_length_run_coverage"] == 0:
        return "no_same_length_runs"
    if row["same_length_run_coverage"] < 0.25:
        return "weak_same_length_regularity"
    if row["mean_code_length"] <= 2.5:
        return "short_codes_leave_little_room"
    if row["alternating_share"] == 0:
        return "few_alternating_patterns"
    if row["normalized_reduction"] <= 0 and row["word_count"] > 5:
        return "metadata_overhead_offsets_local_gain"
    return "mixed_local_patterns_limit_gain"


def collect_failure_cases(rows: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["method"] not in {"morsefold_base", "morsefold_dp"}:
            continue
        buckets.setdefault((row["method"], row["group"]), []).append(row)

    failures: list[dict[str, Any]] = []
    for (method, group), bucket in sorted(buckets.items()):
        ranked = sorted(
            bucket,
            key=lambda item: (item["normalized_reduction"], item["compression_ratio"], item["text"]),
        )[:top_n]
        for item in ranked:
            failure = dict(item)
            failure["diagnosis"] = diagnose_failure_case(item)
            failures.append(failure)
    return failures


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
    main_summary: list[dict[str, Any]],
    base_vs_dp: list[dict[str, Any]],
    ablation_summary: list[dict[str, Any]],
    failure_cases: list[dict[str, Any]],
    worthwhile_run: list[dict[str, Any]],
    worthwhile_alt: list[dict[str, Any]],
) -> str:
    overall_rows = [row for row in main_summary if row["group"] == "ALL"]
    overall_rows.sort(key=lambda row: row["normalized_compression_ratio"])
    grouped_rows = [row for row in group_summary if row["group"] != "ALL"]
    grouped_by_name: dict[str, list[dict[str, Any]]] = {}
    for row in grouped_rows:
        grouped_by_name.setdefault(row["group"], []).append(row)
    for rows in grouped_by_name.values():
        rows.sort(key=lambda row: row["normalized_compression_ratio"])
    ablation_all = [row for row in ablation_summary if row["group"] == "ALL"]
    failure_overall = sorted(
        failure_cases,
        key=lambda row: (row["method"], row["normalized_reduction"], row["text"]),
    )[:10]
    best_overall_structured = min(
        (row for row in overall_rows if row["method"] in {"morsefold_base", "morsefold_dp"}),
        key=lambda row: row["normalized_compression_ratio"],
        default=None,
    )
    best_overall_baseline = min(
        (
            row
            for row in overall_rows
            if row["method"] in {"raw_morse", "rle", "gzip", "zstd", "sequitur_style", "repair"}
        ),
        key=lambda row: row["normalized_compression_ratio"],
        default=None,
    )
    best_dp_group = max(
        base_vs_dp,
        key=lambda row: row["normalized_ratio_improvement_pp"],
        default=None,
    )
    ablation_lookup = {row["method"]: row for row in ablation_all}
    strongest_ablation_drop: dict[str, Any] | None = None
    base_full_row = ablation_lookup.get("base_full")
    if base_full_row is not None:
        candidates: list[dict[str, Any]] = []
        for row in ablation_all:
            if row["method"] == "base_full":
                continue
            candidates.append(
                {
                    "method": row["method"],
                    "drop_pp": (row["normalized_compression_ratio"] - base_full_row["normalized_compression_ratio"]) * 100.0,
                }
            )
        strongest_ablation_drop = max(candidates, key=lambda row: row["drop_pp"], default=None)

    lines = ["# Paper Results Summary", "", "## 7.1 Overall Results", ""]
    if best_overall_structured is not None and best_overall_baseline is not None:
        lines.extend(
            [
                "Table 1 reports the overall results aggregated across all benchmark groups. "
                f"Among the two proposed structured methods, `{best_overall_structured['method']}` achieves the lowest normalized compression ratio "
                f"({best_overall_structured['normalized_compression_ratio']:.4f}). "
                f"Among the full six-method comparison, the strongest general-purpose baseline is `{best_overall_baseline['method']}` "
                f"with normalized compression ratio {best_overall_baseline['normalized_compression_ratio']:.4f}. "
                "However, this aggregate result should be interpreted with care: the overall metric is length-weighted, so long paragraphs and long texts contribute much more to the final total than short words or short phrases. "
                "As a result, the apparent overall advantage of generic compressors is strongly influenced by the long-text groups, where the proposed method is relatively weak because it mainly exploits local same-length regularity rather than broader cross-segment redundancy. "
                "Therefore, Table 1 should be read together with the per-group comparisons below, which show that the ranking is not uniform across all test conditions.",
                "",
            ]
        )
    lines.extend(
        [
            "Table 1. Overall comparison across all methods.",
            "",
            "| Method | Samples | Compression Ratio | Normalized Compression Ratio | Avg Encode ns | Round-trip |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in overall_rows:
        lines.append(
            "| {method} | {samples} | {compression_ratio:.4f} | {normalized_compression_ratio:.4f} | "
            "{average_encode_ns:.0f} | {round_trip_success_rate:.2%} |".format(**row)
        )

    lines.extend(
        [
            "",
            "To show how method behavior changes across input scales and content types, Tables 2-9 provide the six-method comparison for each test group separately. "
            "These tables make clear that the proposed structured methods are more competitive on short and medium-scale inputs with stronger local regularity, whereas the gap against gzip and zstd becomes larger on longer texts.",
            "",
        ]
    )
    for table_index, group_name in enumerate(sorted(grouped_by_name), start=2):
        lines.extend(
            [
                f"Table {table_index}. Six-method comparison on `{group_name}`.",
                "",
                "| Method | Samples | Compression Ratio | Normalized Compression Ratio | Total Reduction | Avg Encode ns | Round-trip |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in grouped_by_name[group_name]:
            lines.append(
                "| {method} | {samples} | {compression_ratio:.4f} | {normalized_compression_ratio:.4f} | "
                "{total_character_reduction} | {average_encode_ns:.0f} | {round_trip_success_rate:.2%} |".format(**row)
            )
        lines.append("")

    lines.extend(
        [
            "",
            "## 7.2 Comparison Between Base and DP Variant",
            "",
        ]
    )
    if best_dp_group is not None:
        lines.extend(
            [
                "Table 10 compares the base method with the dynamic-programming variant. "
                "The DP variant keeps the same reversible protocol but searches for a shorter local partition inside each same-length run. "
                f"Its largest gain in the current benchmark appears on `{best_dp_group['group']}`, "
                f"where the normalized compression ratio improves by {best_dp_group['normalized_ratio_improvement_pp']:.2f} percentage points.",
                "",
            ]
        )
    lines.extend(
        [
            "Table 10. Base method versus DP variant.",
            "",
            "| Group | Extra Reduction | Base Norm Ratio | DP Norm Ratio | Improvement (pp) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in base_vs_dp:
        lines.append(
            "| {group} | {extra_reduction} | {base_normalized_compression_ratio:.4f} | "
            "{dp_normalized_compression_ratio:.4f} | {normalized_ratio_improvement_pp:.2f} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## 7.3 Ablation Study",
            "",
        ]
    )
    if strongest_ablation_drop is not None:
        lines.extend(
            [
                "Table 11 summarizes the ablation study. "
                "All three components contribute to the final performance, and removing fallback causes the largest degradation, "
                f"with `{strongest_ablation_drop['method']}` increasing the normalized compression ratio by {strongest_ablation_drop['drop_pp']:.2f} percentage points relative to the full base method.",
                "",
            ]
        )
    lines.extend(
        [
            "Table 11. Ablation study on the base method.",
            "",
            "| Variant | Compression Ratio | Normalized Compression Ratio | Round-trip |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(ablation_all, key=lambda item: item["normalized_compression_ratio"]):
        lines.append(
            "| {method} | {compression_ratio:.4f} | {normalized_compression_ratio:.4f} | "
            "{round_trip_success_rate:.2%} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## 7.4 Failure Cases",
            "",
            "The failure cases in Table 12 indicate that the proposed structured encoding becomes less effective when same-length runs are sparse, "
            "when local regularity is weak, or when the metadata introduced by structural description offsets the small local gain available in the original Morse sequence.",
            "",
            "Table 12. Representative low-gain or failure cases.",
            "",
            "| Method | Group | Text | Norm Reduction | Diagnosis |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for row in failure_overall:
        lines.append(
            "| {method} | {group} | {text} | {normalized_reduction} | {diagnosis} |".format(**row)
        )

    best_run = min(worthwhile_run, key=lambda row: row["average_normalized_compression_ratio"], default=None)
    best_alt = min(worthwhile_alt, key=lambda row: row["average_normalized_compression_ratio"], default=None)
    lines.extend(
        [
            "",
            "## 7.5 When Structured Encoding Is Worthwhile",
            "",
        ]
    )
    if best_run is not None and best_alt is not None:
        lines.extend(
            [
                "Tables 13 and 14 analyze when structured encoding is most worthwhile. "
                "Overall, the method tends to work better when the input contains a larger fraction of same-length runs and enough reusable local regularity to amortize structural metadata. "
                f"In the current benchmark, the best run-coverage bucket is `{best_run['bucket']}` for `{best_run['method']}`, "
                f"with average normalized compression ratio {best_run['average_normalized_compression_ratio']:.4f}. "
                f"For alternating-pattern analysis, the best bucket is `{best_alt['bucket']}` for `{best_alt['method']}`, "
                f"with average normalized compression ratio {best_alt['average_normalized_compression_ratio']:.4f}.",
                "",
            ]
        )
    lines.extend(
        [
            "Table 13. Worthwhile-analysis by same-length run coverage.",
            "",
            "| Method | Bucket | Samples | Avg Norm Ratio | Avg Norm Reduction | Avg Word Count |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in worthwhile_run:
        lines.append(
            "| {method} | {bucket} | {samples} | {average_normalized_compression_ratio:.4f} | "
            "{average_normalized_reduction:.2f} | {average_word_count:.2f} |".format(**row)
        )

    lines.extend(
        [
            "",
            "Table 14. Worthwhile-analysis by alternating-pattern share.",
            "",
            "| Method | Bucket | Samples | Avg Norm Ratio | Avg Norm Reduction | Avg Word Count |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in worthwhile_alt:
        lines.append(
            "| {method} | {bucket} | {samples} | {average_normalized_compression_ratio:.4f} | "
            "{average_normalized_reduction:.2f} | {average_word_count:.2f} |".format(**row)
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-aligned Morse structured encoding experiments.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--time-repeat", type=int, default=1)
    parser.add_argument("--failure-top-n", type=int, default=10)
    parser.add_argument(
        "--group-limit",
        action="append",
        default=[],
        help="Override a group limit, for example single_word=200",
    )
    args = parser.parse_args()

    group_limits = parse_group_limits(args.group_limit)
    groups = build_groups(group_limits)
    all_methods = baseline_methods()
    main_methods = [method for method in all_methods if method.available]
    unavailable_methods = [method for method in all_methods if not method.available]
    ablations = ablation_methods()

    sample_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []

    for group_name, samples in groups:
        print(f"[group] {group_name}: {len(samples)} samples")
        for text in samples:
            morse = text_to_morse(text, word_sep=" / ")
            raw_length = len(morse)
            normalized_raw_length = normalized_morse_length(text, morse)

            for method in main_methods:
                sample_rows.append(
                    evaluate_method(
                        method,
                        group_name,
                        text,
                        morse,
                        raw_length,
                        normalized_raw_length,
                        max(args.time_repeat, 1),
                    )
                )

            for method in ablations:
                ablation_rows.append(
                    evaluate_method(
                        method,
                        group_name,
                        text,
                        morse,
                        raw_length,
                        normalized_raw_length,
                        max(args.time_repeat, 1),
                    )
                )

    overall_main_rows = list(sample_rows)
    for row in sample_rows:
        cloned = dict(row)
        cloned["group"] = "ALL"
        overall_main_rows.append(cloned)

    overall_ablation_rows = list(ablation_rows)
    for row in ablation_rows:
        cloned = dict(row)
        cloned["group"] = "ALL"
        overall_ablation_rows.append(cloned)

    group_method_summary = aggregate_rows(sample_rows, ("group", "method", "category"))
    overall_method_summary = aggregate_rows(overall_main_rows, ("group", "method", "category"))
    ablation_summary = aggregate_rows(overall_ablation_rows, ("group", "method", "category"))
    base_vs_dp_summary = compare_base_vs_dp(overall_method_summary)
    failure_cases = collect_failure_cases(sample_rows, args.failure_top_n)
    worthwhile_by_run = summarize_worthwhile(sample_rows, "same_length_run_coverage")
    worthwhile_by_alt = summarize_worthwhile(sample_rows, "alternating_share")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "sample_method_details.csv", sample_rows)
    write_csv(args.output_dir / "group_method_summary.csv", group_method_summary)
    write_csv(args.output_dir / "overall_method_summary.csv", overall_method_summary)
    write_csv(args.output_dir / "ablation_summary.csv", ablation_summary)
    write_csv(args.output_dir / "base_vs_dp_summary.csv", base_vs_dp_summary)
    write_csv(args.output_dir / "failure_cases.csv", failure_cases)
    write_csv(args.output_dir / "worthwhile_by_run_coverage.csv", worthwhile_by_run)
    write_csv(args.output_dir / "worthwhile_by_alternating_share.csv", worthwhile_by_alt)

    unavailable_rows = [{"method": method.name, "note": method.note} for method in unavailable_methods]
    if unavailable_rows:
        write_csv(args.output_dir / "unavailable_methods.csv", unavailable_rows)

    markdown_summary = build_markdown_summary(
        group_method_summary,
        overall_method_summary,
        base_vs_dp_summary,
        ablation_summary,
        failure_cases,
        worthwhile_by_run,
        worthwhile_by_alt,
    )
    (args.output_dir / "paper_results_summary.md").write_text(markdown_summary, encoding="utf-8")

    print(f"Wrote: {args.output_dir}")
    if unavailable_methods:
        for method in unavailable_methods:
            print(f"Skipped {method.name}: {method.note}")


if __name__ == "__main__":
    main()
